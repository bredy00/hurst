"""
The learners (Session M). Each one produces an action-value function Q(s, a); the policy is
always `rl.mdp.greedy(Q)`, the argmax over actions.

  "fitted-q"    Fitted Q iteration (Ernst, Geurts & Wehenkel 2005) with a LINEAR basis per
                action -- batch, offline, deterministic given its data. The Bellman target
                y = r + gamma max_a' Q(s', a') is recomputed each sweep and the per-action
                regressions refitted. Linear because a policy you cannot read is a policy you
                cannot check against the analytic one, and because on these problems the
                optimal Q is close to linear in a well-chosen basis -- which is itself a
                finding worth reporting rather than hiding under a network.
  "lspi"        Least-squares policy iteration (Lagoudakis & Parr 2003): the same basis, but
                Q^pi solved directly from the projected Bellman equation (LSTDQ) and the
                policy improved, rather than iterated to a fixed point. Fewer sweeps, and
                the linear system exposes its own conditioning.
  "tabular-q"   Watkins Q-learning on a discretised state. Slow and noisy, kept because it
                is the textbook object and because it converges to the exact tabular answer,
                so it checks the other two on a problem small enough to solve both ways.

All three are offline: they learn from a fixed batch of logged transitions and never
interact with a simulator. That is deliberate. Online exploration would let an agent visit
states the analytic baseline never sees, and then the comparison is not like for like.

Every agent exposes the same three things:

    fit(batch) -> self        batch is an rl.agents.Batch
    q(features) -> (n, n_actions)
    policy(features) -> (n,)  argmax, via rl.mdp.greedy
"""

from dataclasses import dataclass

import numpy as np

import rl
from rl.mdp import greedy


@dataclass
class Batch:
    """
    Logged transitions. `phi` and `phi_next` are the state features; `action` indexes the
    action set; `absorbing` marks transitions INTO an absorbing state, where the bootstrap
    term is dropped and the reward is the whole target.
    """
    phi: np.ndarray            # (n, k)
    action: np.ndarray         # (n,) int
    reward: np.ndarray         # (n,)
    phi_next: np.ndarray       # (n, k)
    absorbing: np.ndarray      # (n,) bool -- the NEXT state is absorbing
    n_actions: int
    episode: np.ndarray = None  # (n,) int, for the Markov test and for grouped scoring
    step: np.ndarray = None     # (n,) int, the time index WITHIN the episode. Needed by the
                                # Markov test whenever the rows are not in episode-then-time
                                # order, which a date-major batch never is.

    def __post_init__(self):
        self.phi = np.atleast_2d(np.asarray(self.phi, float))
        self.phi_next = np.atleast_2d(np.asarray(self.phi_next, float))
        self.action = np.asarray(self.action, int)
        self.reward = np.asarray(self.reward, float)
        self.absorbing = np.asarray(self.absorbing, bool)
        n = len(self.reward)
        for name, arr in (("phi", self.phi), ("phi_next", self.phi_next), ("action", self.action),
                          ("absorbing", self.absorbing)):
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows against {n} rewards")
        if self.action.min(initial=0) < 0 or (n and self.action.max(initial=0) >= self.n_actions):
            raise ValueError(f"actions must be in [0, {self.n_actions})")
        if self.episode is None:
            self.episode = np.zeros(n, int)
        self.episode = np.asarray(self.episode, int)
        if self.step is None:
            self.step = np.zeros(n, int)
            for e in np.unique(self.episode):
                idx = np.flatnonzero(self.episode == e)
                self.step[idx] = np.arange(len(idx))
        self.step = np.asarray(self.step, int)

    def __len__(self):
        return len(self.reward)

    @property
    def k(self):
        return self.phi.shape[1]


def _ridge(X, y, lam):
    """Ridge solve with the intercept unpenalised (column 0 is assumed constant)."""
    k = X.shape[1]
    A = X.T @ X + lam * np.eye(k)
    A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ y)


class _Scaler:
    """
    Standardise every non-constant feature column to unit scale.

    Not cosmetic. These bases mix a delta in [0, 1] with an m^2 in [0, 64], and the normal
    equations of the block design inherit the square of that spread: unscaled, LSPI's system
    came back at a condition number of 1e17, which is a numerically singular solve reported
    as a policy. Scaling is applied inside `fit` and folded back into the weights, so the
    caller's features and the stored W stay in the original units and remain readable.
    """

    def fit(self, X):
        sd = X.std(axis=0)
        const = sd <= 1e-12 * max(float(np.abs(X).max()), 1e-300)
        self.scale = np.where(const, 1.0, np.where(sd > 0, sd, 1.0))
        return self

    def __call__(self, X):
        return X / self.scale


@rl.register_agent("fitted-q")
class FittedQ:
    """
    Fitted Q iteration with one linear model per action.

    `sweeps` Bellman sweeps; `ridge` regularises each per-action regression, which matters
    because an action the logging policy rarely chose has few rows. `converged` reports the
    last sweep's change in Q, so a run that did not settle says so.
    """

    def __init__(self, sweeps=40, gamma=1.0, ridge=1e-6, tol=1e-10):
        self.sweeps, self.gamma, self.ridge, self.tol = int(sweeps), float(gamma), float(ridge), float(tol)
        self.W = None
        self.info = {}

    def fit(self, batch):
        rl.require_enabled("the fitted-Q agent")
        n, k = len(batch), batch.k
        self._sc = _Scaler().fit(batch.phi)
        phi, phi_next = self._sc(batch.phi), self._sc(batch.phi_next)
        self.W = np.zeros((batch.n_actions, k))
        rows = [np.flatnonzero(batch.action == a) for a in range(batch.n_actions)]
        self.info["rows_per_action"] = [len(r) for r in rows]
        live = ~batch.absorbing
        gap = float("inf")
        for sweep in range(1, self.sweeps + 1):
            qn = phi_next @ self.W.T                           # (n, n_actions)
            target = batch.reward.copy()
            if live.any():
                target[live] += self.gamma * qn[live].max(axis=1)
            W_new = self.W.copy()
            for a, idx in enumerate(rows):
                if len(idx) > k:
                    W_new[a] = _ridge(phi[idx], target[idx], self.ridge)
            gap = float(np.max(np.abs(W_new - self.W))) if n else 0.0
            self.W = W_new
            if gap <= self.tol:
                break
        self.info.update(sweeps=sweep, last_change=gap, converged=bool(gap <= self.tol))
        return self

    def q(self, phi):
        return self._sc(np.atleast_2d(np.asarray(phi, float))) @ self.W.T

    def policy(self, phi):
        return greedy(self.q(phi))


@rl.register_agent("lspi")
class LSPI:
    """
    Least-squares policy iteration: LSTDQ for Q^pi, then a greedy step, repeated.

    The basis is block-diagonal over actions (k * n_actions weights), so LSTDQ is one linear
    solve of that size per iteration. `condition` records the system's condition number: an
    LSPI result from an ill-conditioned A is the numerical equivalent of an unvisited state.
    """

    def __init__(self, iters=20, gamma=1.0, ridge=1e-6, tol=1e-10):
        self.iters, self.gamma, self.ridge, self.tol = int(iters), float(gamma), float(ridge), float(tol)
        self.W = None
        self.info = {}

    @staticmethod
    def _block(phi, action, n_actions):
        n, k = phi.shape
        out = np.zeros((n, k * n_actions))
        out[np.arange(n)[:, None], action[:, None] * k + np.arange(k)[None, :]] = phi
        return out

    def fit(self, batch):
        rl.require_enabled("the LSPI agent")
        k, n_a = batch.k, batch.n_actions
        self._sc = _Scaler().fit(batch.phi)
        phi, phi_next_s = self._sc(batch.phi), self._sc(batch.phi_next)
        self.W = np.zeros((n_a, k))
        Phi = self._block(phi, batch.action, n_a)
        live = ~batch.absorbing
        conds = []
        for it in range(1, self.iters + 1):
            pi_next = greedy(phi_next_s @ self.W.T)
            Phi_next = self._block(phi_next_s, pi_next, n_a)
            Phi_next[~live] = 0.0                              # absorbing: no bootstrap
            A = Phi.T @ (Phi - self.gamma * Phi_next) + self.ridge * np.eye(k * n_a)
            b = Phi.T @ batch.reward
            conds.append(float(np.linalg.cond(A)))
            w = np.linalg.solve(A, b).reshape(n_a, k)
            gap = float(np.max(np.abs(w - self.W)))
            self.W = w
            if gap <= self.tol:
                break
        self.info.update(iterations=it, last_change=gap, condition=max(conds),
                         converged=bool(gap <= self.tol))
        return self

    def q(self, phi):
        return self._sc(np.atleast_2d(np.asarray(phi, float))) @ self.W.T

    def policy(self, phi):
        return greedy(self.q(phi))


@rl.register_agent("tabular-q")
class TabularQ:
    """
    Watkins Q-learning over a discretised state -- the textbook learner, kept as a check on
    the linear ones where the problem is small enough to solve exactly.

    `discretiser(phi) -> int` maps features to a state index. The learning rate is
    1 / (1 + visits)^alpha, which satisfies the Robbins-Monro conditions at alpha in
    (0.5, 1], so the tabular answer is the limit rather than a hope.
    """

    def __init__(self, n_states, discretiser, gamma=1.0, alpha=0.7, epochs=40, seed=0):
        self.n_states, self.discretiser = int(n_states), discretiser
        self.gamma, self.alpha, self.epochs, self.seed = float(gamma), float(alpha), int(epochs), int(seed)
        self.Q = None
        self.info = {}

    def fit(self, batch):
        rl.require_enabled("the tabular-Q agent")
        rng = np.random.default_rng(self.seed)
        self.Q = np.zeros((self.n_states, batch.n_actions))
        visits = np.zeros((self.n_states, batch.n_actions))
        s = np.asarray([self.discretiser(p) for p in batch.phi], int)
        sp = np.asarray([self.discretiser(p) for p in batch.phi_next], int)
        n = len(batch)
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                a = batch.action[i]
                visits[s[i], a] += 1
                lr = 1.0 / (1.0 + visits[s[i], a]) ** self.alpha
                boot = 0.0 if batch.absorbing[i] else self.gamma * self.Q[sp[i]].max()
                self.Q[s[i], a] += lr * (batch.reward[i] + boot - self.Q[s[i], a])
        self.info.update(epochs=self.epochs, unvisited=int((visits == 0).sum()),
                         states=self.n_states)
        return self

    def q(self, phi):
        idx = np.asarray([self.discretiser(p) for p in np.atleast_2d(phi)], int)
        return self.Q[idx]

    def policy(self, phi):
        return greedy(self.q(phi))
