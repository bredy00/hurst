"""
The Markov decision process itself: absorbing states, the action-value (return) function,
and the argmax that turns it into a policy (Session M).

A finite MDP is (S, A, P, R, gamma) with a set of ABSORBING states. Two objects matter:

    V(s)     = max_a Q(s, a)                      the value of a state
    Q(s, a)  = R(s, a) + gamma sum_s' P(s'|s, a) V(s')

and the policy is pi(s) = argmax_a Q(s, a) -- the action whose (state, action) pair goes on
to generate the most. `value_iteration` computes Q exactly for a tabular MDP, which is what
every learned agent in this package is scored against: on a problem small enough to solve
exactly there is no excuse for reporting an agent's return without the optimum beside it.

**Absorbing states** are carried explicitly rather than encoded as a self-loop with zero
reward. Both are correct, but the explicit version makes the two commonest mistakes
impossible to write: an absorbing state whose value keeps accumulating (a self-loop with a
non-zero reward), and a terminal reward counted twice (once on entry, once on the loop).
Here Q(s, a) = 0 for every absorbing s and every a, the terminal payoff is earned on the
TRANSITION INTO s, and `value_iteration` asserts it.

Everything is exact and tabular. That is the point: this module is the answer key, not the
learner. The learners live in `rl/agents.py`.
"""

import numpy as np

import rl


def greedy(Q, tol=0.0):
    """
    The argmax policy, with deterministic tie-breaking (the lowest action index wins).

    `tol` treats actions within `tol` of the best as tied, which matters when Q comes from a
    fitted model: reporting a policy that flips between two indistinguishable actions makes
    the policy look unstable when it is the estimate that is.
    """
    Q = np.asarray(Q, dtype=float)
    best = Q.max(axis=-1, keepdims=True)
    return np.argmax(Q >= best - tol, axis=-1)


def policy_value(Q, pi):
    """V under a given (deterministic) policy, read off Q."""
    Q = np.asarray(Q, dtype=float)
    return np.take_along_axis(Q, np.asarray(pi, int)[..., None], axis=-1)[..., 0]


class TabularMDP:
    """
    A finite MDP with explicit absorbing states.

    P       (n_states, n_actions, n_states) transition probabilities; rows over s' sum to 1
    R       (n_states, n_actions) expected reward earned on LEAVING (s, a)
    absorb  (n_states,) bool; no action from an absorbing state earns or transitions
    gamma   discount in (0, 1], 1 allowed only when every episode reaches an absorbing state
    """

    def __init__(self, P, R, absorbing, gamma=1.0, action_names=None, state_names=None):
        self.P = np.asarray(P, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.absorbing = np.asarray(absorbing, dtype=bool)
        self.gamma = float(gamma)
        n_s, n_a = self.R.shape
        if self.P.shape != (n_s, n_a, n_s):
            raise ValueError(f"P must be {(n_s, n_a, n_s)}, got {self.P.shape}")
        if self.absorbing.shape != (n_s,):
            raise ValueError(f"absorbing must be ({n_s},), got {self.absorbing.shape}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        live = ~self.absorbing
        row = self.P[live].sum(axis=-1)
        if live.any() and not np.allclose(row, 1.0, atol=1e-9):
            raise ValueError(f"transition rows must sum to 1; worst {np.max(np.abs(row - 1.0)):.2e}")
        if not self.absorbing.any():
            raise ValueError("an MDP with no absorbing state: give one, or set gamma < 1 deliberately "
                             "and pass absorbing=[False]*n with a self-loop you have checked")
        self.action_names = tuple(action_names) if action_names is not None else tuple(range(n_a))
        self.state_names = tuple(state_names) if state_names is not None else None

    @property
    def n_states(self):
        return self.R.shape[0]

    @property
    def n_actions(self):
        return self.R.shape[1]

    def value_iteration(self, tol=1e-12, max_iter=100_000):
        """
        (Q, V, pi, info) by value iteration. Exact to `tol` in the sup norm.

        With gamma = 1 the contraction argument needs every episode to reach an absorbing
        state; that is checked here (from every live state, the absorbing set must be
        reachable) rather than assumed, because the failure mode is a silent divergence.
        """
        if self.gamma == 1.0:
            self._check_absorbing_reachable()
        n_s = self.n_states
        V = np.zeros(n_s)
        live = ~self.absorbing
        for it in range(1, max_iter + 1):
            Q = self.R + self.gamma * (self.P @ V)
            Q[self.absorbing] = 0.0                      # no action from an absorbing state
            Vn = np.where(live, Q.max(axis=1), 0.0)
            gap = float(np.max(np.abs(Vn - V)))
            V = Vn
            if gap <= tol:
                break
        Q = self.R + self.gamma * (self.P @ V)
        Q[self.absorbing] = 0.0
        assert np.all(Q[self.absorbing] == 0.0), "absorbing states must have zero value"
        return Q, V, greedy(Q), {"iterations": it, "residual": gap}

    def policy_evaluation(self, pi, tol=1e-12, max_iter=100_000):
        """V^pi for a deterministic policy, by iteration (no linear solve, so n can be large)."""
        pi = np.asarray(pi, int)
        n_s = self.n_states
        Ppi = self.P[np.arange(n_s), pi]
        Rpi = np.where(self.absorbing, 0.0, self.R[np.arange(n_s), pi])
        V = np.zeros(n_s)
        for _ in range(max_iter):
            Vn = np.where(self.absorbing, 0.0, Rpi + self.gamma * (Ppi @ V))
            if float(np.max(np.abs(Vn - V))) <= tol:
                return Vn
            V = Vn
        return V

    def _check_absorbing_reachable(self):
        """Every live state must reach the absorbing set under SOME action, or gamma = 1 diverges."""
        reach = self.absorbing.copy()
        for _ in range(self.n_states + 1):
            nxt = reach | np.any(self.P[:, :, reach].sum(axis=-1) > 0, axis=1)
            if np.array_equal(nxt, reach):
                break
            reach = nxt
        if not reach.all():
            bad = int(np.flatnonzero(~reach)[0])
            raise ValueError(f"gamma = 1 but state {bad} cannot reach an absorbing state: value "
                             "iteration would not converge. Set gamma < 1 or fix the transitions.")

    def describe_policy(self, pi, values=None, max_rows=20):
        """The policy as readable lines -- an RL result nobody can read is not a result."""
        pi = np.asarray(pi, int)
        rows = []
        idx = np.flatnonzero(~self.absorbing)[:max_rows]
        for s in idx:
            name = self.state_names[s] if self.state_names else f"state {s}"
            act = self.action_names[pi[s]]
            v = f"  (value {values[s]:+.4f})" if values is not None else ""
            rows.append(f"{name}: {act}{v}")
        return rows


def from_transitions(states, actions, rewards, next_states, n_states, n_actions, absorbing,
                     gamma=1.0):
    """
    Build a TabularMDP by counting logged transitions -- the maximum-likelihood model.

    Unvisited (s, a) pairs get zero reward and a self-transition, which makes them look
    neutral; `info["unvisited"]` counts them, and a policy that chooses one is a policy
    fitted on no data. Checking that count is the caller's job and the studies do.
    """
    s = np.asarray(states, int)
    a = np.asarray(actions, int)
    r = np.asarray(rewards, float)
    sp = np.asarray(next_states, int)
    absorbing = np.asarray(absorbing, bool)
    counts = np.zeros((n_states, n_actions, n_states))
    rsum = np.zeros((n_states, n_actions))
    n = np.zeros((n_states, n_actions))
    np.add.at(counts, (s, a, sp), 1.0)
    np.add.at(rsum, (s, a), r)
    np.add.at(n, (s, a), 1.0)
    seen = n > 0
    P = np.where(seen[..., None], counts / np.maximum(n, 1)[..., None], 0.0)
    for i in np.flatnonzero(~seen.ravel()):
        si, ai = divmod(int(i), n_actions)
        P[si, ai, si] = 1.0
    R = np.where(seen, rsum / np.maximum(n, 1), 0.0)
    R[absorbing] = 0.0
    mdp = TabularMDP(P, R, absorbing, gamma)
    return mdp, {"visits": n, "unvisited": int((~seen & ~absorbing[:, None]).sum())}


def rollout_return(rewards, gamma=1.0):
    """The discounted return of one episode's reward sequence."""
    r = np.asarray(rewards, float)
    if gamma == 1.0:
        return float(r.sum())
    return float(r @ gamma ** np.arange(len(r)))


def require():
    rl.require_enabled("the MDP framework")
