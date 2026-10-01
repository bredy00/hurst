"""
Hedging with proportional transaction costs -- the first problem here that is a genuine MDP
(Session O).

Session M's two problems were bandits: the writer's hedge does not move the market, so the next
state did not depend on the action and bootstrapping could only add noise. Charge for trading and
that changes. The holding carried into the next date IS this date's action, and what trading
costs then depends on it. The optimum is a no-trade band around the frictionless hedge -- a
threshold rule -- which is the problem docs/rl-framework.md left open as the first one an agent
could win.

**The problem.** A one-month at-the-money call written at S0 = K = 1 under Black-Scholes (sigma
known, zero rates, zero drift), rebalanced at n dates. Holding h_k into date k, the writer
chooses a_k, the holding over the next interval, and books

    r_k = - cost * S_k * |a_k - h_k|  -  (risk / 2) * (dC_k - a_k dS_k)^2,        h_{k+1} = a_k,

with C the Black-Scholes value, a martingale here (Session N: the one-step risks add up to the
total only when the mark is one). Summed, the return is minus the total trading cost minus
risk/2 times the realised quadratic variation of the hedging error, whose expectation is the
variance of the terminal hedging error: a mean-variance objective, one number per rule. The
position starts flat and is not liquidated at expiry.

**Why it can be graded.** Under Black-Scholes the state is (k, S, h), two dimensions and a clock,
so the Bellman equation can be solved on a grid (`solve_dp`) -- the answer key. The structure
separates, which is also what makes the grid cheap:

    Q_k(S, h, a) = - cost * S * |a - h|  +  G_k(S, a),
    G_k(S, a)    = - (risk / 2) E[(dC - a dS)^2 | S]  +  E[V_{k+1}(S', a) | S],

so the holding enters only through the cost, which is KNOWN. For small costs Whalley & Wilmott
(1997) give the band in closed form: half-width w = (3/2 cost S Gamma^2 / risk)^(1/3) around the
Black-Scholes delta. Leland (1985) is the other classical rule: rebalance at every date to a
delta computed at an inflated volatility.

**Two learners** (`study_rl.py T`):
  generic       the framework's FittedQ, one linear model per action on generic state features,
                the reward a black box -- how RL is usually done;
  known-cost    `KnownCostFQI`: fitted Q iteration backwards through the dates, keeping the
                KNOWN cost term exact and learning only G_k(S, a) from transitions. Its policy
                has the band structure by construction, and it never needs the holding in its
                data, because G does not depend on it.
"""

import math
from dataclasses import dataclass

import numpy as np

import models.hedging as hd
import rl
from rl.agents import Batch


@dataclass(frozen=True)
class CostProblem:
    sigma: float = 0.2
    T: float = 1.0 / 12
    n_dates: int = 32
    K: float = 1.0
    S0: float = 1.0
    cost: float = 0.002          # proportional: cost * S * |shares traded|
    risk: float = 1000.0         # the (risk / 2) (dC - a dS)^2 penalty

    @property
    def dt(self):
        return self.T / self.n_dates

    def tau(self, k):
        return self.T - k * self.dt

    def mark(self, S, k):
        """The option's value at date k: Black-Scholes, the payoff at expiry."""
        if k >= self.n_dates:
            return np.maximum(S - self.K, 0.0)
        return hd.bs_call(S, self.K, self.sigma, self.tau(k))

    def delta(self, S, k, sigma=None):
        if k >= self.n_dates:
            return (np.asarray(S) > self.K).astype(float)
        return hd.bs_delta(S, self.K, self.sigma if sigma is None else sigma, self.tau(k))

    def gamma(self, S, k):
        tau = self.tau(k)
        st = self.sigma * math.sqrt(max(tau, 1e-12))
        d1 = np.log(np.asarray(S, float) / self.K) / st + 0.5 * st
        return np.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * np.asarray(S, float) * st)

    def price(self):
        return float(hd.bs_call(np.array([self.S0]), self.K, self.sigma, self.T)[0])


def simulate(prob, n_paths, seed=0):
    """Black-Scholes paths at the rebalancing dates, exact: (n_dates + 1, n_paths)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((prob.n_dates, n_paths))
    inc = -0.5 * prob.sigma ** 2 * prob.dt + prob.sigma * math.sqrt(prob.dt) * z
    x = np.vstack([np.zeros((1, n_paths)), np.cumsum(inc, axis=0)])
    return prob.S0 * np.exp(x)


def evaluate(prob, S, choose):
    """
    Run a rule over paths and score it. `choose(k, S_k, h_k) -> a_k`, vectorised over paths.
    Returns the objective per path (the summed reward) and its two parts, per unit premium.
    """
    n_paths = S.shape[1]
    h = np.zeros(n_paths)
    cost = np.zeros(n_paths)
    risk = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    C = prob.mark(S[0], 0)
    for k in range(prob.n_dates):
        a = np.clip(np.asarray(choose(k, S[k], h), float), -1.0, 2.0)
        C1 = prob.mark(S[k + 1], k + 1)
        trade = np.abs(a - h)
        cost += prob.cost * S[k] * trade
        turnover += trade
        err = (C1 - C) - a * (S[k + 1] - S[k])
        risk += 0.5 * prob.risk * err * err
        h, C = a, C1
    p = prob.price()
    total = -(cost + risk)
    return {"J": float(total.mean() / p), "J_se": float(total.std(ddof=1) / math.sqrt(n_paths) / p),
            "cost": float(cost.mean() / p), "risk": float(risk.mean() / p),
            "turnover": float(turnover.mean()), "per_path": total / p}


# ------------------------------------------------------------------ the classical rules
def bs_rule(prob):
    """Rebalance to the Black-Scholes delta at every date, whatever it costs."""
    return lambda k, S, h: prob.delta(S, k)


def leland_rule(prob):
    """Leland (1985): rebalance at every date to the delta at the volatility inflated by the
    round-trip cost, sigma_L^2 = sigma^2 (1 + sqrt(2/pi) * 2 cost / (sigma sqrt(dt)))."""
    le = math.sqrt(2.0 / math.pi) * 2.0 * prob.cost / (prob.sigma * math.sqrt(prob.dt))
    sig_l = prob.sigma * math.sqrt(1.0 + le)
    return lambda k, S, h: prob.delta(S, k, sigma=sig_l)


def ww_halfwidth(prob, S, k):
    """Whalley-Wilmott's asymptotic band half-width, (3/2 cost S Gamma^2 / risk)^(1/3)."""
    return (1.5 * prob.cost * np.asarray(S, float) * prob.gamma(S, k) ** 2 / prob.risk) ** (1.0 / 3.0)


def ww_rule(prob):
    """Hold inside the Whalley-Wilmott band around the delta; outside it, trade to its edge."""
    def choose(k, S, h):
        d, w = prob.delta(S, k), ww_halfwidth(prob, S, k)
        return np.clip(h, d - w, d + w)
    return choose


def static_rule(prob):
    """Buy the initial delta and never trade again."""
    d0 = float(prob.delta(np.array([prob.S0]), 0)[0])
    return lambda k, S, h: np.full_like(np.asarray(S, float), d0) if k == 0 else h


# ------------------------------------------------------------------ the answer key
def _gauss_hermite(n):
    z, w = np.polynomial.hermite_e.hermegauss(n)          # weight exp(-z^2 / 2)
    return z, w / w.sum()


def solve_dp(prob, n_h=101, n_x=401, width=9.0, n_quad=24, h_range=(0.0, 1.0)):
    """
    The Bellman equation on a grid: log S on n_x points spanning `width` standard deviations
    of log S at expiry, holdings and actions on the same n_h points of `h_range`, and the
    one-step expectations by Gauss-Hermite quadrature on the exact log-normal step.

    Returns {"x", "h", "G": [G_k (n_x, n_h)], "V0": V_0 at (S0, h = 0), "policy": callable}.
    The policy maximises the stored G_k, interpolated in log S, minus the exact cost at the
    ACTUAL holding -- so it is defined off the grid too.
    """
    sd = prob.sigma * math.sqrt(prob.T)
    x = math.log(prob.S0) + np.linspace(-width * sd, width * sd, n_x)
    hg = np.linspace(h_range[0], h_range[1], n_h)
    z, wq = _gauss_hermite(n_quad)
    step = -0.5 * prob.sigma ** 2 * prob.dt + prob.sigma * math.sqrt(prob.dt) * z
    S = np.exp(x)
    absdiff = np.abs(hg[None, :] - hg[:, None])            # [h, a]
    V_next = np.zeros((n_x, n_h))                           # V_n = 0: no liquidation
    Gs = [None] * prob.n_dates
    for k in range(prob.n_dates - 1, -1, -1):
        xn = x[:, None] + step[None, :]                     # (n_x, n_quad)
        Sn = np.exp(xn)
        dC = prob.mark(Sn, k + 1) - prob.mark(S, k)[:, None]
        dS = Sn - S[:, None]
        m_cc, m_cs, m_ss = (dC * dC) @ wq, (dC * dS) @ wq, (dS * dS) @ wq
        R = m_cc[:, None] - 2.0 * hg[None, :] * m_cs[:, None] + hg[None, :] ** 2 * m_ss[:, None]
        EV = np.empty((n_x, n_h))
        for j in range(n_h):
            EV[:, j] = np.interp(xn.ravel(), x, V_next[:, j]).reshape(xn.shape) @ wq
        G = -0.5 * prob.risk * R + EV                       # (n_x, n_a)
        Gs[k] = G
        # V_k(x, h) = max_a [G(x, a) - cost S |a - h|], chunked over x to bound memory
        V = np.empty((n_x, n_h))
        for lo in range(0, n_x, 64):
            sl = slice(lo, lo + 64)
            Q = G[sl, None, :] - prob.cost * S[sl, None, None] * absdiff[None, :, :]
            V[sl] = Q.max(axis=2)
        V_next = V
    j0 = int(np.argmin(np.abs(hg)))                         # the flat start
    V0 = float(np.interp(math.log(prob.S0), x, V_next[:, j0]))

    def policy(k, Sk, h, grid=hg):
        xk = np.log(np.asarray(Sk, float))
        Gk = np.stack([np.interp(xk, x, Gs[k][:, j]) for j in range(n_h)], axis=1)   # (n, n_a)
        Q = Gk - prob.cost * np.asarray(Sk, float)[:, None] * np.abs(grid[None, :] - np.asarray(h, float)[:, None])
        return grid[np.argmax(Q, axis=1)]
    return {"x": x, "h": hg, "G": Gs, "V0": V0 / prob.price(), "policy": policy}


def no_trade_band(rule, k, S, h_grid=None):
    """A rule's no-trade interval in the holding at date k, for each price in S: the holdings
    from which it stays put (to within half a grid step). Works for any rule -- the DP's, a
    closed form's or a learner's -- so their bands can be laid side by side."""
    hg = np.linspace(0.0, 1.0, 201) if h_grid is None else np.asarray(h_grid, float)
    out = []
    for s in np.atleast_1d(S):
        a = np.asarray(rule(k, np.full(len(hg), float(s)), hg), float)
        stay = np.abs(a - hg) <= 0.5 * (hg[1] - hg[0]) + 1e-12
        out.append((float(hg[stay].min()), float(hg[stay].max())) if stay.any() else (np.nan, np.nan))
    return np.array(out)


# ------------------------------------------------------------------ learners
def transitions(prob, S, seed=0, n_actions=21):
    """
    Logged transitions under a uniformly random policy on an n_actions grid: every (date, path)
    pair once, the holding carried from the previous random action. Returns the arrays both
    learners need; neither sees the model.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, n_actions)
    n, m = prob.n_dates, S.shape[1]
    act = rng.integers(0, n_actions, (n, m))
    h = np.vstack([np.zeros((1, m)), grid[act[:-1]]])      # holding into each date
    return {"S": S, "act": act, "h": h, "grid": grid}


def generic_features(prob, S, k, h):
    """[1, delta, delta^2, h, h^2, h delta, m, m^2, tau]: the state as a generalist would
    describe it, with no hint of the band."""
    tau = prob.tau(k)
    m = np.clip(np.log(np.asarray(S, float) / prob.K) / (prob.sigma * math.sqrt(max(tau, prob.dt))), -8, 8)
    d = prob.delta(S, k)
    h = np.asarray(h, float) * np.ones_like(d)
    return np.column_stack([np.ones_like(d), d, d * d, h, h * h, h * d, m, m * m, np.full_like(d, tau / prob.T)])


def generic_batch(prob, tr):
    """The transitions as a `Batch` for the framework's agents, the reward a black box."""
    S, act, h, grid = tr["S"], tr["act"], tr["h"], tr["grid"]
    n, m = prob.n_dates, S.shape[1]
    phi, phin, rew, absorb, ep, st, a_all = [], [], [], [], [], [], []
    for k in range(n):
        a = grid[act[k]]
        C, C1 = prob.mark(S[k], k), prob.mark(S[k + 1], k + 1)
        r = -prob.cost * S[k] * np.abs(a - h[k]) - 0.5 * prob.risk * ((C1 - C) - a * (S[k + 1] - S[k])) ** 2
        phi.append(generic_features(prob, S[k], k, h[k]))
        phin.append(generic_features(prob, S[k + 1], min(k + 1, n - 1), a))
        rew.append(r)
        absorb.append(np.full(m, k == n - 1))
        ep.append(np.arange(m))
        st.append(np.full(m, k))
        a_all.append(act[k])
    return Batch(np.vstack(phi), np.concatenate(a_all), np.concatenate(rew), np.vstack(phin),
                 np.concatenate(absorb), len(grid), episode=np.concatenate(ep), step=np.concatenate(st))


def generic_rule(prob, model, grid):
    return lambda k, S, h: grid[model.policy(generic_features(prob, S, k, h))]


class KnownCostFQI:
    """
    Fitted Q iteration that keeps the known part of the reward exact (Session O).

    Q_k(S, h, a) = - cost S |a - h| + G_k(S, a), and only G_k is learned, backwards through the
    dates, from transitions -- model-free in the dynamics, with the band structure built in
    because the cost is exact. It never needs the holding in its data: G does not depend on it.

    How G_k is learned is the other half of Session N's section D. `risk="direct"` regresses the
    whole target, - (risk/2) (dC - a dS)^2 + V_{k+1}(S', a), on a basis in (S, a) -- a squared,
    heavy-tailed target, the inefficiency section D measured. `risk="moments"` (the default)
    completes the square instead,

        E[(dC - a dS)^2 | S] = eps(S) + (a - beta(S))^2 m_SS(S),

    with beta the one-step risk-minimising hedge, fitted the way Hedged Monte Carlo fits it
    (dC on its hedge basis times dS), m_SS = E[dS^2 | S] and eps the residual risk each by their
    own regression. Only the continuation E[V_{k+1}(S', a) | S] is left to a regression in
    (S, a), on powers of u = a - centre up to `degree_a` times [1, delta, pdf], the centre being
    beta ("moments") or the Black-Scholes delta ("direct"). The state terms are Hedged Monte
    Carlo's bounded greeks, not powers of moneyness: a first version on [1, m, ..., m^4] put the
    hedge ratio's sigmoid through a quartic, which runs away in the tails near expiry, and
    scored far worse than the naive fit for that reason alone.

    `degree_a` is the one knob that matters (study_rl.py T). At 32 dates a quartic reaches the
    DP to 0.0009 of the premium. At 128 dates the band is wider than the delta's move between
    dates, the continuation has sharp shoulders at the band's edges, and the gap is the basis's,
    not the data's: 0.035 at degree 4, 0.029 at 6 (the same with three times the paths), 0.024
    at 8. A local basis was tried and dropped: hat functions on 0.1 knots reached 0.026, and on
    0.05 knots overfit and collapsed to -4.1.
    """

    def __init__(self, prob, degree_a=4, n_eval=101, risk="moments", continuation=True):
        if risk not in ("moments", "direct"):
            raise ValueError("risk must be 'moments' or 'direct'")
        self.prob, self.degree_a, self.risk = prob, int(degree_a), risk
        self.continuation = bool(continuation)      # False: myopic, each date for itself
        self.eval_grid = np.linspace(0.0, 1.0, int(n_eval))
        self.coef = [None] * prob.n_dates

    def _cont_design(self, M, u):
        return np.hstack([M * (u ** p)[:, None] for p in range(self.degree_a + 1)])

    def _cont_grid(self, M, U, coef):
        W = coef.reshape(self.degree_a + 1, 3)
        out = np.zeros_like(U)
        for p in range(self.degree_a, -1, -1):              # Horner in u
            out = out * U + (M @ W[p])[:, None]
        return out

    def _greeks(self, S, k):
        """Hedged Monte Carlo's value basis psi and hedge basis chi at date k."""
        psi, chi, _ = hd._features(np.asarray(S, float), self.prob.sigma, self.prob.tau(k), self.prob.K)
        return psi, chi

    def _centre(self, S, k):
        """(state terms [1, delta, pdf], the hedge u is measured from)."""
        _, chi = self._greeks(S, k)
        M = chi[:, :3]
        if self.risk == "moments":
            return M, chi @ self.coef[k]["beta"]
        return M, chi[:, 1]

    def G_grid(self, k, S, grid=None):
        """G_k(S, a) for every a on `grid` at once: (n, n_grid)."""
        g = self.eval_grid if grid is None else np.asarray(grid, float)
        c = self.coef[k]
        M, centre = self._centre(S, k)
        U = g[None, :] - centre[:, None]
        out = self._cont_grid(M, U, c["cont"])
        if self.risk == "moments":
            S = np.asarray(S, float)
            psi, _ = self._greeks(S, k)
            m_ss = np.column_stack([np.ones_like(S), S, S * S]) @ c["m_ss"]
            out = out - 0.5 * self.prob.risk * ((psi @ c["eps"])[:, None] + U * U * m_ss[:, None])
        return out

    def _best(self, k, S, h):
        S, h = np.asarray(S, float), np.asarray(h, float)
        g = self.eval_grid
        Q = self.G_grid(k, S) - self.prob.cost * S[:, None] * np.abs(g[None, :] - h[:, None])
        j = np.argmax(Q, axis=1)
        return g[j], Q[np.arange(len(S)), j]

    def value(self, k, S, h):
        """V_k(S, h) = max over the evaluation grid of G_k(S, a) - cost S |a - h|."""
        return self._best(k, S, h)[1]

    def fit(self, tr):
        rl.require_enabled("the known-cost fitted-Q agent")
        prob, S, act, grid = self.prob, tr["S"], tr["act"], tr["grid"]
        for k in range(prob.n_dates - 1, -1, -1):
            a = grid[act[k]]
            dC = prob.mark(S[k + 1], k + 1) - prob.mark(S[k], k)
            dS = S[k + 1] - S[k]
            live = self.continuation and k + 1 < prob.n_dates
            cont = self.value(k + 1, S[k + 1], a) if live else np.zeros_like(a)
            c = {}
            if self.risk == "moments":
                psi, chi = self._greeks(S[k], k)
                c["beta"] = hd._lstsq(chi * dS[:, None], dC)           # Hedged Monte Carlo's regression
                resid = dC - (chi @ c["beta"]) * dS
                c["m_ss"] = hd._lstsq(np.column_stack([np.ones_like(dS), S[k], S[k] ** 2]), dS * dS)
                c["eps"] = hd._lstsq(psi, resid * resid)
                self.coef[k] = c
                y = cont
            else:
                self.coef[k] = c
                y = -0.5 * prob.risk * (dC - a * dS) ** 2 + cont
            M, centre = self._centre(S[k], k)
            X = self._cont_design(M, a - centre)
            c["cont"] = hd._lstsq(X, y)
        return self

    def rule(self):
        return lambda k, S, h: self._best(k, S, h)[0]
