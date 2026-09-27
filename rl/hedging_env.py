"""
Hedging as an MDP, and the reason this one can be graded (Session M).

The writer of a one-month at-the-money call rebalances every `every` steps. At each date
they choose a hedge ratio; over the next interval their wealth moves by phi dS, and the
option moves against them. Sessions J-K solved this exactly, by backward regression:

    min E[ (C_{k+1} - C_k(z) - phi_k(z) dS)^2 ]      Foellmer-Schweizer local risk minimisation

and `models/hedging.py` reports what it costs and what is left. So an agent here is not
being shown off -- it is being scored against the right answer on the same paths.

**The MDP.**
  state      z = (S, sigma_hat, tau), the SAME state Hedged Monte Carlo regresses on, so a
             gap between them is the learner's, not the state's. `phi_features` is the
             basis both use.
  action     a hedge ratio from a fixed grid spanning [0, 1] (a call's delta lives there);
             the grid is the discretisation an RL agent needs and Hedged Monte Carlo does
             not, and its coarseness is measured, not waved away.
  reward     -(dW)^2, the squared one-step wealth change of the hedged book -- the local
             risk the analytic rule minimises. Summing it over an episode gives (minus) the
             realised quadratic variation of the hedging error, so the agent's objective
             and the baseline's are the same quantity.
  absorbing  expiry. The terminal payoff is earned on the transition INTO it, and no action
             is taken there.
  gamma      1. Every episode reaches expiry in a fixed number of steps, so the undiscounted
             return is finite and is the quantity the desk cares about.

**What the comparison can and cannot show.** The analytic rule chooses a real number; the
agent chooses from a grid, so it starts behind by the grid's own resolution. That floor is
computed (`grid_floor`) and reported beside the gap, because an agent that lands within its
own discretisation error has matched the optimum and should be said to have done so.
"""

import math

import numpy as np

import models.hedging as hd
import rl
from rl.agents import Batch


def action_grid(n_actions=21, lo=0.0, hi=1.0):
    """The hedge ratios on offer. A call's delta is in [0, 1], so the grid spans it."""
    return np.linspace(lo, hi, int(n_actions))


M_CLIP = 8.0      # standardised log-moneyness is clipped here; see phi_features


def phi_features(S, sigma, tau, K):
    """
    The state basis: [1, delta, vega/S, m, m^2], with m the standardised log-moneyness.

    Deliberately the analytic greeks at sigma_hat, which is what `hedging._features` gives
    the regression -- the point is to compare learners on one state, not to find a better
    state for one of them. `delta` alone would already span the Black-Scholes answer, so an
    agent that cannot reach it has no excuse.

    m is CLIPPED at +-M_CLIP. As tau -> 0 the standardisation sigma sqrt(tau) -> 0 and m
    diverges; unclipped it reaches 1e6 on the last date, which does nothing to the value
    (the terminal state is absorbing, so nothing bootstraps through it) but wrecks
    everything that touches the feature matrix -- it put LSPI's normal equations at a
    condition number of 1.6e17 and made a Markov test on "next moneyness" report a 7.4%
    lag effect that was entirely the blow-up. Beyond 8 standard deviations delta is 0 or 1
    to 16 digits, so the clip discards no information the hedge can use.
    """
    st = np.maximum(np.asarray(sigma, float) * math.sqrt(max(float(tau), 1e-12)), 1e-6)
    m = np.clip(np.log(np.asarray(S, float) / K) / st, -M_CLIP, M_CLIP)
    d1 = m + 0.5 * st
    pdf = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    delta = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / math.sqrt(2.0)))
    return np.column_stack([np.ones_like(m), delta, pdf * math.sqrt(max(float(tau), 1e-12)), m, m * m])


def hmc_mark(fit, paths, K):
    """
    The option's value at each rebalancing date as Hedged Monte Carlo fitted it, C_k(z) =
    psi(z) . a_k, with the payoff at expiry: a MARTINGALE-CONSISTENT mark, built by backward
    regression on realised future cash flows (Session N). Pass as `mark=` to `build_batch`.
    """
    n = len(paths["t"]) - 1

    def mark(k):
        if k >= n or paths["tau"][k] <= 1e-12:
            return np.maximum(paths["S"][k] - K, 0.0)
        psi, _, _ = hd._features(paths["S"][k], hd.model_sigma(paths, k), paths["tau"][k], K)
        return psi @ fit["coefs"][k][0]
    return mark


def build_batch(paths, K, every, n_actions=21, seed=0, policy="grid-uniform", mark=None):
    """
    Log one transition per path per rebalancing date, under a fixed logging policy.

    `policy`:
      "grid-uniform"  the action is drawn uniformly from the grid. Maximal coverage, which
                      is what an offline learner needs and what makes the batch honest:
                      every action is tried in every state region.
      "delta-noisy"   the Black-Scholes delta, snapped to the grid, jittered by one grid
                      step. Closer to a real logging policy, and it makes the coverage
                      problem visible -- the far actions are never seen, and the agents say so.

    The reward is -(dW)^2 with dW = dC - phi dS, dC the change in the option's mark. The
    mark is the model's own Black-Scholes value at sigma_hat, which at expiry becomes the
    payoff exactly, so nothing about the terminal condition is approximated.
    """
    rl.require_enabled("the hedging environment")
    rng = np.random.default_rng(seed)
    S, n = paths["S"], len(paths["t"]) - 1
    dates = list(range(0, n, every))
    grid = action_grid(n_actions)
    n_paths = S.shape[1]

    def bs_mark(k):
        tau = paths["tau"][k]
        if tau <= 1e-12:
            return np.maximum(S[k] - K, 0.0)
        return hd.bs_call(S[k], K, hd.model_sigma(paths, k), tau)

    # The mark is the thing the one-step reward differences. The default is Black-Scholes at
    # sigma_hat, which is NOT a martingale under the rough model -- and then minimising each
    # step's squared error is not minimising the variance of the total hedging error, because
    # the steps are serially correlated. Session N measured what that costs: with `hmc_mark`
    # (a martingale-consistent value) the same fitted-Q agent goes from 0.354 of the price to
    # 0.332, 44% of its gap to Hedged Monte Carlo's 0.305. Most of the rest is the agent
    # learning a reward whose form is known (`known_reward_fit`; study_rl.py section D).
    mark = mark or bs_mark

    phi_l, act_l, rew_l, phin_l, abs_l, ep_l, st_l = [], [], [], [], [], [], []
    for j, k in enumerate(dates):
        k2 = min(k + every, n)
        tau, tau2 = paths["tau"][k], paths["tau"][k2]
        f = phi_features(S[k], hd.model_sigma(paths, k), tau, K)
        f2 = phi_features(S[k2], hd.model_sigma(paths, k2), max(tau2, 1e-12), K)
        if policy == "grid-uniform":
            a = rng.integers(0, n_actions, n_paths)
        elif policy == "delta-noisy":
            d = hd.bs_delta(S[k], K, hd.model_sigma(paths, k), tau)
            a = np.clip(np.rint(d * (n_actions - 1)).astype(int)
                        + rng.integers(-1, 2, n_paths), 0, n_actions - 1)
        else:
            raise ValueError(f"unknown logging policy {policy!r}")
        dS = S[k2] - S[k]
        dW = (mark(k2) - mark(k)) - grid[a] * dS
        if k2 >= n:
            f2 = np.zeros_like(f2)     # absorbing: nothing bootstraps through it, and a
                                       # terminal feature row only pollutes the design
        phi_l.append(f)
        act_l.append(a)
        rew_l.append(-dW * dW)
        phin_l.append(f2)
        abs_l.append(np.full(n_paths, k2 >= n))
        ep_l.append(np.arange(n_paths))
        st_l.append(np.full(n_paths, j))
    return Batch(np.vstack(phi_l), np.concatenate(act_l), np.concatenate(rew_l),
                 np.vstack(phin_l), np.concatenate(abs_l), n_actions,
                 episode=np.concatenate(ep_l), step=np.concatenate(st_l)), grid, dates


def apply_policy(paths, K, every, choose, premium):
    """
    The terminal hedging error of a writer following `choose(phi, k) -> hedge ratios`.

    The same accounting as `hedging.hedge_error`, so the numbers sit in one table with the
    analytic rules: premium + the hedge's gains - the payoff.
    """
    S, n = paths["S"], len(paths["t"]) - 1
    gains = np.zeros(S.shape[1])
    for k in range(0, n, every):
        k2 = min(k + every, n)
        f = phi_features(S[k], hd.model_sigma(paths, k), paths["tau"][k], K)
        gains += choose(f, k) * (S[k2] - S[k])
    return premium + gains - np.maximum(S[n] - K, 0.0)


def agent_policy(model, grid):
    """Turn an agent into the callable `apply_policy` wants: argmax over the grid."""
    return lambda f, k: grid[model.policy(f)]


def grid_floor(paths, K, every, premium, n_actions=21, sigma_fixed=None):
    """
    The residual a PERFECT grid policy would leave: the analytic Black-Scholes delta snapped
    to the same grid. Any agent restricted to the grid is judged against this, not against
    the continuous optimum, because the difference is the discretisation and not the
    learning.
    """
    grid = action_grid(n_actions)
    S = paths["S"]

    def choose(f, k):
        sig = sigma_fixed if sigma_fixed is not None else hd.model_sigma(paths, k)
        d = hd.bs_delta(S[k], K, sig, paths["tau"][k])
        return grid[np.clip(np.rint(d * (n_actions - 1)).astype(int), 0, n_actions - 1)]

    return apply_policy(paths, K, every, choose, premium)


def markov_inputs(batch, paths, K, every):
    """
    (features, lagged features, targets, names) for `rl.markov` on the hedging state.

    The targets are the next state's informative coordinates and the reward. The constant
    column is dropped from the design -- a lag block containing a constant is collinear with
    the intercept, which would make the F test meaningless rather than merely wrong.
    """
    from rl.markov import lagged_design
    rows, prev, f_t, f_prev = lagged_design(batch.phi[:, 1:], batch.episode, batch.step)
    live = ~batch.absorbing[rows]          # a terminal row has no next state to predict
    rows, prev = rows[live], prev[live]
    f_t, f_prev = f_t[live], f_prev[live]
    act = batch.action[rows][:, None].astype(float)
    act_prev = batch.action[prev][:, None].astype(float)
    feats = np.hstack([f_t, act])
    lags = np.hstack([f_prev, act_prev])
    tgt = np.column_stack([batch.phi_next[rows, 1], batch.phi_next[rows, 3], batch.reward[rows]])
    return feats, lags, tgt, ["next delta", "next moneyness", "reward"]


def known_reward_fit(paths, K, every, mark=None, basis="phi", pooled=True):
    """
    The hedge fitted as a REGRESSION, using the fact that the reward's dependence on the
    action is known exactly (Session N).

    The one-step reward is -(dC - a dS)^2: a known function of observed quantities, not a
    black box in a. So the risk-minimising a(s) is the least-squares coefficient of dC on
    dS given the state -- fit dC ~ (basis(s) . beta) dS on the logged transitions, and hedge
    with a(s) = basis(s) . beta. No action is ever tried: every transition informs the fit
    whichever action the logging policy took, because the action does not enter it.

    This is what Q-learning cannot do. It treats r(s, a) as an unknown function and must
    learn how it varies with a -- even `QuadraticQ`, which knows the shape, has to model
    E[dS^2 | s] and E[dC dS | s] as functions of the state, and they scale with S^2 V, which
    a linear basis in the state does not span. The regression uses the realised dS as its
    regressor and never models either. Session N measured the difference at 54% of the whole
    gap to Hedged Monte Carlo.

    mark    "bs" (Black-Scholes at sigma_hat, the default everywhere else) or a callable
            mark(k) on these paths, e.g. `hmc_mark(fit, paths, K)`
    basis   "phi" (this module's state features) or "chi" (Hedged Monte Carlo's hedge basis)
    pooled  one coefficient vector for every date, or one per date as Hedged Monte Carlo has

    With mark = hmc_mark, basis = "chi", pooled = False this reproduces Hedged Monte Carlo's
    residual to 0.0002 -- the regression IS the analytic hedge, arrived at from the RL side.
    Returns a rule(paths, k) -> hedge ratios, for `score_rule`.
    """
    rl.require_enabled("the known-reward regression")
    S, n = paths["S"], len(paths["t"]) - 1
    dates = list(range(0, n, every))

    def bs_mark(k):
        tau = paths["tau"][k]
        if tau <= 1e-12:
            return np.maximum(S[k] - K, 0.0)
        return hd.bs_call(S[k], K, hd.model_sigma(paths, k), tau)

    mk = bs_mark if mark in (None, "bs") else mark

    def B(p, k):
        s_, sig, tau = p["S"][k], hd.model_sigma(p, k), p["tau"][k]
        return phi_features(s_, sig, tau, K) if basis == "phi" else hd._features(s_, sig, tau, K)[1]

    Xs, ys, per = [], [], {}
    for k in dates:
        k2 = min(k + every, n)
        dS = S[k2] - S[k]
        X, y = B(paths, k) * dS[:, None], mk(k2) - mk(k)
        if pooled:
            Xs.append(X)
            ys.append(y)
        else:
            per[k] = np.linalg.lstsq(X, y, rcond=None)[0]
    if pooled:
        beta = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]
        return lambda p, k: B(p, k) @ beta
    return lambda p, k: B(p, k) @ per[k]


def score_rule(paths, K, every, rule, premium):
    """Terminal hedging error of rule(paths, k) -> hedge ratios, with apply_policy's accounting."""
    S, n = paths["S"], len(paths["t"]) - 1
    gains = np.zeros(S.shape[1])
    for k in range(0, n, every):
        k2 = min(k + every, n)
        gains += rule(paths, k) * (S[k2] - S[k])
    return premium + gains - np.maximum(S[n] - K, 0.0)
