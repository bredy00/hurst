"""
Is the state you chose actually Markov? (Session M)

Every MDP in this package rests on one assumption: given (s_t, a_t), the future is
independent of the past. Choose the state badly -- drop the volatility from a hedging
state, drop the boundary flag from a filtering state -- and the assumption fails silently.
The agent still trains, still reports a return, and is still wrong, because the thing it
is approximating does not exist.

So this runs BEFORE any learning, on logged trajectories, and it is the honest answer to
"why would an RL framework belong in a project like this at all": it turns an assumption
into a measurement.

**The test.** Under the Markov property, adding lagged information to a predictor of the
next state (or reward) must not help. So fit two nested least-squares models on the logged
transitions,

    restricted   y ~ [1, phi(s_t, a_t)]
    full         y ~ [1, phi(s_t, a_t), phi(s_{t-1}, a_{t-1})]

and compare them with a partial F test, which for nested linear models is exact under
Gaussian errors and asymptotically valid otherwise. `y` is each coordinate of the next
state in turn, and the reward. A significant lag block means the state is not Markov FOR
THIS PREDICTOR -- a necessary condition, tested where it bites, not a proof of the
property, and reported as such.

Two practical points the F statistic cannot see, both checked here:

  serial correlation  transitions from one trajectory are not independent draws, so the
                      naive F test over-rejects. The p-value is therefore computed on
                      `n_eff` effective observations, n scaled by the residuals' lag-1
                      autocorrelation ((1 - rho) / (1 + rho)), and both are reported.
  the whole point     a lag block can be significant and tiny. `partial_r2` says how much
                      of the remaining variance it explains, and a state is judged on that
                      as well as on p -- significance is not size.
"""

import numpy as np
from scipy import stats

import rl


def _lstsq_rss(X, y):
    """Residual sum of squares of the least-squares fit, and the fitted residuals."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    return float(e @ e), e


def _effective_n(e):
    """n scaled by the residuals' lag-1 autocorrelation: (1 - rho) / (1 + rho), floored at 1."""
    n = len(e)
    if n < 3:
        return float(n), 0.0
    ec = e - e.mean()
    denom = float(ec @ ec)
    rho = float(ec[1:] @ ec[:-1]) / denom if denom > 0 else 0.0
    rho = min(max(rho, -0.99), 0.99)
    return max(n * (1.0 - rho) / (1.0 + rho), 1.0), rho


def markov_test(features, lagged, targets, names=None, alpha=0.01):
    """
    The nested F test, one target at a time.

    features  (n, k)   phi(s_t, a_t) for each transition, WITHOUT a constant column
    lagged    (n, k')  phi(s_{t-1}, a_{t-1}) for the same transitions
    targets   (n, m)   what the state should predict: the next state's coordinates, the reward
    names     m labels

    Returns a dict per target with F, its p-value on the naive and effective sample sizes,
    the partial R^2 of the lag block, and a verdict.
    """
    rl.require_enabled("the Markov property test")
    X = np.atleast_2d(np.asarray(features, float))
    L = np.atleast_2d(np.asarray(lagged, float))
    Y = np.atleast_2d(np.asarray(targets, float))
    if Y.shape[0] != X.shape[0]:
        Y = Y.T
    n = X.shape[0]
    if L.shape[0] != n or Y.shape[0] != n:
        raise ValueError(f"features, lagged and targets need the same number of rows: "
                         f"{X.shape[0]}, {L.shape[0]}, {Y.shape[0]}")
    one = np.ones((n, 1))
    Xr = np.hstack([one, X])
    Xf = np.hstack([one, X, L])
    q = L.shape[1]
    df2 = n - Xf.shape[1]
    if df2 <= 0:
        raise ValueError(f"{n} transitions cannot support {Xf.shape[1]} regressors")
    out = {}
    for j in range(Y.shape[1]):
        y = Y[:, j]
        rss_r, _ = _lstsq_rss(Xr, y)
        rss_f, e_f = _lstsq_rss(Xf, y)
        if rss_f <= 0 or rss_r <= rss_f * (1 + 1e-15):
            F, p, part = 0.0, 1.0, 0.0
        else:
            F = ((rss_r - rss_f) / q) / (rss_f / df2)
            p = float(stats.f.sf(F, q, df2))
            part = (rss_r - rss_f) / rss_r
        n_eff, rho = _effective_n(e_f)
        df2_eff = max(n_eff - Xf.shape[1], 1.0)
        p_eff = float(stats.f.sf(F, q, df2_eff)) if F > 0 else 1.0
        name = names[j] if names is not None else f"target {j}"
        out[name] = {"F": float(F), "p": p, "p_effective": p_eff, "partial_r2": float(part),
                     "n": int(n), "n_effective": float(n_eff), "residual_rho1": float(rho),
                     "markov": bool(p_eff > alpha)}
    return out


def report(res, alpha=0.01, r2_small=0.01):
    """
    Readable lines, and the overall verdict: Markov when no target's lag block is both
    significant on the effective sample AND explains more than `r2_small` of the variance.
    """
    lines, ok = [], True
    for name, r in res.items():
        big = r["partial_r2"] > r2_small
        sig = r["p_effective"] <= alpha
        flag = "LAGS MATTER" if (big and sig) else ("significant but small" if sig else "Markov")
        ok &= not (big and sig)
        lines.append(f"{name}: F {r['F']:8.2f}, p {r['p']:.1e} (effective n {r['n_effective']:.0f} "
                     f"of {r['n']}, residual rho1 {r['residual_rho1']:+.2f}: p {r['p_effective']:.1e}), "
                     f"the lags explain {100 * r['partial_r2']:.2f}% -- {flag}")
    return {"ok": bool(ok), "lines": lines}


def lagged_design(features, episode_ids, steps=None):
    """
    (rows, prev_rows, features_t, features_{t-1}) for the transitions that HAVE a
    predecessor in the same episode. The first transition of each episode is dropped rather
    than padded with zeros, which would manufacture a relationship that is not there.

    `steps` is the time index within the episode. Pass it whenever the batch is not stored
    in episode-then-time order -- a batch stacked date-major has consecutive ROWS from
    different episodes, and pairing by row adjacency would then silently find no lags at
    all (or, worse, pair two different episodes). With `steps` the pairing is on
    (episode, step) and does not care how the rows were stacked.
    """
    f = np.atleast_2d(np.asarray(features, float))
    ep = np.asarray(episode_ids)
    if steps is None:
        keep = np.zeros(len(f), bool)
        keep[1:] = ep[1:] == ep[:-1]
        rows = np.flatnonzero(keep)
        return rows, rows - 1, f[rows], f[rows - 1]
    st = np.asarray(steps, int)
    order = np.lexsort((st, ep))                       # episode, then time
    e_o, s_o = ep[order], st[order]
    ok = np.zeros(len(order), bool)
    ok[1:] = (e_o[1:] == e_o[:-1]) & (s_o[1:] == s_o[:-1] + 1)
    pos = np.flatnonzero(ok)
    rows, prev = order[pos], order[pos - 1]
    return rows, prev, f[rows], f[prev]
