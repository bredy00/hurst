"""
The risk model's Hurst index: the rough-volatility forecast of Gatheral, Jaisson & Rosenbaum
(2018), the knob the trial turns.

If log-variance moves like fBm with index H, its best forecast Delta days ahead is a weighted
average of its past (Nuzman & Poor 2000; GJR eq. 5.1):

    E[log s2_{t+Delta} | F_t] = cos(H pi)/pi Delta^{H+1/2} int_0^inf log s2_{t-u} du / ((u + Delta) u^{H+1/2}).

The weights integrate to one. As H -> 1/2 they collapse onto the last observation (a
Brownian log-variance is a martingale); the rougher the process, the further back they
reach, because rough increments are anti-persistent and a rise is more likely undone than
continued. So a LOW H gives a SMOOTH forecast and H near 1/2 a jagged one that follows every
day's noisy realised variance: the forecaster's H sets how jagged the risk model, and
through it the portfolio, is. That is the lever the dual Kalman filter moves.

The integral is taken cell by cell on the daily grid (the first cell's singularity in closed
form, through the hypergeometric function), truncated at n_lags and renormalised. The level
is set separately (level_scale), so H changes the dynamics and not the average variance.
"""

import numpy as np
from scipy import special

H_RANGE = (0.02, 0.49)


def gjr_weights(H, horizon=5, n_lags=504):
    """Weights a_1..a_n (a_1 on the most recent day) of the fBm predictor, summing to one."""
    H = float(min(max(H, H_RANGE[0]), H_RANGE[1]))
    a = H + 0.5
    d = float(horizon)
    first = special.hyp2f1(1.0, 1.0 - a, 2.0 - a, -1.0 / d) / ((1.0 - a) * d)
    x, wq = np.polynomial.legendre.leggauss(8)
    j = np.arange(2, n_lags + 1, dtype=float)[:, None]
    u = j - 0.5 + 0.5 * x[None, :]                         # nodes in the cell [j-1, j]
    rest = 0.5 * (wq[None, :] * u ** (-a) / (u + d)).sum(axis=1)
    cells = np.concatenate([[first], rest])
    return cells / cells.sum()


def forecast_log(y, weights):
    """The predictor at every day t from y[:t]: out[t] = sum_j a_j y[t-j]; NaN until n_lags days exist."""
    y = np.asarray(y, float)
    n = len(weights)
    out = np.full(len(y) + 1, np.nan)
    conv = np.convolve(y, weights)[:len(y)]              # conv[s] = sum_j a_j y[s+1-j] for j = 1..n
    out[n:len(y) + 1] = conv[n - 1:]
    return out


def level_scale(target, forecast_var, window):
    """Trailing mean(target) / mean(forecast) over `window` days: makes the forecast level-unbiased."""
    t = np.asarray(target, float)
    f = np.asarray(forecast_var, float)
    ok = np.isfinite(f)
    ct = np.concatenate([[0.0], np.cumsum(np.where(ok, t, 0.0))])
    cf = np.concatenate([[0.0], np.cumsum(np.where(ok, f, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    out = np.full(len(t), np.nan)
    for s in range(window, len(t) + 1):
        k = cn[s] - cn[s - window]
        if k >= window // 2:
            out[s - 1] = (ct[s] - ct[s - window]) / (cf[s] - cf[s - window])
    return out


def variogram_hurst(y, lags=tuple(range(1, 41)), noise_var=None):
    """
    H of a log-variance series observed with white measurement noise: least squares of the
    variogram gamma(l) = n0 + c l^{2H} over the lags (n0 = twice the noise variance). The
    plain log-log slope is pulled toward zero by the noise; carrying n0 removes that, but
    only if n0 is known: at small H, n0 + c l^{2H} ~ (n0 + c) + 2 H c log l, and noise and
    roughness trade off along a ridge. Pass `noise_var` when it is known -- for realised
    variance from M intraday returns it is about 2 / M in log terms -- and n0 is fixed.
    """
    from scipy.optimize import least_squares
    y = np.asarray(y, float)
    lags = np.asarray(lags, float)
    g = np.array([np.mean((y[int(l):] - y[:-int(l)]) ** 2) for l in lags])

    if noise_var is not None:
        n0 = 2.0 * float(noise_var)

        def res(p):
            c, H = p
            return (n0 + c * lags ** (2 * H)) / g - 1.0

        sol = least_squares(res, x0=[max(g[0] - n0, 1e-6), 0.15], bounds=([1e-10, 0.005], [np.inf, 0.99]))
        return {"H": float(sol.x[1]), "noise_var": float(noise_var), "c": float(sol.x[0])}

    def res(p):
        n0, c, H = p
        return (n0 + c * lags ** (2 * H)) / g - 1.0

    sol = least_squares(res, x0=[0.5 * g[0], 0.5 * g[0], 0.15],
                        bounds=([0.0, 1e-8, 0.005], [np.inf, np.inf, 0.99]))
    return {"H": float(sol.x[2]), "noise_var": float(sol.x[0] / 2), "c": float(sol.x[1])}
