"""
HAR-RV: the heterogeneous autoregressive model of realised variance (Corsi 2009), as a
first-class model (Session N, 27 September 2026).

    RV_{t+h} (average over the next h days) = b0 + b_d RV_t + b_w RV_t^(w) + b_m RV_t^(m) + e,

RV^(w) and RV^(m) the averages over the last 5 and 22 days. Three horizons of past variance
stand in for traders who rebalance daily, weekly and monthly, and the cascade reproduces the
long memory of volatility with an OLS regression -- which is why it is the benchmark every
realised-variance forecast is measured against, and why the T-EGARCH scan (Session L) found
it 17% better in QLIKE than the best returns-only model on this project's own rough Heston.
Until Session N it lived inline in `study_egarch.py`; this module is where it belongs for
the day real realised variance arrives.

  fit(rv, horizon, log)   OLS on the training days, Newey-West standard errors (the
                          residuals of an h-day-ahead average overlap, so plain OLS errors
                          are wrong for h > 1 by construction)
  forecast(model, rv)     the forecast known at the close of each day, NaN before 22 days
  log=True                the log-HAR: fitted on log RV, forecast back in levels with the
                          lognormal correction exp(mu + s^2 / 2), without which it is biased
                          low by the whole of that factor

`qlike` is Patton's loss, the one to rank variance forecasts by.
"""

import math
from dataclasses import dataclass, field

import numpy as np

WINDOWS = (1, 5, 22)


def regressors(rv, windows=WINDOWS):
    """
    (T, 1 + len(windows)) design: [1, mean RV over the last w days for each w], row t known at
    the close of day t. NaN rows before the longest window is full.
    """
    rv = np.asarray(rv, float)
    T = len(rv)
    c = np.concatenate([[0.0], np.cumsum(rv)])
    X = np.full((T, 1 + len(windows)), np.nan)
    X[:, 0] = 1.0
    for j, w in enumerate(windows):
        t = np.arange(w - 1, T)
        X[t, 1 + j] = (c[t + 1] - c[t + 1 - w]) / w
    return X


def targets(rv, horizon=1):
    """y_t = mean(RV_{t+1..t+horizon}): what a forecast made at the close of day t is for."""
    rv = np.asarray(rv, float)
    T = len(rv)
    c = np.concatenate([[0.0], np.cumsum(rv)])
    y = np.full(T, np.nan)
    t = np.arange(0, T - horizon)
    y[t] = (c[t + 1 + horizon] - c[t + 1]) / horizon
    return y


def _nw_cov(X, e, lag):
    n, k = X.shape
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, min(lag, n - 1) + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1.0 - j / (lag + 1.0)) * (G + G.T)
    iXX = np.linalg.inv(X.T @ X)
    return iXX @ S @ iXX


@dataclass
class HAR:
    coef: np.ndarray
    se: np.ndarray
    horizon: int
    log: bool
    windows: tuple
    resid_var: float
    n_obs: int
    names: tuple = field(default=())

    def describe(self):
        return ", ".join(f"{n} {c:+.4f} ({s:.4f})" for n, c, s in zip(self.names, self.coef, self.se))


def fit(rv, horizon=1, log=False, train=None, windows=WINDOWS, lag=None):
    """
    Fit on the days before `train` (all of them if None). Newey-West errors with lag
    max(horizon - 1, the automatic rule): an h-day average target overlaps its neighbours by
    h - 1 days, so the residuals are MA(h - 1) at least.
    """
    rv = np.asarray(rv, float)
    if log and np.any(rv <= 0):
        raise ValueError("log-HAR needs strictly positive realised variance")
    z = np.log(rv) if log else rv
    X = regressors(z, windows)
    y = targets(z, horizon)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if train is not None:
        ok &= np.arange(len(rv)) + horizon < train          # the target must be inside the window
    Xo, yo = X[ok], y[ok]
    if len(yo) < 3 * X.shape[1]:
        raise ValueError(f"{len(yo)} usable days cannot support {X.shape[1]} coefficients")
    coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    e = yo - Xo @ coef
    auto = int(math.floor(4.0 * (len(yo) / 100.0) ** (2.0 / 9.0)))
    L = max(horizon - 1, auto) if lag is None else int(lag)
    se = np.sqrt(np.clip(np.diag(_nw_cov(Xo, e, L)), 0, None))
    names = ("const",) + tuple({1: "daily", 5: "weekly", 22: "monthly"}.get(w, f"{w}d") for w in windows)
    return HAR(coef, se, int(horizon), bool(log), tuple(windows), float(e.var(ddof=Xo.shape[1])),
               int(len(yo)), names)


def forecast(model, rv):
    """
    The forecast of mean RV over the next `horizon` days, known at the close of each day,
    in LEVELS. For the log-HAR the lognormal correction exp(s^2 / 2) is applied -- dropping
    it biases every forecast low by that factor, which QLIKE punishes. Floored at 1e-12.
    """
    rv = np.asarray(rv, float)
    z = np.log(rv) if model.log else rv
    X = regressors(z, model.windows)
    f = X @ model.coef
    if model.log:
        f = np.exp(f + 0.5 * model.resid_var)
    return np.maximum(f, 1e-12)


def qlike(target, forecast_):
    """Patton's (2011) QLIKE, averaged over the finite pairs; 0 at a perfect forecast."""
    t = np.asarray(target, float)
    f = np.asarray(forecast_, float)
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0) & (t > 0)
    r = t[ok] / f[ok]
    return float(np.mean(r - np.log(r) - 1.0))
