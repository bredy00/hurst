"""
The Black-Litterman portfolio, rebalanced with costs (Session L trial).

Every rebalance (weekly) builds the portfolio from information before that day only:

  risk model   B, D   loadings and idiosyncratic variances from a trailing year of OLS
               C      factor correlation over the same year
               sigma  each factor's volatility, forecast with the rough-volatility predictor
                      at Hurst index H (portfolio/rough_forecast.py) from its log realised
                      variance, level-matched over two years
               Sigma_opt = B diag(sigma) C diag(sigma) B' + D, the short-run risk forecast
  equilibrium  Sigma_bar from three years of factor covariance: the market's long-run view,
               so Pi = delta_eq Sigma_bar w_mkt does not move with the day's volatility
  views        refreshed at each signal period (21 days): momentum (12-1 month return) and
               value, each top third minus bottom third, with Q = (2%, 1.5%) a year: modest,
               because a diversified long-short view has little variance and Black-Litterman
               sizes it by that (4% and 3% put the gross exposure above 2)
  weights      Black-Litterman with Omega = diag(P tau Sigma_bar P') / c, then
               w = (delta (Sigma_opt + M))^-1 mu_BL, the rest in cash, gross capped at 3

When Sigma_opt = Sigma_bar and delta = delta_eq this is the market plus the view tilts.
When forecast volatility runs above its long-run level the risky position shrinks, below
it grows: the Black-Litterman portfolio is volatility-managed by construction, and H sets
how jagged that management is. Trades cost `cost_bp` per unit of turnover, optionally only
beyond a no-trade band. Returns are excess of cash (rf = 0 in the demo market).
"""

import math
from dataclasses import dataclass, replace

import numpy as np

import portfolio.black_litterman as bl
import portfolio.rough_forecast as rf

REBALANCE_EVERY = 5
PERIOD = 21
BETA_WINDOW = 252
EQ_WINDOW = 756
FORECAST_LAGS = 504
LEVEL_WINDOW = 504
HORIZON = 5
START = FORECAST_LAGS + LEVEL_WINDOW          # first day the risk model is complete (> EQ_WINDOW)


@dataclass(frozen=True)
class Config:
    H: float = 0.10
    confidence: float = 1.0
    delta: float = 2.5
    delta_eq: float = 2.5
    tau: float = 0.05
    band: float = 0.0
    cost_bp: float = 10.0
    q_mom: float = 0.02
    q_val: float = 0.015
    gross_cap: float = 3.0

    def with_theta(self, theta, names):
        """Config with the filter's coordinates applied: H as is, the rest as logs."""
        kw = {}
        for n, v in zip(names, theta):
            kw[n] = float(v) if n == "H" else float(math.exp(v))
        return replace(self, **kw)


class Engine:
    """Precomputes everything that does not depend on the configuration, once per market."""

    def __init__(self, market):
        self.m = market
        T, N = market.T, market.N
        self.days = np.arange(START, T, REBALANCE_EVERY)
        self.period_starts = np.arange(START, T, PERIOD)
        self._risk = {}
        self._vol_cache = {}
        logrv = np.log(np.maximum(market.RV, 1e-300))
        self.logrv = logrv
        for t in self.days:
            X = np.column_stack([np.ones(BETA_WINDOW), market.F[t - BETA_WINDOW:t]])
            Y = market.R[t - BETA_WINDOW:t]
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            B = coef[1:].T                                          # (N, 4)
            resid = Y - X @ coef
            D = resid.var(axis=0, ddof=5) * 252
            C = np.corrcoef(market.F[t - BETA_WINDOW:t].T)
            Sf_bar = np.cov(market.F[t - EQ_WINDOW:t].T) * 252
            S_bar = B @ Sf_bar @ B.T + np.diag(D)
            self._risk[int(t)] = (B, D, C, S_bar, market.w_mkt(t))
        cum = np.concatenate([np.zeros((1, N)), np.cumsum(np.log1p(market.R), axis=0)])
        self._views = {}
        for t in self.period_starts:
            mom = cum[t - PERIOD] - cum[t - BETA_WINDOW]                # 12-1 month return
            self._views[int(t)] = np.vstack([_long_short(mom), _long_short(market.value)])

    def factor_vol(self, H):
        """(T, 4) annualised factor vols forecast at each day's open, for this H (cached)."""
        key = round(float(H), 6)
        if key not in self._vol_cache:
            a = rf.gjr_weights(H, HORIZON, FORECAST_LAGS)
            out = np.full((self.m.T, 4), np.nan)
            for k in range(4):
                v = np.exp(rf.forecast_log(self.logrv[:, k], a))[:self.m.T]
                scale = rf.level_scale(self.m.RV[:, k], v, LEVEL_WINDOW)
                lv = np.concatenate([[np.nan], scale[:-1]])             # level known before the day
                out[:, k] = np.sqrt(252 * v * lv)
            if len(self._vol_cache) > 64:
                self._vol_cache.pop(next(iter(self._vol_cache)))
            self._vol_cache[key] = out
        return self._vol_cache[key]

    def view_matrix(self, t):
        p0 = self.period_starts[self.period_starts <= t][-1]
        return self._views[int(p0)]

    def target(self, t, cfg):
        B, D, C, S_bar, w_mkt = self._risk[int(t)]
        sig = self.factor_vol(cfg.H)[t]
        Sf = C * np.outer(sig, sig)
        S_opt = B @ Sf @ B.T + np.diag(D)
        P = self.view_matrix(t)
        Q = np.array([cfg.q_mom, cfg.q_val])
        Pi = bl.equilibrium(S_bar, w_mkt, cfg.delta_eq)
        Om = bl.omega(P, S_bar, cfg.tau, cfg.confidence)
        mu, M = bl.posterior(Pi, S_bar, P, Q, Om, cfg.tau)
        w = bl.weights(mu, S_opt, M, cfg.delta)
        g = float(np.abs(w).sum())
        return w * (cfg.gross_cap / g) if g > cfg.gross_cap else w

    def run(self, cfg, start=START, end=None, w0=None, cfg_path=None):
        """
        Daily results from `start` to `end` (exclusive), holdings `w0` at the open of `start`
        (none: the portfolio is built on the first day at full cost). `cfg_path(t)` may return
        a Config per day (the dual filter); otherwise `cfg` throughout.
        """
        m = self.m
        end = m.T if end is None else end
        n = end - start
        out = {k: np.zeros(n) for k in ("gross", "cost", "net", "turnover", "beta_true", "gross_exposure")}
        w = np.zeros(m.N) if w0 is None else np.array(w0, float)
        for i, t in enumerate(range(start, end)):
            c = cfg if cfg_path is None else cfg_path(t)
            if (t - START) % REBALANCE_EVERY == 0:
                tgt = self.target(t, c)
                trade = tgt - w
                if c.band > 0:
                    trade = np.where(np.abs(trade) > c.band, trade, 0.0)
                to = float(np.abs(trade).sum())
                w = w + trade
                cost = to * c.cost_bp * 1e-4
            else:
                to = cost = 0.0
            r = m.R[t]
            gross = float(w @ r)
            out["gross"][i], out["cost"][i], out["net"][i] = gross, cost, gross - cost
            out["turnover"][i] = to
            out["beta_true"][i] = float(w @ m.B[:, 0])
            out["gross_exposure"][i] = float(np.abs(w).sum())
            w = w * (1.0 + r) / (1.0 + gross - cost)
        out["w_end"] = w
        out["days"] = np.arange(start, end)
        return out


def _long_short(score):
    """Equal-weighted top third minus bottom third of a score."""
    n = len(score)
    k = n // 3
    order = np.argsort(score)
    p = np.zeros(n)
    p[order[-k:]] = 1.0 / k
    p[order[:k]] = -1.0 / k
    return p
