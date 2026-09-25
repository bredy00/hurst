"""
Fama-French-Carhart attribution for the trial, through falsify when it is there.

falsify (github.com/bredy00/falsify, a sibling checkout at ../falsify or FALSIFY_PATH) is the
backtester whose job is to try to kill the strategy. Three of its parts are used as they are:

  falsify.attribution.fit_factors     OLS on MKT, SMB, HML, MOM with Newey-West (Bartlett)
                                      standard errors, lag floor(4 (T/100)^(2/9))
  falsify.deflated.deflated_sharpe    the Sharpe ratio deflated by every trial the ledger holds
  falsify.cscv.cscv + ArgMax          the probability of backtest overfitting (CSCV)

`ols_nw` is this package's own copy of the regression, so the trial runs (and CI tests it)
without falsify installed; test_portfolio.py checks the two agree to rounding when falsify is
importable. Every fit records which one produced it.
"""

import math
import os
import pathlib
import sys

import numpy as np

NAMES = ("MKT", "SMB", "HML", "MOM")


def _find_falsify():
    here = pathlib.Path(__file__).resolve().parents[2]
    for p in (os.environ.get("FALSIFY_PATH"), here / "falsify"):
        if p and (pathlib.Path(p) / "falsify" / "attribution.py").exists():
            if str(p) not in sys.path:
                sys.path.append(str(p))
            try:
                import falsify.attribution
                import falsify.cscv
                import falsify.deflated
                assert (falsify.attribution.fit_factors and falsify.cscv.cscv
                        and falsify.deflated.deflated_sharpe)
                return str(p)
            except Exception:                 # pragma: no cover - a broken checkout
                return None
    return None


FALSIFY = _find_falsify()


def nw_lag(n):
    """falsify.metrics.newey_west_lag: floor(4 (n/100)^(2/9))."""
    return math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))


def ols_nw(y, F, lag=None):
    """(coef [alpha, b_MKT..b_MOM], standard errors, covariance) with Newey-West (Bartlett)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(F, float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    L = nw_lag(len(y)) if lag is None else lag
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, min(L, len(y) - 1) + 1):
        G = Xe[j:].T @ Xe[:-j]
        S = S + (1 - j / (L + 1)) * (G + G.T)
    iXX = np.linalg.inv(X.T @ X)
    cov = iXX @ S @ iXX
    return coef, np.sqrt(np.clip(np.diag(cov), 0, None)), cov


def fit(y, F, use_falsify=True):
    """One FF4 fit: dict with alpha, betas, their SEs, n, and the source that computed it."""
    if use_falsify and FALSIFY:
        from falsify.attribution import fit_factors
        f = fit_factors(np.asarray(y, float), np.asarray(F, float), NAMES)
        return {"alpha": f.alpha, "betas": np.array(f.betas), "se_alpha": f.alpha_stderr,
                "se": np.array(f.beta_stderrs), "n": f.n_obs, "source": "falsify"}
    coef, se, _ = ols_nw(y, F)
    return {"alpha": float(coef[0]), "betas": coef[1:], "se_alpha": float(se[0]), "se": se[1:],
            "n": len(y), "source": "internal"}


def period_fits(net, F, days, period_starts, use_falsify=True):
    """An FF4 fit per signal period: net daily returns on `days`, periods starting at `period_starts`."""
    out = []
    idx = {int(d): i for i, d in enumerate(days)}
    for k, p0 in enumerate(period_starts):
        p1 = period_starts[k + 1] if k + 1 < len(period_starts) else days[-1] + 1
        sel = [idx[d] for d in range(int(p0), int(p1)) if d in idx]
        if len(sel) < 10:
            continue
        f = fit(net[sel], F[days[sel]], use_falsify)
        f["period"], f["first_day"] = k, int(p0)
        out.append(f)
    return out


def deflated_sharpe(net, all_trial_sharpes):
    """falsify's DSR when available; otherwise the same formula here (Bailey & Lopez de Prado 2014)."""
    if FALSIFY:
        from falsify.deflated import deflated_sharpe as dsr
        return float(dsr(np.asarray(net, float), np.asarray(all_trial_sharpes, float))), "falsify"
    from scipy.stats import kurtosis, norm, skew
    r = np.asarray(net, float)
    tr = np.asarray(all_trial_sharpes, float)
    g = 0.5772156649015329
    n = tr.size
    sr0 = math.sqrt(np.var(tr, ddof=1)) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e))) \
        if n >= 2 else 0.0
    sr = r.mean() / r.std(ddof=1)
    v = 1 - skew(r, bias=False) * sr + (kurtosis(r, fisher=False, bias=False) - 1) / 4 * sr * sr
    return float(norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / math.sqrt(v))), "internal"


def pbo(returns_matrix, n_blocks=16):
    """falsify's CSCV probability of backtest overfitting, ArgMax rule; None without falsify."""
    if not FALSIFY:
        return None
    from falsify.cscv import cscv
    from falsify.selection import ArgMax
    res = cscv(np.asarray(returns_matrix, float), ArgMax(), n_blocks=n_blocks)
    return {"pbo": res.pbo(), "median_logit": res.median_logit(),
            "degradation": res.performance_degradation(), "splits": res.n_splits}
