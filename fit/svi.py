"""
Raw SVI slice fitting, with no scipy.

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

Five parameters replace fifteen-odd noisy quotes, extrapolate sanely into the
wings, and -- once the Durrleman condition holds -- give a slice that cannot be
butterfly-arbitraged. That last part is why this is worth doing at all: the raw
grid can be dented by a single bad quote, and a fitted arbitrage-free slice
cannot.

Why no scipy: this module is imported by the live app, and `import scipy.optimize`
costs ~2.8s on this machine. It is not needed. Holding (m, sigma) fixed makes the
problem LINEAR in the remaining three parameters -- substitute y = (k-m)/sigma and

    w = a + c*y + d*sqrt(y^2 + 1),      c = b*sigma*rho,  d = b*sigma

so the inner solve is one numpy lstsq, and only a 2-D search over (m, sigma)
remains. That is the Zeliade quasi-explicit reduction, and a coarse grid plus a
compass search handles two dimensions comfortably.
"""

import numpy as np

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


def raw_svi(k, a, b, rho, m, sigma):
    """Total variance w(k) under raw SVI."""
    y = np.asarray(k, dtype=float) - m
    return a + b * (rho * y + np.sqrt(y * y + sigma * sigma))


def svi_derivatives(k, a, b, rho, m, sigma):
    """(w, w', w'') with respect to k, analytically."""
    y = np.asarray(k, dtype=float) - m
    r = np.sqrt(y * y + sigma * sigma)
    w = a + b * (rho * y + r)
    dw = b * (rho + y / r)
    d2w = b * (sigma * sigma) / (r ** 3)
    return w, dw, d2w


def durrleman_function(k, params):
    """
    Gatheral's g(k). The slice is free of butterfly arbitrage iff g >= 0
    everywhere:

        g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2
    """
    w, dw, d2w = svi_derivatives(k, **{p: params[p] for p in PARAM_NAMES})
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (1.0 - np.asarray(k, dtype=float) * dw / (2.0 * w)) ** 2
        term2 = (dw * dw / 4.0) * (1.0 / w + 0.25)
        return term1 - term2 + d2w / 2.0


def durrleman_min(params, tau, k_span):
    """Minimum of g(k) over |k| <= k_span. Negative means butterfly arbitrage."""
    p = {name: float(params[name]) for name in PARAM_NAMES}
    g = durrleman_function(np.linspace(-k_span, k_span, 2001), p)
    g = g[np.isfinite(g)]
    return float(np.min(g)) if g.size else float('nan')


def durrleman_ok(params, tau, k_span=None, tol=-1e-10):
    """
    No-butterfly check: the simple necessary conditions AND g(k) >= 0.

    The four textbook conditions (b > 0, |rho| < 1, minimum variance >= 0,
    b(1+|rho|) <= 4/tau) are necessary but NOT sufficient -- a parameter set can
    satisfy every one and still give a negative risk-neutral density -- so g(k)
    is evaluated too.

    `k_span` matters more than it looks. Judging g over a fixed +/-2 in
    log-moneyness tests a region where a short-dated slice has no data at all:
    a 42-day slice fitted over |k| <= 0.196 was scored at min g = -216092
    purely on extrapolation, while over its own data range it sits at +0.113.
    An unconstrained least-squares fit is only meaningfully arbitrage-free where
    it was fitted. Default to twice the fitted span when the fit recorded one,
    and let callers ask about the wings explicitly.
    """
    p = {name: float(params[name]) for name in PARAM_NAMES}
    if not (p["b"] > 0 and p["sigma"] > 0 and abs(p["rho"]) < 1):
        return False
    if p["a"] + p["b"] * p["sigma"] * math_sqrt(1.0 - p["rho"] ** 2) < 0:
        return False
    if tau and p["b"] * (1.0 + abs(p["rho"])) > 4.0 / tau:
        return False
    if k_span is None:
        span = params.get("k_span") if isinstance(params, dict) else None
        k_span = 2.0 * span if span else 2.0
    return durrleman_min(p, tau, k_span) >= tol


def math_sqrt(x):
    return float(np.sqrt(max(x, 0.0)))


def _solve_linear(k, w, m, sigma, sw):
    """
    Inner problem: with (m, sigma) fixed, (a, c, d) is a linear least squares.
    Returns (params_dict, weighted_sse) or (None, inf).
    """
    y = (k - m) / sigma
    A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    Aw = A * sw[:, None]
    try:
        coef, *_ = np.linalg.lstsq(Aw, w * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None, np.inf
    a, c, d = (float(v) for v in coef)
    if d <= 0 or abs(c) >= d:
        return None, np.inf                     # implies b <= 0 or |rho| >= 1
    resid = (A @ coef - w) * sw
    return {"a": a, "b": d / sigma, "rho": c / d, "m": float(m),
            "sigma": float(sigma)}, float(resid @ resid)


def fit_slice(k, w, weights=None, tau=None, m_range=(-0.5, 0.5),
              sigma_range=(1e-3, 2.0), n_m=25, n_sigma=25, refine_steps=60):
    """
    Fit raw SVI to one slice of total variance.

    Returns a dict of the five parameters plus 'rmse' and 'durrleman', or None
    when there are too few points to determine five parameters.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(k) & np.isfinite(w) & (w > 0)
    if weights is None:
        ww = np.ones_like(k)
    else:
        # Broadcast so a scalar weight is accepted rather than crashing on a
        # 0-d index; callers do pass one by mistake.
        ww = np.broadcast_to(np.asarray(weights, dtype=float), k.shape).astype(float)
        ok &= np.isfinite(ww) & (ww > 0)
    k, w, ww = k[ok], w[ok], ww[ok]
    if len(k) < 5:
        return None
    sw = np.sqrt(ww / ww.mean())

    # Coarse grid: m across the observed range, sigma log-spaced
    best, best_sse = None, np.inf
    m_lo = max(m_range[0], k.min() - 0.5 * np.ptp(k))
    m_hi = min(m_range[1], k.max() + 0.5 * np.ptp(k))
    for m in np.linspace(m_lo, m_hi, n_m):
        for sigma in np.geomspace(sigma_range[0], sigma_range[1], n_sigma):
            cand, sse = _solve_linear(k, w, m, sigma, sw)
            if sse < best_sse:
                best, best_sse = cand, sse
    if best is None:
        return None

    # Compass search on (m, log sigma); two dimensions do not need more
    m, log_s = best["m"], np.log(best["sigma"])
    step_m, step_s = 0.5 * (m_hi - m_lo) / n_m, 0.5 * np.log(
        sigma_range[1] / sigma_range[0]) / n_sigma
    for _ in range(refine_steps):
        improved = False
        for dm, ds in ((step_m, 0), (-step_m, 0), (0, step_s), (0, -step_s)):
            cand, sse = _solve_linear(k, w, m + dm, np.exp(log_s + ds), sw)
            if sse < best_sse - 1e-18:
                best, best_sse, m, log_s = cand, sse, m + dm, log_s + ds
                improved = True
                break
        if not improved:
            step_m *= 0.5
            step_s *= 0.5
            if step_m < 1e-10 and step_s < 1e-10:
                break

    best["rmse"] = float(np.sqrt(best_sse / len(k)))
    best["n"] = int(len(k))
    best["k_span"] = float(max(abs(k.min()), abs(k.max())))
    if tau:
        # Where the market actually is, and how far the fit can be trusted
        best["durrleman"] = durrleman_ok(best, tau)
        best["g_min_data"] = durrleman_min(best, tau, best["k_span"])
        best["g_min_wide"] = durrleman_min(best, tau, 2.0)
        best["durrleman_wide"] = best["g_min_wide"] >= -1e-10
    else:
        best["durrleman"] = None
    return best


def essvi_calendar_ok(slices, taus, k_grid=None):
    """
    Across-slice check: total variance must not decrease with maturity at fixed
    k. Fitting each slice independently does not guarantee this, which is
    exactly what eSSVI's shared parameterisation buys; until that is built, the
    honest move is to measure the violation rather than assume it away.
    """
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 121)
    order = np.argsort(taus)
    ws = [raw_svi(k_grid, **{p: slices[i][p] for p in PARAM_NAMES}) for i in order]
    bad = []
    for i in range(1, len(ws)):
        viol = np.where(ws[i] < ws[i - 1] - 1e-12)[0]
        if viol.size:
            bad.append((int(order[i]), k_grid[viol].tolist()))
    return bad
