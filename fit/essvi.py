"""
eSSVI: a volatility surface with no calendar or butterfly arbitrage by construction, and no
scipy (Session O).

Raw SVI (fit/svi.py) fits each expiry on its own, so nothing stops two fitted slices from
crossing; `svi.essvi_calendar_ok` has measured that since Session A without preventing it.
The extended SSVI of Hendriks & Martini (2019) describes a slice by three numbers,

    w(k) = 1/2 [ theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2)) ],

theta the ATM total variance, rho the skew's sign and shape, psi = theta phi the scale of the
wings. Each slice is a raw SVI (`to_raw_svi`), so fit/svi.py's Durrleman check applies to it.

**Butterfly.** A slice is free of butterfly arbitrage if psi (1 + |rho|) < 4 and
psi^2 (1 + |rho|) <= 4 theta (Gatheral & Jacquier 2014, Theorem 4.2; sufficient). Checked here
on 20,000 random slices satisfying it, converted to raw SVI: Durrleman's g stayed above +0.21.

**Calendar.** Consecutive slices must not cross: w_{i+1}(k) >= w_i(k) for every k. Two
necessary conditions are cheap -- theta non-decreasing, and the wings' asymptotic slopes
ordered, psi_{i+1} - psi_i >= |rho_{i+1} psi_{i+1} - rho_i psi_i| -- but they are NOT
sufficient: of 200,000 random pairs satisfying both, 37,483 still crossed, by up to 0.19 in
total variance (a slice with rho near 1 whose left wing's intercept collapses). So a candidate
here must also clear the previous slice on a dense grid spanning |k| <= 10, far past any
quote, and beyond that as far as it takes to PROVE the order holds for good (`_wing_limit`):
ordered slopes alone do not keep it when the new wing's intercept is lower.

**The fit** is sequential in maturity (the approach of Corbetta, Cohort, Laachir & Martini
2019): each slice is fitted with the one before it as a constraint, so every candidate the
search considers is already arbitrage-free, and so is the surface that comes out. A coarse grid
over (theta, rho, psi) inside the feasible set, then a compass search that refuses infeasible
steps -- three dimensions, no scipy, for the same reason as svi.py.
"""

import math

import numpy as np

import fit.svi as svi

K_CHECK = np.concatenate([-np.geomspace(10.0, 1e-4, 400), [0.0], np.geomspace(1e-4, 10.0, 400)])
RHO_MAX = 0.999


def essvi_w(k, theta, rho, psi):
    """Total variance of one eSSVI slice at log-moneyness k."""
    k = np.asarray(k, float)
    return 0.5 * (theta + rho * psi * k + np.sqrt((psi * k + theta * rho) ** 2 + theta * theta * (1.0 - rho * rho)))


def to_raw_svi(theta, rho, psi):
    """The same slice as raw SVI parameters (Gatheral & Jacquier 2014, Lemma 3.2)."""
    phi = psi / theta
    return {"a": 0.5 * theta * (1.0 - rho * rho), "b": 0.5 * psi, "rho": rho, "m": -rho / phi,
            "sigma": math.sqrt(1.0 - rho * rho) / phi}


def butterfly_ok(theta, rho, psi):
    return theta > 0 and psi > 0 and abs(rho) < 1 and psi * (1 + abs(rho)) < 4 and psi * psi * (1 + abs(rho)) <= 4 * theta


def _wing_limit(prev, theta, rho, psi, side):
    """
    |k| beyond which the new slice provably stays above the previous one on one side (+1 right,
    -1 left), or inf when that cannot be proved. On each side a slice lies ABOVE its asymptote
    a + s |k| and at most theta sqrt(1 - rho^2) / 2 above it (sqrt(x^2 + c^2) <= |x| + c), both
    once k is past the asymptote's kink at -theta rho / psi. So the new slice is above the old
    wherever its asymptote clears the old one's by that margin -- a linear condition in |k|.
    """
    t0, r0, p0 = prev["theta"], prev["rho"], prev["psi"]
    a_new, s_new = 0.5 * theta * (1 + side * rho), 0.5 * psi * (1 + side * rho)
    a_old, s_old = 0.5 * t0 * (1 + side * r0), 0.5 * p0 * (1 + side * r0)
    kink = max(0.0, side * (-theta * rho / psi), side * (-t0 * r0 / p0))
    need = a_old - a_new + 0.5 * t0 * math.sqrt(max(1.0 - r0 * r0, 0.0))
    if need <= 0:
        return kink
    if s_new - s_old <= 0:
        return math.inf
    return max(kink, need / (s_new - s_old))


def calendar_ok(prev, theta, rho, psi, tol=1e-12):
    """
    No crossing with the previous slice, anywhere: the two necessary conditions, the dense grid
    on |k| <= 10, and -- where the wings' asymptotes only separate far out -- a grid out to the
    point where `_wing_limit` proves the order holds for good. (A first version stopped at
    |k| = 10 on the claim that ordered asymptotic slopes keep the order beyond it; they do not
    when the new wing's intercept is lower, and a random pair crossed between k = 8 and 23.)
    """
    if prev is None:
        return True
    t0, r0, p0 = prev["theta"], prev["rho"], prev["psi"]
    if theta < t0 - tol or (psi - p0) < abs(rho * psi - r0 * p0) - tol:
        return False
    if not np.all(essvi_w(K_CHECK, theta, rho, psi) >= essvi_w(K_CHECK, t0, r0, p0) - tol):
        return False
    for side in (1, -1):
        lim = _wing_limit(prev, theta, rho, psi, side)
        if not math.isfinite(lim):
            return False
        if lim > K_CHECK[-1]:
            k = side * np.geomspace(K_CHECK[-1], lim * 1.01, 400)
            if not np.all(essvi_w(k, theta, rho, psi) >= essvi_w(k, t0, r0, p0) - tol):
                return False
    return True


def feasible(prev, theta, rho, psi):
    return butterfly_ok(theta, rho, psi) and calendar_ok(prev, theta, rho, psi)


def _sse(params, k, iv, tau, sw):
    w = essvi_w(k, *params)
    model = np.sqrt(np.maximum(w, 0.0) / tau)
    return float(np.sum((sw * (model - iv)) ** 2))


def _atm_total_variance(k, iv, tau):
    order = np.argsort(k)
    return float(np.interp(0.0, k[order], (iv[order] ** 2) * tau))


def fit_slice(k, iv, tau, prev=None, weights=None, n_theta=9, n_rho=41, n_psi=41, refine_steps=80):
    """
    One slice, constrained by the previous one. Fits implied vols (weighted least squares).
    Returns {theta, rho, psi, tau, rmse_vp, n} or None when no feasible point exists or there
    are fewer than 3 quotes for 3 parameters.
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    ww = np.ones_like(k) if weights is None else np.broadcast_to(np.asarray(weights, float), k.shape).astype(float)
    ok &= np.isfinite(ww) & (ww > 0)
    k, iv, ww = k[ok], iv[ok], ww[ok]
    if len(k) < 3:
        return None
    sw = np.sqrt(ww / ww.mean())
    th_obs = _atm_total_variance(k, iv, tau)
    th_lo = max(0.7 * th_obs, prev["theta"] if prev else 0.0)
    th_hi = max(1.3 * th_obs, th_lo * 1.05)
    best, best_sse = None, np.inf
    for theta in np.linspace(th_lo, th_hi, n_theta):
        psi_cap = math.sqrt(4 * theta)
        for rho in np.linspace(-RHO_MAX, RHO_MAX, n_rho):
            hi = min(4.0 / (1 + abs(rho)) * (1 - 1e-9), math.sqrt(4 * theta / (1 + abs(rho))), psi_cap)
            lo = 1e-4 * hi
            if prev is not None:
                lo = max(lo, prev["psi"] * (1 + prev["rho"]) / (1 + rho), prev["psi"] * (1 - prev["rho"]) / (1 - rho))
            if lo >= hi:
                continue
            for psi in np.geomspace(lo, hi, n_psi):
                if not calendar_ok(prev, theta, rho, psi):
                    continue
                e = _sse((theta, rho, psi), k, iv, tau, sw)
                if e < best_sse:
                    best, best_sse = [theta, rho, psi], e
    if best is None:
        return None
    # compass search in (theta, rho, log psi), refusing any step that leaves the feasible set
    steps = [0.5 * (th_hi - th_lo) / n_theta, RHO_MAX / n_rho, 0.5 * math.log(1e4) / n_psi]
    x = [best[0], best[1], math.log(best[2])]
    for _ in range(refine_steps):
        moved = False
        for i in range(3):
            for sgn in (1, -1):
                cand = list(x)
                cand[i] += sgn * steps[i]
                th, rh, ps = cand[0], cand[1], math.exp(cand[2])
                if abs(rh) >= 1 or not feasible(prev, th, rh, ps):
                    continue
                e = _sse((th, rh, ps), k, iv, tau, sw)
                if e < best_sse - 1e-18:
                    x, best_sse, moved = cand, e, True
                    break
            if moved:
                break
        if not moved:
            steps = [s * 0.5 for s in steps]
            if max(steps) < 1e-10:
                break
    theta, rho, psi = x[0], x[1], math.exp(x[2])
    model = np.sqrt(essvi_w(k, theta, rho, psi) / tau)
    return {"theta": theta, "rho": rho, "psi": psi, "tau": float(tau), "n": int(len(k)),
            "rmse_vp": float(100 * np.sqrt(np.mean((model - iv) ** 2)))}


def fit_surface(slices, weights=None):
    """
    `slices`: [(tau, k, iv), ...] in any order. Fitted shortest first, each slice constrained by
    the one before; a slice with no feasible fit is skipped (and reported), never forced.
    """
    order = sorted(range(len(slices)), key=lambda i: slices[i][0])
    out, prev = [], None
    for i in order:
        tau, k, iv = slices[i][:3]
        wts = None if weights is None else weights[i]
        f = fit_slice(k, iv, tau, prev=prev, weights=wts)
        if f is None:
            out.append({"tau": float(tau), "skipped": True})
            continue
        out.append(f)
        prev = f
    return out


def raw_svi_rmse_vp(fit, k, iv, tau):
    """A raw SVI slice's fit error in implied-vol points. (fit/svi.py's own `rmse` is in total
    variance, which is not comparable across maturities and is not a vol.)"""
    if fit is None:
        return float("nan")
    w = svi.raw_svi(np.asarray(k, float), **{p: fit[p] for p in svi.PARAM_NAMES})
    return float(100 * np.sqrt(np.mean((np.sqrt(np.maximum(w, 0.0) / tau) - np.asarray(iv, float)) ** 2)))


def audit_raw_svi(raw_fits, taus, k_wide=1.0):
    """The same audit for independently fitted raw SVI slices (fit/svi.py): the worst Durrleman
    g over |k| <= k_wide, and the worst crossing between consecutive maturities."""
    live = [(t, f) for t, f in sorted(zip(taus, raw_fits), key=lambda z: z[0]) if f is not None]
    if not live:
        return {"durrleman_min": float("nan"), "worst_crossing": float("nan"), "crossing_pairs": 0}
    g = min(svi.durrleman_min(f, t, k_wide) for t, f in live)
    k = np.linspace(-k_wide, k_wide, 801)
    ws = [svi.raw_svi(k, **{p: f[p] for p in svi.PARAM_NAMES}) for _, f in live]
    diffs = [float((b - a).min()) for a, b in zip(ws, ws[1:])]
    return {"durrleman_min": g, "worst_crossing": min([0.0] + diffs),
            "crossing_pairs": int(sum(d < -1e-12 for d in diffs))}


def audit(fits, k_grid=None):
    """The surface's arbitrage, measured the same way whatever produced it: the worst Durrleman
    g over the slices (butterfly) and the worst crossing between consecutive slices (calendar)."""
    k_grid = np.linspace(-1.0, 1.0, 801) if k_grid is None else np.asarray(k_grid, float)
    live = [f for f in fits if not f.get("skipped")]
    g = min(svi.durrleman_min(to_raw_svi(f["theta"], f["rho"], f["psi"]), f["tau"], float(np.max(np.abs(k_grid))))
            for f in live) if live else float("nan")
    cross = 0.0
    for a, b in zip(live, live[1:]):
        d = essvi_w(k_grid, b["theta"], b["rho"], b["psi"]) - essvi_w(k_grid, a["theta"], a["rho"], a["psi"])
        cross = min(cross, float(d.min()))
    return {"durrleman_min": g, "worst_crossing": cross}
