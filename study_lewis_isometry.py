"""
The Lewis contour, fBm, the Ito isometry and the tail at the cut-off (Session G).

Four questions from the 2026-09-15 review, answered with measurements:

 1. Does the (checked) Lewis contour "solve the fBm model"?
    No, and it does not need to. Rough Heston's variance is driven by a
    Riemann-Liouville kernel K(t) = t^(H-1/2) / Gamma(H+1/2) against a Brownian
    motion -- not by fractional Brownian motion increments. The characteristic
    function comes from the lifted Riccati system; Lewis integrates it on
    Im u = -1/2. fBm (Davies-Harte) appears only in the fixtures that test the
    roughness ESTIMATORS. What the lift must reproduce is the kernel's local
    roughness: section B measures the variogram exponent of the lifted process.

 2. Is it Ito-isometric?
    The lift is a finite-dimensional Ito diffusion, so the isometry holds for it
    exactly: Var[int_0^t K_N(t-s) dW] = int_0^t K_N(s)^2 ds. The question that
    matters is how close that is to the TRUE kernel's isometry,
    int_0^t K(s)^2 ds = t^(2H) / (2H Gamma(H+1/2)^2). Section A.

 3. Does it need to read the tail at the cut-off?
    Yes. |phi(u - i/2)| <= 1 gives a model-free bound on the truncated Lewis mass,
    e^(k/2) / (pi U), which is useless (U ~ 3e5 for 1e-6), so the pricer reads
    the integrand where it stops. Section D shows what a fixed cut-off does to
    short-dated implied vols -- exactly where H is identified.

 4. What does this do to analytics on real data? See the Session G report.

Also section C: the cf itself carries the volatility drag. E[X_T] = -i phi'(0)
must equal -1/2 E[int_0^T V dt], known in closed form from the affine mean, and
phi(-i) = 1 is the martingale property.

    python study_lewis_isometry.py        (~2 minutes) -> captures/lewis_isometry.png, .json
"""

import json
import math
import pathlib
import time

import numpy as np

import models.rough_heston as rh
import pricing.fourier as fo

ROOT = pathlib.Path(__file__).parent
P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
YEAR_MIN = 365.0 * 24 * 60


def isometry_exact(t, H):
    a = H + 0.5
    return np.asarray(t, float) ** (2 * H) / (2 * H * math.gamma(a) ** 2)


def isometry_lift(t, w, x):
    r = x[:, None] + x[None, :]
    t = np.atleast_1d(np.asarray(t, float))
    ww = np.outer(w, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(r[None] > 0, -np.expm1(-r[None] * t[:, None, None]) / np.where(r > 0, r, 1.0)[None],
                     t[:, None, None])
    return (ww[None] * g).sum(axis=(1, 2))


def section_a():
    """Ito isometry: lifted vs true kernel, and a Monte Carlo check of the lift's own isometry."""
    ts = np.exp(np.linspace(math.log(1.0 / YEAR_MIN), math.log(2.0), 300))
    out = {"t": ts.tolist(), "curves": {}}
    for H in (0.05, 0.12, 0.30):
        for N in (24, 32):
            w, x = rh.lift_nodes(H, N)
            rel = isometry_lift(ts, w, x) / isometry_exact(ts, H) - 1.0
            out["curves"][f"H={H} N={N}"] = rel.tolist()
    w, x = rh.lift_nodes(0.12, 24)
    band = (ts >= 1.0 / 365) & (ts <= 2.0)
    hour = (ts >= 1.0 / (365 * 24)) & (ts <= 2.0)
    rel = isometry_lift(ts, w, x) / isometry_exact(ts, 0.12) - 1.0
    out["max_rel_day_2y"] = float(np.max(np.abs(rel[band])))
    out["max_rel_hour_2y"] = float(np.max(np.abs(rel[hour])))
    out["rel_at_5min"] = float(isometry_lift([5 / YEAR_MIN], w, x)[0] / isometry_exact(5 / YEAR_MIN, 0.12) - 1)
    # the lift is its own exact isometry: simulated variance vs the formula
    dt, n_steps, n_paths = 1.0 / (365 * 24), 24 * 30, 40_000
    rng = np.random.default_rng(7)
    dZ = rng.standard_normal((n_steps, n_paths)) * math.sqrt(dt)
    X = rh.simulate_lifted_gaussian(w, x, dZ, dt)
    t_end = n_steps * dt
    # discrete sum: Var = sum_k K_N(t_n - t_k)^2 dt with lags dt ... n dt (the recursion's own convolution)
    lags = np.arange(1, n_steps + 1) * dt
    kv = rh.kernel_approx(lags, w, x)
    var_formula = float(np.sum(kv ** 2) * dt)
    var_mc = float(X[-1].var())
    se = var_mc * math.sqrt(2.0 / n_paths)
    out["mc_var"], out["mc_formula"], out["mc_z"] = var_mc, var_formula, (var_mc - var_formula) / se
    out["mc_vs_continuous_isometry"] = float(var_formula / isometry_lift([t_end], w, x)[0] - 1)
    return out


def section_b():
    """Local roughness: the variogram exponent of the lifted process vs 2H."""
    H = 0.12
    t = 1.0
    lags = np.exp(np.linspace(math.log(1.0 / YEAR_MIN), math.log(30.0 / 365), 60))
    s = np.exp(np.linspace(math.log(1e-9), math.log(t), 4000))
    out = {"lag": lags.tolist()}
    for N in (24, 32):
        w, x = rh.lift_nodes(H, N)
        v = []
        for d in lags:
            first = isometry_lift([d], w, x)[0]
            k1 = rh.kernel_approx(s + d, w, x) - rh.kernel_approx(s, w, x)
            second = float(np.trapezoid(k1 ** 2, s))
            v.append(first + second)
        v = np.array(v)
        slope = np.gradient(np.log(v), np.log(lags))
        out[f"slope_N{N}"] = slope.tolist()
    a = H + 0.5
    ex = []
    for d in lags:
        first = d ** (2 * H) / (2 * H * math.gamma(a) ** 2)
        k1 = ((s + d) ** (a - 1) - s ** (a - 1)) / math.gamma(a)
        ex.append(first + float(np.trapezoid(k1 ** 2, s)))
    out["slope_exact"] = np.gradient(np.log(ex), np.log(lags)).tolist()
    sel = (lags >= 1.0 / (365 * 24)) & (lags <= 30.0 / 365)
    out["mean_slope_hour_month_N24"] = float(np.mean(np.array(out["slope_N24"])[sel]))
    out["slope_at_1min_N24"] = float(out["slope_N24"][0])
    return out


def section_c():
    """The cf carries the martingale property and the volatility drag."""
    out = []
    w, x = rh.lift_nodes(P.H, rh.N_DEFAULT)
    for d_ in (1, 30, 365):
        tau = d_ / 365
        eps = 1e-3
        lp = rh.log_char_func(np.array([eps, 0.0, -eps], dtype=complex), tau, P, steps_mult=2.0)
        mean_cf = float(((lp[0] - lp[2]) / (2 * eps)).imag)
        var_cf = float(-((lp[0] - 2 * lp[1] + lp[2]) / eps ** 2).real)
        st = rh.LiftedAffineStep(w, x, P.v0, P.kappa, P.theta, P.xi, tau)
        drag = -0.5 * st.EI_const
        mart = abs(complex(rh.char_func(-1j, tau, P)) - 1.0)
        out.append({"days": d_, "E[X] from cf": mean_cf, "-1/2 E[int V] closed form": drag,
                    "rel": abs(mean_cf / drag - 1), "Var[X] from cf": var_cf, "|phi(-i) - 1|": mart})
    Xq = rh.simulate_qe(P, 30 / 365, 200_000, 120, seed=11)
    out[1]["Var[X] QE Monte Carlo"] = float(Xq.var())
    out[1]["E[X] QE Monte Carlo"] = float(Xq.mean())
    out[1]["E[X] MC se"] = float(Xq.std() / math.sqrt(len(Xq)))
    return out


def lewis_truncated(ks, tau, p, U, n_nodes=4000):
    """Lewis price with the integral cut at U, on a fine grid (reference for the truncation effect)."""
    x, wq = fo._leggauss(64)
    edges = np.concatenate([[0.0], np.geomspace(1.0, U, 60)])
    lo, hi = edges[:-1], edges[1:]
    half = 0.5 * (hi - lo)
    v = (half[:, None] * (x[None, :] + 1.0) + lo[:, None]).ravel()
    wv = (half[:, None] * wq[None, :]).ravel()
    phi = rh._safe_exp(rh.log_char_func(v - 0.5j, tau, p))
    dens = phi * (1.0 / (v * v + 0.25))
    return 1.0 - np.exp(0.5 * ks) / math.pi * fo._real_transform(ks, v, dens, wv), v, phi


def section_d():
    out = {"decay": {}, "truncation": {}}
    for d_ in (1, 7, 30, 365):
        tau = d_ / 365
        u = np.geomspace(0.5, 4000.0, 160)
        phi = np.abs(rh._safe_exp(rh.log_char_func(u - 0.5j, tau, P)))
        out["decay"][d_] = {"u": u.tolist(), "absphi": phi.tolist(),
                            "u_max_for": rh.u_max_for(P, tau)}
    for d_ in (1, 7, 30):
        tau = d_ / 365
        band = 2.5 * math.sqrt(P.v0 * tau)
        ks = np.array([-band, 0.0, band])
        ref, info = rh.lewis_prices(ks, tau, P, tol=1e-12)
        iv_ref = fo.implied_vols_from_calls(ref, ks, tau)
        rows = []
        for U in (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0):
            c, _, _ = lewis_truncated(ks, tau, P, U)
            iv = fo.implied_vols_from_calls(c, ks, tau)
            rows.append({"U": U, "price_err": np.abs(c - ref).tolist(),
                         "iv_err_vp": (100 * np.abs(iv - iv_ref)).tolist(),
                         "model_free_bound": (np.exp(0.5 * ks) / (math.pi * U)).tolist()})
        out["truncation"][d_] = {"k": ks.tolist(), "rows": rows, "checked_u_max": info["u_max"],
                                 "checked_extended": info["extended"]}
    return out


def section_e():
    """
    Does the isometry gap move PRICES? The lift against the true rough Heston cf
    (fractional Adams, models.rough_heston_adams), priced through the same Lewis
    grid, for the default lift and a finer one. Adams' own error: M vs 2M.
    """
    import models.rough_heston_adams as ad
    out = {}
    lifts = (("N=24, eta_N=1e5 (shipped)", dict(N=24, eta_N=1e5)),
             ("N=40, eta_N=1e8", dict(N=40, eta_N=1e8)))
    for d_, M in ((1, 1500), (7, 2000), (30, 3000), (90, 3000)):
        tau = d_ / 365
        s = math.sqrt(P.v0 * tau)
        ks = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0]) * s
        um = 1.3 * rh.u_max_for(P, tau)
        t0 = time.perf_counter()
        ref = ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, M), um)
        ref2 = ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, 2 * M), um)
        iv_ref = fo.implied_vols_from_calls(ref2, ks, tau)
        iv_ref1 = fo.implied_vols_from_calls(ref, ks, tau)
        row = {"k_over_sigma_sqrt_tau": [-2.5, -1.0, -0.5, 0.0, 0.5, 1.0],
               "adams_iv": iv_ref.tolist(), "adams_self_error_vp": (100 * np.abs(iv_ref - iv_ref1)).tolist(),
               "adams_seconds": time.perf_counter() - t0}
        skew = lambda iv: (iv[4] - iv[2]) / ks[4]
        row["adams_atm_skew"] = float(skew(iv_ref))
        for name, kw in lifts:
            c = ad.lewis_prices_from_logcf(ks, lambda z: rh.log_char_func(z, tau, P, steps_mult=2.0, **kw), um)
            iv = fo.implied_vols_from_calls(c, ks, tau)
            row[name] = {"iv_err_vp": (100 * (iv - iv_ref)).tolist(), "atm_skew": float(skew(iv)),
                         "skew_err_pct": float(100 * (skew(iv) / skew(iv_ref) - 1))}
        out[d_] = row
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    t = np.array(res["A"]["t"])
    for name, col in (("H=0.05 N=24", "#9ecae1"), ("H=0.12 N=24", "#1f77b4"), ("H=0.3 N=24", "#08306b"),
                      ("H=0.12 N=32", "#2ca02c")):
        short = -100 * np.array(res["A"]["curves"][name])
        a.loglog(t * YEAR_MIN, np.maximum(short, 1e-3), color=col, label=name)
    a.set_ylim(0.3, 100)
    for lab, tm in (("5 min", 5), ("1 h", 60), ("1 d", 1440), ("1 y", 525600)):
        a.axvline(tm, color="0.8", lw=0.8)
        a.text(tm * 1.1, 0.4, lab, fontsize=7, color="0.35")
    a.set_xlabel("t (minutes)")
    a.set_ylabel("shortfall of the lift's isometry (%)")
    a.set_title("(A) Ito isometry: 1 - int K_N^2 / int K^2   (21.6% at 1 day, N = 24)")
    a.legend(fontsize=8)
    a = ax[0, 1]
    lag = np.array(res["B"]["lag"]) * YEAR_MIN
    a.semilogx(lag, res["B"]["slope_exact"], color="0.3", lw=2, label="true Riemann-Liouville kernel")
    a.semilogx(lag, res["B"]["slope_N24"], color="#1f77b4", label="lift, N = 24")
    a.semilogx(lag, res["B"]["slope_N32"], "--", color="#2ca02c", label="lift, N = 32")
    a.axhline(2 * 0.12, color="#d62728", ls=":", label="2H = 0.24")
    a.axhline(1.0, color="0.7", ls=":", lw=1)
    a.set_xlabel("lag (minutes)")
    a.set_ylabel("local exponent d log E[dX^2] / d log lag")
    a.set_title("(B) roughness: the variogram exponent, H = 0.12")
    a.legend(fontsize=8)
    a = ax[1, 0]
    for d_, col in ((1, "#d62728"), (7, "#ff7f0e"), (30, "#2ca02c"), (365, "#1f77b4")):
        dd = res["D"]["decay"].get(d_) or res["D"]["decay"][str(d_)]   # JSON keys are strings
        a.loglog(dd["u"], dd["absphi"], color=col, label=f"{d_} d")
        a.axvline(dd["u_max_for"], color=col, ls=":", lw=1)
    a.set_ylim(1e-16, 2)
    a.set_xlabel("u")
    a.set_ylabel("|phi(u - i/2)|  (<= 1 for any martingale)")
    a.set_title("(D1) the Lewis integrand's tail by maturity (dotted: starting cut-off)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    for d_, col in ((1, "#d62728"), (7, "#ff7f0e"), (30, "#2ca02c")):
        tr = res["D"]["truncation"].get(d_) or res["D"]["truncation"][str(d_)]
        U = [r["U"] for r in tr["rows"]]
        a.loglog(U, [max(max(r["iv_err_vp"][0], r["iv_err_vp"][2]), 1e-12) for r in tr["rows"]], "o-", color=col,
                 label=f"{d_} d, 2.5-sigma wings")
        a.loglog(U, [max(r["iv_err_vp"][1], 1e-12) for r in tr["rows"]], "s--", color=col, alpha=0.6,
                 label=f"{d_} d, ATM")
    a.axhspan(0.1, 5.0, color="#fdd0a2", alpha=0.4, label="real quote noise, 0.1-5 vp")
    a.set_xlabel("fixed cut-off U")
    a.set_ylabel("implied-vol error from truncation (vol points)")
    a.set_title("(D2) what a fixed cut-off does -- short maturities, where H lives")
    a.legend(fontsize=7)
    fig.suptitle("Lewis contour, Ito isometry of the lift, and the tail at the cut-off (rough Heston, H = 0.12)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "lewis_isometry.png", dpi=125)


def main():
    t0 = time.perf_counter()
    res = {}
    res["A"] = section_a()
    A = res["A"]
    print(f"A  isometry, H=0.12 N=24: max |rel| {100*A['max_rel_day_2y']:.2f}% on [1 d, 2 y], "
          f"{100*A['max_rel_hour_2y']:.2f}% on [1 h, 2 y], {100*A['rel_at_5min']:+.1f}% at 5 minutes; "
          f"MC of the lift's own isometry z = {A['mc_z']:+.2f} (discrete vs continuous {100*A['mc_vs_continuous_isometry']:+.2f}%)  "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    res["B"] = section_b()
    print(f"B  variogram exponent, N=24: mean {res['B']['mean_slope_hour_month_N24']:.3f} over [1 h, 1 month] "
          f"(2H = 0.24); {res['B']['slope_at_1min_N24']:.2f} at one minute  ({time.perf_counter()-t0:.0f}s)", flush=True)
    res["C"] = section_c()
    for r in res["C"]:
        print(f"C  {r['days']:3d} d: E[X] cf {r['E[X] from cf']:+.8f} vs -1/2 E[int V] {r['-1/2 E[int V] closed form']:+.8f} "
              f"(rel {r['rel']:.1e}); |phi(-i)-1| {r['|phi(-i) - 1|']:.1e}; Var[X] cf {r['Var[X] from cf']:.6f}"
              + (f"; QE MC Var {r['Var[X] QE Monte Carlo']:.6f}, E[X] {r['E[X] QE Monte Carlo']:+.6f} +/- {r['E[X] MC se']:.6f}"
                 if "Var[X] QE Monte Carlo" in r else ""), flush=True)
    res["D"] = section_d()
    res["E"] = section_e()
    for d_, row in res["E"].items():
        print(f"E  {d_:2d} d: Adams ATM iv {100*row['adams_iv'][3]:.3f}% (own error <= {max(row['adams_self_error_vp']):.1e} vp); "
              + "; ".join(f"{name}: iv error {min(row[name]['iv_err_vp']):+.2f}..{max(row[name]['iv_err_vp']):+.2f} vp, "
                          f"ATM skew {row[name]['skew_err_pct']:+.1f}%"
                          for name in row if name.startswith("N=")), flush=True)
    for d_, tr in res["D"]["truncation"].items():
        print(f"D  {d_:2d} d: checked cut-off {tr['checked_u_max']:.0f} (extended {tr['checked_extended']}x); "
              "wing IV error by fixed U: " + ", ".join(
                  f"U={r['U']:g}: {max(r['iv_err_vp'][0], r['iv_err_vp'][2]):.2e} vp" for r in tr["rows"]), flush=True)
    plot(res)
    (ROOT / "captures" / "lewis_isometry.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    print(f"({time.perf_counter()-t0:.0f}s)")
    return res


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        plot(json.loads((ROOT / "captures" / "lewis_isometry.json").read_text(encoding="utf-8")))
    else:
        main()
