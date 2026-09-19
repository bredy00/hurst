"""
Should the lift go from N = 24 (fastest node 1e5/y, ~5 minutes) to N = 40 (1e8/y, ~0.3 s)?
(Session H decision brief.)

Candidates: the shipped lift (N = 24, eta_N = 1e5), the node count the calibration
already refines to below H ~ 0.08 (N = 32, same fastest node), and the finer lift
(N = 40, eta_N = 1e8). Every consumer of the lift -- pricer, calibration, Kalman,
particle and Fourier filters, simulators -- reads rh.lift_nodes at call time, so
`rh.using_lift` swaps the lift for the whole stack at once: the numbers below are what
adopting each one would actually do, not a re-implementation of it.

  A  price accuracy against the TRUE rough Heston cf (fractional Adams, M and 2M
     steps for its own error), H = 0.05 and 0.12, at 1, 7 and 30 days
  B  the kernel itself: sup-norm error on [1 h, 2 y] and [1 d, 2 y], Ito-isometry
     shortfall at 1 h / 1 d / 1 month / 1 y, local variogram exponent at 1 h and 1 d
     against the exact kernel's
  C  cost on a quiet machine: one Riccati step, the 10-expiry objective through the
     real pricer (with its checks), the exact-moment step build, a QE step for 10k
     paths, Kalman (spot and RV) per day, adapted particle filter and Fourier filter
     per day
  D  the H the pipeline would report:
     D1 calibration -- a 10-expiry surface priced by Adams (the true model), fitted
        from the truth with each lift, H = 0.12 and 0.05
     D2 the realised-variance filter -- 500 days of 5-minute returns simulated with
        a much finer lift (N = 64, eta_N = 1e10) standing in for the true process,
        H profiled with each lift

    python study_finer_lift.py A B D      (~40 minutes)
    python study_finer_lift.py C          (~10 minutes; run with nothing else loading the CPU)
    -> captures/finer_lift.json (sections merge), captures/finer_lift.log
"""

import json
import math
import pathlib
import sys
import time

import numpy as np

import filters.kalman as kf
import models.rough_heston as rh
import models.rough_heston_adams as ad
import pricing.fourier as fo

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "finer_lift.json"
LIFTS = {"N=24, 1e5 (shipped)": (24, 1e5), "N=32, 1e5": (32, 1e5), "N=40, 1e8": (40, 1e8)}
YEAR_S = 365.0 * 24 * 3600
EXPIRIES = (1, 2, 3, 7, 14, 30, 60, 90, 180, 365)
_ORIG_LIFT = rh._LIFT_NODES
lift_as = rh.using_lift            # the whole stack on one lift; see models/rough_heston.py


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ A: prices
def section_a():
    out = {}
    for H in (0.05, 0.12):
        P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, H)
        for d_, M in ((1, 1500), (7, 2000), (30, 3000)):
            tau = d_ / 365
            s = math.sqrt(P.v0 * tau)
            ks = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0]) * s
            um = 1.3 * rh.u_max_for(P, tau)
            t0 = time.perf_counter()
            ref1 = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, M), um), ks, tau)
            ref = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, 2 * M), um), ks, tau)
            skew = lambda iv: (iv[4] - iv[2]) / ks[4]
            row = {"adams_self_error_vp": float(100 * np.max(np.abs(ref - ref1))), "adams_atm_iv": float(ref[3]),
                   "adams_seconds": time.perf_counter() - t0}
            for name, (N, eta) in LIFTS.items():
                iv = fo.implied_vols_from_calls(
                    ad.lewis_prices_from_logcf(ks, lambda z: rh.log_char_func(z, tau, P, N=N, eta_N=eta, steps_mult=2.0), um), ks, tau)
                row[name] = {"worst_vp": float(100 * np.max(np.abs(iv - ref))), "atm_vp": float(100 * (iv[3] - ref[3])),
                             "skew_pct": float(100 * (skew(iv) / skew(ref) - 1))}
            out[f"H={H} {d_}d"] = row
            say(f"A  H={H} {d_:2d}d (Adams own error {row['adams_self_error_vp']:.1e} vp): " + "; ".join(
                f"{n}: worst {r['worst_vp']:.3f} vp, ATM {r['atm_vp']:+.3f} vp, skew {r['skew_pct']:+.1f}%"
                for n, r in row.items() if isinstance(r, dict)))
    return out


# ------------------------------------------------------------------ B: kernel
def iso_lift(t, w, x):
    r = x[:, None] + x[None, :]
    return float((np.outer(w, w) * (-np.expm1(-r * t)) / r).sum())


def section_b():
    out = {}
    s = np.exp(np.linspace(math.log(1e-13), 0.0, 8000))          # t = 1 y of history
    for H in (0.05, 0.12):
        a = H + 0.5

        def incvar_exact(lag):
            k1 = ((s + lag) ** (a - 1) - s ** (a - 1)) / math.gamma(a)
            return lag ** (2 * H) / (2 * H * math.gamma(a) ** 2) + float(np.trapezoid(k1 ** 2, s))
        lags = {"1h": 1 / (365 * 24), "1d": 1 / 365}
        exact_expo = {lab: math.log(incvar_exact(lag * 1.2) / incvar_exact(lag / 1.2)) / math.log(1.44) for lab, lag in lags.items()}
        for name, (N, eta) in LIFTS.items():
            w, x = rh.lift_nodes(H, N, eta_N=eta)
            iso = {lab: 100 * (1 - iso_lift(t, w, x) / (t ** (2 * H) / (2 * H * math.gamma(a) ** 2)))
                   for lab, t in (("1h", 1 / (365 * 24)), ("1d", 1 / 365), ("1m", 30 / 365), ("1y", 1.0))}

            def incvar(lag):
                k1 = rh.kernel_approx(s + lag, w, x) - rh.kernel_approx(s, w, x)
                return iso_lift(lag, w, x) + float(np.trapezoid(k1 ** 2, s))
            expo = {lab: math.log(incvar(lag * 1.2) / incvar(lag / 1.2)) / math.log(1.44) for lab, lag in lags.items()}
            r = {"kernel_err_1d_2y": rh.kernel_error(H, N, eta_N=eta)[0],
                 "kernel_err_1h_2y": rh.kernel_error(H, N, t_lo=1 / (365 * 24), eta_N=eta)[0],
                 "kernel_err_5min_2y": rh.kernel_error(H, N, t_lo=5 / (365 * 24 * 60), eta_N=eta)[0],
                 "isometry_shortfall_pct": iso, "variogram_exponent": expo, "variogram_exponent_exact": exact_expo,
                 "K_N(0)": float(w.sum()), "fastest_node_seconds": float(YEAR_S / x.max())}
            out[f"H={H} {name}"] = r
            say(f"B  H={H} {name}: kernel err 5min/1h/1d-2y {100*r['kernel_err_5min_2y']:.1f}% / {100*r['kernel_err_1h_2y']:.2f}% / "
                f"{100*r['kernel_err_1d_2y']:.2f}%; isometry short " + ", ".join(f"{k} {v:.1f}%" for k, v in iso.items())
                + f"; variogram exponent 1h {expo['1h']:.3f} (exact {exact_expo['1h']:.3f}), 1d {expo['1d']:.3f} "
                f"(exact {exact_expo['1d']:.3f}); K_N(0) {r['K_N(0)']:.0f}, fastest node {r['fastest_node_seconds']:.2g} s")
    return out


# ------------------------------------------------------------------ C: cost
def best_of(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(min(ts))


def section_c():
    import filters.fourier as ff
    import filters.particle as pfm
    from calibrate.fit import blas_single_thread
    P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
    pf = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.01 ** 2)
    rng = np.random.default_rng(5)
    yv = np.abs(0.04 + 0.012 * rng.standard_normal(250))
    out = {}
    for name, (N, eta) in LIFTS.items():
        row = {}
        with lift_as(N, eta), blas_single_thread():
            u = np.linspace(0.0, 100.0, 640) - 0.5j
            row["riccati_step_us_640u"] = 1e6 * best_of(lambda: rh.log_char_func(u, 30 / 365, P, steps=64, N=N)) / 64
            flags = []

            def objective():
                flags.clear()
                for d_ in EXPIRIES:
                    tau = d_ / 365
                    band = 3 * 0.2 * math.sqrt(tau)
                    _, info = rh.call_prices(np.linspace(-band, band, 13), tau, P, N=N)
                    flags.append((bool(info["ok"]), int(bool(info.get("refined", False))), float(info["phi_max"])))
            row["objective_10x13_s"] = best_of(objective, reps=3)
            row["objective_all_ok"] = all(f[0] for f in flags)
            row["objective_phi_max"] = max(f[2] for f in flags)
            w, x = rh.lift_nodes(0.12, N)
            row["exact_step_build_ms"] = 1e3 * best_of(lambda: rh.LiftedAffineStep(w, x, 0.04, 3.0, 0.04, 0.3, 1 / 252), reps=3)
            st = rh.LiftedAffineStep(w, x, 0.04, 3.0, 0.04, 0.3, 1 / (252 * 78))
            Y = np.repeat(st.y_star[:, None], 10000, axis=1)
            g = np.random.default_rng(0)
            row["qe_step_10k_paths_ms"] = 1e3 * best_of(lambda: st.step_y(Y, g))
            m = kf.LiftedRoughModel(1 / 252, H=0.12, v0=0.04)
            row["kalman_spot_per_day_us"] = 1e6 * best_of(lambda: kf.kalman_filter(m, pf, yv), reps=3) / len(yv)
            mr = kf.LiftedRoughRVModel(1 / 252, H=0.12, v0=0.04)
            row["kalman_rv_per_day_us"] = 1e6 * best_of(lambda: kf.kalman_filter(mr, dict(pf, R=1e-6), yv), reps=3) / len(yv)
            apf = pfm.AdaptedParticleFilter(1 / 252, H=0.12, v0=0.04, n_particles=500, seed=1)
            row["particle_500_per_day_ms"] = 1e3 * best_of(lambda: apf.loglik(pf, yv), reps=2) / len(yv)
            fil = ff.FourierFilter(1 / 252, H=0.12, v0=0.04)
            t0 = time.perf_counter()
            fil._coefficients(pf)
            row["fourier_coefficients_s"] = time.perf_counter() - t0
            r = fil.run(pf, yv)
            row["fourier_per_day_ms"] = 1e3 * best_of(lambda: fil.run(pf, yv), reps=2) / len(yv)
            row["fourier_loglik_finite"] = bool(np.isfinite(r["loglik"]))
        out[name] = row
        say(f"C  {name}: Riccati step {row['riccati_step_us_640u']:.0f} us (640 u); objective 10x13 {row['objective_10x13_s']:.2f} s "
            f"(checks ok {row['objective_all_ok']}, max|phi| {row['objective_phi_max']:.6f}); exact-step build {row['exact_step_build_ms']:.1f} ms; "
            f"QE 10k paths {row['qe_step_10k_paths_ms']:.1f} ms; Kalman spot {row['kalman_spot_per_day_us']:.0f} us/day, RV {row['kalman_rv_per_day_us']:.0f} us/day; "
            f"particle(500) {row['particle_500_per_day_ms']:.1f} ms/day; Fourier {row['fourier_per_day_ms']:.2f} ms/day "
            f"+ {row['fourier_coefficients_s']:.1f} s per parameter set (finite {row['fourier_loglik_finite']})")
    return out


# ------------------------------------------------------------------ D: H
def adams_surface(P):
    """10 expiries x 13 strikes priced by fractional Adams (the true model), equal weights."""
    from calibrate.objective import MarketSurface
    tau_a, k_a, iv_a = [], [], []
    for d_ in EXPIRIES:
        tau = d_ / 365
        band = 3 * 0.2 * math.sqrt(tau)
        ks = np.linspace(-band, band, 13)
        um = 1.3 * rh.u_max_for(P, tau)
        M = 1500 if d_ <= 7 else 3000
        iv = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, M), um), ks, tau)
        tau_a += [tau] * len(ks)
        k_a += list(ks)
        iv_a += list(iv)
    iv_a = np.array(iv_a)
    return MarketSurface(np.array(tau_a), np.array(k_a), iv_a, np.ones(len(iv_a)))


def section_d1():
    from calibrate.fit import calibrate_rough_heston
    out = {}
    for H in (0.12, 0.05):
        P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, H)
        t0 = time.perf_counter()
        S = adams_surface(P)
        say(f"D1 H={H}: Adams surface priced ({time.perf_counter()-t0:.0f}s), {len(S)} quotes, finite {bool(np.all(np.isfinite(S.iv)))}")
        for name, (N, eta) in LIFTS.items():
            with lift_as(N, eta):
                t0 = time.perf_counter()
                res = calibrate_rough_heston(S, starts=(P,), max_iter=40, max_seconds=1500)
                q = res["params"]
                r = {"H": q.H, "kappa": q.kappa, "theta": q.theta, "xi": q.xi, "rho": q.rho, "v0": q.v0,
                     "rmse_vol_vp": 100 * res["rmse_vol"], "seconds": time.perf_counter() - t0,
                     "n_fev": int(res.get("n_fev", -1)), "reason": res.get("reason"), "ok": bool(res["ok"]),
                     "kernel_error": res["kernel_error"]}
            out[f"H={H} {name}"] = r
            say(f"D1 H={H} {name}: H_hat {r['H']:.4f} (truth {H}); rho {r['rho']:+.3f}, xi {r['xi']:.3f}, v0 {r['v0']:.4f}; "
                f"RMSE {r['rmse_vol_vp']:.3f} vp; {r['n_fev']} evaluations, {r['seconds']:.0f}s, {r['reason']}; ok {r['ok']}")
    return out


def section_d2():
    H_true = 0.12
    p = dict(kappa=3.0, theta=0.06, xi=0.15)
    w64, x64 = _ORIG_LIFT(H_true, 64, eta_N=1e10)
    bars, n_days = 78, 500
    st = rh.LiftedAffineStep(w64, x64, p["theta"], p["kappa"], p["theta"], p["xi"], 1.0 / (252 * bars))
    rng = np.random.default_rng(11)
    y = st.y_star[:, None].copy()
    rv, iv = [], []
    t0 = time.perf_counter()
    for _ in range(n_days):
        r2 = I = 0.0
        for _j in range(bars):
            y, V, dI, dZ, _fix = st.step_y(y, rng, with_log_price=True)
            ret = float(dZ[0] - 0.5 * dI[0])
            r2 += ret * ret
            I += float(dI[0])
        rv.append(r2 * 252)
        iv.append(I * 252)
    rv, iv = np.array(rv), np.array(iv)
    say(f"D2 simulated {n_days} days x {bars} bars with N=64, eta_N=1e10 ({time.perf_counter()-t0:.0f}s); "
        f"mean RV {rv.mean():.4f}, corr(RV, IV) {np.corrcoef(rv, iv)[0, 1]:.3f}")
    grid = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30)
    out = {"truth": H_true, "grid": list(grid)}
    for name, (N, eta) in LIFTS.items():
        with lift_as(N, eta):
            t0 = time.perf_counter()
            prof = kf.profile_h(rv, grid, dict(p, R=1e-8), 1 / 252, p["theta"], names=("xi",), model_cls=kf.LiftedRoughRVModel)
            r = {"H": prof["H_hat"], "se": prof["se_quadratic"], "ci95": list(prof["ci95"]),
                 "loglik": prof["loglik"].tolist(), "seconds": time.perf_counter() - t0}
        out[name] = r
        say(f"D2 {name}: H from daily RV {r['H']:.3f} +/- {r['se']:.3f}, 95% [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
            f"(truth {H_true}); {r['seconds']:.0f}s")
    return out


def main(argv):
    wanted = [a.upper() for a in argv] or ["A", "B", "D"]
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    funcs = {"A": section_a, "B": section_b, "C": section_c, "D1": section_d1, "D2": section_d2}
    for sec in wanted:
        for key in (["D2", "D1"] if sec == "D" else [sec]):
            t0 = time.perf_counter()
            res[key] = funcs[key]()
            say(f"[{key} done in {time.perf_counter()-t0:.0f}s]")
            OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])


def plot(res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = res or json.loads(OUT.read_text(encoding="utf-8"))
    cols = {"N=24, 1e5 (shipped)": "#d62728", "N=32, 1e5": "#ff7f0e", "N=40, 1e8": "#1f77b4"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    ts = np.exp(np.linspace(math.log(1.0 / (365 * 24 * 60)), math.log(2.0), 400))
    minutes = ts * 365 * 24 * 60
    a = ax[0, 0]
    for name, (N, eta) in LIFTS.items():
        w, x = _ORIG_LIFT(0.12, N, eta_N=eta)
        rel = np.abs(rh.kernel_approx(ts, w, x) / rh.fractional_kernel(ts, 0.12) - 1)
        a.loglog(minutes, 100 * rel, color=cols[name], label=name)
    for lab, mins in (("1 min", 1), ("1 h", 60), ("1 d", 1440), ("1 y", 525600)):
        a.axvline(mins, color="0.85", lw=0.8, zorder=0)
        a.text(mins, 50, lab, fontsize=7, color="0.5")
    a.axhline(1.0, color="0.5", ls=":", lw=0.8)
    a.set_xlabel("lag (minutes)")
    a.set_ylabel("|lifted kernel / true - 1| (%)")
    a.set_title("(a) the kernel, H = 0.12: the fastest node sets where it fails")
    a.legend(fontsize=8)
    a = ax[0, 1]
    for name, (N, eta) in LIFTS.items():
        for H, ls in ((0.12, "-"), (0.05, "--")):
            w, x = _ORIG_LIFT(H, N, eta_N=eta)
            short = [100 * (1 - iso_lift(t, w, x) / (t ** (2 * H) / (2 * H * math.gamma(H + 0.5) ** 2))) for t in ts[::8]]
            a.semilogx(minutes[::8], short, ls, color=cols[name], label=f"{name}, H = {H}")
    a.axvline(1440, color="0.85", lw=0.8, zorder=0)
    a.set_xlabel("horizon (minutes)")
    a.set_ylabel("Ito isometry shortfall of the driver (%)")
    a.set_title("(b) variance of the Volterra driver the lift misses")
    a.legend(fontsize=7, ncol=2)
    a = ax[1, 0]
    A = res.get("A", {})
    xs = np.arange(6)
    labels = []
    for i, (name, _) in enumerate(LIFTS.items()):
        vals = []
        for H in (0.12, 0.05):
            for d_ in (1, 7, 30):
                row = A.get(f"H={H} {d_}d", {})
                vals.append(row.get(name, {}).get("worst_vp", np.nan))
        a.bar(xs + (i - 1) * 0.27, vals, 0.27, color=cols[name], label=name)
    labels = [f"H={H}\n{d_} d" for H in (0.12, 0.05) for d_ in (1, 7, 30)]
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=8)
    a.set_ylabel("worst IV error vs fractional Adams (vol points)")
    a.set_title("(c) option prices against the true rough Heston")
    a.legend(fontsize=8)
    a = ax[1, 1]
    D1, D2 = res.get("D1", {}), res.get("D2", {})
    groups = [("calibration, H = 0.12", lambda n: D1.get(f"H=0.12 {n}", {}).get("H", np.nan) - 0.12),
              ("calibration, H = 0.05", lambda n: D1.get(f"H=0.05 {n}", {}).get("H", np.nan) - 0.05),
              ("daily RV filter, H = 0.12", lambda n: D2.get(n, {}).get("H", np.nan) - 0.12)]
    for i, (name, _) in enumerate(LIFTS.items()):
        vals = [g(name) for _, g in groups]
        xb = np.arange(3) + (i - 1) * 0.27
        a.bar(xb, vals, 0.27, color=cols[name], label=name)
        for xv, v in zip(xb, vals):
            if np.isfinite(v):
                a.text(xv, v + (0.0012 if v >= 0 else -0.0012), f"{v:+.4f}", ha="center",
                       va="bottom" if v >= 0 else "top", fontsize=7, rotation=90)
    if D2:
        se = D2.get("N=24, 1e5 (shipped)", {}).get("se", np.nan)
        a.errorbar([2], [0.0], yerr=[se], fmt="none", ecolor="0.4", capsize=4)
        a.text(2.05, se, f" filter SE {se:.3f}", fontsize=7, color="0.4")
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xticks(np.arange(3))
    a.set_xticklabels([g[0] for g in groups], fontsize=8)
    a.set_ylabel("H estimate - truth")
    a.set_title("(d) the H the pipeline reports")
    a.legend(fontsize=8)
    fig.suptitle("The finer lift: N = 40 with the fastest node at 1e8/y vs the shipped N = 24 at 1e5/y", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "finer_lift.png", dpi=120)
