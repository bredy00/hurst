"""
Session G in one figure -> captures/session_g.png

  (a) rough variance on one seed: Euler goes below zero and is floored; the
      positivity-preserving step never does
  (b) QML kappa, one Newton step from the truth, in SE: the floor's bias, gone
      away from zero, and what remains near it
  (c) the Riccati stability guard: an under-stepped solve blows through |phi| <= 1;
      the guard now refuses it
  (d) H from the time series: profile log-likelihood with xi re-fitted at every H
  (e) realised variance is an integral: the spot-variance filter's H is far off,
      the integrated-variance filter's is not
  (f) calibration weights for H: RMS error in H when the long or the short end of
      the surface is wrong

    python capture_session_g.py        (~3 minutes)
"""

import math
import pathlib
import time

import numpy as np

import filters.kalman as kf
import models.rough_heston as rh
import sources.history as hist
import sources.synthetic as syn
import study_h_weighting as shw

ROOT = pathlib.Path(__file__).parent
DT = 1.0 / 252


def newton_z(model, p, y, d=0.15):
    ll = [kf.kalman_filter(model, dict(p, kappa=p["kappa"] + e), y)["loglik"] for e in (-d, 0.0, d)]
    info = -(ll[2] - 2 * ll[1] + ll[0]) / (d * d)
    return (ll[2] - ll[0]) / (2 * d) / math.sqrt(info) if info > 0 else float("nan")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (a) paths
    pz = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.01 ** 2)
    me = kf.LiftedRoughModel(DT, H=0.12, v0=0.04, discretisation="euler")
    mq = kf.LiftedRoughModel(DT, H=0.12, v0=0.04)
    Ve, _ = kf.simulate_rough(me, pz, 3000, seed=61, substeps=1, scheme="euler")
    Vq, _ = kf.simulate_rough(mq, pz, 3000, seed=61)
    a = ax[0, 0]
    sl = slice(1000, 1250)
    a.plot(np.arange(250), Ve[sl], color="#d62728", lw=0.9, label=f"exponential Euler: < 0 on {100*np.mean(Ve < 0):.0f}% of days")
    a.plot(np.arange(250), Vq[sl], color="#1f77b4", lw=0.9, label=f"positivity-preserving (QE): min {Vq.min():.0e}")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("day")
    a.set_ylabel("rough variance V")
    a.set_title("(a) H = 0.12, xi = 0.3: the scheme, not the model, went negative")
    a.legend(fontsize=8)

    # (b) QML kappa bias
    nf = dict(kappa=3.0, theta=0.09, xi=0.08, R=0.01 ** 2)
    mn = kf.LiftedRoughModel(DT, H=0.12, v0=0.09)
    zs = {"Euler data,\nfloor binds": [], "QE data, exact\nfilter, V near 0": [], "QE data, exact\nfilter, V away from 0": []}
    for seed in (61, 63, 65, 67, 69, 71):
        eps = np.random.default_rng(600 + seed).normal(0.0, 0.01, 3000)
        V, _ = kf.simulate_rough(me, pz, 3000, seed=seed, substeps=1, scheme="euler")
        zs["Euler data,\nfloor binds"].append(newton_z(me, pz, V + eps))
        V, _ = kf.simulate_rough(mq, pz, 3000, seed=seed)
        zs["QE data, exact\nfilter, V near 0"].append(newton_z(mq, pz, V + eps))
        V, _ = kf.simulate_rough(mn, nf, 3000, seed=seed)
        zs["QE data, exact\nfilter, V away from 0"].append(newton_z(mn, nf, V + eps))
    a = ax[0, 1]
    for i, (k, v) in enumerate(zs.items()):
        a.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=["#d62728", "#ff7f0e", "#1f77b4"][i])
        a.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color="k", lw=2)
        a.text(i, max(v) + 0.3, f"mean {np.mean(v):+.2f}", ha="center", fontsize=8)
    a.axhspan(-2, 2, color="0.9", zorder=0)
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xticks(range(3))
    a.set_xticklabels(list(zs), fontsize=8)
    a.set_ylabel("kappa: score / sqrt(information) at the truth (SE)")
    a.set_title("(b) QML kappa bias, 6 seeds")
    print(f"(a,b) {time.perf_counter()-t0:.0f}s", flush=True)

    # (c) stability guard
    P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
    u = np.linspace(0.0, 1200.0, 600) - 0.5j
    tau = 90 / 365
    bad = np.abs(rh.char_func(u, tau, P, scheme="exptrap", steps=120, check_stability=False, check_invariant=False))
    m_t = max(rh.AUTO_EXPTRAP_STEPS, rh.stability_steps(tau, 1200.0, P, scheme="exptrap"))
    good = np.abs(rh.char_func(u, tau, P, scheme="exptrap", steps=m_t))
    try:
        rh.char_func(u, tau, P, scheme="exptrap", steps=120)
        msg = "not refused"
    except rh.StabilityError as exc:
        msg = "StabilityError: " + str(exc).split(";")[-1].strip()
    a = ax[0, 2]
    a.semilogy(u.real, np.maximum(bad, 1e-300), color="#d62728", label="120 implicit steps (guards off)")
    a.semilogy(u.real, np.maximum(good, 1e-300), color="#1f77b4", label=f"{m_t} steps (stability-sized)")
    a.axhline(1.0, color="k", ls="--", lw=1)
    a.text(20, 3.0, "|phi(u - i/2)| <= 1 for any martingale model", fontsize=8)
    a.text(20, 1e-12, "default now: " + msg, fontsize=8, color="#d62728")
    a.set_ylim(1e-16, 1e40)
    a.set_xlabel("u")
    a.set_title("(c) the stability constants, now hard assertions")
    a.legend(fontsize=8, loc="upper right")
    print(f"(c) {time.perf_counter()-t0:.0f}s", flush=True)

    # (d) H from the time series
    grid = (0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.49)
    a = ax[1, 0]
    pH = dict(kappa=3.0, theta=0.06, xi=0.15, R=0.005 ** 2)
    for H_true, col in ((0.12, "#1f77b4"), (0.30, "#2ca02c")):
        m = kf.LiftedRoughModel(DT, H=H_true, v0=0.06)
        V, _ = kf.simulate_rough(m, pH, 1500, seed=1)
        y = V + np.random.default_rng(901).normal(0.0, 0.005, 1500)
        prof = kf.profile_h(y, grid, pH, DT, 0.06, names=("xi",))
        a.plot(grid, prof["loglik"] - prof["loglik"].max(), "o-", color=col,
               label=f"true H {H_true}: H_hat {prof['H_hat']:.3f} +/- {prof['se_quadratic']:.3f}")
        a.axvline(H_true, color=col, ls=":", lw=1)
    a.set_ylim(-150, 5)
    a.set_xlabel("H")
    a.set_ylabel("log-likelihood minus maximum (xi re-fitted)")
    a.set_title("(d) roughness is filterable: the profile likelihood over H")
    a.legend(fontsize=8)
    print(f"(d) {time.perf_counter()-t0:.0f}s", flush=True)

    # (e) realised variance
    truth = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)
    h = syn.synthetic_history(truth, n_days=500, seed=3)
    integ = np.array(h["true_daily_integrated_variance"])
    rv = hist.realised_variance(h["bars5m"])["rv"]
    close = np.array(h["true_close_variance"])
    g2 = (0.03, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.49)
    a = ax[1, 1]
    rng = np.random.default_rng(5)
    for label, y, cls, col, ls in (
            ("spot V at the close, spot filter", close + rng.normal(0, 0.003, len(close)), None, "0.4", "-"),
            ("daily integrated variance, spot filter", integ, None, "#d62728", "--"),
            ("daily integrated variance, integrated-variance filter", integ, kf.LiftedRoughRVModel, "#1f77b4", "-"),
            ("5-minute realised variance, integrated-variance filter", rv, kf.LiftedRoughRVModel, "#2ca02c", "-")):
        p0 = dict(kappa=2.0, theta=float(np.mean(y)), xi=0.4, R=(0.003 ** 2 if cls is None and "close" in label else 1e-10))
        prof = kf.profile_h(y, g2, p0, DT, p0["theta"], names=("xi",), model_cls=cls)
        a.plot(g2, prof["loglik"] - prof["loglik"].max(), "o" + ls, color=col, label=f"{label}: {prof['H_hat']:.2f}")
    a.axvline(0.10, color="k", ls=":", lw=1)
    a.set_ylim(-80, 5)
    a.set_xlabel("H")
    a.set_ylabel("log-likelihood minus maximum")
    a.set_title("(e) realised variance is an integral: model the observation (truth H = 0.10)")
    a.legend(fontsize=7)
    print(f"(e) {time.perf_counter()-t0:.0f}s", flush=True)

    # (f) weights
    out = shw.run(verbose=False)
    d, sp = "recorder surface, 10 expiries x 9", "spread-based noise (tick + 2.5% of value)"
    schemes = ("vega2", "inverse_variance", "hybrid", "hybrid x25")
    long_ = [math.hypot(out[(d, sp, s)]["se_H"], out[(d, sp, s)]["bias_H"]) for s in schemes]
    short = [math.hypot(out[(d, sp, s)]["se_H"], out[(d, sp, s)]["bias_H_short"]) for s in schemes]
    a = ax[1, 2]
    xs = np.arange(len(schemes))
    a.bar(xs - 0.2, long_, 0.4, color="#9467bd", label="long end misspecified")
    a.bar(xs + 0.2, short, 0.4, color="#ff7f0e", label="short end misspecified")
    for x_, v in zip(xs - 0.2, long_):
        a.text(x_, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for x_, v in zip(xs + 0.2, short):
        a.text(x_, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(xs)
    a.set_xticklabels(["vega^2", "inverse\nvariance", "hybrid\n(skew anchors)", "hybrid,\nanchors x25"], fontsize=8)
    a.set_ylabel("RMS error in H (SE and bias)")
    a.set_title("(f) weights for H: 90-quote surface, realistic spreads")
    a.legend(fontsize=8)
    fig.suptitle("Session G: positivity, stability, H weighting, and roughness learned by the filter", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "session_g.png", dpi=120)
    print(f"wrote captures/session_g.png ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
