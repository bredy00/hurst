"""
The real-data pipeline (Session G): recorded IBKR chains -> calibration, recorded
history -> filters, and four independent readings of H side by side.

    .venv/Scripts/python.exe run_real_data.py                  # newest recording + newest history
    .venv/Scripts/python.exe run_real_data.py --snapshot PATH --history PATH
    .venv/Scripts/python.exe run_real_data.py --synthetic      # same pipeline, known answer (~5 min)

Chains, per snapshot:
  checks     parity r2 per expiry, vega units (recorded IBKR vega vs Black-Scholes
             vega at the recorded IV), tau convention (IBKR's own IV vs the IV
             implied by the mid price under OUR tau and forward), usable quotes
  surface    OTM by forward, IV from the mid under our tau, noise = half-spread / vega,
             weights by scheme (hybrid by default: inverse variance balanced per
             expiry + short-end ATM-skew anchors; see calibrate/weights.py)
  fits       vanilla Heston, then rough Heston with its kernel and stability checks
  clock      the variance clock the short end is priced on (Session H,
             calibrate/clock.py): omega, the price of a night- or weekend-second
             relative to a session-second, from ATM total variance, pooled over the
             day's snapshots, with event days found and kept out of it
  H          (1) from the market's ATM-skew term structure (skew ~ tau^(H - 1/2)),
                 on calendar time and again on the measured trading clock
             (2) from the rough Heston calibration

History:
  series     daily realised variance from 5-minute bars (Garman-Klass if too few)
  filters    CIR by maximum likelihood; the lifted rough filter with H profiled on a
             grid (xi re-fitted at each H), a bank-of-filters posterior over H
  boundary   the zero-boundary protocol at the profile maximum (filters/protocol.py,
             Session I): the Kalman filter unless its predicted variance is within two
             sd of zero on more than 5% of days; then the cf filter takes the state
             estimate, confirmed by a 16-substep particle filter (escalated to 64
             substeps when they disagree); --quick skips the confirmation
  H          (3) from the filter's profile likelihood
             (4) from the structure function of log realised volatility

Writes captures/real/report/<stamp>/{report.json, report.md, figure.png}.
"""

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

import volsurf_core as vc
import volatility_surface_3 as v3
import sources.replay as replay
import sources.history as hist
import calibrate.weights as wt
from calibrate.objective import MarketSurface, per_expiry_report
import models.rough_heston as rh

ROOT = pathlib.Path(__file__).parent
H_GRID = (0.03, 0.05, 0.08, 0.12, 0.17, 0.25, 0.35, 0.49)


# ------------------------------------------------------------------ chains
def latest(pattern, root=ROOT / "captures" / "real"):
    files = sorted(root.rglob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def chain_checks(app, ctxs):
    ratios, iv_gaps, parity = [], [], {}
    for q in app.quotes.values():
        ctx = ctxs.get(q.expiry)
        if ctx is None or not (np.isfinite(q.iv) and np.isfinite(q.vega) and np.isfinite(ctx.forward)):
            continue
        bs = float(vc.bs_vega(ctx.forward, q.strike, q.iv, ctx.tau, ctx.discount))
        if bs > 0:
            ratios.append(q.vega / bs)
        mid = q.mid
        if np.isfinite(mid) and mid > 0 and q.right == vc.otm_right(q.strike, ctx.forward):
            iv_mid = vc.implied_vol(mid, ctx.forward, q.strike, ctx.tau, ctx.discount, q.right)
            if iv_mid is not None and np.isfinite(iv_mid):
                iv_gaps.append(iv_mid - q.iv)
    for exp, ctx in ctxs.items():
        parity[exp] = {"tau_days": ctx.tau * 365, "forward": ctx.forward, "r2": ctx.parity_r2}
    return {"vega_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "vega_units_ok": bool(ratios) and 0.8 < float(np.median(ratios)) < 1.25,
            "iv_mid_minus_ibkr_median_vp": 100 * float(np.median(iv_gaps)) if iv_gaps else float("nan"),
            "n_quotes": len(app.quotes), "parity": parity,
            "weak_parity": [e for e, v in parity.items() if v["r2"] < 0.99]}


def surface_from_snapshot(app, ctxs, scheme="hybrid", max_err=0.05, min_tau_days=0.9, iv_source="mid"):
    tau, k, iv, vega, hs, rows = [], [], [], [], [], []
    for q in app.quotes.values():
        ctx = ctxs.get(q.expiry)
        if ctx is None or ctx.tau * 365 < min_tau_days or not np.isfinite(ctx.forward):
            continue
        if q.right != vc.otm_right(q.strike, ctx.forward):
            continue
        if iv_source == "mid":
            mid = q.mid
            if not (np.isfinite(mid) and mid > 0):
                continue
            sig = vc.implied_vol(mid, ctx.forward, q.strike, ctx.tau, ctx.discount, q.right)
        else:
            sig = q.iv
        if sig is None or not np.isfinite(sig) or sig <= 0:
            continue
        veg = float(vc.bs_vega(ctx.forward, q.strike, sig, ctx.tau, ctx.discount))
        if veg <= 0 or q.half_spread / veg > max_err:
            continue
        tau.append(ctx.tau)
        k.append(math.log(q.strike / ctx.forward))
        iv.append(sig)
        vega.append(veg)
        hs.append(q.half_spread)
        rows.append({"expiry": q.expiry, "strike": q.strike, "right": q.right})
    tau, k, iv, vega, hs = (np.array(a, float) for a in (tau, k, iv, vega, hs))
    w, anchors = wt.scheme_weights(scheme, tau, k, iv, vega, hs)
    return MarketSurface(tau, k, iv, w, anchors=anchors), rows


def skew_term_structure(S):
    """Market ATM skew per expiry (local fit) and H from its log-log slope."""
    taus, skews, errs = [], [], []
    for t, idx in S.by_expiry:
        ks, ivs = S.k[idx], S.iv[idx]
        if len(idx) < 5:
            continue
        atm = idx[np.argmin(np.abs(ks))]
        z = ks / (S.iv[atm] * math.sqrt(t))
        s, se, n = vc.atm_skew_from_slice(ks, ivs, zs=z, weights=S.weight[idx], window=1.5)
        if s is not None and np.isfinite(s) and s < 0:
            taus.append(t)
            skews.append(s)
            errs.append(se if se else abs(s))
    if len(taus) < 3:
        return {"H": None, "taus": taus, "skews": skews}
    H, H_err, icpt, r2 = vc.estimate_hurst(taus, skews, errs)
    return {"H": H, "H_err": H_err, "r2": r2, "taus": taus, "skews": skews, "errs": errs}


def calibrate_chain(S, max_seconds=900, rough_starts=None):
    from calibrate.fit import calibrate_heston, calibrate_rough_heston, DEFAULT_ROUGH_STARTS, heston_cf_factory
    out = {}
    t0 = time.perf_counter()
    hes = calibrate_heston(S, max_seconds=max_seconds / 3)
    out["heston"] = {"params": hes["params"].__dict__, "rmse_vp": 100 * hes["rmse_vol"],
                     "seconds": time.perf_counter() - t0,
                     "by_expiry": per_expiry_report(hes["params"], S, heston_cf_factory)}
    t0 = time.perf_counter()
    starts = rough_starts or DEFAULT_ROUGH_STARTS
    ro = calibrate_rough_heston(S, starts=starts, max_seconds=max_seconds, max_iter=40)
    pricer = rh.RoughPricer()
    out["rough"] = {"params": ro["params"].__dict__, "rmse_vp": 100 * ro["rmse_vol"],
                    "seconds": time.perf_counter() - t0, "ok": ro["ok"],
                    "kernel_error": ro["kernel_error"], "N": ro["N"], "stability": ro["stability"],
                    "data_cost": ro["data_cost"],
                    "by_expiry": per_expiry_report(ro["params"], S, lambda q, t: rh.cf_factory(q, t, N=ro["N"]),
                                                   pricer=pricer)}
    return out


def trading_clock(snapshot, max_peers=12):
    """
    The variance clock from this snapshot and the other snapshots of the same New
    York day in its folder (evenly thinned to max_peers): pooled omega with its 95%
    interval, each snapshot's own fit, and how far the calendar clock is from the data.
    """
    import calibrate.clock as ck
    snapshot = pathlib.Path(snapshot)
    app, ctxs = replay.load(snapshot)
    t_main = replay.recorded_utc(app, ctxs)
    day = vc.new_york_date(t_main)
    files = sorted(f for f in snapshot.parent.glob("*.json") if f.name != "session.jsonl")
    if len(files) > max_peers:
        files = [files[int(round(i))] for i in np.linspace(0, len(files) - 1, max_peers)]
    if snapshot not in files:
        files.append(snapshot)
    snaps = []
    for f in files:
        try:
            a, c = replay.load(f)
            t = replay.recorded_utc(a, c)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if vc.new_york_date(t) == day:
            snaps.append((t, ck.atm_total_variance(a, c)))
    main = ck.estimate_omega([(t_main, ck.atm_total_variance(app, ctxs))])
    pooled = ck.estimate_omega(snaps) if len(snaps) > 1 else main
    return {"t0_utc": t_main.isoformat(), "n_snapshots": len(snaps), "pooled": pooled, "this_snapshot": main}, t_main


# ------------------------------------------------------------------ history
def analyse_history(data, dt=1.0 / 252, max_days=750, profile_names=("xi",), confirm=True):
    import filters.kalman as kf
    import filters.protocol as proto
    rv = hist.realised_variance(data["bars5m"])
    if len(rv["date"]) >= 120:
        y, dates, source = rv["rv"], rv["date"], "realised variance (5-minute bars)"
        m_equiv = int(np.median(rv["bars"]))              # returns per day: RV noise 2/M
    else:
        d = hist.daily_series(data["daily"])
        y, dates, source = hist.garman_klass(d["open"], d["high"], d["low"], d["close"]), d["date"], "Garman-Klass"
        m_equiv = 7                                       # GK is ~7.4x a squared return
    y, dates = np.asarray(y)[-max_days:], list(dates)[-max_days:]
    out = {"source": source, "n_days": int(len(y)), "mean_vol": float(np.sqrt(np.mean(y)))}

    # (4) structure function of log realised volatility
    sf = vc.hurst_from_structure(np.log(np.sqrt(np.maximum(y, 1e-10))))
    out["H_structure"] = sf.get("H")
    out["H_structure_r2"] = sf.get("linearity_r2")

    theta0 = float(np.mean(y))
    R0 = float(np.var(np.diff(y))) / 2.0                 # the lag-1 jump of noise is mostly R
    cm = kf.CIRModel(dt)
    fit = kf.fit_mle_multistart(cm, y, dict(kappa=5.0, theta=theta0, xi=0.5, R=R0), values=(2.0, 10.0, 40.0))
    out["cir"] = {"params": fit["params"], "se": fit["se"], "loglik": fit["loglik"]}

    # Daily realised variance measures the INTEGRAL of V over the day: the filter
    # must observe that, with RV's own sampling noise, or H comes out far too high
    # (0.39 for a true 0.10 on synthetic data with the spot-variance filter).
    # RV's sampling error at the prior mean (Session G's model). Not at the observed RV: that
    # weights each day by its own noise realisation, and on the synthetic history (planted
    # H = 0.10) it moved the profile's H from 0.130 to 0.183 (Session I). The zero-boundary
    # protocol below does use the observed RV -- for state estimation, where it keeps the
    # non-Gaussian filters from breaking on days whose RV jumps above a small forecast.
    rv_model = lambda dt_, H, v0, N=None: kf.LiftedRoughRVModel(dt_, H=H, v0=v0, N=N, bars_per_day=m_equiv)
    p0 = dict(kappa=3.0, theta=theta0, xi=0.3, R=1e-10)
    prof = kf.profile_h(y, H_GRID, p0, dt, theta0, names=profile_names, model_cls=rv_model)
    bank = kf.filter_bank(prof["runs"], H_GRID)
    out["rough_profile"] = {"H": list(H_GRID), "loglik_minus_max": (prof["loglik"] - prof["loglik"].max()).tolist(),
                            "H_hat": prof["H_hat"], "se": prof["se_quadratic"], "ci95": prof["ci95"],
                            "xi_by_H": [q[profile_names[0]] for q in prof["params"]] if profile_names else None,
                            "bank_mean": float(bank["mean"][-1]), "bank_sd": float(bank["sd"][-1]),
                            "loglik_vs_cir": float(prof["loglik"].max() - fit["loglik"])}
    out["_series"] = {"dates": dates, "y": y.tolist(), "bank_mean_path": bank["mean"].tolist()}

    # (5) the zero-boundary protocol at the profile maximum (Session I): the Kalman filter
    # unless the boundary binds, then the cf filter, confirmed by the particle filter
    j = int(np.argmax(prof["loglik"]))
    pr = proto.run(prof["params"][j], y, dt=dt, H=float(H_GRID[j]), v0=theta0, bars_per_day=m_equiv,
                   confirm=confirm)
    zb = {"H": float(H_GRID[j]), "boundary_share": pr["boundary"]["share"], "binds": pr["boundary"]["binds"],
          "state_estimate": pr["state_estimate"], "kalman_loglik": pr["kalman"]["loglik"]}
    if "cf" in pr:
        zb.update(cf_loglik=pr["cf"]["loglik"], cf_fallback_days=pr["cf"]["fallback_days"],
                  cf_capped_days=pr["cf"]["capped_days"])
        out["_series"]["cf_filtered_rv"] = pr["cf"]["filtered_rv"]
    if "confirmation" in pr:
        c = pr["confirmation"]
        zb.update(confirmed=c["confirmed"], verdict=c["verdict"], checks=c["checks"],
                  particle_loglik=c["particle_loglik_mean"],
                  cf_minus_particle_per_day=c["cf_minus_particle_per_day"],
                  kalman_minus_particle=c["kalman_minus_particle"],
                  median_rel_rv_diff=c["median_rel_rv_diff_boundary_days"])
        if "escalation" in c:
            zb["escalation"] = {k: c["escalation"][k] for k in ("substeps", "loglik", "cf_minus_particle_per_day",
                                                                 "particle16_minus_particle")}
    out["zero_boundary"] = zb
    return out


# ------------------------------------------------------------------ report
def plot(report, S, fits, path):
    """
    The run's figure, in the ui.theme light system (Session I): maturities on the ordinal
    blue ramp, one hue per entity, reference lines in muted ink with their own labels,
    and no second y-axis -- the history's volatility and its H path are two panels that
    share the time axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from ui import theme as th
    t = th.apply("light")
    blue, orange = th.series(0), th.series(1)
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.62, 0.62], hspace=0.55, wspace=0.28)
    fig.text(0.012, 0.975, "Real-data run" + ("  (synthetic, known answer)" if report.get("synthetic") else ""),
             fontsize=15, fontweight="semibold", color=t["ink"], va="top")
    fig.text(0.012, 0.945, f"generated {report.get('generated', '')}", fontsize=9.5, color=t["ink2"], va="top")

    a = fig.add_subplot(gs[0, 0])
    if S is not None:
        # maturity is ordered: the ordinal blue ramp from step 250 (clears 2:1 on the surface) to 700
        ramp = LinearSegmentedColormap.from_list("ordinal_blue", th.BLUE_RAMP[3:])
        cols = ramp(np.linspace(0, 1, S.n_expiries))
        for (tt, idx), c in zip(S.by_expiry, cols):
            a.plot(S.k[idx], 100 * S.iv[idx], "o", ms=4, color=c, mec=t["surface"], mew=0.6, label=f"{tt*365:.0f} d")
        a.set_xlabel("log-moneyness k")
        a.set_ylabel("implied vol (%)")
        a.set_title("(a) The recorded surface: OTM quotes, IV from the mid under our τ")
        a.legend(ncol=2, title="maturity (light = short)", title_fontsize=8, loc="upper right")
    a = fig.add_subplot(gs[0, 1])
    ts = report.get("chain", {}).get("skew", {})
    if ts.get("taus"):
        tt = np.array(ts["taus"]) * 365
        a.loglog(tt, -np.array(ts["skews"]), "o", ms=6, color=blue, mec=t["surface"], mew=0.8,
                 label="market ATM skew per expiry")
        if ts.get("H") is not None:
            xs = np.geomspace(tt.min(), tt.max(), 50)
            ref = -np.array(ts["skews"])[0] * (xs / tt[0]) ** (ts["H"] - 0.5)
            a.loglog(xs, ref, "--", color=blue, lw=1.6, label=f"power law τ^(H−½), H = {ts['H']:.3f}")
        a.set_xlabel("τ (days)")
        a.set_ylabel("−∂σ/∂k")
        a.set_title("(b) ATM skew term structure")
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        a.legend(loc="upper right")
    hs = report.get("history")
    if hs:
        yy = np.array(hs["_series"]["y"])
        a = fig.add_subplot(gs[1, 0])
        a.plot(np.sqrt(np.maximum(yy, 0.0)), color=blue, lw=1.0, label=f"√{hs['source']}")
        cfp = hs["_series"].get("cf_filtered_rv")
        if cfp is not None:
            a.plot(np.sqrt(np.maximum(np.array(cfp), 0.0)), color=orange, lw=1.2,
                   label="cf filter, filtered (zero-boundary protocol)")
        a.set_ylabel("volatility")
        a.set_title("(c) The history")
        a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.3)
        a.tick_params(labelbottom=False)
        b = fig.add_subplot(gs[2, 0], sharex=a)
        b.plot(hs["_series"]["bank_mean_path"], color=blue, lw=1.6)
        b.set_ylabel("posterior mean H")
        b.set_xlabel("trading day")
        b.set_title("(d) H learned online by the filter bank")
    a = fig.add_subplot(gs[1:, 1])
    names, vals, errs = [], [], []
    hc = report.get("H", {})
    for key, lab in (("skew_term_structure", "(1) surface skew slope"),
                     ("skew_term_structure_trading_clock", "(1b) skew slope, trading clock"),
                     ("rough_calibration", "(2) rough Heston fit"),
                     ("filter_profile", "(3) filter profile likelihood"), ("structure_function", "(4) structure function")):
        v = hc.get(key)
        if v and v.get("H") is not None and np.isfinite(v["H"]):
            names.append(lab)
            vals.append(v["H"])
            errs.append(v.get("se") if v.get("se") is not None and np.isfinite(v.get("se") or np.nan) else 0.0)
    if names:
        yv = np.arange(len(names))[::-1]
        a.axvline(0.5, color=t["muted"], ls=":", lw=1.2)
        a.text(0.5, yv.max() + 0.55, " H = ½ (Brownian)", color=t["ink2"], fontsize=8, va="bottom")
        truth = report.get("truth_H")
        if truth is not None:
            a.axvline(truth, color=t["ink2"], ls="--", lw=1.2)
            a.text(truth, yv.max() + 0.55, f" truth {truth}", color=t["ink2"], fontsize=8, va="bottom")
        a.errorbar(vals, yv, xerr=errs, fmt="o", ms=7, color=blue, ecolor=t["muted"], elinewidth=1.4, capsize=3,
                   mec=t["surface"], mew=0.8)
        for v, y_, e in zip(vals, yv, errs):
            a.text(v + (e or 0.0) + 0.012, y_, f"{v:.3f}" + (f" ± {e:.3f}" if e else ""), color=t["ink2"],
                   fontsize=8, ha="left", va="center")
        a.set_yticks(yv)
        a.set_yticklabels(names)
        a.set_ylim(-0.6, yv.max() + 1.1)
        a.set_xlim(min(0.0, min(v - (e or 0.0) for v, e in zip(vals, errs)) - 0.05), 0.62)
        a.axvline(0.0, color=t["axis"], lw=0.8)
        a.set_xlabel("H")
        a.grid(axis="y", visible=False)
        a.set_title("(e) Independent readings of H")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07)
    fig.savefig(path, dpi=120)


def run(snapshot=None, history=None, scheme="hybrid", out_dir=None, synthetic=False, max_seconds=900, quick=False):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(out_dir or ROOT / "captures" / "real" / "report" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    w_lift, x_lift = rh.lift_nodes(0.12)
    report = {"generated": stamp, "scheme": scheme, "synthetic": synthetic, "H": {},
              "lift": {"N": len(w_lift), "fastest_node_per_year": float(x_lift.max())}}
    S = fits = None
    if snapshot:
        app, ctxs = replay.load(snapshot)
        report["snapshot"] = str(snapshot)
        report["chain"] = {"checks": chain_checks(app, ctxs)}
        c = report["chain"]["checks"]
        print(f"chain: {c['n_quotes']} quotes; vega units ratio {c['vega_ratio_median']:.3f} "
              f"({'ok' if c['vega_units_ok'] else 'CHECK'}); IV(mid, our tau) - IBKR IV median "
              f"{c['iv_mid_minus_ibkr_median_vp']:+.2f} vp; weak parity on {len(c['weak_parity'])} expiries", flush=True)
        S, rows = surface_from_snapshot(app, ctxs, scheme)
        report["chain"]["surface"] = {"quotes": len(S), "expiries": S.n_expiries, "anchors": len(S.anchors),
                                      "effective_quotes": wt.effective_quotes(S.weight)}
        ts = skew_term_structure(S)
        report["chain"]["skew"] = ts
        if ts.get("H") is not None:
            report["H"]["skew_term_structure"] = {"H": ts["H"], "se": ts.get("H_err")}
        print(f"surface: {len(S)} quotes on {S.n_expiries} expiries, {len(S.anchors)} skew anchors; "
              f"skew slope H = {ts.get('H')}", flush=True)
        try:
            clock, t_snap = trading_clock(snapshot)
        except ValueError as e:                       # no usable instant or short end
            clock, t_snap = {"error": str(e)}, None
        report["chain"]["clock"] = clock
        pc = clock.get("pooled", {})
        if pc.get("identified"):
            import calibrate.clock as ck
            S_v, _ = surface_from_snapshot(app, ck.retime(ctxs, t_snap, pc["omega"]), scheme)
            ts_v = skew_term_structure(S_v)
            report["chain"]["skew_trading_clock"] = ts_v
            if ts_v.get("H") is not None:
                report["H"]["skew_term_structure_trading_clock"] = {"H": ts_v["H"], "se": ts_v.get("H_err")}
            print(f"clock: omega {pc['omega']:.3f} (95% {pc['ci95'][0]:.3f}-{pc['ci95'][1]:.3f}) from "
                  f"{clock['n_snapshots']} snapshot(s), {pc['n_expiries']} short expiries; calendar clock "
                  f"{pc['calendar_chi2']:.1f} noise variances worse; events "
                  f"{sorted({e for q in pc['per_snapshot'] for e in q['events']})}; skew slope H on it = {ts_v.get('H')}",
                  flush=True)
        else:
            print(f"clock: not identified ({pc.get('reason') or clock.get('error')})", flush=True)
        if not quick:
            fits = calibrate_chain(S, max_seconds)
            report["chain"]["fits"] = fits
            rp = fits["rough"]["params"]
            report["H"]["rough_calibration"] = {"H": rp["H"], "se": None}
            print(f"fits: Heston rmse {fits['heston']['rmse_vp']:.3f} vp; rough rmse {fits['rough']['rmse_vp']:.3f} vp, "
                  f"H {rp['H']:.3f}, ok {fits['rough']['ok']}", flush=True)
    if history:
        data = hist.load(history) if not isinstance(history, dict) else history
        report["history_file"] = str(history) if not isinstance(history, dict) else "in memory"
        hs = analyse_history(data, max_days=500 if quick else 750, confirm=not quick)
        report["history"] = hs
        report["H"]["filter_profile"] = {"H": hs["rough_profile"]["H_hat"], "se": hs["rough_profile"]["se"]}
        report["H"]["structure_function"] = {"H": hs["H_structure"], "se": None}
        print(f"history: {hs['n_days']} days of {hs['source']}; CIR kappa {hs['cir']['params']['kappa']:.1f}; "
              f"filter H {hs['rough_profile']['H_hat']:.3f} (se {hs['rough_profile']['se']:.3f}, bank "
              f"{hs['rough_profile']['bank_mean']:.3f}); structure-function H {hs['H_structure']}", flush=True)
        if isinstance(data, dict) and data.get("truth"):
            report["truth_H"] = data["truth"]["H"]
    plot(report, S, fits, out_dir / "figure.png")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=_json_default), encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {out_dir}")
    return report, out_dir


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def markdown(r):
    lines = [f"# Real-data run {r['generated']}" + (" (synthetic, known answer)" if r.get("synthetic") else ""), ""]
    ch = r.get("chain")
    if ch:
        c = ch["checks"]
        lines += ["## Chain", "", f"- snapshot: `{r.get('snapshot')}`",
                  f"- vega units: IBKR x 100 / Black-Scholes = {c['vega_ratio_median']:.3f} ({'consistent' if c['vega_units_ok'] else '**CHECK**'})",
                  f"- IV from mid under our tau minus IBKR IV: median {c['iv_mid_minus_ibkr_median_vp']:+.2f} vol points",
                  f"- surface: {ch['surface']['quotes']} quotes, {ch['surface']['expiries']} expiries, "
                  f"{ch['surface']['anchors']} skew anchors, {ch['surface']['effective_quotes']:.1f} effective quotes"]
        pc = ch.get("clock", {}).get("pooled", {})
        if pc.get("identified"):
            evs = sorted({e for q in pc["per_snapshot"] for e in q["events"]})
            lines += [f"- variance clock: omega = {pc['omega']:.3f} (95% {pc['ci95'][0]:.3f}-{pc['ci95'][1]:.3f}) from "
                      f"{ch['clock']['n_snapshots']} snapshot(s), {pc['n_expiries']} short expiries; the calendar clock is "
                      f"{pc['calendar_chi2']:.1f} noise variances worse; event days priced: {', '.join(evs) or 'none'}"]
        elif ch.get("clock"):
            lines += [f"- variance clock: not identified ({pc.get('reason') or ch['clock'].get('error')})"]
        if ch.get("fits"):
            f = ch["fits"]
            lines += [f"- Heston: rmse {f['heston']['rmse_vp']:.3f} vp; rough Heston: rmse {f['rough']['rmse_vp']:.3f} vp, "
                      f"ok = {f['rough']['ok']} (kernel {f['rough']['kernel_error']:.4f}, phi_max {f['rough']['stability']['phi_max']:.6f})",
                      "", "| parameter | Heston | rough Heston |", "|---|---|---|"]
            for k in ("v0", "kappa", "theta", "xi", "rho", "H"):
                hv = f["heston"]["params"].get(k, "")
                lines.append(f"| {k} | {hv if hv == '' else f'{hv:.4f}'} | {f['rough']['params'][k]:.4f} |")
        lines.append("")
    hs = r.get("history")
    if hs:
        rp = hs["rough_profile"]
        lines += ["## History", "", f"- {hs['n_days']} days of {hs['source']}, mean vol {hs['mean_vol']:.3f}",
                  f"- CIR MLE kappa {hs['cir']['params']['kappa']:.2f} "
                  + (f"(se {hs['cir']['se']['kappa']:.2f})" if np.isfinite(hs['cir']['se']['kappa'])
                     else "(se not available: the numerical Hessian at the optimum gives none)"),
                  f"- lifted rough filter: H profile maximum {rp['H_hat']:.3f} (se {rp['se']:.3f}, 95% {rp['ci95'][0]:.3f}-{rp['ci95'][1]:.3f}); "
                  f"bank posterior {rp['bank_mean']:.3f} +/- {rp['bank_sd']:.3f}; log-likelihood over CIR {rp['loglik_vs_cir']:+.1f}",
                  f"- structure function of log RV: H {hs['H_structure']} (r2 {hs['H_structure_r2']})"]
        zb = hs.get("zero_boundary")
        if zb:
            line = (f"- zero boundary (at H = {zb['H']:.2f}): predicted variance within 2 sd of zero on "
                    f"{100 * zb['boundary_share']:.1f}% of days -> state estimate: **{zb['state_estimate']}**")
            if "cf_loglik" in zb:
                line += (f"; cf filter log-likelihood {zb['cf_loglik']:.1f} vs Kalman {zb['kalman_loglik']:.1f} "
                         f"({zb['cf_fallback_days']} fallback days)")
            if "confirmed" in zb:
                verdict = zb["verdict"] if zb["confirmed"] else f"**{zb['verdict']}**"
                line += (f"; particle-filter confirmation: {verdict} (16 substeps: cf - particle "
                         f"{zb['cf_minus_particle_per_day']:+.3f} nats/day, filtered RV median difference "
                         f"{100 * zb['median_rel_rv_diff']:.1f}%")
                if "escalation" in zb:
                    e = zb["escalation"]
                    line += f"; at {e['substeps']} substeps: cf - particle {e['cf_minus_particle_per_day']:+.3f} nats/day"
                line += ")"
            lines.append(line)
        lines.append("")
    lines += ["## Four readings of H", "", "| estimator | H | se |", "|---|---|---|"]
    for key, v in r["H"].items():
        se = v.get("se")
        lines.append(f"| {key} | {v['H']:.3f} | {'' if se is None or not np.isfinite(se) else f'{se:.3f}'} |")
    if r.get("truth_H") is not None:
        lines += ["", f"Truth (synthetic): H = {r['truth_H']}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ synthetic
def synthetic_inputs(out_dir, quick=True):
    """A fake-exchange recording priced by rough Heston, and a simulated history."""
    import contextlib
    import io
    import fake_ib
    import record_chains as rc
    import sources.synthetic as syn
    truth = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)
    acquire, probe = v3.RateLimiter.acquire, v3.probe_tws
    v3.RateLimiter.acquire = lambda self: None
    v3.probe_tws = lambda *a, **k: True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cls = fake_ib.make_fake(v3.LiveSurfaceApp, spot=650.0, iv_fn=syn.model_iv_fn(truth))
            app = v3.connect_app("SPY", port=4001, app_factory=cls, spot_timeout=2.0)
            today = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
            targets = (2, 7, 14, 30, 90) if quick else rc.TARGET_DAYS
            exps = v3.select_expiries_by_target(app.expirations, today, targets, 1)
            path, _ = rc.record_cycle(app, "SPY", exps, out_dir, 0, max_strikes=11 if quick else 21)
    finally:
        v3.RateLimiter.acquire, v3.probe_tws = acquire, probe
    history = syn.synthetic_history(truth, n_days=500 if quick else 750, seed=3)
    return truth, path, history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot")
    ap.add_argument("--history")
    ap.add_argument("--scheme", default="hybrid", choices=("vega2", "inverse_variance", "hybrid"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip calibration (checks, skew H, filters only)")
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--lift", default=None, metavar="N:ETA_N",
                    help="run the whole pipeline on another lift, e.g. 24:1e5 (the Sessions A-H lift); "
                         "the default is 44:1e8 (Session L; Session I's was 40:1e8)")
    args = ap.parse_args(argv)
    if args.lift:
        n_, eta_ = args.lift.split(":")
        with rh.using_lift(int(n_), float(eta_)):
            return _main(args)
    return _main(args)


def _main(args):
    if args.synthetic:
        out = ROOT / "captures" / "real" / "synthetic"
        truth, snap, history = synthetic_inputs(out)
        report, _ = run(snap, history, args.scheme, synthetic=True, max_seconds=args.max_seconds, quick=args.quick)
        return 0
    snap = args.snapshot or latest("SPY_*ET.json")
    history = args.history or latest("history_*.json")
    if not snap and not history:
        print("No recording found under captures/real. Record first: docs/tutorial-ibkr-recording.md")
        return 1
    run(snap, history, args.scheme, max_seconds=args.max_seconds, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
