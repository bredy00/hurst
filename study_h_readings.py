"""
Every reading of H the report prints, graded -- and whether the history's bias is finite-
sample or structural (Session O, 29 September 2026).

The real-data report puts several readings of H side by side. On the synthetic inputs
(planted H = 0.10) they read: skew slope -0.031 +/- 0.007, filter profile 0.197 +/- 0.053,
filter bank 0.177 +/- 0.023, structure function 0.138. A reader of the first real report
needs to know which of those to believe, and the only way to know is to grade each one
where the answer is known.

  A  the skew slope. The synthetic chain is priced exactly by the model, so its reading
     has no noise to blame: whatever it is off by is the estimator's. Graded against the
     model's own ATM skew (central differences of the exact prices), over the report's
     window and over the short end only, so the error splits into the local fit's and the
     power law's.
  B  the history readings over Session N's 21 histories of 500 days (seeds 3-23): the
     filter profile with both error bars, the filter bank as reported and tempered by the
     sandwich ratio (`filters.kalman.filter_bank(temper=)`), and the structure function.
  C  the filter profile as the history grows -- 250, 1000 and 2000 days, 10 seeds each --
     which says whether Session N's +0.046 bias shrinks with data (finite-sample) or stays
     (the estimator is inconsistent for this model), and whether the robust SE stays honest.

    python study_h_readings.py A           (~1 minute)
    python study_h_readings.py B           (~25 minutes; B and C can run side by side)
    python study_h_readings.py C           (~60 minutes)
    python study_h_readings.py summary     (merges the parts, prints the scorecard)
    -> captures/h_readings_{A,B,C}.json, captures/h_readings.json, captures/h_readings.log
"""

import contextlib
import io
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np

import filters.kalman as kf
import models.rough_heston as rh
import pricing.fourier as fo
import run_real_data as rrd
import sources.history as hist
import sources.synthetic as syn
import volsurf_core as vc

ROOT = pathlib.Path(__file__).parent
CAP = ROOT / "captures"
TRUTH = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)     # = synthetic_inputs' truth
EDGE = (min(rrd.H_GRID), max(rrd.H_GRID))
SEEDS_B = tuple(range(3, 24))                     # Session N's 21 histories
SEEDS_C = tuple(range(3, 13))
LENGTHS_C = (250, 1000, 2000)


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ A: the skew slope
def exact_atm_skew(p, tau):
    """d sigma / dk at k = 0 from the model's exact call prices: a central difference in
    standardised steps, refined once by Richardson so the step cannot pass for signal."""
    atm = math.sqrt(max(p.v0, 1e-8))
    out = []
    for frac in (0.10, 0.05):
        h = frac * atm * math.sqrt(tau)
        ks = np.array([-h, 0.0, h])
        c, _ = rh.call_prices(ks, tau, p)
        v = fo.implied_vols_from_calls(c, ks, tau)
        out.append((v[2] - v[0]) / (2 * h))
    return float((4 * out[1] - out[0]) / 3), float(abs(out[1] - out[0]))


def part_a():
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        truth, snap, _ = rrd.synthetic_inputs(pathlib.Path(d))
        import sources.replay as replay
        app, ctxs = replay.load(snap)
        S, _ = rrd.surface_from_snapshot(app, ctxs, "hybrid")
    assert truth == TRUTH
    full = rrd.skew_term_structure(S)
    short = rrd.skew_term_structure(S, max_tau_days=16)
    taus = np.array(full["taus"])
    exact = [exact_atm_skew(TRUTH, t) for t in taus]
    ex = np.array([e[0] for e in exact])
    H_ex_full = vc.estimate_hurst(taus, ex)[0]
    s_mask = taus * 365 <= 16
    H_ex_short = vc.estimate_hurst(taus[s_mask], ex[s_mask])[0] if s_mask.sum() >= 3 else None
    dense = np.geomspace(1.0 / 365, 0.5, 14)
    dsk = np.array([exact_atm_skew(TRUTH, t)[0] for t in dense])
    local = 0.5 + np.diff(np.log(np.abs(dsk))) / np.diff(np.log(dense))
    mid_days = 365 * np.sqrt(dense[1:] * dense[:-1])
    res = {"taus_days": (taus * 365).tolist(), "chain_skews": full["skews"], "exact_skews": ex.tolist(),
           "exact_skew_step_error": [e[1] for e in exact],
           "reading_full": full["H"], "reading_full_se": full.get("H_err"),
           "reading_short": short.get("H"), "reading_short_se": short.get("H_err"),
           "exact_law_full": H_ex_full, "exact_law_short": H_ex_short,
           "local_H": {"days": mid_days.tolist(), "H": local.tolist()}, "seconds": time.perf_counter() - t0}
    say_a(res)
    return res



def say_a(res):
    """Part A's report, from its result -- so `summary` can print it without re-recording a chain
    (whose expiries, and so whose numbers, move with the calendar)."""
    taus = np.array(res["taus_days"])
    say("A  the skew slope on the synthetic chain (priced exactly by the model; planted H = 0.10)")
    say(f"   expiries (days)            {np.round(taus, 1).tolist()}")
    say(f"   chain's ATM skews           {np.round(res['chain_skews'], 4).tolist()}")
    say(f"   the model's exact skews     {np.round(res['exact_skews'], 4).tolist()}")
    say(f"   reading, every expiry       H {res['reading_full']:+.3f} (se {res['reading_full_se'] or float('nan'):.3f})   "
        f"the exact skews give {res['exact_law_full']:+.3f}")
    if res.get("reading_short") is not None:
        say(f"   reading, <= 16 days         H {res['reading_short']:+.3f} (se {res['reading_short_se'] or float('nan'):.3f})   "
            f"the exact skews give {res['exact_law_short']:+.3f}")
    say("   the power law's own exponent between neighbouring maturities (exact skews):")
    say("     " + "  ".join(f"{m:5.1f}d {h:+.3f}" for m, h in zip(res["local_H"]["days"], res["local_H"]["H"])))


# ------------------------------------------------------------------ B, C: the history readings
def history_readings(n_days, seed):
    """The pipeline's history readings on one synthetic history -- analyse_history's settings,
    with the parts that do not bear on H left out."""
    data = syn.synthetic_history(TRUTH, n_days=n_days, seed=seed)
    rv = hist.realised_variance(data["bars5m"])
    y = np.asarray(rv["rv"])[-n_days:]
    m_eq = int(np.median(rv["bars"]))
    theta0 = float(np.mean(y))
    model = lambda dt_, H, v0, N=None: kf.LiftedRoughRVModel(dt_, H=H, v0=v0, N=N, bars_per_day=m_eq)
    with contextlib.redirect_stdout(io.StringIO()):
        prof = kf.profile_h(y, rrd.H_GRID, dict(kappa=3.0, theta=theta0, xi=0.3, R=1e-10), 1.0 / 252,
                            theta0, names=("xi",), model_cls=model)
    rob = prof.get("se_robust_detail") or {}
    ratio = rob.get("ratio", float("nan"))
    bank = kf.filter_bank(prof["runs"], rrd.H_GRID)
    row = {"n_days": n_days, "seed": seed, "H": prof["H_hat"], "se_curvature": prof["se_quadratic"],
           "se_robust": prof.get("se_robust", float("nan")), "ratio": ratio,
           "loglik": prof["loglik"].tolist(), "bank_mean": float(bank["mean"][-1]), "bank_sd": float(bank["sd"][-1])}
    if np.isfinite(ratio) and ratio > 0:
        bt = kf.filter_bank(prof["runs"], rrd.H_GRID, temper=1.0 / ratio ** 2)
        row.update(bank_t_mean=float(bt["mean"][-1]), bank_t_sd=float(bt["sd"][-1]))
    sf = vc.hurst_from_structure(np.log(np.sqrt(np.maximum(y, 1e-10))))
    row["H_structure"] = sf.get("H")
    return row


def run_rows(tag, jobs):
    rows, t0 = [], time.perf_counter()
    for n_days, seed in jobs:
        ts = time.perf_counter()
        r = history_readings(n_days, seed)
        rows.append(r)
        say(f"  {tag} {n_days:4d} days, seed {seed:2d}: H {r['H']:.3f}  se {r['se_curvature']:.3f} / robust "
            f"{r['se_robust']:.3f}  bank {r['bank_mean']:.3f} +/- {r['bank_sd']:.3f}"
            + (f"  tempered {r['bank_t_mean']:.3f} +/- {r['bank_t_sd']:.3f}" if "bank_t_mean" in r else "")
            + f"  structure {r['H_structure'] if r['H_structure'] is None else round(r['H_structure'], 3)}"
            f"  ({time.perf_counter() - ts:.0f}s)")
        (CAP / f"h_readings_{tag}.json").write_text(json.dumps({"rows": rows, "seconds": time.perf_counter() - t0},
                                                               indent=1, default=float), encoding="utf-8")
    return rows


def part_b():
    return run_rows("B", [(500, s) for s in SEEDS_B])


def part_c():
    return run_rows("C", [(n, s) for n in LENGTHS_C for s in SEEDS_C])


# ------------------------------------------------------------------ the scorecard
def grade(rows, truth=TRUTH.H):
    """Bias, spread and 95% coverage of every history reading over a set of histories."""
    H = np.array([r["H"] for r in rows])
    inside = np.array([EDGE[0] + 1e-9 < h < EDGE[1] - 1e-9 for h in H])
    out = {"n": len(rows), "edge_hits": int((~inside).sum())}

    def cover(est, se, mask):
        est, se = np.asarray(est, float), np.asarray(se, float)
        ok = mask & np.isfinite(est) & np.isfinite(se)
        return float(np.mean(np.abs(est[ok] - truth) <= 1.96 * se[ok])) if ok.any() else float("nan")

    Hi = H[inside]
    out["profile"] = {"mean": float(Hi.mean()), "median": float(np.median(Hi)), "sd": float(Hi.std(ddof=1)),
                      "bias": float(Hi.mean() - truth), "median_bias": float(np.median(Hi) - truth),
                      "coverage_curvature": cover(H, [r["se_curvature"] for r in rows], inside),
                      "coverage_robust": cover(H, [r["se_robust"] for r in rows], inside),
                      "median_se_robust": float(np.nanmedian([r["se_robust"] for r in rows]))}
    bm = np.array([r["bank_mean"] for r in rows])
    out["bank"] = {"mean": float(bm.mean()), "sd_of_means": float(bm.std(ddof=1)),
                   "median_reported_sd": float(np.median([r["bank_sd"] for r in rows])),
                   "coverage": cover(bm, [r["bank_sd"] for r in rows], np.ones(len(rows), bool))}
    tm = [r for r in rows if "bank_t_mean" in r]
    if tm:
        tmm = np.array([r["bank_t_mean"] for r in tm])
        out["bank_tempered"] = {"n": len(tm), "mean": float(tmm.mean()), "sd_of_means": float(tmm.std(ddof=1)),
                                "median_reported_sd": float(np.median([r["bank_t_sd"] for r in tm])),
                                "coverage": cover(tmm, [r["bank_t_sd"] for r in tm], np.ones(len(tm), bool))}
    hs = np.array([np.nan if r["H_structure"] is None else r["H_structure"] for r in rows])
    hs = hs[np.isfinite(hs)]
    out["structure"] = {"n": int(hs.size), "mean": float(hs.mean()), "sd": float(hs.std(ddof=1)),
                        "bias": float(hs.mean() - truth)}
    return out


def summary():
    res = {}
    for part in "ABC":
        f = CAP / f"h_readings_{part}.json"
        if f.exists():
            res[part] = json.loads(f.read_text(encoding="utf-8"))
    if "A" in res:
        say_a(res["A"])
    if "B" in res:
        res["scorecard_500"] = grade(res["B"]["rows"])
        g = res["scorecard_500"]
        say(f"\nB  the history readings over {g['n']} synthetic histories of 500 days (planted H = 0.10; "
            f"{g['edge_hits']} on the grid's edge, left out of the profile's spread)")
        p = g["profile"]
        say(f"   filter profile        mean {p['mean']:.3f}  median {p['median']:.3f}  sd {p['sd']:.3f}  "
            f"95% coverage: curvature {100 * p['coverage_curvature']:.0f}%, robust {100 * p['coverage_robust']:.0f}%")
        b = g["bank"]
        say(f"   filter bank           mean {b['mean']:.3f}  sd of the means {b['sd_of_means']:.3f}  "
            f"reported sd (median) {b['median_reported_sd']:.3f}  95% coverage {100 * b['coverage']:.0f}%")
        if "bank_tempered" in g:
            t = g["bank_tempered"]
            say(f"   ...tempered           mean {t['mean']:.3f}  sd of the means {t['sd_of_means']:.3f}  "
                f"reported sd (median) {t['median_reported_sd']:.3f}  95% coverage {100 * t['coverage']:.0f}%")
        s = g["structure"]
        say(f"   structure function    mean {s['mean']:.3f}  sd {s['sd']:.3f}  (no error bar of its own)")
    if "C" in res:
        by = {}
        rows_all = list(res["C"]["rows"]) + (list(res["B"]["rows"]) if "B" in res else [])
        for n in sorted({r["n_days"] for r in rows_all}):
            sub = [r for r in rows_all if r["n_days"] == n and (n != 500 or r["seed"] in SEEDS_C)]
            if len(sub) >= 3:
                by[n] = grade(sub)
        res["by_length"] = {str(k): v for k, v in by.items()}
        say("\nC  the filter profile as the history grows (seeds 3-12 at every length)")
        say("   days   mean   median   sd     bias    robust se   coverage robust / curvature   edge")
        for n, g in by.items():
            p = g["profile"]
            say(f"   {n:4d}  {p['mean']:.3f}  {p['median']:.3f}  {p['sd']:.3f}  {p['bias']:+.3f}   "
                f"{p['median_se_robust']:.3f}       {100 * p['coverage_robust']:3.0f}% / "
                f"{100 * p['coverage_curvature']:3.0f}%                    {g['edge_hits']}")
        ns = np.array(sorted(by))
        bias = np.array([by[n]["profile"]["bias"] for n in ns])
        if len(ns) >= 3 and np.all(bias > 0):
            slope = float(np.polyfit(np.log(ns), np.log(bias), 1)[0])
            res["bias_rate"] = slope
            say(f"   the bias moves like T^{slope:+.2f} (a consistent estimator's finite-sample bias goes like T^-1 "
                f"or T^-1/2; a flat one does not go)")
    (CAP / "h_readings.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main(argv):
    want = [a.upper() for a in argv] or ["A", "SUMMARY"]
    if "A" in want:
        (CAP / "h_readings_A.json").write_text(json.dumps(part_a(), indent=1, default=float), encoding="utf-8")
    if "B" in want:
        part_b()
    if "C" in want:
        part_c()
    if "SUMMARY" in want:
        summary()


if __name__ == "__main__":
    main(sys.argv[1:])
