"""
Is the pipeline's error bar on H honest? (Session N, 27 September 2026)

Session L found that the history filter's H, estimated from 500 days of synthetic realised
variance with a planted H = 0.10, spreads with sd 0.079 across seeds while the profile
likelihood's curvature reports a median SE of 0.028 -- the reported precision about three
times too good. That matters the moment real data arrives: the report would state an H to
+/- 0.03 that is really known to +/- 0.08.

`filters.kalman.robust_profile_se` is the quasi-likelihood correction (the sandwich, with
the per-day scores the filters report since Session M and a Newey-West long-run variance).
This study grades it against the only thing that can grade an error bar: the spread of the
estimate over many independent histories with a known answer.

Pre-registered before the run: the lag is the automatic rule floor(4 (T/100)^(2/9)) -- 5
for 500 days -- and the robust SE is judged honest if its median is within 25% of the
interior estimates' spread. Lags 2x and 4x are reported as sensitivity only.

    python study_h_error_bars.py [n_seeds]        (~50 s a seed here; 21 seeds + lags ~35 min)
    -> captures/h_error_bars.json, captures/h_error_bars.log
"""

import contextlib
import io
import json
import pathlib
import sys
import time

import numpy as np

import filters.kalman as kf
import models.rough_heston as rh
import run_real_data as rrd
import sources.synthetic as syn

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "h_error_bars.json"
TRUTH = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)
EDGE = (min(rrd.H_GRID), max(rrd.H_GRID))


def say(msg):
    print(msg, flush=True)


def one(seed):
    data = syn.synthetic_history(TRUTH, n_days=500, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        hs = rrd.analyse_history(data, max_days=500, confirm=False)
    return hs


def main(argv):
    n = int(argv[0]) if argv else 21
    rows = []
    t0 = time.perf_counter()
    for seed in range(3, 3 + n):
        ts = time.perf_counter()
        hs = one(seed)
        rp = hs["rough_profile"]
        rows.append({"seed": seed, "H": rp["H_hat"], "se_curvature": rp["se"],
                     "se_robust": rp.get("se_robust", float("nan")), "ratio": rp.get("se_ratio", float("nan"))})
        say(f"  seed {seed:2d}: H {rp['H_hat']:.3f}  curvature se {rp['se']:.3f}  robust se "
            f"{rp.get('se_robust', float('nan')):.3f}  ({time.perf_counter() - ts:.0f}s)")
    H = np.array([r["H"] for r in rows])
    interior = np.array([EDGE[0] + 1e-9 < h < EDGE[1] - 1e-9 for h in H])
    se_c = np.array([r["se_curvature"] for r in rows])
    se_r = np.array([r["se_robust"] for r in rows])
    ok_c, ok_r = np.isfinite(se_c) & interior, np.isfinite(se_r) & interior
    spread = float(H[interior].std(ddof=1))
    bias = float(H[interior].mean() - TRUTH.H)
    med_c, med_r = float(np.median(se_c[ok_c])), float(np.median(se_r[ok_r]))
    cover_c = float(np.mean(np.abs(H[ok_c] - TRUTH.H) <= 1.96 * se_c[ok_c]))
    cover_r = float(np.mean(np.abs(H[ok_r] - TRUTH.H) <= 1.96 * se_r[ok_r]))
    verdict = abs(med_r / spread - 1) <= 0.25
    say(f"\n{len(rows)} histories of 500 days, planted H = {TRUTH.H}; {int((~interior).sum())} landed on the "
        f"grid's edge ({EDGE[0]} or {EDGE[1]}) and are excluded from the spread")
    say(f"  the estimate's actual spread (interior): sd {spread:.4f}; its mean bias {bias:+.4f}")
    say(f"  curvature SE: median {med_c:.4f} = {med_c / spread:.2f}x the spread; 95% intervals cover the "
        f"truth in {100 * cover_c:.0f}% of histories")
    say(f"  robust SE:    median {med_r:.4f} = {med_r / spread:.2f}x the spread; 95% intervals cover the "
        f"truth in {100 * cover_r:.0f}% of histories")
    say(f"  pre-registered test (robust median within 25% of the spread): {'PASS' if verdict else 'FAIL'}")

    # sensitivity to the lag, reported, not used for the verdict
    sens = {}
    say("  sensitivity to the Newey-West lag (not used for the verdict):")
    for mult in (1, 2, 4):
        vals = []
        for seed in range(3, 3 + min(n, 8)):
            data = syn.synthetic_history(TRUTH, n_days=500, seed=seed)
            with contextlib.redirect_stdout(io.StringIO()):
                import sources.history as hist
                rv = hist.realised_variance(data["bars5m"])
                y = np.asarray(rv["rv"])[-500:]
                m_eq = int(np.median(rv["bars"]))
            theta0 = float(np.mean(y))
            model = lambda dt_, H, v0, N=None: kf.LiftedRoughRVModel(dt_, H=H, v0=v0, N=N, bars_per_day=m_eq)
            prof = kf.profile_h(y, rrd.H_GRID, dict(kappa=3.0, theta=theta0, xi=0.3, R=1e-10), 1 / 252, theta0,
                                names=("xi",), model_cls=model)
            j = int(np.argmax(prof["loglik"]))
            d = kf.robust_profile_se(rrd.H_GRID, prof["runs"], j, prof["H_hat"],
                                     lag=mult * kf.newey_west_lag(len(y)))
            if d:
                vals.append(d["se"])
        sens[f"{mult}x"] = float(np.median(vals)) if vals else float("nan")
        say(f"    lag {mult}x the rule: median robust SE {sens[f'{mult}x']:.4f} "
            f"({sens[f'{mult}x'] / spread:.2f}x the spread; first {min(n, 8)} seeds)")
    OUT.write_text(json.dumps({"rows": rows, "spread": spread, "bias": bias, "median_se_curvature": med_c,
                               "median_se_robust": med_r, "coverage_curvature": cover_c,
                               "coverage_robust": cover_r, "edge_hits": int((~interior).sum()),
                               "pass": bool(verdict), "lag_sensitivity": sens,
                               "seconds": time.perf_counter() - t0}, indent=1, default=float), encoding="utf-8")
    say(f"\n[done in {time.perf_counter() - t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
