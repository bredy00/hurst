"""
Particle filter vs the characteristic-function filter at the zero boundary (Session H).

Same data as Session G's flag: lifted rough variance (H = 0.12, kappa = 3) from the
positivity-preserving simulator, observed daily with noise R = 0.01^2.

  near zero     theta = 0.04, xi = 0.3: V exactly zero on ~13% of days
  away          theta = 0.09, xi = 0.08: V never near zero (all filters should agree)

For each seed and filter: kappa by profile likelihood on a grid (theta, xi, R at
the truth), the maximiser and Wald SE from a quadratic through the best three
points, z = (kappa_hat - 3) / SE. Also the log-likelihood at the truth, the
filtered-variance RMSE, 90% band coverage (where the filter gives a band) and the
cost of one likelihood.

    python study_zero_boundary.py          (~25 minutes) -> captures/zero_boundary.json, .png
"""

import json
import math
import pathlib
import time

import numpy as np

import filters.fourier as ff
import filters.kalman as kf
import filters.particle as pfm

ROOT = pathlib.Path(__file__).parent
DT = 1.0 / 252
T = 1500
GRID = (2.0, 2.5, 3.0, 3.5, 4.0)
CASES = {
    "near zero": (dict(kappa=3.0, theta=0.04, xi=0.3, R=0.01 ** 2), (61, 63, 65, 67, 69, 71)),
    "away from zero": (dict(kappa=3.0, theta=0.09, xi=0.08, R=0.01 ** 2), (61, 63, 65, 67)),
}


def profile(loglik, grid):
    grid = list(grid)
    lls = [loglik(k) for k in grid]
    while int(np.argmax(lls)) == 0 and grid[0] > 0.6:          # extend if the maximum sits on an edge
        grid.insert(0, grid[0] - 0.5)
        lls.insert(0, loglik(grid[0]))
    while int(np.argmax(lls)) == len(grid) - 1 and grid[-1] < 9.0:
        grid.append(grid[-1] + 0.5)
        lls.append(loglik(grid[-1]))
    g, l = np.array(grid), np.array(lls)
    j = int(np.argmax(l))
    a, b, _ = np.polyfit(g[j - 1:j + 2], l[j - 1:j + 2], 2)
    hat = float(-b / (2 * a)) if a < 0 else float(g[j])
    se = float(1.0 / math.sqrt(-2 * a)) if a < 0 else float("nan")
    return {"grid": g.tolist(), "loglik": l.tolist(), "hat": hat, "se": se, "z": (hat - 3.0) / se}


def run():
    out = {}
    for case, (p, seeds) in CASES.items():
        rows = []
        for seed in seeds:
            m = kf.LiftedRoughModel(DT, H=0.12, v0=p["theta"])
            V, _ = kf.simulate_rough(m, p, T, seed=seed)
            y = V + np.random.default_rng(600 + seed).normal(0.0, math.sqrt(p["R"]), T)
            row = {"seed": seed, "V_at_zero": float(np.mean(np.abs(V) < 1e-12))}

            t0 = time.perf_counter()
            r = kf.kalman_filter(m, p, y)
            row["kalman"] = {"cost": time.perf_counter() - t0, "loglik_truth": r["loglik"],
                             "rmse": float(np.sqrt(np.mean((r["estimate"] - V) ** 2))),
                             **profile(lambda k: kf.kalman_filter(m, dict(p, kappa=k), y)["loglik"], GRID)}

            fil = ff.FourierFilter(DT, H=0.12, v0=p["theta"])
            t0 = time.perf_counter()
            r = fil.run(p, y, keep_path=True)
            band = (V >= r["mean_V"] - 1.645 * r["sd_V"]) & (V <= r["mean_V"] + 1.645 * r["sd_V"])
            row["fourier"] = {"cost": time.perf_counter() - t0, "loglik_truth": r["loglik"],
                              "rmse": float(np.sqrt(np.mean((r["mean_V"] - V) ** 2))),
                              "coverage90_gaussian_band": float(np.mean(band)),
                              **profile(lambda k: ff.FourierFilter(DT, H=0.12, v0=p["theta"]).loglik(dict(p, kappa=k), y), GRID)}

            apf = pfm.AdaptedParticleFilter(DT, H=0.12, v0=p["theta"], n_particles=500, seed=seed)
            t0 = time.perf_counter()
            r = apf.run(p, y, keep_path=True)
            row["particle"] = {"cost": time.perf_counter() - t0, "loglik_truth": r["loglik"],
                               "rmse": float(np.sqrt(np.mean((r["mean_V"] - V) ** 2))),
                               "coverage90": float(np.mean((V >= r["q05"]) & (V <= r["q95"]))),
                               "ess_median": float(np.median(r["ess"])),
                               **profile(lambda k: apf.loglik(dict(p, kappa=k), y), GRID)}
            rows.append(row)
            print(f"{case} seed {seed}: " + "  ".join(
                f"{k}: kappa {row[k]['hat']:.2f}+/-{row[k]['se']:.2f} (z {row[k]['z']:+.2f}), ll {row[k]['loglik_truth']:.1f}, "
                f"rmse {row[k]['rmse']:.4f}, {row[k]['cost']:.1f}s" for k in ("kalman", "fourier", "particle")), flush=True)
        out[case] = rows
    summary(out)
    (ROOT / "captures" / "zero_boundary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    plot(out)
    return out


def summary(out):
    for case, rows in out.items():
        print(f"\n{case}:")
        for k in ("kalman", "fourier", "particle"):
            zs = np.array([r[k]["z"] for r in rows])
            print(f"   {k:9s} mean z {zs.mean():+.2f}, sd {zs.std(ddof=1):.2f}; mean kappa {np.mean([r[k]['hat'] for r in rows]):.2f}; "
                  f"loglik vs Kalman {np.mean([r[k]['loglik_truth'] - r['kalman']['loglik_truth'] for r in rows]):+.1f}; "
                  f"rmse {np.mean([r[k]['rmse'] for r in rows]):.5f}; cost {np.mean([r[k]['cost'] for r in rows]):.1f}s per likelihood")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    cols = {"kalman": "#d62728", "fourier": "#1f77b4", "particle": "#2ca02c"}
    names = {"kalman": "Gaussian (exact-moment Kalman)", "fourier": "characteristic function", "particle": "adapted particle filter"}
    for ci, (case, rows) in enumerate(out.items()):
        a = ax[ci]
        for i, k in enumerate(("kalman", "fourier", "particle")):
            zs = [r[k]["z"] for r in rows]
            a.plot(np.full(len(zs), i) + np.linspace(-0.15, 0.15, len(zs)), zs, "o", color=cols[k])
            a.plot([i - 0.25, i + 0.25], [np.mean(zs)] * 2, color="k", lw=2)
            a.text(i, max(zs) + 0.25, f"mean {np.mean(zs):+.2f}", ha="center", fontsize=8)
        a.axhspan(-2, 2, color="0.92", zorder=0)
        a.axhline(0, color="0.5", lw=0.8)
        a.set_xticks(range(3))
        a.set_xticklabels([names[k] for k in ("kalman", "fourier", "particle")], fontsize=8)
        a.set_ylabel("kappa: (estimate - truth) / SE")
        a.set_title(f"({'ab'[ci]}) {case}: kappa bias, {len(rows)} seeds x {T} days")
    a = ax[2]
    near = out["near zero"]
    labels = ["log-lik gain\nover Kalman", "filtered RMSE\n(x 1e3)", "seconds per\nlikelihood"]
    for i, k in enumerate(("kalman", "fourier", "particle")):
        vals = [np.mean([r[k]["loglik_truth"] - r["kalman"]["loglik_truth"] for r in near]),
                1e3 * np.mean([r[k]["rmse"] for r in near]), np.mean([r[k]["cost"] for r in near])]
        a.bar(np.arange(3) + (i - 1) * 0.27, vals, 0.27, color=cols[k], label=names[k])
        for x_, v in zip(np.arange(3) + (i - 1) * 0.27, vals):
            a.text(x_, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(range(3))
    a.set_xticklabels(labels, fontsize=8)
    a.set_yscale("symlog", linthresh=1.0)
    a.set_title("(c) near zero: fit, accuracy, cost")
    a.legend(fontsize=8)
    fig.suptitle("The zero boundary: Gaussian vs characteristic-function vs particle filtering of rough variance", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "zero_boundary.png", dpi=125)


if __name__ == "__main__":
    run()
