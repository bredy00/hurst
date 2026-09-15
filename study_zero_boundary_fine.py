"""
Is the characteristic-function filter's low kappa near zero its gamma approximation,
or the benchmark's data? (Session H follow-up to study_zero_boundary.py.)

The benchmark simulated 4 QE substeps a day and the adapted particle filter uses the
same 4-substep QE law, so the particle filter was exactly specified while the CF filter
carried the continuous lift's transition law. Here the data use 64 substeps a day --
close to the continuous law -- and the particle filter runs at 4 and 16 substeps.

    python study_zero_boundary_fine.py     (~15 minutes) -> captures/zero_boundary_fine.json, .log
"""

import json
import math
import pathlib
import time

import numpy as np

import filters.fourier as ff
import filters.kalman as kf
import filters.particle as pfm
from study_zero_boundary import profile, GRID, DT, T

ROOT = pathlib.Path(__file__).parent
P = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.01 ** 2)


def main(seeds=(61, 63, 65)):
    out = []
    for seed in seeds:
        m = kf.LiftedRoughModel(DT, H=0.12, v0=P["theta"])
        t0 = time.perf_counter()
        V, _ = kf.simulate_rough(m, P, T, seed=seed, substeps=64)
        y = V + np.random.default_rng(600 + seed).normal(0.0, math.sqrt(P["R"]), T)
        row = {"seed": seed, "V_at_zero": float(np.mean(V < 1e-12)), "V_below_0.005": float(np.mean(V < 0.005))}
        row["kalman"] = profile(lambda k: kf.kalman_filter(m, dict(P, kappa=k), y)["loglik"], GRID)
        fil = ff.FourierFilter(DT, H=0.12, v0=P["theta"])
        row["fourier"] = profile(lambda k: fil.loglik(dict(P, kappa=k), y), GRID)
        for S in (4, 16):
            apf = pfm.AdaptedParticleFilter(DT, H=0.12, v0=P["theta"], n_particles=500, substeps=S, seed=seed)
            row[f"particle_{S}"] = profile(lambda k: apf.loglik(dict(P, kappa=k), y), GRID)
        out.append(row)
        print(f"seed {seed} (64-substep data; V exactly 0 on {100*row['V_at_zero']:.1f}% of days, below 0.005 on "
              f"{100*row['V_below_0.005']:.1f}%): " + "  ".join(
                  f"{k}: kappa {row[k]['hat']:.2f}+/-{row[k]['se']:.2f}" for k in ("kalman", "fourier", "particle_4", "particle_16"))
              + f"  ({time.perf_counter()-t0:.0f}s)", flush=True)
    for k in ("kalman", "fourier", "particle_4", "particle_16"):
        d = [r[k]["hat"] - r["particle_16"]["hat"] for r in out]
        print(f"   {k:12s} mean kappa {np.mean([r[k]['hat'] for r in out]):.2f}; minus particle(16 substeps) per seed "
              f"{', '.join(f'{x:+.2f}' for x in d)}", flush=True)
    (ROOT / "captures" / "zero_boundary_fine.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
