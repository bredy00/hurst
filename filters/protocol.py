"""
The zero-boundary filtering protocol for daily realised variance (Session I).

Decision of the analytics review of 18 September 2026: the characteristic-function
filter does the state estimation where the zero boundary binds, the 16-substep
particle filter is only an independent confirmation, and the Kalman filter is used
everywhere away from zero, where it is cheap and adequate.

    1. Kalman. The exact-moment Kalman filter for daily RV (LiftedRoughRVModel) at the
       given parameters -- the pipeline's maximum-likelihood estimates. Cheap.
    2. Does the boundary bind? From the Kalman filter's own one-step predictions: the
       share of days whose predicted spot variance lies within two predictive standard
       deviations of zero (mean - 2 sd <= 0). Above BOUNDARY_SHARE (5%) the Gaussian
       predictive law is not trusted there: Session G measured kappa +1.79 SE with
       R = 0.01^2 and -0.65 SE with R = 0.002^2 on a host that sits at zero 13% of days.
    3. Where it binds: the cf filter (filters/fourier.py, observation "rv") is the state
       estimate -- the filtered spot variance and the filtered RV of each day.
    4. Confirmation: the bootstrap particle filter with 16 QE substeps a day, on two
       seeds, at the same parameters. It is exact up to Monte Carlo error, so it grades
       the cf filter's approximations (a gamma posterior for V, Gaussian factors given V)
       where they matter. Confirmed when the two seeds agree with each other, the cf
       log-likelihood is within LL_TOL_PER_DAY per day of the particle filter's, and the
       filtered RV paths agree to PATH_TOL (median relative difference on boundary days).
    5. Escalation. At 16 substeps the particle filter is exact for the QE-DISCRETISED
       law, and at a hard boundary QE's atom at zero makes that law depend on the step.
       Measured on 300 days simulated with 256 substeps from a host at zero on 64% of days
       (theta 0.02, xi 0.9): the particle filter's log-likelihood is 1004.5 at 16 substeps,
       1442.2 at 64 and 1478.8 at 256 -- converging, and the 16-substep filter the worst
       of all the filters; the cf filter 1351.2 (0.43 nats a day short of converged), the
       Kalman filter 729.8. So when the cf and 16-substep filters disagree, the particle
       filter is rerun at 4x the substeps (one seed) and the verdict is taken against
       that: "confirmed at 64 substeps" or "not confirmed". A hard boundary like that one
       comes out "not confirmed": the cf filter's gamma posterior is the better estimate
       on offer, but it carries a measured approximation cost there.

All three filters use the same observation model: y = I / dt + N(0, R + 2/M y^2), realised
variance's sampling error at the observed RV (Barndorff-Nielsen & Shephard's feasible
version, rv_noise="observed"). With the error at the prior mean instead, a day whose RV
jumps above a small forecast becomes an observation of 16% relative precision at the
forecast's scale; on a synthetic host at zero on 64% of days that broke the cf filter
(non-positive densities on 12 days) and one particle-filter seed (log-likelihood -4152
against +533 on another). The observed version is for STATE estimation only: it weights
each day by its own noise realisation, which biases parameter estimates (the pipeline's H
profile read 0.183 with it and 0.130 without, planted 0.10), so the parameters handed in
come from the prior-mean model. See docs/session-i-report.md.
"""

import math
import time

import numpy as np

import filters.kalman as kf

BOUNDARY_SHARE = 0.05       # share of days with predicted V within 2 predictive sd of zero
LL_TOL_PER_DAY = 0.05       # |cf - particle| log-likelihood, nats per day
PATH_TOL = 0.05             # median relative difference of the filtered RV on boundary days
SEED_TOL = 5.0              # the two particle-filter seeds must agree to this many nats


def boundary_days(model, p, y):
    """
    (flags, kalman result): flags[t] is True where the Kalman filter's predicted spot
    variance for day t is within two predictive standard deviations of zero.
    """
    res = kf.kalman_filter(model, p, y)
    n = len(model.w)
    w = model.w
    mean_V = model.v0 + res["x_prior"][:, :n] @ w
    sd_V = np.sqrt(np.maximum(np.einsum("i,tij,j->t", w, res["P_prior"][:, :n, :n], w), 0.0))
    return mean_V - 2.0 * sd_V <= 0.0, res, mean_V, sd_V


def run(p, y, dt=1.0 / 252, H=0.12, v0=0.04, bars_per_day=78, confirm=True, n_particles=2000,
        seeds=(1, 2), substeps=16, escalate=True, verbose=False):
    """
    The protocol on one RV series at parameters p (kappa, theta, xi, R). Returns a dict with
    the Kalman result, the boundary diagnostic, and -- if the boundary binds -- the cf
    state estimate and the particle-filter confirmation.
    """
    from filters.fourier import FourierFilter
    from filters.particle import ParticleFilter

    y = np.asarray(y, float)
    T = len(y)
    out = {"n_days": T, "params": dict(p)}
    t0 = time.perf_counter()
    km = kf.LiftedRoughRVModel(dt, H=H, v0=v0, bars_per_day=bars_per_day, rv_noise="observed")
    flags, kres, mV, sV = boundary_days(km, p, y)
    share = float(flags.mean())
    out["kalman"] = {"loglik": float(kres["loglik"]), "seconds": time.perf_counter() - t0,
                     "filtered_rv": kres["estimate"].tolist()}
    out["boundary"] = {"share": share, "threshold": BOUNDARY_SHARE, "binds": share > BOUNDARY_SHARE,
                       "days": np.flatnonzero(flags).tolist()}
    if verbose:
        print(f"  Kalman: loglik {kres['loglik']:.2f}; boundary binds on {100 * share:.1f}% of days", flush=True)
    if not out["boundary"]["binds"]:
        out["state_estimate"] = "kalman"
        return out

    t0 = time.perf_counter()
    # one run at fixed parameters: size the frequency panels' meshes for them (25% margin)
    # rather than for the estimation caps; the step count grows like xi^1.6
    cf = FourierFilter(dt, H=H, v0=v0, observation="rv", bars_per_day=bars_per_day, rv_noise="observed",
                       xi_cap=1.25 * p["xi"], kappa_cap=1.25 * p["kappa"] + 1.0)
    cres = cf.run(p, y, keep_path=True)
    out["cf"] = {"loglik": float(cres["loglik"]), "seconds": time.perf_counter() - t0,
                 "fallback_days": int(cres["fallback_days"]), "capped_days": int(cres["capped_days"]),
                 "panels": int(cres["panels"]), "filtered_V": cres["mean_V"].tolist(),
                 "filtered_V_sd": cres["sd_V"].tolist(), "filtered_rv": cres["post_mean_rv"].tolist()}
    out["state_estimate"] = "cf"
    if verbose:
        print(f"  cf filter: loglik {cres['loglik']:.2f} ({out['cf']['seconds']:.0f}s; fallback days "
              f"{cres['fallback_days']}, capped days {cres['capped_days']})", flush=True)
    if not confirm:
        return out

    runs = []
    for sd in seeds:
        t0 = time.perf_counter()
        pf = ParticleFilter(dt, H=H, v0=v0, n_particles=n_particles, substeps=substeps, observation="rv",
                            bars_per_day=bars_per_day, rv_noise="observed", seed=sd)
        r = pf.run(p, y, keep_path=True)
        runs.append({"seed": sd, "loglik": float(r["loglik"]), "min_ess": float(r["ess"].min()),
                     "filtered_rv": r["mean_V"].tolist(), "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"  particle filter seed {sd}: loglik {r['loglik']:.2f} (min ESS {r['ess'].min():.0f}, "
                  f"{runs[-1]['seconds']:.0f}s)", flush=True)
    ll_pf = np.array([r["loglik"] for r in runs])
    pf_path = np.mean([r["filtered_rv"] for r in runs], axis=0)
    cf_path = np.asarray(cres["post_mean_rv"])
    on = flags if flags.any() else np.ones(T, bool)
    rel = np.abs(cf_path[on] - pf_path[on]) / np.maximum(np.abs(pf_path[on]), 1e-12)
    seed_gap = float(ll_pf.max() - ll_pf.min())
    ll_gap = float(cres["loglik"] - ll_pf.mean())
    checks = {"particle seeds agree": seed_gap <= SEED_TOL,
              "cf log-likelihood within tolerance of the particle filter": abs(ll_gap) / T <= LL_TOL_PER_DAY,
              "filtered RV agrees on boundary days": float(np.median(rel)) <= PATH_TOL}
    out["confirmation"] = {"particle_runs": runs, "particle_loglik_mean": float(ll_pf.mean()),
                           "seed_gap": seed_gap, "cf_minus_particle": ll_gap,
                           "cf_minus_particle_per_day": ll_gap / T,
                           "median_rel_rv_diff_boundary_days": float(np.median(rel)),
                           "p90_rel_rv_diff_boundary_days": float(np.quantile(rel, 0.9)),
                           "kalman_minus_particle": float(kres["loglik"] - ll_pf.mean()),
                           "checks": checks, "confirmed": all(checks.values()),
                           "verdict": "confirmed" if all(checks.values()) else "not confirmed"}
    if not all(checks.values()) and escalate:
        # The 16-substep particle filter can be the worse of the two at a hard boundary
        # (module docstring): rerun it at 4x the substeps, one seed, and judge against that.
        t0 = time.perf_counter()
        pf = ParticleFilter(dt, H=H, v0=v0, n_particles=n_particles, substeps=4 * substeps, observation="rv",
                            bars_per_day=bars_per_day, rv_noise="observed", seed=seeds[0])
        r = pf.run(p, y, keep_path=True)
        gap = float(cres["loglik"] - r["loglik"])
        rel4 = np.abs(cf_path[on] - np.asarray(r["mean_V"])[on]) / np.maximum(np.abs(np.asarray(r["mean_V"])[on]), 1e-12)
        ok4 = abs(gap) / T <= LL_TOL_PER_DAY and float(np.median(rel4)) <= PATH_TOL
        out["confirmation"]["escalation"] = {
            "substeps": 4 * substeps, "loglik": float(r["loglik"]), "seconds": time.perf_counter() - t0,
            "cf_minus_particle": gap, "cf_minus_particle_per_day": gap / T,
            "particle16_minus_particle": float(ll_pf.mean() - r["loglik"]),
            "median_rel_rv_diff_boundary_days": float(np.median(rel4))}
        out["confirmation"]["verdict"] = (f"confirmed at {4 * substeps} substeps (the {substeps}-substep particle "
                                          "filter under-resolves this boundary)" if ok4 else "not confirmed")
        out["confirmation"]["confirmed"] = ok4
    if verbose:
        c = out["confirmation"]
        print(f"  confirmation: cf - particle {ll_gap:+.2f} ({ll_gap / T:+.4f}/day), Kalman - particle "
              f"{c['kalman_minus_particle']:+.2f}; seeds {seed_gap:.2f} apart; filtered RV median rel diff "
              f"{100 * c['median_rel_rv_diff_boundary_days']:.1f}% on boundary days -> "
              f"{c['verdict']}", flush=True)
        if "escalation" in c:
            e = c["escalation"]
            print(f"  escalation to {e['substeps']} substeps: cf - particle {e['cf_minus_particle']:+.2f} "
                  f"({e['cf_minus_particle_per_day']:+.4f}/day); the {substeps}-substep filter was "
                  f"{e['particle16_minus_particle']:+.2f} from it", flush=True)
    return out
