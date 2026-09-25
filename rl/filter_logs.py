"""
Producing the logs the filter-selection agent learns from (Session M).

Each series is filtered ONCE by each of the three filters, and what is recorded is what a
desk would actually have afterwards: every day's log-likelihood contribution under each
filter, and what the run cost in seconds. The MDP is then built over that table, so the
agent's counterfactual -- "what if this day had used the cf filter" -- is a logged row and
not a guess.

The reference is the 64-substep particle filter, which Session I measured as the converged
answer at a hard boundary (1004.5 nats at 16 substeps, 1442.2 at 64, 1478.8 at 256). It is
logged as `truth_loglik` so every policy can be scored against the best available estimate
rather than against each other.

This is the expensive part of the whole framework and it is why the framework is off by
default: one series of 300 days costs a minute or two, almost all of it particle filtering.
`captures/rl_filter_logs.json` caches the result so the study does not pay twice.
"""

import json
import pathlib
import time

import numpy as np

import filters.kalman as kf
import filters.protocol as proto
import models.rough_heston as rh
import rl
from rl.filter_env import FILTERS, day_features

CACHE = pathlib.Path(__file__).parent.parent / "captures" / "rl_filter_logs.json"


def per_day_loglik(res, T):
    """
    A filter's per-day log-likelihood contribution.

    The filters report a total; where a per-day array is carried (`loglik_t`) it is used,
    and otherwise the total is spread evenly, which is honest for a policy that chooses a
    filter for a WHOLE series and a stated approximation for one that switches per day.
    `docs/rl-framework.md` records which filters give which.
    """
    if "loglik_t" in res and res["loglik_t"] is not None:
        a = np.asarray(res["loglik_t"], float)
        if len(a) == T:
            return a, True
    return np.full(T, float(res["loglik"]) / max(T, 1)), False


def simulate_series(p_true, n_days, seed, bars_per_day=78, substeps=16):
    """One synthetic RV series from the lifted model, with its true daily integrated variance."""
    import sources.synthetic as syn
    data = syn.synthetic_history(p_true, n_days=n_days, bars_per_day=bars_per_day, seed=seed)
    import sources.history as hist
    rv = hist.realised_variance(data["bars5m"])["rv"]
    return np.asarray(rv, float), np.asarray(data["true_daily_integrated_variance"], float)


def log_one(y, p, dt=1.0 / 252, H=0.12, v0=0.04, bars_per_day=78, n_particles=1000,
            substeps=16, truth_substeps=64, seed=1, verbose=False):
    """
    Filter one series with all three filters and return the log entry.

    `p` is (kappa, theta, xi, R) -- the pipeline's estimates, not refitted here: the question
    is which filter to USE at known parameters, which is the decision the protocol makes.
    """
    rl.require_enabled("the filter logger")
    from filters.fourier import FourierFilter
    from filters.particle import ParticleFilter

    y = np.asarray(y, float)
    T = len(y)
    out = {"loglik": {}, "seconds": {}, "per_day_exact": {}}

    km = kf.LiftedRoughRVModel(dt, H=H, v0=v0, bars_per_day=bars_per_day, rv_noise="observed")
    t0 = time.perf_counter()
    flags, kres, mV, sV = proto.boundary_days(km, p, y)
    out["seconds"]["kalman"] = time.perf_counter() - t0
    ll_k, exact_k = per_day_loglik(kres, T)
    out["loglik"]["kalman"], out["per_day_exact"]["kalman"] = ll_k, exact_k

    t0 = time.perf_counter()
    cf = FourierFilter(dt, H=H, v0=v0, observation="rv", bars_per_day=bars_per_day,
                       rv_noise="observed", xi_cap=1.25 * p["xi"], kappa_cap=1.25 * p["kappa"] + 1.0)
    cres = cf.run(p, y, keep_path=True)
    out["seconds"]["cf"] = time.perf_counter() - t0
    ll_c, exact_c = per_day_loglik(cres, T)
    out["loglik"]["cf"], out["per_day_exact"]["cf"] = ll_c, exact_c

    t0 = time.perf_counter()
    pf = ParticleFilter(dt, H=H, v0=v0, n_particles=n_particles, substeps=substeps,
                        observation="rv", bars_per_day=bars_per_day, rv_noise="observed", seed=seed)
    pres = pf.run(p, y, keep_path=True)
    out["seconds"]["particle"] = time.perf_counter() - t0
    ll_p, exact_p = per_day_loglik(pres, T)
    out["loglik"]["particle"], out["per_day_exact"]["particle"] = ll_p, exact_p

    t0 = time.perf_counter()
    pf64 = ParticleFilter(dt, H=H, v0=v0, n_particles=n_particles, substeps=truth_substeps,
                          observation="rv", bars_per_day=bars_per_day, rv_noise="observed", seed=seed + 100)
    tres = pf64.run(p, y, keep_path=True)
    out["truth_seconds"] = time.perf_counter() - t0
    out["truth_loglik"], _ = per_day_loglik(tres, T)

    out["features"] = day_features(mV, sV, y, np.arange(T) / max(T - 1, 1))
    out["boundary_share"] = float(flags.mean())
    if verbose:
        tot = {k: float(np.sum(v)) for k, v in out["loglik"].items()}
        print(f"    {T} days, boundary {100 * out['boundary_share']:.0f}%: "
              + ", ".join(f"{k} {tot[k]:.1f} ({out['seconds'][k]:.0f}s)" for k in FILTERS)
              + f", reference(64) {float(np.sum(out['truth_loglik'])):.1f} ({out['truth_seconds']:.0f}s)",
              flush=True)
    return out


def build_logs(hosts, n_days=250, seeds=(1, 2, 3), use_cache=True, verbose=True):
    """
    Log every (host, seed) series. `hosts` is a list of (label, RoughHestonParams, filter params).

    Cached in `captures/rl_filter_logs.json`: this is minutes of particle filtering and the
    study reruns often.
    """
    rl.require_enabled("the filter logger")
    key = json.dumps([[h[0], [float(getattr(h[1], n)) for n in rh.RoughHestonParams.NAMES],
                       {k: float(v) for k, v in sorted(h[2].items())}] for h in hosts]
                     + [n_days, list(seeds)], sort_keys=True)
    if use_cache and CACHE.exists():
        blob = json.loads(CACHE.read_text(encoding="utf-8"))
        if blob.get("key") == key:
            if verbose:
                print(f"  (cached: {CACHE.name})", flush=True)
            return [_from_json(e) for e in blob["logs"]]
    logs = []
    for label, p_true, p_filt in hosts:
        for sd in seeds:
            if verbose:
                print(f"  {label}, seed {sd}:", flush=True)
            y, _ = simulate_series(p_true, n_days, sd)
            e = log_one(y, p_filt, H=p_true.H, v0=p_true.theta, seed=sd, verbose=verbose)
            e["host"], e["seed"] = label, sd
            logs.append(e)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"key": key, "logs": [_to_json(e) for e in logs]}), encoding="utf-8")
    return logs


def _to_json(e):
    d = dict(e)
    d["features"] = np.asarray(e["features"]).tolist()
    d["loglik"] = {k: np.asarray(v).tolist() for k, v in e["loglik"].items()}
    d["truth_loglik"] = np.asarray(e["truth_loglik"]).tolist()
    return d


def _from_json(d):
    e = dict(d)
    e["features"] = np.asarray(d["features"], float)
    e["loglik"] = {k: np.asarray(v, float) for k, v in d["loglik"].items()}
    e["truth_loglik"] = np.asarray(d["truth_loglik"], float)
    return e
