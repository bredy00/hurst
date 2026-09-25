"""
The T-EGARCH scan (Session L, 25 September 2026): EGARCH with Student-t innovations as the
returns-only volatility model to keep for when realised volatility arrives.

What a backup has to show is how close it gets to the model it backs up. So the scan runs on
ten years of daily returns simulated from this project's own rough Heston (the lifted model,
QE step, 78 five-minute bars a day; H = 0.10, xi = 0.4, rho = -0.7, the synthetic pipeline's
truth), where the answer is known: the true integrated variance of every day.

  1  the scan: every spec in models.egarch.DEFAULT_SCAN (EGARCH(p,q) for p, q in {1, 2} with
     normal and t innovations, Beta-t-EGARCH(1,1), GARCH(1,1) normal and t) fitted on the
     first seven years, ranked by BIC;
  2  out of sample, the last three years: one-day variance forecasts scored by QLIKE against
     the TRUE integrated variance (and against 5-minute realised variance, the proxy real
     data will have), beside EWMA (RiskMetrics, lambda 0.94) and HAR-RV (Corsi 2009, fitted
     on the same seven years of realised variance), with Diebold-Mariano statistics
     (Newey-West, 5 lags) against the best returns-only model;
  3  the same with price jumps added (compound Poisson, 12 a year, N(-2%, 3%)), where one
     extreme return is what separates Nelson's EGARCH from the score-driven Beta-t;
  0  first, a well-specified control: data simulated FROM EGARCH(1,1)-t (nu = 7), where the
     t model must win out of sample too, or the machinery is wrong.

    python study_egarch.py          (~6 minutes)
    -> captures/egarch.json, captures/egarch.log
"""

import contextlib
import io
import json
import math
import pathlib
import time

import numpy as np

import models.egarch as eg
import models.rough_heston as rh
import sources.history as hist
import sources.synthetic as syn

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "egarch.json"
TRUTH = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)
N_DAYS, N_TRAIN = 2520, 1764
JUMPS = dict(rate=12.0, mean=-0.02, sd=0.03)


def say(msg):
    print(msg, flush=True)


def simulate(seed=21):
    with contextlib.redirect_stdout(io.StringIO()):
        data = syn.synthetic_history(TRUTH, n_days=N_DAYS, seed=seed)
    close = np.array([d["close"] for d in data["daily"]])
    opens = np.array([d["open"] for d in data["daily"]])
    r = np.log(close / opens)                               # no overnight move in the simulation
    rv = np.asarray(hist.realised_variance(data["bars5m"])["rv"]) / 252.0
    iv = np.asarray(data["true_daily_integrated_variance"]) / 252.0
    return r, rv, iv


def add_jumps(r, rv, iv, seed=5):
    rng = np.random.default_rng(seed)
    n = rng.poisson(JUMPS["rate"] / 252.0, len(r))
    J = np.array([rng.normal(JUMPS["mean"], JUMPS["sd"], k).sum() for k in n])
    return r + J, rv + J * J, iv + J * J, int(n.sum())


def ewma(r, lam=0.94):
    s2 = np.empty(len(r))
    s2[0] = float(np.var(r[:250]))
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return s2


def har_forecasts(rv, n_train):
    """HAR-RV: RV_t on (RV_{t-1}, mean RV_{t-5..t-1}, mean RV_{t-22..t-1}), OLS on the training days."""
    rv = np.asarray(rv, float)
    c = np.concatenate([[0.0], np.cumsum(rv)])
    mean_back = lambda t, k: (c[t] - c[t - k]) / k
    X = np.array([[1.0, rv[t - 1], mean_back(t, 5), mean_back(t, 22)] for t in range(22, len(rv))])
    y = rv[22:]
    b = np.linalg.lstsq(X[:n_train - 22], y[:n_train - 22], rcond=None)[0]
    f = np.full(len(rv), np.nan)
    f[22:] = np.maximum(X @ b, 1e-12)
    return f, b


def dm_stat(loss_a, loss_b, lags=5):
    """Diebold-Mariano: mean(a - b) / its Newey-West SE; negative means a is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * float(dc[k:] @ dc[:-k]) / n
    return float(d.mean() / math.sqrt(lrv / n))


def qlike_series(target, s2):
    ratio = np.maximum(np.asarray(target) / np.asarray(s2), 1e-300)
    return ratio - np.log(ratio) - 1.0


def run_scenario(label, r, rv, iv):
    t0 = time.perf_counter()
    targets = {"true IV": iv, "RV (5 min)": rv}
    rows = eg.scan(r, split=N_TRAIN, targets=targets, se=True, n_random=1)
    say(f"\n[{label}] scan on {N_TRAIN} days, scored on {len(r) - N_TRAIN} ({time.perf_counter() - t0:.0f}s)")
    for z in rows:
        say(f"  {z['spec']:22s} loglik {z['loglik']:9.1f}  k {z['k']}  BIC {z['bic']:9.1f}  persistence "
            f"{z['persistence']:.4f}  nu {z['nu']:6.1f}  QLIKE vs IV {z['oos']['true IV']:.4f}, vs RV {z['oos']['RV (5 min)']:.4f}")
    best = rows[0]
    say(f"  BIC picks {best['spec']}: " + ", ".join(
        f"{n} {v:.4g} ({s:.2g})" for n, v, s in zip(best['fit']['spec'].param_names, best['fit']['theta'], best['fit']['se'])))

    # benchmarks and Diebold-Mariano against the best returns-only model out of sample
    test = slice(N_TRAIN, len(r))
    paths = {z["spec"]: eg.variance_path(z["fit"]["theta"], r, z["fit"]["spec"])[:len(r)] for z in rows}
    paths["EWMA (0.94)"] = ewma(r)
    har, b = har_forecasts(rv, N_TRAIN)
    paths["HAR-RV (uses RV)"] = har
    table = {}
    for name, s2 in paths.items():
        table[name] = {tn: float(np.mean(qlike_series(tg[test], s2[test]))) for tn, tg in targets.items()}
    ret_only = [n for n in table if n != "HAR-RV (uses RV)"]
    best_oos = min(ret_only, key=lambda n: table[n]["true IV"])
    dm = {n: {tn: dm_stat(qlike_series(tg[test], paths[n][test]), qlike_series(tg[test], paths[best_oos][test]))
              for tn, tg in targets.items()} for n in paths if n != best_oos}
    say(f"  out of sample, QLIKE (lower is better; DM > 0: worse than {best_oos}, the best returns-only model):")
    for n in sorted(table, key=lambda n: table[n]["true IV"]):
        d = dm.get(n)
        say(f"    {n:22s} vs IV {table[n]['true IV']:.4f}" + (f" (DM {d['true IV']:+.1f})" if d else " (reference)")
            + f"   vs RV {table[n]['RV (5 min)']:.4f}" + (f" (DM {d['RV (5 min)']:+.1f})" if d else ""))
    gap = table[best_oos]["true IV"] / table["HAR-RV (uses RV)"]["true IV"]
    say(f"  the best returns-only model's QLIKE is {gap:.2f}x HAR-RV's against the true variance")
    return {"rows": [{k: v for k, v in z.items() if k != "fit"} | {"theta": z["fit"]["theta"].tolist(),
                                                                    "se": z["fit"]["se"].tolist(),
                                                                    "params": list(z["fit"]["spec"].param_names)}
                     for z in rows],
            "bic_pick": best["spec"], "oos_qlike": table, "best_returns_only_oos": best_oos, "dm": dm,
            "har_coefficients": b.tolist(), "qlike_ratio_best_returns_only_to_har": gap}


def control(seed=33):
    spec = eg.Spec("nelson", 1, 1, "t")
    truth = np.array([-0.25, 0.12, -0.07, 0.975, 7.0])
    r, s2 = eg.simulate(truth, spec, N_DAYS, np.random.default_rng(seed))
    rows = eg.scan(r, specs=(spec, eg.Spec("nelson", 1, 1, "normal"), eg.Spec("garch", 1, 1, "normal"),
                             eg.Spec("garch", 1, 1, "t"), eg.Spec("beta-t")),
                   split=N_TRAIN, targets={"true variance": s2}, se=False, n_random=1)
    say(f"[control: EGARCH(1,1)-t data, nu = 7] BIC and out-of-sample QLIKE against the true conditional variance:")
    for z in rows:
        say(f"  {z['spec']:22s} BIC {z['bic']:9.1f}  nu {z['nu']:6.1f}  QLIKE {z['oos']['true variance']:.4f}")
    best = min(rows, key=lambda z: z["oos"]["true variance"])
    say(f"  out of sample the best is {best['spec']}")
    return {"rows": [{k: v for k, v in z.items() if k != "fit"} for z in rows], "best_oos": best["spec"],
            "bic_pick": rows[0]["spec"]}


def main():
    t0 = time.perf_counter()
    out_control = control()
    r, rv, iv = simulate()
    say(f"simulated {N_DAYS} days of rough Heston (H = {TRUTH.H}, xi = {TRUTH.xi}, rho = {TRUTH.rho}) in "
        f"{time.perf_counter() - t0:.0f}s: daily vol {math.sqrt(252 * r.var()):.3f}, corr(RV, IV) "
        f"{np.corrcoef(rv, iv)[0, 1]:.3f}, return kurtosis {float(((r - r.mean()) ** 4).mean() / r.var() ** 2):.2f}")
    out = {"truth": {n: getattr(TRUTH, n) for n in rh.RoughHestonParams.NAMES}, "n_days": N_DAYS, "n_train": N_TRAIN,
           "jump_model": JUMPS}
    out["control"] = out_control
    out["rough"] = run_scenario("rough Heston", r, rv, iv)
    rj, rvj, ivj, n_j = add_jumps(r, rv, iv)
    say(f"\nadded {n_j} jumps over {N_DAYS} days: return kurtosis {float(((rj - rj.mean()) ** 4).mean() / rj.var() ** 2):.2f}")
    out["jumps"] = run_scenario("rough Heston + jumps", rj, rvj, ivj)
    out["jumps"]["n_jumps"] = n_j
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    say(f"\n[done in {time.perf_counter() - t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
