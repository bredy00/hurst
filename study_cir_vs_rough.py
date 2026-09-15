"""
A plain CIR filter against the filter built for the rough model (Session G).

Session F found that a CIR filter fitted by maximum likelihood to rough variance
forecasts it at 5 days within -0.6% to +2.8% of the true 24-factor lifted model,
by pushing kappa to 43-54. The 2026-09-15 review asked three things about that:

  1. What does each filter COST? (per step, per likelihood, per fit, against N)
  2. Is "3% at 5 days" still good once volatility drag is accounted for -- the
     -1/2 int V dt in every log return -- and its stochastic part under Heston-type
     dynamics, Var[1/2 int V dt]?
  3. Is kappa ~ 50 robust? Try 30, 35, 40, 45, 50, 55, 60, 70.

Forecast targets, from every filtered state, at 1, 5, 10 and 21 trading days:

  variance     V_{t+h}
  volatility   sqrt(V_{t+h}). Forecast as sqrt(E V) ("naive") and Jensen-corrected,
               E sqrt(V) for a gamma law with the model's conditional mean m and
               variance s^2 of V_{t+h} (state uncertainty included):
               sqrt(s^2/m) Gamma(m^2/s^2 + 1/2) / Gamma(m^2/s^2). The delta-method
               version sqrt(m) - s^2 / (8 m^1.5) was tried first and is useless at one
               day, where rough variance's conditional SD exceeds its mean. The two
               models disagree about s^2 even when they agree about m, so this is where
               they can split.
  drag         D_h = 1/2 sum_{j=1..h} V_{t+j} dt: the expected volatility drag on the
               log return over the horizon.
  drag risk    the model's SD of D_h, from the full cross-covariance of the path
               (Cov(V_i, V_j) = H' A^(j-i) C_i H), against the realised dispersion
               of the forecast errors, and the coverage of a 90% interval.

All conditional moments are exact for each model's own dynamics (affine
recursions; Q evaluated along the mean path), vectorised across origins.
Data: the positivity-preserving simulator, true H = 0.12, kappa = 3.

    python study_cir_vs_rough.py          (~3 minutes) -> captures/cir_vs_rough.png, .json
"""

import json
import math
import pathlib
import time

import numpy as np
from scipy.special import gammaln

import filters.kalman as kf

ROOT = pathlib.Path(__file__).parent
DT = 1.0 / 252
TRUE = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.03 ** 2)
N_DAYS, BURN = 2000, 300
HORIZONS = (1, 5, 10, 21)
KAPPAS = (30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0)
SEEDS = (10, 11, 12)


# ------------------------------------------------------------------ costs
def per_step_cost(model, p, y, reps=3, policy=None):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        kf.kalman_filter(model, p, y, policy=policy)
        best = min(best, time.perf_counter() - t0)
    return best / len(y), best


def costs(y):
    out = {"dims": [], "exact": [], "euler": []}
    cm = kf.CIRModel(DT)
    pc = dict(kappa=50.0, theta=0.04, xi=1.0, R=TRUE["R"])
    out["cir_scalar"] = per_step_cost(cm, pc, y)[0]
    out["cir_matrix"] = per_step_cost(cm, pc, y, policy=kf.Robust(threshold=1e12))[0]
    for N in (4, 8, 12, 16, 24, 32, 48):
        out["dims"].append(N)
        out["exact"].append(per_step_cost(kf.LiftedRoughModel(DT, H=0.12, v0=0.04, N=N), TRUE, y, reps=2)[0])
        out["euler"].append(per_step_cost(kf.LiftedRoughModel(DT, H=0.12, v0=0.04, N=N, discretisation="euler"),
                                          TRUE, y, reps=2)[0])
    t0 = time.perf_counter()
    kf.fit_mle(cm, y, dict(kappa=3.0, theta=float(np.mean(y)), xi=0.3, R=TRUE["R"]))
    out["fit_cir_all4"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    kf.fit_mle(cm, y, dict(kappa=3.0, theta=0.04, xi=0.3, R=TRUE["R"]), names=("kappa",))
    out["fit_cir_kappa"] = time.perf_counter() - t0
    rm = kf.LiftedRoughModel(DT, H=0.12, v0=0.04)
    t0 = time.perf_counter()
    kf.fit_mle(rm, y, dict(TRUE, kappa=1.0), names=("kappa",))
    out["fit_rough_kappa"] = time.perf_counter() - t0
    out["lik_cir"] = per_step_cost(cm, pc, y)[1]
    out["lik_rough_exact"] = per_step_cost(rm, TRUE, y, reps=2)[1]
    return out


# ------------------------------------------------------------------ forecasts
def batched_Q(model, p, X):
    """Process covariance at many states X (n, d) at once."""
    if isinstance(model, kf.CIRModel):
        k, th, xi = p["kappa"], p["theta"], p["xi"]
        e = math.exp(-k * model.dt)
        q = np.maximum(X[:, 0], 0.0) * xi * xi * (e - e * e) / k + th * xi * xi * (1.0 - e) ** 2 / (2.0 * k)
        return np.maximum(q, 1e-18)[:, None, None]
    st = model.stepper(p)
    CU = st._covU_const[None] + np.einsum("ijk,nk->nij", st._covU_lin, X)
    ev, vec = np.linalg.eigh(0.5 * (CU + np.transpose(CU, (0, 2, 1))))
    return (vec * np.maximum(ev, 0.0)[:, None, :]) @ np.transpose(vec, (0, 2, 1))


def forecast_moments(model, p, r, origins, hmax):
    """
    For each origin k: mean and variance of V_{k+h} (h = 1..hmax) and of the
    running sum S_h = sum_{j<=h} V_{k+j}, from the filtered (x, P) at k.
    """
    A, b = model.transition(p)
    Hv, c = model.observation(p)
    X = r["x_post"][origins].copy()
    C = r["P_post"][origins].copy()
    n = len(origins)
    mV = np.empty((n, hmax))
    vV = np.empty((n, hmax))
    mS = np.empty((n, hmax))
    vS = np.empty((n, hmax))
    G = []                       # propagated C_i H for the cross-covariances
    run_m = np.zeros(n)
    run_v = np.zeros(n)
    for j in range(hmax):
        Q = batched_Q(model, p, X)
        C = A[None] @ C @ A.T[None] + Q
        X = X @ A.T + b[None]
        mj = X @ Hv + c
        CH = C @ Hv
        vj = CH @ Hv
        G = [g @ A.T for g in G]                         # C_i H -> A^(j-i) C_i H
        cross = sum((g @ Hv) for g in G) if G else 0.0   # sum_i Cov(V_j, V_i), i < j
        G.append(CH)
        run_m = run_m + mj
        run_v = run_v + vj + 2.0 * cross
        mV[:, j], vV[:, j], mS[:, j], vS[:, j] = mj, vj, run_m, run_v
    return mV, vV, mS, vS


def evaluate(V, mV, vV, mS, vS, origins):
    """RMSE, bias and drag-risk calibration per horizon."""
    res = {}
    for h in HORIZONS:
        j = h - 1
        tgt_V = V[origins + h]
        tgt_D = 0.5 * DT * np.array([V[k + 1:k + h + 1].sum() for k in origins])
        m, s2 = mV[:, j], np.maximum(vV[:, j], 0.0)
        mpos = np.maximum(m, 1e-6)
        vol_naive = np.sqrt(mpos)
        shape = mpos * mpos / np.maximum(s2, 1e-300)
        vol_jensen = np.where(s2 > 1e-300, np.sqrt(np.maximum(s2, 1e-300) / mpos)
                              * np.exp(gammaln(shape + 0.5) - gammaln(shape)), np.sqrt(mpos))
        vol_true = np.sqrt(np.maximum(tgt_V, 0.0))
        D_m = 0.5 * DT * mS[:, j]
        D_sd = 0.5 * DT * np.sqrt(np.maximum(vS[:, j], 0.0))
        err_D = tgt_D - D_m
        res[h] = {
            "rmse_var": float(np.sqrt(np.mean((tgt_V - m) ** 2))), "bias_var": float(np.mean(m - tgt_V)),
            "rmse_vol_naive": float(np.sqrt(np.mean((vol_true - vol_naive) ** 2))),
            "bias_vol_naive": float(np.mean(vol_naive - vol_true)),
            "rmse_vol_jensen": float(np.sqrt(np.mean((vol_true - vol_jensen) ** 2))),
            "bias_vol_jensen": float(np.mean(vol_jensen - vol_true)),
            "rmse_drag": float(np.sqrt(np.mean(err_D ** 2))), "bias_drag": float(np.mean(-err_D)),
            "drag_sd_model": float(np.sqrt(np.mean(D_sd ** 2))),
            "drag_coverage90": float(np.mean(np.abs(err_D) <= 1.645 * D_sd)),
            "mean_drag": float(np.mean(tgt_D)),
        }
    return res


def run(verbose=True):
    t_all = time.perf_counter()
    rm = kf.LiftedRoughModel(DT, H=0.12, v0=TRUE["theta"])
    cm = kf.CIRModel(DT)
    hmax = max(HORIZONS)
    per_seed = []
    cost = None
    for s_i, seed in enumerate(SEEDS):
        V, _ = kf.simulate_rough(rm, TRUE, N_DAYS, seed=seed)
        y = V + np.random.default_rng(1000 + seed).normal(0.0, math.sqrt(TRUE["R"]), N_DAYS)
        if s_i == 0:
            t0 = time.perf_counter()
            cost = costs(y)
            if verbose:
                print(f"costs measured ({time.perf_counter()-t0:.0f}s)", flush=True)
        origins = np.arange(BURN, N_DAYS - hmax - 1)
        row = {"seed": seed}
        r = kf.kalman_filter(rm, TRUE, y)
        row["lifted"] = evaluate(V, *forecast_moments(rm, TRUE, r, origins, hmax), origins)
        row["lifted_loglik"] = r["loglik"]
        fit = kf.fit_mle(cm, y, dict(kappa=3.0, theta=float(np.mean(y)), xi=0.3, R=TRUE["R"]))
        pk = fit["params"]
        row["cir_mle"] = {"params": pk, "se": fit["se"], "loglik": fit["loglik"]}
        row["cir_mle"]["eval"] = evaluate(V, *forecast_moments(cm, pk, kf.kalman_filter(cm, pk, y), origins, hmax),
                                          origins)
        row["sweep"] = {}
        for kap in KAPPAS:
            f = kf.fit_mle(cm, y, dict(pk, kappa=kap), names=("theta", "xi", "R"), fixed={"kappa": kap})
            p = f["params"]
            row["sweep"][kap] = {"params": p, "loglik": f["loglik"],
                                 "eval": evaluate(V, *forecast_moments(cm, p, kf.kalman_filter(cm, p, y), origins, hmax),
                                                  origins)}
        per_seed.append(row)
        if verbose:
            L = row["lifted"][5]
            print(f"seed {seed}: CIR MLE kappa {pk['kappa']:.1f} +/- {fit['se']['kappa']:.1f}; 5-day RMSE "
                  f"lifted/CIR: var {L['rmse_var']:.4f}/{row['cir_mle']['eval'][5]['rmse_var']:.4f}, "
                  f"drag {L['rmse_drag']:.2e}/{row['cir_mle']['eval'][5]['rmse_drag']:.2e}  "
                  f"({time.perf_counter()-t_all:.0f}s)", flush=True)
    out = {"costs": cost, "seeds": per_seed, "true": TRUE, "horizons": HORIZONS, "kappas": KAPPAS}
    summarise(out, verbose)
    return out


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def summarise(out, verbose=True):
    """Seed-averaged tables; CIR error relative to the lifted model's, in percent."""
    seeds = out["seeds"]
    keys = ("rmse_var", "rmse_vol_naive", "rmse_vol_jensen", "rmse_drag")
    table = {}
    for h in HORIZONS:
        lif = {k: np.mean([s["lifted"][h][k] for s in seeds]) for k in keys}
        rows = {"CIR MLE": {k: np.mean([s["cir_mle"]["eval"][h][k] for s in seeds]) for k in keys}}
        for kap in KAPPAS:
            rows[f"CIR kappa {kap:g}"] = {k: np.mean([s["sweep"][kap]["eval"][h][k] for s in seeds]) for k in keys}
        table[h] = {"lifted": lif, "rows": {name: {k: pct(v[k], lif[k]) for k in keys} for name, v in rows.items()}}
    prof = {kap: float(np.mean([s["sweep"][kap]["loglik"] - s["cir_mle"]["loglik"] for s in seeds])) for kap in KAPPAS}
    drag_cal = {}
    for name, getter in [("lifted", lambda s, h: s["lifted"][h])] + \
            [(f"CIR kappa {k:g}", (lambda kk: lambda s, h: s["sweep"][kk]["eval"][h])(k)) for k in KAPPAS]:
        drag_cal[name] = {h: {"sd_model_over_rmse": float(np.mean([getter(s, h)["drag_sd_model"] / getter(s, h)["rmse_drag"]
                                                                    for s in seeds])),
                              "coverage90": float(np.mean([getter(s, h)["drag_coverage90"] for s in seeds]))}
                          for h in HORIZONS}
    out["table"], out["profile"], out["drag_calibration"] = table, prof, drag_cal
    if not verbose:
        return
    c = out["costs"]
    print(f"\nCost per filter step: CIR scalar {1e6*c['cir_scalar']:.1f} us, CIR matrix path {1e6*c['cir_matrix']:.1f} us, "
          f"lifted N=24 exact {1e6*c['exact'][c['dims'].index(24)]:.0f} us, Euler {1e6*c['euler'][c['dims'].index(24)]:.0f} us")
    print(f"Likelihood over {N_DAYS} days: CIR {1e3*c['lik_cir']:.1f} ms, lifted exact {1e3*c['lik_rough_exact']:.0f} ms. "
          f"Fit kappa: CIR {c['fit_cir_kappa']:.2f} s, lifted {c['fit_rough_kappa']:.1f} s; CIR all four: {c['fit_cir_all4']:.2f} s")
    print("\nCIR profile log-likelihood, kappa fixed, relative to the CIR MLE (95% CI needs > -1.92): "
          + ", ".join(f"{k:g}: {v:+.1f}" for k, v in prof.items()))
    for h in HORIZONS:
        t = table[h]
        print(f"\n{h}-day: lifted RMSE var {t['lifted']['rmse_var']:.4f}, vol {t['lifted']['rmse_vol_jensen']:.4f}, "
              f"drag {t['lifted']['rmse_drag']:.2e}.  CIR relative to it (+ = worse):")
        for name, v in t["rows"].items():
            print(f"   {name:15s} var {v['rmse_var']:+6.1f}%   vol naive {v['rmse_vol_naive']:+6.1f}%   "
                  f"vol Jensen {v['rmse_vol_jensen']:+6.1f}%   drag {v['rmse_drag']:+6.1f}%")
    print("\nDrag risk: model SD / realised RMSE of the drag forecast (1 = calibrated), and 90% coverage:")
    for name, v in drag_cal.items():
        print(f"   {name:15s} " + "  ".join(f"{h}d {v[h]['sd_model_over_rmse']:.2f} ({100*v[h]['coverage90']:.0f}%)"
                                            for h in HORIZONS))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = out["costs"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    a = ax[0, 0]
    dims = np.array(c["dims"])
    a.loglog(dims, 1e6 * np.array(c["exact"]), "o-", color="#1f77b4", label="lifted rough, exact moments")
    a.loglog(dims, 1e6 * np.array(c["euler"]), "s--", color="#aec7e8", label="lifted rough, Euler (Session F)")
    a.axhline(1e6 * c["cir_scalar"], color="#d62728", label="CIR, scalar path")
    a.axhline(1e6 * c["cir_matrix"], color="#ff9896", ls="--", label="CIR, generic matrix path")
    a.axvline(24, color="0.6", ls=":", lw=1)
    a.set_xlabel("state dimension N (lifted factors)")
    a.set_ylabel("microseconds per filter step")
    a.set_title("(a) cost per step")
    a.legend(fontsize=8)

    a = ax[0, 1]
    labels = ["likelihood\n2000 days", "fit kappa", "fit all 4\n(CIR)"]
    cir = [c["lik_cir"], c["fit_cir_kappa"], c["fit_cir_all4"]]
    rough = [c["lik_rough_exact"], c["fit_rough_kappa"], np.nan]
    xs = np.arange(3)
    a.bar(xs - 0.2, cir, 0.4, color="#d62728", label="CIR")
    a.bar(xs + 0.2, rough, 0.4, color="#1f77b4", label="lifted rough (exact, N=24)")
    for x_, v in zip(xs - 0.2, cir):
        a.text(x_, v, f"{v:.2g}s", ha="center", va="bottom", fontsize=8)
    for x_, v in zip(xs + 0.2, rough):
        if np.isfinite(v):
            a.text(x_, v, f"{v:.2g}s", ha="center", va="bottom", fontsize=8)
    a.set_yscale("log")
    a.set_xticks(xs)
    a.set_xticklabels(labels)
    a.set_ylabel("seconds")
    a.set_title("(b) cost of a likelihood and of a fit")
    a.legend(fontsize=8)

    a = ax[0, 2]
    ks = np.array(KAPPAS)
    prof = np.array([out["profile"][k] for k in KAPPAS])
    a.plot(ks, prof, "o-", color="#d62728")
    a.axhline(-1.92, color="0.4", ls="--", lw=1)
    a.text(ks[0], -1.92, " 95% profile interval", va="bottom", fontsize=8, color="0.3")
    mle = [s["cir_mle"]["params"]["kappa"] for s in out["seeds"]]
    for m in mle:
        a.axvline(m, color="#ff9896", lw=0.8)
    a.set_xlabel("CIR kappa (theta, xi, R refitted)")
    a.set_ylabel("log-likelihood minus the CIR MLE's (3 seeds)")
    a.set_title("(c) is kappa ~ 50 robust? the profile likelihood")

    a = ax[1, 0]
    t5 = out["table"][5]["rows"]
    for key, col, lab in (("rmse_var", "#1f77b4", "variance"), ("rmse_vol_naive", "#ff7f0e", "vol, sqrt(E V)"),
                          ("rmse_vol_jensen", "#2ca02c", "vol, Jensen-corrected"), ("rmse_drag", "#9467bd", "drag 1/2 int V")):
        a.plot(ks, [t5[f"CIR kappa {k:g}"][key] for k in KAPPAS], "o-", color=col, label=lab)
    a.axhline(0.0, color="0.5", lw=0.8)
    a.set_xlabel("CIR kappa")
    a.set_ylabel("CIR 5-day RMSE vs the lifted model (%)")
    a.set_title("(d) 5 days: what the kappa choice costs, by target")
    a.legend(fontsize=8)

    a = ax[1, 1]
    hs = np.array(HORIZONS)
    for key, col, lab in (("rmse_var", "#1f77b4", "variance"), ("rmse_vol_jensen", "#2ca02c", "vol (Jensen)"),
                          ("rmse_drag", "#9467bd", "drag")):
        a.plot(hs, [out["table"][h]["rows"]["CIR MLE"][key] for h in HORIZONS], "o-", color=col, label=lab)
    a.axhline(0.0, color="0.5", lw=0.8)
    a.set_xscale("log")
    a.set_xticks(hs)
    a.set_xticklabels([str(h) for h in hs])
    a.set_xlabel("horizon (trading days)")
    a.set_ylabel("CIR (MLE kappa) RMSE vs lifted (%)")
    a.set_title("(e) by horizon: the drag accumulates what variance forgives")
    a.legend(fontsize=8)

    a = ax[1, 2]
    dc = out["drag_calibration"]
    a.plot(hs, [dc["lifted"][h]["sd_model_over_rmse"] for h in HORIZONS], "o-", color="#1f77b4", lw=2, label="lifted rough")
    for kap, col in ((30.0, "#fcbba1"), (50.0, "#ef3b2c"), (70.0, "#67000d")):
        a.plot(hs, [dc[f"CIR kappa {kap:g}"][h]["sd_model_over_rmse"] for h in HORIZONS], "s--", color=col,
               label=f"CIR kappa {kap:g}")
    a.axhline(1.0, color="0.5", lw=0.8)
    a.set_xscale("log")
    a.set_xticks(hs)
    a.set_xticklabels([str(h) for h in hs])
    a.set_xlabel("horizon (trading days)")
    a.set_ylabel("model SD of the drag / realised RMSE")
    a.set_title("(f) stochastic drag: does the model know its own uncertainty?")
    a.legend(fontsize=8)
    fig.suptitle("Plain CIR filter vs the lifted rough filter: cost, forecasts, volatility drag (true H = 0.12, kappa = 3)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "captures" / "cir_vs_rough.png", dpi=125)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    res = run()
    plot(res)
    (ROOT / "captures" / "cir_vs_rough.json").write_text(json.dumps(_jsonable(res), indent=1), encoding="utf-8")
