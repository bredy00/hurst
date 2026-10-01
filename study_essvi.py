"""
eSSVI against raw SVI, graded where the true surface is known (Session O, 29 September 2026).

`fit/essvi.py` builds a surface that cannot be arbitraged; `fit/svi.py` fits each expiry alone.
Whether the constraint costs fit, and whether raw SVI actually produces arbitrage on surfaces
like this project's, are questions with measurable answers:

  A  a planted eSSVI surface is recovered -- the fitter finds the optimum when one exists;
  B  the synthetic chain's own surface, rough Heston at H = 0.10 (the pipeline's truth), with
     exact vols: fit error in vol points, butterfly (Durrleman's g) and calendar (slice
     crossings) on the quotes' range and far outside it;
  C  the same with 0.3 vol points of quote noise, 20 draws: arbitrage frequency of each method,
     and the error against the TRUE surface, where a constraint can pay for itself by refusing
     to fit noise.

    python study_essvi.py            (~10 minutes)
    -> captures/essvi.json, captures/essvi.log
"""

import json
import math
import pathlib
import sys
import time

import numpy as np

import fit.essvi as es
import fit.svi as svi
import models.rough_heston as rh
import sources.synthetic as syn

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "essvi.json"
TRUTH = rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10)
TAUS_DAYS = (2, 7, 14, 30, 90)            # the synthetic chain's quick-mode targets
Z_SPAN = 2.5
N_STRIKES = 11


def say(msg):
    print(msg, flush=True)


def planted():
    taus = [d / 365 for d in TAUS_DAYS]
    fits, prev, sl = [], None, []
    for t in taus:
        th = 0.04 * t + 0.002 * math.sqrt(t)
        f = dict(theta=th, rho=-0.7 + 0.2 * t, psi=0.35 * math.sqrt(th))
        assert es.feasible(prev, f["theta"], f["rho"], f["psi"])
        fits.append(f)
        prev = f
        k = np.linspace(-3, 3, 13) * math.sqrt(th)
        sl.append((t, k, np.sqrt(es.essvi_w(k, **f) / t)))
    got = es.fit_surface(sl)
    err = max(max(abs(g[p] - f[p]) / abs(f[p]) for p in ("theta", "rho", "psi")) for g, f in zip(got, fits))
    return {"worst_relative_error": err, "worst_rmse_vp": max(g["rmse_vp"] for g in got), "audit": es.audit(got)}


def model_surface():
    iv_fn = syn.model_iv_fn(TRUTH)
    out = []
    for d in TAUS_DAYS:
        t = d / 365
        atm = iv_fn(0.0, t)
        k = np.linspace(-Z_SPAN, Z_SPAN, N_STRIKES) * atm * math.sqrt(t)
        iv = np.array([iv_fn(x, t) for x in k])
        k_dense = np.linspace(k.min(), k.max(), 81)
        out.append({"tau": t, "k": k, "iv": iv, "k_dense": k_dense, "iv_dense": np.array([iv_fn(x, t) for x in k_dense])})
    return out


def raw_fits(surface, iv_key="iv"):
    return [svi.fit_slice(s["k"], s[iv_key] ** 2 * s["tau"], tau=s["tau"]) for s in surface]


def raw_audit(fits, surface, k_wide=1.0):
    return es.audit_raw_svi(fits, [s["tau"] for s in surface], k_wide)


def err_vs_truth(surface, w_fn):
    """RMS error in vol points against the TRUE surface on a dense grid inside the quotes."""
    e = []
    for i, s in enumerate(surface):
        w = w_fn(i, s["k_dense"])
        e.append(np.sqrt(np.maximum(w, 0) / s["tau"]) - s["iv_dense"])
    return float(100 * np.sqrt(np.mean(np.concatenate(e) ** 2)))


def compare(surface, iv_key):
    t0 = time.perf_counter()
    ef = es.fit_surface([(s["tau"], s["k"], s[iv_key]) for s in surface])
    t_es = time.perf_counter() - t0
    rf = raw_fits(surface, iv_key)
    ea, ra = es.audit(ef), raw_audit(rf, surface)
    e_es = err_vs_truth(surface, lambda i, k: es.essvi_w(k, ef[i]["theta"], ef[i]["rho"], ef[i]["psi"]))
    e_raw = err_vs_truth(surface, lambda i, k: svi.raw_svi(k, **{p: rf[i][p] for p in svi.PARAM_NAMES}))
    return {"essvi": {"fits": ef, "audit": ea, "err_vs_truth_vp": e_es, "seconds": t_es,
                      "rmse_vp": [f["rmse_vp"] for f in ef]},
            "raw_svi": {"audit": ra, "err_vs_truth_vp": e_raw,
                        "rmse_vp": [es.raw_svi_rmse_vp(f, s["k"], s[iv_key], s["tau"]) for f, s in zip(rf, surface)]}}


def main(argv):
    n_noise = int(argv[0]) if argv else 20
    res = {}
    t0 = time.perf_counter()
    res["A"] = planted()
    say(f"A  a planted eSSVI surface recovered: worst relative parameter error {res['A']['worst_relative_error']:.1e}, "
        f"worst fit {res['A']['worst_rmse_vp']:.1e} vp; audit {res['A']['audit']}")

    surf = model_surface()
    b = compare(surf, "iv")
    res["B"] = b
    say(f"B  rough Heston at H = 0.10, exact vols, {len(TAUS_DAYS)} expiries x {N_STRIKES} strikes (+/- {Z_SPAN} sd):")
    say(f"     eSSVI     error vs the true surface {b['essvi']['err_vs_truth_vp']:.3f} vp; per slice "
        f"{np.round(b['essvi']['rmse_vp'], 3).tolist()}; Durrleman min {b['essvi']['audit']['durrleman_min']:+.3f}, "
        f"worst crossing {b['essvi']['audit']['worst_crossing']:+.2e} ({b['essvi']['seconds']:.0f}s)")
    say(f"     raw SVI   error vs the true surface {b['raw_svi']['err_vs_truth_vp']:.3f} vp; Durrleman min "
        f"{b['raw_svi']['audit']['durrleman_min']:+.3f}, worst crossing {b['raw_svi']['audit']['worst_crossing']:+.2e}")

    rng = np.random.default_rng(7)
    rows = []
    for draw in range(n_noise):
        for s in surf:
            s["iv_noisy"] = s["iv"] + 0.003 * rng.standard_normal(len(s["iv"]))
        c = compare(surf, "iv_noisy")
        rows.append({"essvi_err": c["essvi"]["err_vs_truth_vp"], "raw_err": c["raw_svi"]["err_vs_truth_vp"],
                     "essvi_audit": c["essvi"]["audit"], "raw_audit": c["raw_svi"]["audit"]})
        say(f"   C draw {draw:2d}: error vs truth eSSVI {rows[-1]['essvi_err']:.3f} vp, raw SVI {rows[-1]['raw_err']:.3f} vp; "
            f"raw SVI g min {rows[-1]['raw_audit']['durrleman_min']:+.3f}, crossing {rows[-1]['raw_audit']['worst_crossing']:+.1e}")
    ee = np.array([r["essvi_err"] for r in rows])
    er = np.array([r["raw_err"] for r in rows])
    arb_raw = [r["raw_audit"]["durrleman_min"] < -1e-10 or r["raw_audit"]["worst_crossing"] < -1e-12 for r in rows]
    arb_es = [r["essvi_audit"]["durrleman_min"] < -1e-10 or r["essvi_audit"]["worst_crossing"] < -1e-12 for r in rows]
    bfly_raw = [r["raw_audit"]["durrleman_min"] < -1e-10 for r in rows]
    cal_raw = [r["raw_audit"]["worst_crossing"] < -1e-12 for r in rows]
    res["C"] = {"rows": rows, "essvi_err_mean": float(ee.mean()), "raw_err_mean": float(er.mean()),
                "essvi_better_share": float(np.mean(ee < er)), "raw_arbitrage_share": float(np.mean(arb_raw)),
                "raw_butterfly_share": float(np.mean(bfly_raw)), "raw_calendar_share": float(np.mean(cal_raw)),
                "essvi_arbitrage_share": float(np.mean(arb_es))}
    c = res["C"]
    say(f"C  {n_noise} draws of 0.3 vp quote noise: error vs the TRUE surface eSSVI {c['essvi_err_mean']:.3f} vp, "
        f"raw SVI {c['raw_err_mean']:.3f} vp (eSSVI closer in {100 * c['essvi_better_share']:.0f}% of draws)")
    say(f"   arbitrage on |k| <= 1: raw SVI {100 * c['raw_arbitrage_share']:.0f}% of surfaces (butterfly "
        f"{100 * c['raw_butterfly_share']:.0f}%, calendar {100 * c['raw_calendar_share']:.0f}%); eSSVI "
        f"{100 * c['essvi_arbitrage_share']:.0f}%")
    res["seconds"] = time.perf_counter() - t0
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)),
                   encoding="utf-8")
    say(f"\n[done in {res['seconds']:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
