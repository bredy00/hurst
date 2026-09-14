"""
Which calibration weights let an option surface pin H? (Session G)

For each surface design, noise model and weighting scheme, at the true
parameters, with J = d(model iv)/d(params) and the scheme's residual map M
(residuals = M (model_iv - market_iv), anchors included) and an optional
Gaussian prior on H of standard deviation s_H:

  A = J'M'MJ + e_H e_H' / s_H^2
  covariance   A^-1 (J'M'M S M'MJ + e_H e_H' / s_H^2) A^-1       S = true quote noise
  bias         A^-1 J'M'M delta                                   first order, for a
               market that departs from the model by delta (prior centred on the truth)

The misspecification delta is what a real surface does to a model that is right
at the short end: a term-structure hump around six months (+1 vol point) and
extra smile curvature beyond 90 days (+1.5 vp at the band edge). H should come
from the short-end skew; the question is which weights let the long end
overrule it.

    python study_h_weighting.py            (~70 s: two Jacobians, cached in captures/)
"""

import math
import pathlib
import time

import numpy as np

import volsurf_core as vc
import study_h_identifiability as study
import calibrate.weights as wt
import models.rough_heston as rh

np.seterr(all="ignore")
NAMES = rh.RoughHestonParams.NAMES
jH, jXI = NAMES.index("H"), NAMES.index("xi")
CACHE = pathlib.Path("captures/h_weighting_jacobians.npz")

DESIGNS = {
    "E1 test surface, 4 expiries x 5": ((2, 14, 90, 365), 5),
    "recorder surface, 10 expiries x 9": ((1, 2, 3, 7, 14, 30, 60, 90, 180, 365), 9),
}
SCHEMES = ("vega2", "inverse_variance", "hybrid", "hybrid x25", "short-end taper",
           "inverse_variance + H prior 0.03", "hybrid + H prior 0.03")


def misspecification(tau, k):
    """Wrong at the LONG end: a six-month term-structure hump and extra long-dated curvature."""
    band = 3.0 * math.sqrt(study.TRUE.v0) * np.sqrt(tau)
    hump = 0.010 * np.exp(-np.log(tau / 0.5) ** 2 / (2 * 0.6 ** 2))
    curv = 0.015 * (k / band) ** 2 * (tau >= 90 / 365)
    return hump + curv


def short_end_misspecification(tau, k):
    """Wrong at the SHORT end: 1-3 day wings 2 vp rich, as overnight/event premia make them."""
    band = 3.0 * math.sqrt(study.TRUE.v0) * np.sqrt(tau)
    return 0.02 * np.abs(k / band) * (tau <= 3 / 365)


def analyse(J, M, noise_var, delta, prior_sd=None, delta2=None):
    W = M.T @ M
    P = np.zeros((J.shape[1], J.shape[1]))
    if prior_sd:
        P[jH, jH] = 1.0 / prior_sd ** 2
    A = J.T @ W @ J + P
    Ai = np.linalg.inv(A)
    B = J.T @ W @ np.diag(noise_var) @ W @ J + P
    cov = Ai @ B @ Ai
    se = np.sqrt(np.diag(cov))
    bias = Ai @ (J.T @ (W @ delta))
    bias2 = Ai @ (J.T @ (W @ delta2)) if delta2 is not None else np.full_like(bias, np.nan)
    return {"se_H": se[jH], "corr_H_xi": cov[jH, jXI] / (se[jH] * se[jXI]),
            "bias_H": bias[jH], "bias_H_short": bias2[jH], "se": se, "bias": bias}


def jacobians():
    """(rows, iv, J) per design, cached: the Jacobian is the only slow part."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item()
    data = {}
    for dname, (days, n_k) in DESIGNS.items():
        rows = study.quotes(days, n_k)
        iv, J = study.jacobian(rows)
        data[dname] = (rows, iv, J)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, data=np.array(data, dtype=object))
    return data


def scheme_residual_map(scheme, tau, k, iv, vega, sd, n_per):
    """(M, prior_sd, effective quotes, n_anchors) for one scheme."""
    prior = 0.03 if "H prior" in scheme else None
    if scheme == "vega2":
        w = wt.vega2_weights(vega)
        return wt.residual_map(len(tau), w), prior, wt.effective_quotes(w), 0
    w = wt.inverse_variance_weights(sd, np.ones_like(sd), wt.IV_FLOOR)     # 1/(sd^2 + floor^2)
    anchors = []
    if scheme.startswith("hybrid"):
        mult = 25 if "x25" in scheme else 1
        w = wt.balance_by_expiry(tau, w)
        # these designs put 5-9 strikes on +/-3 sigma, so the anchor's local fit
        # needs a +/-3.5 sigma window (a real $1-strike chain does not)
        anchors = wt.build_anchors(tau, k, iv, np.sqrt(sd ** 2 + wt.IV_FLOOR ** 2),
                                   window=3.5, weight=mult * n_per)
    if scheme == "short-end taper":
        w = w * wt.maturity_taper(tau)
    return wt.residual_map(len(tau), w, anchors), prior, wt.effective_quotes(w), len(anchors)


def run(verbose=True):
    t0 = time.perf_counter()
    out = {}
    for dname, (rows, iv, J) in jacobians().items():
        days = DESIGNS[dname][0]
        tau = np.array([t for t, _ in rows])
        k = np.array([kk for _, kk in rows])
        vega = np.array([float(vc.bs_vega(1.0, math.exp(kk), v, t)) for (t, kk), v in zip(rows, iv)])
        price = np.array([float(vc.bs_price(1.0, math.exp(kk), v, t, 1.0, vc.otm_right(math.exp(kk), 1.0)))
                          for (t, kk), v in zip(rows, iv)])
        hs = wt.realistic_half_spread(price)
        delta = misspecification(tau, k)
        delta2 = short_end_misspecification(tau, k)
        if verbose:
            print(f"\n{dname}: {len(rows)} quotes")
        for noise_name, sd in (("homoscedastic 0.5 vp noise", np.full(len(rows), 0.005)),
                               ("spread-based noise (tick + 2.5% of value)", wt.iv_noise_sd(hs, vega))):
            if verbose:
                print(f"  {noise_name}: median {100*np.median(sd):.2f} vp")
            for scheme in SCHEMES:
                M, prior, eff, na = scheme_residual_map(scheme, tau, k, iv, vega, sd, len(rows) // len(days))
                a = analyse(J, M, sd ** 2, delta, prior, delta2)
                a.update(effective_quotes=eff, n_anchors=na)
                out[(dname, noise_name, scheme)] = a
                if verbose:
                    print(f"    {scheme:31s} SE(H) {a['se_H']:.4f}  corr(H,xi) {a['corr_H_xi']:+.3f}  "
                          f"long-end wrong: bias {a['bias_H']:+.4f} RMS {math.hypot(a['se_H'], a['bias_H']):.4f}  "
                          f"short-end wrong: bias {a['bias_H_short']:+.4f} RMS {math.hypot(a['se_H'], a['bias_H_short']):.4f}  "
                          f"eff. quotes {eff:.1f}  anchors {na}")
    if verbose:
        print(f"\n({time.perf_counter()-t0:.0f}s)")
    return out


if __name__ == "__main__":
    run()
