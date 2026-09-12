"""
Is H identified by an option surface -- and by WHICH surface?

A Session E calibration under 0.5 vol point quote noise returned H = 0.50,
0.25 and 0.12 on three seeds of the same 20-quote surface, and in all three the
weighted cost at the fitted parameters was LOWER than at the truth. So the
optimiser did not fail: the data did not contain H. This study says why, with
the Gauss-Newton covariance of the parameters at the truth, for several designs.

    cov = (J' W J)^-1 (J' W S W J) (J' W J)^-1        (sandwich)

with J = d(model iv)/d(params) in natural units, W the fit weights and S the
covariance of the injected noise. When W = S^-1 (the weights say exactly how
noisy each quote is) this collapses to (J' W J)^-1, the Cramer-Rao bound.

Designs:
  A  the E1 test surface: 2, 14, 90, 365 d x 5 strikes; vega^2 weights from a
     0.5%-of-forward half-spread; homoscedastic 0.5 vp noise.
  B  same quotes, noise CONSISTENT with the weights (sigma_i = half_spread/vega_i)
     at a realistic half-spread of 1e-4 of the forward.
  C  same quotes, unweighted fit, homoscedastic 0.5 vp noise.
  D  a real short end: 1, 2, 3, 7, 14, 30, 90, 365 d x 7 strikes, unweighted,
     homoscedastic 0.5 vp noise.
  E  design D with consistent weights and noise as in B.

    python study_h_identifiability.py
"""

import math
import time

import numpy as np

import volsurf_core as vc
import pricing.fourier as fo
import models.rough_heston as rh

np.seterr(all="ignore")
TRUE = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
NAMES = rh.RoughHestonParams.NAMES


def quotes(days, n_k):
    rows = []
    for d in days:
        tau = d / 365.0
        band = 3.0 * math.sqrt(TRUE.v0) * math.sqrt(tau)
        for k in np.linspace(-band, band, n_k):
            rows.append((tau, float(k)))
    return rows


def model_iv_vector(p, rows):
    out = np.empty(len(rows))
    taus = sorted({t for t, _ in rows})
    for tau in taus:
        idx = [i for i, (t, _) in enumerate(rows) if t == tau]
        ks = np.array([rows[i][1] for i in idx])
        c, info = rh.call_prices(ks, tau, p, tol=1e-10, steps_mult=2.0)
        out[idx] = fo.implied_vols_from_calls(c, ks, tau)
    return out


def jacobian(rows, rel=1e-3):
    """Central differences in natural units; H and rho by absolute steps."""
    base = model_iv_vector(TRUE, rows)
    J = np.empty((len(rows), len(NAMES)))
    for j, n in enumerate(NAMES):
        v = getattr(TRUE, n)
        h = 1e-3 if n in ("rho", "H") else rel * abs(v)
        up = {m: getattr(TRUE, m) for m in NAMES}
        dn = dict(up)
        up[n] = v + h
        dn[n] = v - h
        J[:, j] = (model_iv_vector(rh.RoughHestonParams(**up), rows)
                   - model_iv_vector(rh.RoughHestonParams(**dn), rows)) / (2 * h)
    return base, J


def covariance(J, w, noise_var):
    A = J.T @ (w[:, None] * J)
    Ainv = np.linalg.inv(A)
    B = J.T @ ((w * noise_var * w)[:, None] * J)
    return Ainv @ B @ Ainv


def report(label, J, w, noise_var):
    cov = covariance(J, w, noise_var)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    jH = NAMES.index("H")
    top = sorted(((abs(corr[jH, j]), NAMES[j]) for j in range(len(NAMES)) if j != jH), reverse=True)[:2]
    eff = float((w.sum() ** 2) / (w * w).sum())
    print(f"  {label:58} SE(H) {se[jH]:7.4f}   SE(rho) {se[NAMES.index('rho')]:.4f}   "
          f"SE(v0) {se[NAMES.index('v0')]:.5f}   |corr(H, .)| max: "
          + ", ".join(f"{n} {c:.2f}" for c, n in top) + f"   effective quotes {eff:.1f}/{len(w)}")
    return se[jH]


def main():
    t0 = time.perf_counter()
    out = {}
    for name, days, n_k in (("E1 surface (4 x 5)", (2, 14, 90, 365), 5),
                            ("short-end surface (8 x 7)", (1, 2, 3, 7, 14, 30, 90, 365), 7)):
        rows = quotes(days, n_k)
        iv, J = jacobian(rows)
        vega = np.array([float(vc.bs_vega(1.0, math.exp(k), v, t)) for (t, k), v in zip(rows, iv)])
        print(f"\n{name}: {len(rows)} quotes, Jacobian in {time.perf_counter()-t0:.0f}s")
        homo = np.full(len(rows), 0.005 ** 2)
        w_spread = (vega / 0.005) ** 2
        w_real = (vega / 1e-4) ** 2
        out[(name, "A")] = report("vega^2 weights, homoscedastic 0.5 vp noise", J, w_spread, homo)
        out[(name, "B")] = report("vega^2 weights, noise consistent (1e-4 F half-spread)", J,
                                  w_real, 1.0 / w_real)
        out[(name, "C")] = report("unweighted, homoscedastic 0.5 vp noise", J,
                                  np.ones(len(rows)), homo)
    print(f"\n({time.perf_counter()-t0:.0f}s)")
    return out


if __name__ == "__main__":
    main()
