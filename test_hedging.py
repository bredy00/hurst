"""
Session J -- discrete hedging on the simulated rough-Heston paths (models/hedging.py).

  paths        the forward variance to maturity is the exact integrated-variance moment;
               the simulated forward is a martingale
  BS limit     with nearly deterministic variance the risk-minimising hedge IS the
               Black-Scholes delta, and the discrete-hedging error follows the
               Bertsimas-Kogan-Lo law, sd proportional to sqrt(dt)
  rough vol    Hedged Monte Carlo prices the call at the cf price; no hedging rule loses
               money on average; the risk-minimising rule beats Black-Scholes delta out of
               sample; and a floor remains that rebalancing does not remove
  var swap     the variance swap is a martingale that pays the realised variance, its
               hedge ratio is the option's sensitivity to the forward variance, and adding
               it (which completes the lifted model) cuts the floor by more than half

    python test_hedging.py
"""

import math
import sys
import time

import numpy as np

import models.hedging as hd
import models.rough_heston as rh
import pricing.fourier as fo

PASS, FAIL = [], []
T, N, K = 1.0 / 12, 128, 1.0


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_paths():
    print("\nPaths: forward variance and the martingale")
    P = rh.RoughHestonParams(0.03, 2.0, 0.05, 0.3, -0.7, 0.12)
    w, x = rh.lift_nodes(P.H)
    worst = 0.0
    for tau in (1 / 365, 0.1, 0.5):
        st = rh.LiftedAffineStep(w, x, P.v0, P.kappa, P.theta, P.xi, tau, P.rho)
        A, B = hd.forward_variance_coefficients(st, [tau])
        worst = max(worst, abs(A[0] - st.EI_const) / st.EI_const,
                    float(np.max(np.abs(B[0] - st.EI_lin))) / float(np.max(np.abs(st.EI_lin))))
    check("forward variance coefficients = the step's integrated-variance moments at h = tau", worst < 1e-12,
          f"worst relative {worst:.1e}")
    paths = hd.simulate_paths(P, T, N, 6000, seed=3)
    ST = paths["S"][-1]
    z = (ST.mean() - 1.0) / (ST.std() / math.sqrt(len(ST)))
    check("the simulated forward is a martingale: E[S_T] = 1 within 3 SE", abs(z) < 3.0, f"z = {z:+.2f}")
    check("forward variance at t = 0 is the exact E[int_0^T V dt]", abs(paths["FV"][0, 0] - A_exact(P)) < 1e-12,
          f"{paths['FV'][0, 0]:.10f}")


def A_exact(P):
    w, x = rh.lift_nodes(P.H)
    st = rh.LiftedAffineStep(w, x, P.v0, P.kappa, P.theta, P.xi, T, P.rho)
    return st.EI_const


def test_bs_limit():
    print("\nNearly deterministic variance (xi = 0.01): Black-Scholes is the right model")
    P = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.01, -0.7, 0.12)
    tr = hd.simulate_paths(P, T, N, 6000, seed=5)
    te = hd.simulate_paths(P, T, N, 6000, seed=6)
    fit = hd.hmc_fit(tr, K, 1)
    _, chi, _ = hd._features(np.array([1.0]), np.array([math.sqrt(te["FV"][0, 0] / T)]), T, K)
    phi0 = float((chi @ fit["coefs"][0][1])[0])
    d0 = float(hd.bs_delta(1.0, K, math.sqrt(te["FV"][0, 0] / T), T))
    check("the risk-minimising hedge at t = 0 is the Black-Scholes delta (within 0.01)", abs(phi0 - d0) < 0.01,
          f"HMC {phi0:.4f} vs BS {d0:.4f}")
    price = float(rh.call_prices(np.array([0.0]), T, P)[0][0])
    rows = []
    for every in (1, 2, 4, 8, 16):
        e = hd.hedge_error(te, K, every, price, "bs_model")
        rows.append((every * T / N, e.std()))
    slope = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
    check("discrete-hedging error sd scales like sqrt(dt): log-log slope 0.5 +/- 0.1", abs(slope - 0.5) < 0.1,
          f"slope {slope:.3f}")


def test_rough():
    print("\nRough Heston (H = 0.12, xi = 0.3, rho = -0.7): hedging with the underlying alone")
    P = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.3, -0.7, 0.12)
    price = float(rh.call_prices(np.array([0.0]), T, P)[0][0])
    iv = float(fo.implied_vol_from_call(price, 0.0, T))
    tr = hd.simulate_paths(P, T, N, 8000, seed=7)
    te = hd.simulate_paths(P, T, N, 8000, seed=8)
    fit1 = hd.hmc_fit(tr, K, 1)
    z = (fit1["C0"] - price) / fit1["C0_se"]
    check("Hedged Monte Carlo prices the call at the cf price (3 SE)", abs(z) < 3.0,
          f"{fit1['C0']:.6f} +/- {fit1['C0_se']:.6f} vs {price:.6f}")
    worst_mean, better, rows = 0.0, True, []
    for every in (1, 4, 16):
        fit = fit1 if every == 1 else hd.hmc_fit(tr, K, every)
        e = {r: hd.hedge_error(te, K, every, price, r, fit=fit, sigma_fixed=iv) for r in ("bs_fixed", "hmc")}
        for v in e.values():
            worst_mean = max(worst_mean, abs(v.mean()) / (v.std() / math.sqrt(len(v))))
        better &= e["hmc"].std() < e["bs_fixed"].std()
        rows.append(f"every {every}: BS {e['bs_fixed'].std() / price:.3f}, HMC {e['hmc'].std() / price:.3f}")
    check("no rule loses money on average: every mean error within 3.5 SE of zero", worst_mean < 3.5,
          f"worst |z| {worst_mean:.2f}")
    check("the risk-minimising hedge beats Black-Scholes delta out of sample at every interval", better,
          "; ".join(rows))
    Pc = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.01, -0.7, 0.12)
    pc = float(rh.call_prices(np.array([0.0]), T, Pc)[0][0])
    tec = hd.simulate_paths(Pc, T, N, 6000, seed=9)
    ec = hd.hedge_error(tec, K, 1, pc, "bs_model")
    eh = hd.hedge_error(te, K, 1, price, "hmc", fit=fit1)
    ratio = (eh.std() / price) / (ec.std() / pc)
    check("volatility risk leaves a floor: at the finest interval the rough residual is > 3x the control's",
          ratio > 3.0, f"{eh.std() / price:.3f} vs {ec.std() / pc:.3f} of the price ({ratio:.1f}x)")


def test_variance_swap():
    print("\nHedging with the underlying and a variance swap (the lifted model is then complete)")
    P = rh.RoughHestonParams(0.04, 2.0, 0.04, 0.3, -0.7, 0.12)
    price = float(rh.call_prices(np.array([0.0]), T, P)[0][0])
    tr = hd.simulate_paths(P, T, N, 8000, seed=11)
    te = hd.simulate_paths(P, T, N, 8000, seed=12)
    M = te["M"]
    z = (M[-1].mean() - M[0, 0]) / (M[-1].std() / math.sqrt(M.shape[1]))
    check("the variance swap is a martingale: E[M_T] = M_0 within 3 SE", abs(z) < 3.0,
          f"{M[-1].mean():.6f} vs {M[0, 0]:.6f}, z = {z:+.2f}")
    check("...and it pays the realised variance: M_T = int_0^T V dt", np.allclose(M[-1], te["RV"][-1]),
          f"max |diff| {float(np.max(np.abs(M[-1] - te['RV'][-1]))):.1e}")
    f1 = hd.hmc_fit(tr, K, 1)
    f2 = hd.hmc_fit(tr, K, 1, instruments=("S", "M"))
    e1 = hd.hedge_error(te, K, 1, price, "hmc", fit=f1)
    e2 = hd.hedge_error(te, K, 1, price, "hmc2", fit=f2)
    check("adding the variance swap cuts the residual by more than half at the finest interval",
          e2.std() < 0.5 * e1.std(),
          f"{e2.std() / price:.3f} vs {e1.std() / price:.3f} of the price")
    check("it still prices the call at the cf price (3 SE)", abs(f2["C0"] - price) < 3 * f2["C0_se"],
          f"{f2['C0']:.6f} +/- {f2['C0_se']:.6f} vs {price:.6f}")
    sig0 = math.sqrt(te["FV"][0, 0] / T)
    _, _, chi_m = hd._features(np.array([1.0]), np.array([sig0]), T, K)
    j = hd._features(np.array([1.0]), np.array([sig0]), T, K)[1].shape[1]
    phi_m = float((chi_m @ f2["coefs"][0][1][j:])[0])
    vega_v = float(hd.bs_call(1.0, K, sig0 * 1.001, T) - hd.bs_call(1.0, K, sig0, T)) / (0.001 * sig0)
    theory = vega_v / (2 * sig0 * T)                      # dC/d(forward variance) at sigma_hat
    check("the variance-swap hedge ratio is the option's sensitivity to forward variance (within 30%)",
          abs(phi_m / theory - 1) < 0.3, f"fitted {phi_m:.4f} vs Black-Scholes {theory:.4f}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session J -- discrete hedging on simulated rough-Heston paths")
    print("=" * 74)
    t0 = time.perf_counter()
    test_paths()
    test_bs_limit()
    test_rough()
    test_variance_swap()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter()-t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
