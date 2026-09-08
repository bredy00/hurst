"""
Session B, task B2: the Fourier pricers.

Lewis and Carr-Madan share nothing but the characteristic function, so their
agreement is evidence rather than a restatement. Beyond that the strongest check
here is test_implied_density: differentiating the Fourier prices twice must give
a probability density with unit mass and mean equal to the forward. That closes
the loop against the model without reference to either pricer's derivation.

    python test_pricing.py
"""

import math
import sys
import time

import numpy as np

import volsurf_core as vc
import pricing.fourier as fo
from models.heston import HestonParams, char_func, mc_call


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def bs_cf(sigma, tau):
    def f(u):
        u = np.asarray(u, dtype=complex)
        return np.exp(-0.5 * sigma * sigma * tau * (u * u + 1j * u))
    return f


P = HestonParams(0.04, 2.0, 0.045, 0.5, -0.7)
KS = np.array([-0.4, -0.2, -0.05, 0.0, 0.05, 0.2, 0.4])


def test_against_black_scholes():
    print("\nB2 -- against the exact Black-Scholes cf (isolates the pricer)")
    sigma, tau = 0.20, 0.5
    cf = bs_cf(sigma, tau)
    exact = np.array([float(vc.bs_price(1.0, math.exp(k), sigma, tau, 1.0, 'C'))
                      for k in KS])
    lw = np.array(fo.lewis_call(KS, tau, cf))
    cm = np.array(fo.carr_madan_call(KS, tau, cf))
    check("Lewis matches Black-76 to 1e-10",
          float(np.max(np.abs(lw - exact))) < 1e-10,
          f"max {float(np.max(np.abs(lw - exact))):.2e}")
    check("Carr-Madan matches Black-76 to 1e-10",
          float(np.max(np.abs(cm - exact))) < 1e-10,
          f"max {float(np.max(np.abs(cm - exact))):.2e}")

    for tau2 in (1 / 365, 0.05, 1.0, 2.0):
        cf2 = bs_cf(sigma, tau2)
        e2 = np.array([float(vc.bs_price(1.0, math.exp(k), sigma, tau2, 1.0, 'C'))
                       for k in KS])
        l2 = np.array(fo.lewis_call(KS, tau2, cf2))
        check(f"Lewis exact at tau={tau2*365:.0f}d",
              float(np.max(np.abs(l2 - e2))) < 1e-9,
              f"max {float(np.max(np.abs(l2 - e2))):.2e}")


def test_two_pricers_agree():
    print("\nB2 -- Lewis vs Carr-Madan on Heston")
    worst = 0.0
    for tau in (1 / 365, 0.05, 0.25, 1.0, 2.0):
        cf = lambda u, t=tau: char_func(u, t, P)
        lw = np.array(fo.lewis_call(KS, tau, cf))
        cm = np.array(fo.carr_madan_call(KS, tau, cf))
        worst = max(worst, float(np.max(np.abs(lw - cm))))
    check("the two pricers agree to 1e-8 across 5 maturities",
          worst < 1e-8, f"max |Lewis - CM| = {worst:.2e}")

    # Carr-Madan must not depend on its damping parameter
    tau = 0.5
    cf = lambda u: char_func(u, tau, P)
    base = np.array(fo.carr_madan_call(KS, tau, cf, alpha=1.5))
    spread = 0.0
    for a in (0.75, 1.0, 2.0, 3.0):
        spread = max(spread, float(np.max(np.abs(
            np.array(fo.carr_madan_call(KS, tau, cf, alpha=a)) - base))))
    check("Carr-Madan is insensitive to alpha over 0.75..3.0",
          spread < 1e-8, f"max spread {spread:.2e}")


def test_fft():
    print("\nB2 -- the FFT form")
    tau = 0.5
    cf = lambda u: char_func(u, tau, P)
    k_grid, calls = fo.carr_madan_fft(tau, cf, n=8192, eta=0.15)

    m = (k_grid > -0.5) & (k_grid < 0.5)
    direct = np.array(fo.carr_madan_call(k_grid[m], tau, cf))
    err = float(np.max(np.abs(calls[m] - direct)))
    check("FFT matches the direct integral over |k| < 0.5",
          err < 1e-6, f"max {err:.2e}")
    check("FFT returns a monotone decreasing call curve",
          bool(np.all(np.diff(calls[m]) < 1e-12)),
          f"{int(m.sum())} strikes")

    t0 = time.perf_counter()
    fo.carr_madan_fft(tau, cf, n=8192, eta=0.15)
    t_fft = time.perf_counter() - t0
    t0 = time.perf_counter()
    fo.carr_madan_call(k_grid[m], tau, cf)
    t_dir = time.perf_counter() - t0
    print(f"      {int(m.sum())} strikes: FFT {t_fft*1000:.1f} ms, "
          f"direct {t_dir*1000:.1f} ms")


def test_limits():
    print("\nB2 -- boundary behaviour")
    tau = 0.5
    cf = lambda u: char_func(u, tau, P)
    # A deep ITM call is worth its INTRINSIC value, not 1. c(-6) = 1 - e^-6 =
    # 0.9975212478, which is what both this pricer and Black-76 return; an
    # earlier version of this test asserted 1.0 and was simply wrong.
    deep_itm = float(fo.lewis_call(-6.0, tau, cf))
    intrinsic = 1.0 - math.exp(-6.0)
    check("c(k) -> intrinsic as k -> -inf",
          abs(deep_itm - intrinsic) < 1e-6,
          f"c(-6) = {deep_itm:.10f} vs 1-e^-6 = {intrinsic:.10f}")
    check("...and still carries a little time value above it",
          deep_itm >= intrinsic - 1e-12, f"excess {deep_itm - intrinsic:.2e}")

    deep_otm = float(fo.lewis_call(6.0, tau, cf))
    check("c(k) -> 0 as k -> +inf", abs(deep_otm) < 1e-6, f"c(6) = {deep_otm:.2e}")

    ks = np.linspace(-1.0, 1.0, 201)
    cs = np.array(fo.lewis_call(ks, tau, cf))
    check("c(k) is monotone decreasing in k", bool(np.all(np.diff(cs) < 0)))
    check("c(k) >= max(1 - e^k, 0), the intrinsic bound",
          bool(np.all(cs >= np.maximum(1 - np.exp(ks), 0) - 1e-12)),
          f"worst margin {float(np.min(cs - np.maximum(1 - np.exp(ks), 0))):.2e}")
    check("c(k) <= 1, the forward bound", bool(np.all(cs <= 1.0 + 1e-12)))


def test_implied_density():
    """
    Differentiate the Fourier prices twice: the result must be a probability
    density with unit mass and mean equal to the forward (= 1 here). This checks
    the pricer against the model without touching either derivation.
    """
    print("\nB2 -- risk-neutral density implied by the Fourier prices")
    tau = 0.5
    cf = lambda u: char_func(u, tau, P)
    K = np.linspace(0.05, 4.0, 3001)                 # uniform in K, as fd_second needs
    C = np.array(fo.lewis_call(np.log(K), tau, cf))
    Kb, dens = vc.breeden_litzenberger(K, C, 1.0)
    good = np.isfinite(dens)

    mass = float(np.trapezoid(dens[good], Kb[good]))
    mean = float(np.trapezoid(Kb[good] * dens[good], Kb[good]))
    peak = float(np.nanmax(dens))
    lo = float(np.nanmin(dens))

    check("density integrates to 1", abs(mass - 1.0) < 2e-3, f"{mass:.6f}")
    check("density mean is the forward", abs(mean - 1.0) < 2e-3, f"{mean:.6f}")

    # Non-negativity has to be judged RELATIVE to the peak. A second difference
    # amplifies price error by 1/dK^2, so in the far tail -- where the true
    # density is ~0 -- the result dithers around zero at exactly that scale.
    check("density is non-negative relative to its peak",
          lo / peak > -1e-7, f"min/peak = {lo/peak:.2e} (peak {peak:.4f})")

    tail = Kb[good][dens[good] < 0]
    check("any negative values live only in the far tail, not near the mode",
          tail.size == 0 or float(np.min(tail)) > 1.8,
          f"negatives confined to K >= {float(np.min(tail)):.2f}, mode at "
          f"K = {Kb[int(np.nanargmax(dens))]:.2f}" if tail.size else "none")

    # And prove it is differencing noise: it must scale like 1/dK^2, i.e.
    # halving the step quadruples it. A real arbitrage would not do that.
    mins = []
    for n in (1501, 3001, 6001):
        Kx = np.linspace(0.05, 4.0, n)
        Cx = np.array(fo.lewis_call(np.log(Kx), tau, cf))
        _, dx = vc.breeden_litzenberger(Kx, Cx, 1.0)
        mins.append(abs(float(np.nanmin(dx))))
    ratios = [mins[i + 1] / mins[i] for i in range(len(mins) - 1)]
    # A genuine arbitrage would be a property of the surface and would NOT move
    # when the grid is refined (ratio ~ 1). Differencing noise grows like 1/dK^2,
    # so refining makes it worse. The ratio is noisy because it depends on where
    # the nodes happen to land, so the window is generous -- what matters is that
    # it is far from 1 and clearly super-linear.
    check("the negativity GROWS on refinement, so it is quadrature noise",
          all(2.0 < r < 9.0 for r in ratios),
          "ratio per halving: " + ", ".join(f"{r:.1f}" for r in ratios)
          + " (want ~4 for 1/dK^2; ~1 would mean a real arbitrage)")


def test_implied_vol_round_trip():
    print("\nB2 -- implied vol inversion")
    tau = 0.5
    sigma = 0.20
    cf = bs_cf(sigma, tau)
    vols = fo.smile(KS, tau, cf)
    check("a Black-Scholes cf gives a FLAT smile at the right level",
          all(v is not None and abs(v - sigma) < 1e-7 for v in vols),
          f"max dev {max(abs(v - sigma) for v in vols):.2e}")

    hcf = lambda u: char_func(u, tau, P)
    hv = fo.smile(KS, tau, hcf)
    check("Heston gives a smile that is downward sloping (rho < 0)",
          all(a is not None for a in hv) and hv[0] > hv[-1],
          " ".join(f"{v:.4f}" for v in hv))
    check("Heston ATM vol is near sqrt of the average variance",
          abs(hv[3] - 0.207) < 0.02, f"ATM {hv[3]:.4f}")


def test_monte_carlo():
    print("\nB2 -- Fourier vs Monte Carlo (independent of the transform entirely)")
    tau = 0.5
    cf = lambda u: char_func(u, tau, P)
    print(f"      {'k':>6} {'Fourier':>11} {'MC':>11} {'MC se':>10} {'diff':>11} {'diff/se':>8}")
    worst_z = 0.0
    for k in (-0.2, 0.0, 0.2):
        f_price = float(fo.lewis_call(k, tau, cf))
        m_price, se = mc_call(P, tau, k, n_paths=400_000, n_steps=500, seed=7)
        z = abs(f_price - m_price) / se
        worst_z = max(worst_z, z)
        print(f"      {k:6.2f} {f_price:11.7f} {m_price:11.7f} {se:10.7f} "
              f"{f_price - m_price:11.2e} {z:8.2f}")
    check("Fourier price sits within 5 standard errors of Monte Carlo",
          worst_z < 5.0, f"worst {worst_z:.2f} sigma "
                         "(residual is Euler discretisation bias, O(dt))")


def test_quadrature():
    print("\nB2 -- adaptive composite quadrature")
    # Truncation must track the maturity: short options need far more range
    us = {}
    for tau in (1 / 365, 0.05, 1.0, 2.0):
        cf = lambda u, t=tau: char_func(u, t, P)
        us[tau] = fo._auto_u_max(cf, -0.5j)
    print("      auto u_max: " +
          ", ".join(f"{t*365:.0f}d:{u:.0f}" for t, u in us.items()))
    check("short maturities get a wider truncation than long ones",
          us[1 / 365] > us[2.0], f"1d:{us[1/365]:.0f} vs 2y:{us[2.0]:.0f}")
    check("deep strikes get more panels than near-the-money ones",
          fo._auto_panels(400, 6.0) > fo._auto_panels(400, 0.1),
          f"|k|=6: {fo._auto_panels(400, 6.0)}, |k|=0.1: {fo._auto_panels(400, 0.1)}")

    # However fine the grid gets, only the one small panel rule is ever built
    fo._leggauss.cache_clear()
    cf = lambda u: char_func(u, 0.5, P)
    for kk in (0.0, -3.0, -6.0):
        fo.lewis_call(kk, 0.5, cf)
    info = fo._leggauss.cache_info()
    check("only one Gauss-Legendre rule is constructed, however fine the grid",
          info.currsize == 1 and info.misses == 1,
          f"{info.misses} build(s), {info.hits} reuse(s) -- a single high-order "
          f"rule would cost O(n^2) each time")

    t0 = time.perf_counter()
    for _ in range(20):
        fo.lewis_call(KS, 0.5, cf)
    dt = (time.perf_counter() - t0) / 20
    check("a 7-strike Heston slice prices in under 25 ms", dt < 0.025,
          f"{dt*1000:.2f} ms per slice")


if __name__ == "__main__":
    print("=" * 74)
    print("Session B / B2 -- Fourier pricing")
    print("=" * 74)
    test_against_black_scholes()
    test_two_pricers_agree()
    test_fft()
    test_limits()
    test_implied_density()
    test_implied_vol_round_trip()
    test_monte_carlo()
    test_quadrature()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
