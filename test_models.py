"""
Session B, task B1: the Heston characteristic function.

The load-bearing test here is test_little_heston_trap. Everything else checks
identities that a wrong implementation could still satisfy; that one is the only
check that distinguishes the correct branch of the complex logarithm from the
one the 1993 paper uses.

    python test_models.py
"""

import math
import sys

import numpy as np

from models.heston import HestonParams, char_func, char_func_trap, simulate


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def bs_cf(sigma, tau):
    """Exact Black-Scholes characteristic function of log(S/F)."""
    def f(u):
        u = np.asarray(u, dtype=complex)
        return np.exp(-0.5 * sigma * sigma * tau * (u * u + 1j * u))
    return f


PARAM_SETS = [
    HestonParams(0.04, 2.0, 0.045, 0.5, -0.7),
    HestonParams(0.09, 1.2, 0.06, 0.8, -0.5),
    HestonParams(0.01, 5.0, 0.02, 0.3, -0.9),
    HestonParams(0.16, 0.5, 0.20, 1.0, 0.1),
    HestonParams(0.02, 3.5, 0.03, 0.15, -0.3),
]


def test_identities():
    print("\nB1 -- characteristic function identities")
    rng = np.random.default_rng(3)
    worst0, worstm = 0.0, 0.0
    for p in PARAM_SETS:
        for tau in (1 / 365, 0.05, 0.25, 1.0, 2.0):
            worst0 = max(worst0, abs(complex(char_func(0.0, tau, p)) - 1.0))
            worstm = max(worstm, abs(complex(char_func(-1j, tau, p)) - 1.0))
    check("phi(0) = 1 across 5 parameter sets x 5 maturities",
          worst0 < 1e-12, f"max |phi(0)-1| = {worst0:.2e}")
    check("phi(-i) = 1, the martingale condition",
          worstm < 1e-10, f"max |phi(-i)-1| = {worstm:.2e}")

    # Reality: phi(-u) must be the conjugate of phi(u) for real u
    p = PARAM_SETS[0]
    u = np.linspace(0.1, 30, 200)
    a = char_func(u, 0.5, p)
    b = char_func(-u, 0.5, p)
    check("phi(-u) = conj(phi(u)) for real u",
          float(np.max(np.abs(b - np.conj(a)))) < 1e-12,
          f"max {float(np.max(np.abs(b - np.conj(a)))):.2e}")

    check("|phi(u)| <= 1 for real u",
          float(np.max(np.abs(a))) <= 1.0 + 1e-12,
          f"max |phi| = {float(np.max(np.abs(a))):.6f}")

    # 2 kappa theta > xi^2. Note the DEFAULT parameters fail it (0.18 < 0.25) --
    # which is realistic, and exactly why it is reported rather than enforced.
    check("Feller true when 2*kappa*theta > xi^2",
          HestonParams(0.04, 3.0, 0.04, 0.3, -0.7).feller, "2kt=0.240 > xi2=0.090")
    check("Feller false when it is not",
          not HestonParams(0.04, 0.5, 0.02, 0.9, -0.7).feller, "2kt=0.020 < xi2=0.810")
    check("the default parameter set violates Feller, and is still usable",
          not HestonParams().feller and HestonParams().valid(),
          "2kt=0.180 < xi2=0.250 -- reported, never clamped")


def test_little_heston_trap():
    """
    The one that matters. The reciprocal g crosses the branch cut of the complex
    log; the shipped form does not. Demonstrated, not asserted on authority.
    """
    print("\nB1 -- the little Heston trap")
    p = HestonParams(v0=0.04, kappa=0.5, theta=0.045, xi=0.9, rho=-0.7)
    taus = np.linspace(0.01, 3.0, 4000)

    # Compare each implementation against ITSELF: max step / median step. For a
    # continuous function on a fine grid that ratio is O(1) and grows only slowly
    # with u as the cf genuinely oscillates faster; a branch-cut crossing makes
    # one step enormous relative to its neighbours. Comparing the two
    # implementations' absolute steps instead would confound the discontinuity
    # with the legitimate high-u oscillation, which is what an earlier version of
    # this test did -- it reported only 17x at u=20 and looked like a near-miss.
    def jumpiness(f, u):
        v = np.array([complex(f(u, t, p)) for t in taus])
        d = np.abs(np.diff(v))
        return float(np.max(d) / max(float(np.median(d)), 1e-300))

    for u in (4.0, 8.0):
        r_good, r_bad = jumpiness(char_func, u), jumpiness(char_func_trap, u)
        check(f"shipped phi has no jump at u={u:g}",
              r_good < 50.0, f"max/median step = {r_good:.1f}")
        check(f"the reciprocal form DOES jump at u={u:g} (the trap is real)",
              r_bad > 500.0 and r_bad > 100 * r_good,
              f"trap {r_bad:.0f} vs shipped {r_good:.1f}")

    for u, tol in ((1.5, 1e-3), (8.0, 5e-3), (20.0, 1e-1)):
        s = float(np.max(np.abs(np.diff(
            np.array([complex(char_func(u, t, p)) for t in taus])))))
        check(f"shipped max step stays small at u={u:g}", s < tol, f"{s:.2e}")

    # Continuity must hold out to two years for every parameter set
    worst = 0.0
    for p in PARAM_SETS:
        t = np.linspace(0.01, 2.0, 1500)
        v = np.array([complex(char_func(4.0, x, p)) for x in t])
        worst = max(worst, float(np.max(np.abs(np.diff(v)))))
    check("continuous in tau to 2 years for all 5 parameter sets",
          worst < 5e-3, f"worst step {worst:.2e}")


def test_bs_degeneracy():
    """
    xi -> 0. The cf must tend to Black-Scholes at the rate xi^2 -- that is the
    genuine vol-of-vol effect -- all the way down to the floating-point floor.
    The literal textbook evaluation cannot do this (cancellation in
    kappa*theta/xi^2), which is measured on `char_func_naive` rather than stated.
    """
    print("\nB1 -- xi -> 0 degenerates to Black-Scholes")
    import models.heston as mh
    sigma, tau = 0.20, 0.5
    exact = bs_cf(sigma, tau)
    u = np.linspace(0.0, 30.0, 300)

    errs, naive = {}, {}
    for xi in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12):
        q = HestonParams(v0=sigma ** 2, kappa=2.0, theta=sigma ** 2, xi=xi, rho=0.0)
        errs[xi] = float(np.max(np.abs(char_func(u, tau, q) - exact(u))))
        naive[xi] = float(np.max(np.abs(mh.char_func_naive(u, tau, q) - exact(u))))
    print("      shipped: " + ", ".join(f"{x:.0e}:{e:.1e}" for x, e in errs.items()))
    print("      naive  : " + ", ".join(f"{x:.0e}:{e:.1e}" for x, e in naive.items()))

    check("cf matches Black-Scholes at xi = 1e-4", errs[1e-4] < 1e-8,
          f"max |diff| {errs[1e-4]:.2e}")
    # Between 1e-4 and 1e-6 the remaining difference is the MODEL's O(xi^2)
    # vol-of-vol effect, so it must shrink by 1e4, not plateau.
    ratio = errs[1e-6] / errs[1e-4]
    check("the deviation from BS scales as xi^2 (it is model, not error)",
          0.5e-4 < ratio < 2e-4, f"err(1e-6)/err(1e-4) = {ratio:.2e}, expect 1e-4")
    check("machine precision from xi = 1e-8 down to 1e-12",
          max(errs[1e-8], errs[1e-10], errs[1e-12]) < 1e-14,
          f"{errs[1e-8]:.1e}, {errs[1e-10]:.1e}, {errs[1e-12]:.1e}")
    check("the literal evaluation DOES lose precision (measured, not described)",
          naive[1e-6] > 1e-6 and naive[1e-10] > 0.1,
          f"naive {naive[1e-6]:.1e} at 1e-6, {naive[1e-10]:.1e} at 1e-10")
    # Where the naive form is healthy the two must be the same function. Compare
    # the cfs directly rather than two maxima -- at xi = 1e-3 the naive form
    # already carries ~1e-10 of cancellation, which is the whole point.
    uu = np.linspace(0.0, 60.0, 300) - 1.5j
    same = max(float(np.max(np.abs(char_func(uu, t, q) - mh.char_func_naive(uu, t, q))))
               for q in (HestonParams(0.04, 2.0, 0.045, 0.5, -0.7),
                         HestonParams(0.09, 1.2, 0.06, 0.05, -0.5))
               for t in (1 / 365, 0.5, 2.0))
    check("...and agrees with the shipped form where it is healthy",
          same < 1e-12, f"max |diff| {same:.1e} at xi = 0.5 and 0.05")

    # The two branches of the small-xi switch must agree across the switch
    saved = mh.XI_TAYLOR
    worst = 0.0
    try:
        for xi in (1e-3, 1e-4, 1e-5):
            q = HestonParams(0.04, 2.0, 0.045, xi, -0.7)
            for t in (1 / 365, 0.5, 2.0):
                uu = np.linspace(0.0, 60.0, 300) - 1.5j
                mh.XI_TAYLOR = 0.0
                a = char_func(uu, t, q)
                mh.XI_TAYLOR = 1.0
                b = char_func(uu, t, q)
                worst = max(worst, float(np.max(np.abs(a - b))))
    finally:
        mh.XI_TAYLOR = saved
    check("Taylor branch and log1p branch agree across the switch",
          worst < 1e-14, f"max |diff| {worst:.2e} over xi = 1e-3, 1e-4, 1e-5")

    # xi = 0 exactly: the deterministic-variance limit, rho and theta irrelevant
    q0 = HestonParams(0.04, 2.0, 0.09, 0.0, -0.7)
    V = 0.04 * (1 - math.exp(-2.0 * 0.7)) / 2.0 + 0.09 * (0.7 - (1 - math.exp(-2.0 * 0.7)) / 2.0)
    lim = np.exp(-0.5 * (u * u + 1j * u) * V)
    e0 = float(np.max(np.abs(char_func(u, 0.7, q0) - lim)))
    check("xi = 0 exactly returns the deterministic-variance cf", e0 < 1e-15,
          f"{e0:.1e}")
    # and xi = 1e-9 with rho != 0 is within its genuine O(xi) term of that limit
    q9 = HestonParams(0.04, 2.0, 0.09, 1e-9, -0.7)
    e9 = float(np.max(np.abs(char_func(u, 0.7, q9) - lim)))
    check("xi = 1e-9, rho = -0.7 sits within O(xi) of the limit", e9 < 1e-7,
          f"{e9:.1e}")


def test_monte_carlo():
    print("\nB1 -- Monte Carlo agreement (full-truncation Euler)")
    p = HestonParams(0.04, 2.0, 0.045, 0.5, -0.7)
    tau = 0.5

    x = simulate(p, tau, n_paths=200_000, n_steps=400, seed=1)
    mean_exp = float(np.mean(np.exp(x)))
    se = float(np.std(np.exp(x), ddof=1) / math.sqrt(len(x)))
    check("simulated E[S/F] = 1 within MC error",
          abs(mean_exp - 1.0) < 4 * se,
          f"{mean_exp:.6f} +/- {se:.6f}")

    # The cf's own second moment must match the sample
    var_model = float(np.real(-(char_func(1e-4, tau, p) - 2 + char_func(-1e-4, tau, p))
                              / 1e-8))
    check("Var[X] from the cf matches the simulation within 3%",
          abs(var_model - float(np.var(x))) / float(np.var(x)) < 0.03,
          f"cf {var_model:.6f} vs mc {float(np.var(x)):.6f}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session B / B1 -- Heston characteristic function")
    print("=" * 74)
    test_identities()
    test_little_heston_trap()
    test_bs_degeneracy()
    test_monte_carlo()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
