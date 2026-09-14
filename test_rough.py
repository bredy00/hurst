"""
Session D: the Volterra kernel, its Markovian lift, and the lifted rough Heston
characteristic function.

The load-bearing tests are in D4: with a single node at x = 0 the lift must
reproduce closed-form Heston to rounding-plus-ODE-error, and as H -> 1/2 the
full N-node lift must converge to it. Together they prove the lift is a
generalisation of Heston and not a different model. Everything else here
measures how good the approximation is and where it is not.

    python test_rough.py
"""

import math
import sys
import time

import numpy as np

import models.rough_heston as rh
import pricing.fourier as fo
from models.heston import HestonParams, char_func as heston_cf


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


np.seterr(all="ignore")
P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.12)
PH = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, 0.5)
HP = PH.heston()


# --- D1: the kernel and its completely-monotone representation -------------
def test_kernel_representation():
    print("\nD1 -- K(t) = int e^{-xt} mu(dx)")
    from scipy.integrate import quad
    worst = 0.0
    for H in (0.10, 0.12, 0.30):
        for t in (1 / 365, 0.05, 0.5, 2.0):
            num, _ = quad(lambda x: math.exp(-x * t) * float(rh.mu_density(x, H)),
                          0, np.inf, limit=400)
            exact = float(rh.fractional_kernel(t, H))
            worst = max(worst, abs(num - exact) / exact)
    check("numerical integral of mu reproduces K(t) to 1e-8 on [1 day, 2 years]",
          worst < 1e-8, f"worst relative error {worst:.2e}")

    # the closed-form cell integrals used to build the nodes
    worst = 0.0
    for H in (0.12, 0.30):
        for lo, hi in ((0.0, 0.1), (0.1, 1.0), (30.0, 100.0)):
            m, _ = quad(lambda x: float(rh.mu_density(x, H)), lo, hi)
            m1, _ = quad(lambda x: x * float(rh.mu_density(x, H)), lo, hi)
            worst = max(worst, abs(m - float(rh.mu_mass(lo, hi, H))) / m,
                        abs(m1 - float(rh.mu_first_moment(lo, hi, H))) / m1)
    check("closed-form cell mass and first moment match quadrature", worst < 1e-9,
          f"{worst:.2e}")
    check("K(t) is singular at 0 for H < 1/2 and equals 1 for H = 1/2",
          float(rh.fractional_kernel(1e-6, 0.12)) > 100
          and abs(float(rh.fractional_kernel(0.37, 0.5)) - 1.0) < 1e-15)


# --- D2: the sum of exponentials ---------------------------------------------
def test_sum_of_exponentials():
    print("\nD2 -- K(t) ~ sum w_i exp(-x_i t)")
    e_day, _ = rh.kernel_error(0.12)
    e_hour, _ = rh.kernel_error(0.12, t_lo=1 / (365 * 24))
    check(f"default N = {rh.N_DEFAULT} reproduces K to <1% on [1 day, 2 years]",
          e_day < 0.01, f"max relative error {e_day:.3%}")
    check("...and on [1 hour, 2 years], where a one-day option lives",
          e_hour < 0.01, f"{e_hour:.3%}")
    e30, _ = rh.kernel_error(0.30)
    check("same at H = 0.30", e30 < 0.01, f"{e30:.3%}")
    # The roughest H the node range was built for; measured 1.05%, and the
    # bound says so rather than pretending the default reaches 1% there.
    e05, _ = rh.kernel_error(0.05)
    check("H = 0.05 (the rough edge of the range) within 1.2%", e05 < 0.012, f"{e05:.3%}")

    errs = [rh.kernel_error(0.12, N)[0] for N in (5, 10, 20, 40)]
    check("error decreases monotonically in N over {5, 10, 20, 40}",
          all(a > b for a, b in zip(errs, errs[1:])),
          ", ".join(f"{e:.2%}" for e in errs))

    w, x = rh.lift_nodes(0.12)
    check("weights positive, nodes positive and increasing",
          np.all(w > 0) and np.all(x > 0) and np.all(np.diff(x) > 0))
    check("the first node is slow (x_1 < 0.05) -- the long-memory mass is kept",
          x[0] < 0.05, f"x_1 = {x[0]:.4f}, w_1 = {w[0]:.3f}")
    w1, x1 = rh.lift_nodes(0.5)
    check("H = 1/2 gives the single node (w, x) = (1, 0)",
          len(w1) == 1 and w1[0] == 1.0 and x1[0] == 0.0)

    # Laplace domain. The plan asked for <1% on z in [0.1, 100]; that band
    # probes t ~ 10 years at its low end, outside the fitted [1 hour, 2 years],
    # and the kernel is not fitted there. Reported as measured, with the
    # diagnostic that the error is at small z (long memory), not at the
    # pricing scales.
    def lap_err(lo, hi):
        z = np.exp(np.linspace(math.log(lo), math.log(hi), 200))
        return float(np.max(np.abs(rh.laplace_approx(z, w, x) - rh.laplace_kernel(z, 0.12))
                            / rh.laplace_kernel(z, 0.12)))

    e_plan, e_long, e_mid, e_fast = (lap_err(0.1, 100), lap_err(0.1, 1.0),
                                     lap_err(1.0, 100), lap_err(100, 1e4))
    print(f"      Laplace-domain error: [0.1,100] {e_plan:.2%}   [0.1,1] {e_long:.2%}   "
          f"[1,100] {e_mid:.2%}   [100,1e4] {e_fast:.2%}")
    check("Laplace error on z in [1, 100] (a year down to a few days) < 3%",
          e_mid < 0.03, f"{e_mid:.2%}")
    check("the planned [0.1, 100] band fails 1% BECAUSE of z < 1 (memory beyond 2 years)",
          e_plan > 0.01 and e_long > e_mid,
          f"{e_plan:.2%} overall, {e_long:.2%} at z < 1 vs {e_mid:.2%} at z in [1, 100]")
    # At large z the transform is dominated by t -> 0, where no finite lift
    # follows the singularity: at z = 730 a quarter of z^-alpha comes from the
    # first hour. That is the sub-cell limitation of D3 seen from the other side.
    check("...and z > 100 is worse again, from the sub-hour singularity (documented)",
          e_fast > e_mid, f"{e_fast:.2%}")


# --- D3: the lift as a simulation ------------------------------------------
def test_lifted_simulation():
    print("\nD3 -- lifted factors vs the direct Volterra convolution")
    rng = np.random.default_rng(0)
    n, dt = 2000, 1.0 / 365.0
    dZ = rng.normal(0.0, math.sqrt(dt), size=(n, 8))
    w, x = rh.lift_nodes(0.12)
    lags = np.arange(1, n + 1) * dt

    V_lift = rh.simulate_lifted_gaussian(w, x, dZ, dt)
    V_soe = rh.convolve_kernel(rh.kernel_approx(lags, w, x), dZ)
    d = float(np.max(np.abs(V_lift - V_soe)))
    check("lifted recursion == convolution with the SAME sum-of-exponentials kernel (identity)",
          d < 1e-12, f"max |diff| {d:.1e} on paths of scale {float(np.std(V_soe)):.2f}")

    Kbar_N = np.array([float((w * np.exp(-x * (j * dt)) * rh._phi1(-x * dt)).sum())
                       for j in range(n)])
    d2 = float(np.max(np.abs(rh.simulate_lifted_gaussian(w, x, dZ, dt, cell_mean=True)
                             - rh.convolve_kernel(Kbar_N, dZ))))
    check("same identity for the cell-averaged convention", d2 < 1e-12, f"{d2:.1e}")

    V_true = rh.convolve_kernel(rh.fractional_kernel(lags, 0.12), dZ)
    rel = math.sqrt(float(np.mean((V_lift - V_true) ** 2)) / float(np.mean(V_true ** 2)))
    check("lifted path vs TRUE-kernel Volterra path from the same Brownian increments: <1% in L2",
          rel < 0.01, f"relative L2 {rel:.3%} (N = {len(w)}, lags >= 1 day)")

    # Where the lift is approximate: the singular part of K inside the first
    # cell. Including the lag-0 cell mean in both convolutions exposes it.
    a = 0.62
    Kbar_true = np.array([((((j + 1) * dt) ** a - (j * dt) ** a) / (a * dt)) / math.gamma(a)
                          for j in range(n)])
    V_lift_cm = rh.simulate_lifted_gaussian(w, x, dZ, dt, cell_mean=True)
    V_true_cm = rh.convolve_kernel(Kbar_true, dZ)
    rel_cm = math.sqrt(float(np.mean((V_lift_cm - V_true_cm) ** 2)) / float(np.mean(V_true_cm ** 2)))
    print(f"      with the lag-0 cell mean included (sub-day singularity): {rel_cm:.2%}")
    check("the sub-cell singularity is the only place the lift is worse than 1%",
          rel_cm < 0.05 and rel_cm > rel, f"{rel_cm:.2%} vs {rel:.2%}")

    # Each factor alone is an OU process
    i = 10
    E = math.exp(-x[i] * dt)
    z = np.random.default_rng(1).normal(0.0, math.sqrt(dt), 200_000)
    U = np.empty_like(z)
    u = 0.0
    for k in range(len(z)):
        u = E * (u + z[k])
        U[k] = u
    s = U[5000:]
    worst = 0.0
    for m in (1, 5, 20):
        ac = float(np.corrcoef(s[m:], s[:-m])[0, 1])
        worst = max(worst, abs(ac - math.exp(-x[i] * m * dt)))
    check(f"factor i={i} (x = {x[i]:.2f}) has OU autocorrelation e^(-x m dt) at lags 1, 5, 20",
          worst < 0.01, f"worst deviation {worst:.4f}")


# --- D4: the characteristic function ----------------------------------------
def test_reduces_to_heston():
    """N = 1, x = 0: the lift IS Heston. Solve the same Riccati numerically and compare."""
    print("\nD4 -- N = 1 at x = 0 reproduces closed-form Heston")
    for tau in (1 / 365, 0.5, 2.0):
        uu = np.linspace(0.0, 60.0, 121) - 1.5j
        ex = heston_cf(uu, tau, HP)
        errs = {}
        for M in (50, 100, 200):
            # a convergence-ORDER study: 50 steps sits just past the operating
            # stability constant at u = 60 on purpose, so the guard is off here
            errs[M] = float(np.max(np.abs(rh.char_func(uu, tau, PH, N=1, steps=M, scheme="etdrk4",
                                                       check_stability=False) - ex)))
        check(f"tau = {tau:.4f}: ETDRK4 at 200 steps within 1e-9 of Heston",
              errs[200] < 1e-9, f"{errs[200]:.2e}")
        if errs[200] > 1e-13:
            order = math.log2(errs[50] / errs[200]) / 2
            check(f"tau = {tau:.4f}: observed order ~4", 3.6 < order < 4.4, f"{order:.2f}")

    tau = 0.5
    uu = np.linspace(0.0, 60.0, 121) - 1.5j
    ex = heston_cf(uu, tau, HP)
    e_tr = {M: float(np.max(np.abs(rh.char_func(uu, tau, PH, N=1, steps=M, scheme="exptrap",
                                                 check_stability=False) - ex)))
            for M in (50, 200)}
    e_ri = {M: float(np.max(np.abs(rh.char_func(uu, tau, PH, N=1, steps=M, scheme="exptrap",
                                                 richardson=True, check_stability=False) - ex)))
            for M in (50, 200)}
    o_tr = math.log2(e_tr[50] / e_tr[200]) / 2
    o_ri = math.log2(e_ri[50] / e_ri[200]) / 2
    check("implicit exponential-trapezoidal scheme is second order", 1.9 < o_tr < 2.1,
          f"order {o_tr:.2f}, {e_tr[200]:.1e} at 200 steps")
    check("...and fourth order with Richardson", 3.7 < o_ri < 4.3,
          f"order {o_ri:.2f}, {e_ri[200]:.1e} at 200 steps")


def test_h_to_half():
    """The single most important test: the full N-node lift converges to Heston as H -> 1/2."""
    print("\nD4 -- H -> 1/2 with the full lift converges to Heston")
    uu = np.linspace(0.0, 60.0, 121) - 1.5j
    dist = {}
    for H in (0.45, 0.49, 0.499, 0.4999):
        q = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, H)
        dist[H] = max(float(np.max(np.abs(rh.char_func(uu, tau, q, steps_mult=2.0, scheme="etdrk4")
                                          - heston_cf(uu, tau, HP))))
                      for tau in (1 / 365, 0.5, 2.0))
    print("      " + "  ".join(f"H={H}: {d:.2e}" for H, d in dist.items()))
    check("at H = 0.4999 the lifted cf is within 1e-4 of vanilla Heston",
          dist[0.4999] < 1e-4, f"{dist[0.4999]:.2e}")
    r1, r2 = dist[0.49] / dist[0.499], dist[0.499] / dist[0.4999]
    check("the distance shrinks linearly in (1/2 - H): 10x per decade",
          8.0 < r1 < 12.0 and 8.0 < r2 < 12.0, f"ratios {r1:.1f}, {r2:.1f}")
    check("...and is NOT small at H = 0.45 -- roughness is a real difference",
          dist[0.45] > 1e-3, f"{dist[0.45]:.2e}")


def test_cf_properties():
    print("\nD4 -- properties of the rough cf at H = 0.12")
    w0 = wm = 0.0
    for tau in (1 / 365, 0.5, 2.0):
        w0 = max(w0, abs(complex(rh.char_func(0.0, tau, P)) - 1.0))
        wm = max(wm, abs(complex(rh.char_func(-1j, tau, P)) - 1.0))
    check("phi(0) = 1", w0 < 1e-12, f"{w0:.1e}")
    check("phi(-i) = 1 (martingale)", wm < 1e-12, f"{wm:.1e}")

    u = np.linspace(0.1, 30.0, 150)
    a = rh.char_func(u, 0.5, P)
    b = rh.char_func(-u, 0.5, P)
    check("phi(-u) = conj(phi(u))", float(np.max(np.abs(b - np.conj(a)))) < 1e-12)
    check("|phi(u)| <= 1", float(np.max(np.abs(a))) <= 1.0 + 1e-12)

    # No branch-cut style jumps in tau (the Heston trap has no analogue here,
    # but the ODE must not have one either)
    taus = np.linspace(0.01, 2.0, 400)
    v = np.array([complex(rh.char_func(4.0, t, P, steps=60, scheme="exptrap")) for t in taus])
    d = np.abs(np.diff(v))
    ratio = float(np.max(d) / np.median(d))
    check("continuous in tau (max/median step < 20 on a 400-point grid)", ratio < 20.0,
          f"{ratio:.1f}")

    # Step convergence and scheme agreement at realistic u ranges
    worst_step = worst_scheme = 0.0
    for d_ in (1, 30, 365):
        tau = d_ / 365
        um = rh.u_max_for(P, tau)
        uu = np.linspace(0.0, um, 300) - 2.5j
        a1 = rh.char_func(uu, tau, P, scheme="etdrk4")
        a2 = rh.char_func(uu, tau, P, scheme="etdrk4", steps_mult=2.0)
        b = rh.char_func(uu, tau, P, scheme="exptrap", steps=200, richardson=True)
        worst_step = max(worst_step, float(np.max(np.abs(a1 - a2))))
        worst_scheme = max(worst_scheme, float(np.max(np.abs(a2 - b))))
    check("doubling the ETDRK4 steps changes the cf by < 1e-6", worst_step < 1e-6,
          f"{worst_step:.1e}")
    check("ETDRK4 and implicit-trapezoidal+Richardson agree to 5e-6 (independent schemes)",
          worst_scheme < 5e-6, f"{worst_scheme:.1e}")

    # 'auto' picks the cheap scheme
    um1 = rh.u_max_for(P, 1 / 365)
    um365 = rh.u_max_for(P, 1.0)
    s1, m1 = rh.choose_scheme(1 / 365, um1, P)
    s365, m365 = rh.choose_scheme(1.0, um365, P)
    check("auto: ETDRK4 at one day, implicit at one year (cost at stable step count decides)",
          s1 == "etdrk4" and s365 == "exptrap",
          f"1 d: {s1} x {m1} steps;  1 y: {s365} x {m365} (+2x for Richardson)")

    # phi-functions: series and direct branches agree at the switch
    c = np.array([-0.0999, -0.1001, -0.5, -5.0, -1e-6])
    p1, p2, p3 = rh._phi123(c)
    ex1 = np.expm1(c) / c
    check("phi_1 exact across the series/direct switch",
          float(np.max(np.abs(p1 - ex1))) < 1e-15)
    check("phi_3 series matches the direct formula at |c| = 0.1 to 1e-14",
          abs(p3[0] - (math.expm1(-0.0999) + 0.0999 - 0.0999 ** 2 / 2) / (-0.0999) ** 3) < 1e-13
          and abs(p3[1] - (math.expm1(-0.1001) + 0.1001 - 0.1001 ** 2 / 2) / (-0.1001) ** 3) < 1e-13)


def test_pricing_wrapper():
    print("\nD4 -- call_prices: Lewis contour, self-sized and self-checked")
    import models.rough_heston as rhm
    for d_ in (1, 30, 365):
        tau = d_ / 365
        band = 3 * 0.2 * math.sqrt(tau)
        ks = np.linspace(-band, band, 9)
        c, info = rh.call_prices(ks, tau, P)
        # Independent reference: the generic uniform-grid Lewis pricer, twice
        # the range, twice the panels, ETDRK4 at three times its stable steps.
        um = 2.0 * info["u_max"]
        cf = rh.cf_factory(P, tau, scheme="etdrk4", steps_mult=3.0)
        ref = np.asarray(fo.lewis_call(ks, tau, cf, u_max=um,
                                       n_panels=2 * rh.n_panels_for(um, band)))
        iv, iv_ref = fo.implied_vols_from_calls(c, ks, tau), fo.implied_vols_from_calls(ref, ks, tau)
        e = float(np.nanmax(np.abs(iv - iv_ref)))
        check(f"{d_:3d}d: agrees with an independent brute-force Lewis price to 0.005 vol points",
              e < 5e-5, f"max |IV diff| {e*100:.1e} vp; {info['scheme']} x {info['steps']}, "
                        f"{info['n_nodes']} u-nodes")
        check(f"{d_:3d}d: every solve checked -- ok, tail below tolerance, |phi(u - i/2)| <= 1",
              info["ok"] and info["tail"] < 1e-9 and info["phi_max"] <= 1.0 + 1e-6,
              f"tail {info['tail']:.1e}, max |phi| {info['phi_max']:.6f}")
        intrinsic = np.maximum(1.0 - np.exp(ks), 0.0)
        check(f"{d_:3d}d: prices decreasing in k and inside the no-arbitrage band",
              np.all(np.diff(c) < 0) and np.all(c > intrinsic) and np.all(c < 1.0))

    _, tight = rh.call_prices(np.array([0.0]), 7 / 365, P, tol=1e-13)
    check("a tight tail tolerance triggers range extension",
          tight["extended"] >= 1, f"extended {tight['extended']}x to u_max {tight['u_max']:.0f}")

    pricer = rh.RoughPricer()
    pr = pricer(np.array([-0.1, 0.0, 0.1]), 0.25, rh.cf_factory(P, 0.25))
    check("RoughPricer prices through cf.params and keeps statistics",
          pr.shape == (3,) and np.all(np.isfinite(pr)) and pricer.stats["solves"] >= 1,
          f"{pricer.stats}")


def test_solver_robustness():
    """
    The three failures a Session E calibration found, each pinned down.

    1. The implicit scheme is NOT unconditionally stable (an earlier docstring
       said it was). Demonstrated, and the stability-sized step count shown to
       fix it for both schemes out to u = 1200.
    2. A solve that is under-stepped is CAUGHT by the checks and refined, rather
       than returned.
    3. At the parameters the calibration ran away to, the old Carr-Madan pricer
       returned 1e18; the Lewis pricer returns valid prices.
    """
    print("\nE0 -- solver robustness (found by the Session E calibration)")
    u = np.linspace(0.0, 1200.0, 1200) - 0.5j
    tau = 90 / 365
    bad = float(np.max(np.abs(rh.char_func(u, tau, P, scheme="exptrap", steps=120,
                                           check_stability=False, check_invariant=False))))
    check("exptrap at 120 steps blows up at u = 1200, 90 d, H = 0.12 -- NOT unconditionally stable",
          bad > 1e3, f"max |phi(u - i/2)| = {bad:.1e}, must be <= 1 (both guards off to show it)")

    # Session G: the measured constants are now HARD assertions
    need = rh.stability_steps(tau, 1200.0, P, min_steps=1, scheme="exptrap")
    try:
        rh.char_func(u, tau, P, scheme="exptrap", steps=120)
        refused, msg = False, "returned a value"
    except rh.StabilityError as exc:
        refused, msg = True, str(exc)
    check("the precondition refuses the under-stepped solve before doing any work (StabilityError)",
          refused and f"needs >= {need}" in msg, msg[:140])
    try:
        rh.char_func(u, tau, P, scheme="exptrap", steps=120, check_stability=False)
        caught = False
    except rh.StabilityError as exc:
        caught = "strip" in str(exc)
    check("with the precondition off, the |phi| <= 1 postcondition still refuses the result",
          caught)
    for sc in ("etdrk4", "exptrap"):
        h_edge = rh.max_stable_step(P, 1200.0, sc)
        M_ok = rh.stability_steps(tau, 1200.0, P, min_steps=1, scheme=sc)
        ok_edge = abs(tau / M_ok - h_edge) <= h_edge and tau / M_ok <= h_edge * (1 + 1e-9)
        check(f"{sc}: stability_steps and the guard use the same edge (step {tau / M_ok:.2e} <= {h_edge:.2e})",
              ok_edge)
    worst = 0.0
    for q in (P, rh.RoughHestonParams(0.02, 0.1868, 0.8828, 0.6347, -0.5574, 0.02)):
        for d_ in (30, 90):
            t = d_ / 365
            m_t = max(rh.AUTO_EXPTRAP_STEPS, rh.stability_steps(t, 1200.0, q, scheme="exptrap"))
            worst = max(worst, float(np.max(np.abs(rh.char_func(u, t, q, scheme="exptrap",
                                                                steps=m_t)))))
    check("stability-sized exptrap keeps |phi(u - i/2)| <= 1 to u = 1200 (H = 0.12 and 0.02)",
          worst <= 1.0 + 1e-9, f"worst {worst:.6f}")

    # 2. Force an unstable first pass by lying about the stability constant; the
    #    invariant / step-halving checks must refuse it and refine.
    saved = dict(rh.C_STAB)
    try:
        rh.C_STAB["exptrap"] = 1e4
        rh.C_STAB["etdrk4"] = 1e4
        ks = np.linspace(-0.3, 0.3, 7)
        c, info = rh.lewis_prices(ks, 1.0, P, u_max=1200.0, max_extend=0)
    finally:
        rh.C_STAB.clear()
        rh.C_STAB.update(saved)
    ref, _ = rh.lewis_prices(ks, 1.0, P, u_max=1200.0, max_extend=0)
    check("an under-stepped solve is caught and refined, not returned",
          info["refined"] >= 1 and (not info["ok"] or float(np.max(np.abs(c - ref))) < 1e-7),
          f"refined {info['refined']}x, ok={info['ok']}, "
          f"max |price - reference| {float(np.nanmax(np.abs(c - ref))) if info['ok'] else float('nan'):.1e}")

    # 3. The runaway parameters
    run = rh.RoughHestonParams(0.0200, 0.1868, 0.8828, 0.6347, -0.5574, 0.0200)
    lin = 2.5 * run.rho * run.xi - run.kappa
    check("at the runaway parameters E[S^2.5] explodes (Heston discriminant < 0) -- Carr-Madan invalid",
          lin * lin - run.xi ** 2 * 2.5 * 1.5 < 0, f"discriminant {lin * lin - run.xi ** 2 * 3.75:+.3f}")
    ok_all = True
    detail = []
    for d_ in (30, 90, 180, 365):
        t = d_ / 365
        band = 3 * 0.2 * math.sqrt(t)
        ks = np.linspace(-band, band, 9)
        c, info = rh.call_prices(ks, t, run)
        intrinsic = np.maximum(1.0 - np.exp(ks), 0.0)
        good = info["ok"] and np.all((c > intrinsic) & (c < 1.0))
        ok_all &= bool(good)
        detail.append(f"{d_}d {'ok' if good else 'BAD'}")
    check("...and the Lewis pricer returns in-band prices at 30, 90, 180 and 365 days",
          ok_all, ", ".join(detail) + " (the Carr-Madan pricer returned 1.9e18 and 2.9e28 at 30 and 90)")


def test_monte_carlo():
    """
    The independent check on the multi-factor coupling: the lifted simulation
    must agree with the lifted cf. The simulator itself is validated first at
    H = 1/2 against closed-form Heston. Rough-vol Monte Carlo converges slowly
    in the step count, so the rough case is checked for convergence TOWARDS the
    Fourier price as the steps double, not for agreement at a fixed step.
    """
    print("\nD4 -- Monte Carlo (lifted simulation) vs Fourier (lifted cf)")
    tau, k = 0.25, 0.0
    t0 = time.perf_counter()
    um, npan = rh.pricer_settings(PH, tau, 0.3)
    f_h = float(fo.carr_madan_call(k, tau, lambda u: heston_cf(u, tau, HP), u_max=um, n_panels=npan))
    m, se = rh.mc_call(PH, tau, k, n_paths=100_000, n_steps=500, seed=5)
    check("H = 1/2 through the lifted simulator matches the Heston price within 4 SE + Euler bias",
          abs(m - f_h) < 4 * se + 3e-4, f"MC {m:.6f} +/- {se:.6f} vs {f_h:.6f}  ({time.perf_counter()-t0:.0f}s)")

    # Mild vol-of-vol: the variance rarely touches zero, so the full-truncation
    # bias is negligible and the two must simply agree. Measured z = +0.22,
    # -0.14, -0.40, -0.12 at 250/500/1000/2000 steps.
    mild = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.2, -0.7, 0.12)
    f_r, _ = rh.call_prices(np.array([k]), tau, mild, tol=1e-10, steps_mult=2.0)
    f_r = float(f_r[0])
    t0 = time.perf_counter()
    m, se = rh.mc_call(mild, tau, k, n_paths=100_000, n_steps=500, seed=5)
    check("rough H = 0.12, xi = 0.2: lifted MC agrees with the lifted cf within 3 SE",
          abs(m - f_r) < 3 * se,
          f"MC {m:.6f} +/- {se:.6f} vs Fourier {f_r:.6f}, z = {(m-f_r)/se:+.2f}  ({time.perf_counter()-t0:.0f}s)")

    # Default vol-of-vol: at H = 0.12 and xi = 0.5 the variance hits zero often
    # and full truncation biases the price UP by O(dt^beta). Measured gaps
    # +8.2e-4, +5.3e-4, +3.2e-4, -0.8e-4 at 250/500/1000/2000 steps: it
    # converges to the Fourier price, slowly. Assert the convergence.
    f_d, _ = rh.call_prices(np.array([k]), tau, P, tol=1e-10, steps_mult=2.0)
    f_d = float(f_d[0])
    gaps = []
    for steps in (250, 1000):
        t0 = time.perf_counter()
        m, se = rh.mc_call(P, tau, k, n_paths=100_000, n_steps=steps, seed=5)
        gaps.append((steps, m - f_d, se, time.perf_counter() - t0))
    print("      xi = 0.5: " + "   ".join(f"{s} steps: gap {g:+.6f} (SE {e:.6f}, {t:.0f}s)"
                                         for s, g, e, t in gaps))
    check("rough H = 0.12, xi = 0.5: the MC-Fourier gap shrinks when the steps quadruple",
          abs(gaps[1][1]) < abs(gaps[0][1]) - gaps[1][2],
          f"{gaps[0][1]:+.2e} -> {gaps[1][1]:+.2e} (full-truncation bias, not a cf error)")


def test_positivity_scheme():
    """
    Session G, flag 3: the positivity-preserving step (LiftedAffineStep).
    Its closed forms against brute-force quadrature, its draws against its own
    exact moments -- one step, and 252 steps from far out of equilibrium -- and
    its prices against the Fourier cf on the case where Euler is most biased.
    """
    print("\nG -- positivity-preserving lifted step: exact moments, QE draw, pricing")
    from scipy import integrate
    w, x = rh.lift_nodes(0.12, 24)
    v0, kap, th, xi, h = 0.04, 3.0, 0.04, 0.3, 1 / 252
    st = rh.LiftedAffineStep(w, x, v0, kap, th, xi, h)
    U_star = st.U_from_y(st.y_star[:, None])[:, 0]
    g = rh._phi1(-x * h)
    U0 = U_star + g * ((0.08 - st.V_star) / float(w @ g))          # V0 = 2 theta, shock through the driver
    y0 = st.T @ U0
    lam, c = st.lam, st.c

    def EV(s_):
        return v0 + float(c @ (np.exp(-lam * s_) * y0 + st.alpha * (-np.expm1(-lam * s_)) / lam))

    def integ(r):
        pts = sorted(set([0.0, h] + [h - h * 10.0 ** (-j) for j in range(1, 9)] + [h * 10.0 ** (-j) for j in range(1, 9)]))
        return sum(integrate.quad(lambda s_: math.exp(-r * (h - s_)) * EV(s_), a, b, epsrel=1e-12, limit=200)[0]
                   for a, b in zip(pts[:-1], pts[1:]))
    Cy = st._cov_const + st._cov_lin @ y0
    worst = 0.0
    for i, j in ((0, 0), (0, 23), (5, 11), (12, 12), (17, 3), (23, 23)):
        q = xi ** 2 * c[i] * c[j] * integ(lam[i] + lam[j])
        worst = max(worst, abs(Cy[i, j] - q) / abs(q))
    check("covariance closed form (the Ito isometry in eigen-coordinates) = quadrature, 1e-10",
          worst < 1e-10, f"worst relative error {worst:.1e}")
    ei = st.EI_const + st.EI_lin @ y0
    ei_q = integrate.quad(EV, 0, h, epsrel=1e-13, points=[h * 1e-6, h * 1e-3])[0]
    check("E[integrated variance] closed form = quadrature, 1e-10", abs(ei - ei_q) / ei_q < 1e-10,
          f"{ei:.12e} vs {ei_q:.12e}")

    n = 400_000
    Yn, V = st.step_y(np.repeat(y0[:, None], n, axis=1), np.random.default_rng(3))
    m = v0 + c @ (st.e * y0 + st.mean_add)
    s2 = st.s2_const + st.s2_lin @ y0
    check("one step from V0 = 2 theta: E[V_h] matches the exact mean (4 SE)",
          abs(V.mean() - m) < 4 * math.sqrt(s2 / n), f"{V.mean():.6f} vs {m:.6f}")
    check("...and Var[V_h] matches the exact variance to 3%", abs(V.var() / s2 - 1) < 0.03,
          f"{100*(V.var()/s2-1):+.2f}%")
    check("V_h is never negative (a QE draw, not a floor)", float(V.min()) >= 0.0, f"min {V.min():.2e}")
    check("V_h = v0 + c'y_h exactly: the surprise is split without leaking into V",
          float(np.max(np.abs(v0 + c @ Yn - V))) < 1e-12)

    st2 = rh.LiftedAffineStep(w, x, 0.02, kap, 0.06, xi, 1 / 1008)
    A, b = st2.transition_U()
    n2 = 40_000
    yy = np.zeros((24, n2))
    mu, C = np.zeros(24), np.zeros((24, 24))
    rng = np.random.default_rng(4)
    rows, vmin = [], np.inf
    for k in range(1, 253):
        C = A @ C @ A.T + st2.cov_U(mu)
        mu = A @ mu + b
        yy, Vk = st2.step_y(yy, rng)
        vmin = min(vmin, float(Vk.min()))
        if k in (21, 252):
            Vm, Vv = 0.02 + w @ mu, float(w @ C @ w)
            rows.append((k, (Vk.mean() - Vm) / math.sqrt(Vv / n2), Vk.var() / Vv - 1))
    check("252 steps from v0 = 0.02 towards theta = 0.06: mean within 4 SE, variance within 5% of exact",
          all(abs(z) < 4 and abs(r) < 0.05 for _, z, r in rows) and vmin >= 0.0,
          "; ".join(f"step {k}: z {z:+.1f}, var {100*r:+.1f}%" for k, z, r in rows) + f"; min V {vmin:.1e}")

    tau = 0.25
    ks = np.array([-0.10, 0.0, 0.08])
    f, _ = rh.call_prices(ks, tau, P, tol=1e-10, steps_mult=2.0)
    t0 = time.perf_counter()
    Xq = rh.simulate_qe(P, tau, 100_000, 250, seed=5)
    tq = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xe = rh.simulate(P, tau, 100_000, 250, seed=5)
    te = time.perf_counter() - t0
    Sq, Se = np.exp(Xq), np.exp(Xe)
    zq, ze = [], []
    for kk, fk in zip(ks, f):
        for S_, out in ((Sq, zq), (Se, ze)):
            pay = np.maximum(S_ - math.exp(kk), 0.0)
            out.append(((pay.mean() - fk), pay.std() / math.sqrt(len(pay))))
    print("      250 steps, xi = 0.5, H = 0.12: " + "  ".join(
        f"k={kk:+.2f}: QE {gq:+.1e} ({gq/sq:+.1f} SE) vs Euler {ge:+.1e} ({ge/se_:+.1f} SE)"
        for kk, (gq, sq), (ge, se_) in zip(ks, zq, ze)) + f"  ({tq:.0f}s / {te:.0f}s)")
    check("QE prices match the Fourier cf within 4.5 SE at every strike (250 steps)",
          all(abs(gq / sq) < 4.5 for gq, sq in zq))
    check("QE's worst price bias is < 1/3 of full-truncation Euler's at the same steps",
          max(abs(gq) for gq, _ in zq) < max(abs(ge) for ge, _ in ze) / 3.0,
          f"{max(abs(gq) for gq, _ in zq):.1e} vs {max(abs(ge) for ge, _ in ze):.1e}")
    check("QE is a martingale: E[S] = F within 3 SE", abs(Sq.mean() - 1) < 3 * Sq.std() / math.sqrt(len(Sq)),
          f"{Sq.mean() - 1:+.1e}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session D -- Volterra kernel, Markovian lift, lifted rough Heston cf")
    print("=" * 74)
    t0 = time.perf_counter()
    test_kernel_representation()
    test_sum_of_exponentials()
    test_lifted_simulation()
    test_reduces_to_heston()
    test_h_to_half()
    test_cf_properties()
    test_pricing_wrapper()
    test_solver_robustness()
    test_monte_carlo()
    test_positivity_scheme()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter()-t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
