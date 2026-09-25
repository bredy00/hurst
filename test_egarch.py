"""
Session L -- EGARCH with Student-t innovations, Beta-t-EGARCH, and GARCH (models/egarch.py).

  densities     the unit-variance Student-t integrates to one, has unit variance, and
                E|z| matches its closed form; the normal limit is recovered at large nu
  maps          the parameter maps are inverse to each other; every unconstrained point
                gives a stationary log-variance autoregression (p = 2 through the PACF)
  likelihood    the filter's log-likelihood equals a direct evaluation on a short series
  recovery      simulated EGARCH(1,1)-t, Beta-t-EGARCH and GARCH(1,1) are recovered within
                3 SE of the planted parameters
  selection     BIC prefers the t to the normal on t data, and EGARCH to GARCH on data
                with leverage
  robustness    one extreme return moves Beta-t's log-scale by a bounded amount and
                Nelson's without bound
  forecasts     GARCH's closed-form forecast agrees with simulation; the EGARCH
                simulation's first step is the filter's own forecast

    python test_egarch.py
"""

import math
import sys

import numpy as np
from scipy import integrate

import models.egarch as eg

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_densities():
    print("\ndensities")
    for nu in (3.5, 6.0, 30.0):
        f = lambda z: math.exp(float(eg.log_density(z, "t", nu)))
        mass = integrate.quad(f, -np.inf, np.inf)[0]
        var = integrate.quad(lambda z: z * z * f(z), -np.inf, np.inf)[0]
        mabs = integrate.quad(lambda z: abs(z) * f(z), -np.inf, np.inf)[0]
        check(f"unit-variance t(nu = {nu}): mass 1, variance 1, E|z| closed form",
              abs(mass - 1) < 1e-8 and abs(var - 1) < 1e-6 and abs(mabs - eg.abs_moment("t", nu)) < 1e-8,
              f"mass {mass:.10f}, variance {var:.8f}, E|z| {mabs:.8f} vs {eg.abs_moment('t', nu):.8f}")
    check("the t tends to the normal: E|z| at nu = 1e6 within 1e-6 of sqrt(2/pi)",
          abs(eg.abs_moment("t", 1e6) - math.sqrt(2 / math.pi)) < 1e-6)


def test_maps():
    print("\nparameter maps")
    rng = np.random.default_rng(0)
    worst = 0.0
    stationary = True
    for spec in (eg.Spec("nelson", 2, 2, "t"), eg.Spec("nelson", 1, 1, "normal"), eg.Spec("beta-t"),
                 eg.Spec("garch", 1, 1, "t")):
        for _ in range(200):
            x = rng.uniform(-3, 3, spec.k)
            th = eg.to_natural(x, spec)
            worst = max(worst, float(np.max(np.abs(eg.to_unconstrained(th, spec) - x))))
            if spec.family == "nelson" and spec.p == 2:
                roots = np.roots([1.0, -th[5], -th[6]])          # z^2 - b1 z - b2
                stationary &= bool(np.all(np.abs(roots) < 1))
            if spec.family == "garch":
                stationary &= eg.persistence(th, spec) < 1 and bool(np.all(th[1:3] > 0))
    check("unconstrained -> natural -> unconstrained is the identity", worst < 1e-9, f"worst {worst:.1e}")
    check("every unconstrained point is admissible (stationary AR(2), GARCH persistence < 1)", stationary)


def test_likelihood():
    print("\nlikelihood")
    r = np.array([0.01, -0.02, 0.005, 0.03, -0.015])
    spec = eg.Spec("nelson", 1, 1, "t")
    th = np.array([-0.4, 0.1, -0.05, 0.96, 6.0])
    ez = eg.abs_moment("t", 6.0)
    ls = math.log(np.var(r))
    direct = 0.0
    z_prev, a_prev = 0.0, ez
    for t in range(len(r)):
        ls = th[0] + th[3] * ls + th[1] * (a_prev - ez) + th[2] * z_prev
        z = r[t] / math.exp(0.5 * ls)
        direct += float(eg.log_density(z, "t", 6.0)) - 0.5 * ls
        z_prev, a_prev = z, abs(z)
    got = eg.loglik(th, r, spec)
    check("EGARCH(1,1)-t log-likelihood equals a direct evaluation", abs(got - direct) < 1e-12,
          f"{got:.12f} vs {direct:.12f}")
    spec_b = eg.Spec("beta-t")
    tb = np.array([-4.0, 0.95, 0.06, 0.02, 5.0])
    lam, direct = tb[0], 0.0
    nu = tb[4]
    for y in r:
        direct += (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * nu) - lam
                   - 0.5 * (nu + 1) * math.log(1 + y * y / (nu * math.exp(2 * lam))))
        u = (nu + 1) * y * y / (nu * math.exp(2 * lam) + y * y) - 1
        lam = tb[0] * (1 - tb[1]) + tb[1] * lam + tb[2] * u + tb[3] * math.copysign(1.0, -y) * (u + 1)
    got = eg.loglik(tb, r, spec_b)
    check("Beta-t-EGARCH log-likelihood equals a direct evaluation", abs(got - direct) < 1e-12,
          f"{got:.12f} vs {direct:.12f}")


def test_recovery():
    print("\nrecovery (4000 simulated days each)")
    rng = np.random.default_rng(11)
    cases = [(eg.Spec("nelson", 1, 1, "t"), np.array([-0.25, 0.12, -0.07, 0.975, 7.0])),
             (eg.Spec("beta-t"), np.array([-4.8, 0.98, 0.05, 0.02, 6.0])),
             (eg.Spec("garch", 1, 1, "normal"), np.array([2e-6, 0.08, 0.9]))]
    for spec, truth in cases:
        r, _ = eg.simulate(truth, spec, 4000, rng)
        f = eg.fit(r, spec)
        z = (f["theta"] - truth) / f["se"]
        # omega and the persistence are nearly collinear in EGARCH; the pair's combination
        # omega / (1 - beta), the mean log-variance, is what the data pin down
        check(f"{spec.name}: every parameter within 3 SE of the truth", bool(np.all(np.abs(z) < 3)),
              ", ".join(f"{n} {v:.4g} ({zz:+.1f} SE)" for n, v, zz in zip(spec.param_names, f["theta"], z)))


def test_selection():
    print("\nselection by BIC")
    rng = np.random.default_rng(5)
    truth = np.array([-0.25, 0.12, -0.08, 0.975, 5.0])
    r, _ = eg.simulate(truth, eg.Spec("nelson", 1, 1, "t"), 3000, rng)
    rows = eg.scan(r, specs=(eg.Spec("nelson", 1, 1, "t"), eg.Spec("nelson", 1, 1, "normal"),
                             eg.Spec("garch", 1, 1, "t"), eg.Spec("garch", 1, 1, "normal")), se=False)
    check("on EGARCH(1,1)-t data with leverage, BIC ranks EGARCH(1,1)-t first", rows[0]["spec"] == "EGARCH(1,1)-t",
          "; ".join(f"{z['spec']} {z['bic']:.1f}" for z in rows))


def test_robustness():
    print("\nrobustness to one extreme return")
    r = np.random.default_rng(2).standard_normal(300) * 0.01
    nelson = eg.Spec("nelson", 1, 1, "t")
    th_n = np.array([-0.3, 0.12, -0.05, 0.97, 6.0])
    beta = eg.Spec("beta-t")
    th_b = np.array([-4.6, 0.97, 0.05, 0.02, 6.0])
    jumps = {}
    for size in (0.1, 1.0, 10.0):
        rr = r.copy()
        rr[150] = -size
        dn = math.log(eg.variance_path(th_n, rr, nelson)[151]) - math.log(eg.variance_path(th_n, r, nelson)[151])
        db = math.log(eg.variance_path(th_b, rr, beta)[151]) - math.log(eg.variance_path(th_b, r, beta)[151])
        jumps[size] = (dn, db)
    # the news term kappa u + kappa_s sgn (u + 1) lies in [-kappa - |kappa_s|(nu+1), kappa nu +
    # |kappa_s|(nu+1)], so replacing one return moves the log-scale by at most
    # (nu + 1)(kappa + 2|kappa_s|), and the log-variance by twice that
    bound = 2 * 7.0 * (0.05 + 2 * 0.02)
    check("Beta-t's response to a -10 (1000 sigma) return is within its bound 2 (nu+1)(kappa + 2|kappa_s|)",
          jumps[10.0][1] <= bound + 1e-12, f"{jumps[10.0][1]:.3f} <= {bound:.3f}")
    # (at -10 Nelson's log-variance hits the filter's clamp, LOG_S2_CLAMP = 60)
    check("...while Nelson's grows in proportion (x10 return, >8x response)",
          jumps[1.0][0] > 8 * jumps[0.1][0] > 0,
          "; ".join(f"return -{s:g}: Nelson {a:+.2f}, Beta-t {b:+.3f}" for s, (a, b) in jumps.items()))


def test_forecasts():
    print("\nforecasts")
    rng = np.random.default_rng(8)
    spec = eg.Spec("garch", 1, 1, "normal")
    th = np.array([2e-6, 0.08, 0.9])
    r, _ = eg.simulate(th, spec, 500, rng)
    closed = eg.forecast(th, r, spec, 10)
    # simulate forward from the same state
    s2 = eg.variance_path(th, r, spec)[-1]
    n = 200_000
    s = np.full(n, s2)
    sims = []
    for _ in range(10):
        sims.append(s.mean())
        y = np.sqrt(s) * rng.standard_normal(n)
        s = th[0] + th[1] * y * y + th[2] * s
    rel = float(np.max(np.abs(np.array(sims) / closed - 1)))
    check("GARCH(1,1) closed-form forecast = simulation within 1% over 10 days", rel < 0.01, f"{rel:.2%}")
    spec_n = eg.Spec("nelson", 2, 2, "t")
    th_n = np.array([-0.2, 0.08, 0.04, -0.05, -0.02, 0.6, 0.37, 7.0])
    rn, _ = eg.simulate(th_n, spec_n, 800, rng)
    f = eg.forecast(th_n, rn, spec_n, 20)
    first = eg.variance_path(th_n, rn, spec_n)[-1]
    check("EGARCH(2,2)-t simulated forecast starts at the filter's one-step forecast",
          abs(f[0] / first - 1) < 1e-12, f"{f[0]:.6e} vs {first:.6e}")
    lr = math.exp(th_n[0] / (1 - th_n[5] - th_n[6]))
    check("...and moves toward the unconditional level over 20 days",
          abs(math.log(f[-1] / lr)) < abs(math.log(f[0] / lr)) + 0.05,
          f"day 1 {f[0]:.3e}, day 20 {f[-1]:.3e}, exp(mean log s2) {lr:.3e}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session L -- EGARCH-t, Beta-t-EGARCH, GARCH")
    print("=" * 74)
    test_densities()
    test_maps()
    test_likelihood()
    test_recovery()
    test_selection()
    test_robustness()
    test_forecasts()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
