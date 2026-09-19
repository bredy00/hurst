"""
Session I -- the zero-boundary filtering protocol for daily realised variance.

  transform     the joint transform of (V_h, int V) is the variance cf's Riccati with a
                running term: on one factor (Heston) it matches an independent ODE solve
  panels        a day with a small RV needs frequencies zeta ~ 1e7; the spot filter's
                240-step mesh blows up there, the stability-sized panel mesh does not
  rv filter     near zero the cf filter's RV likelihood is within 0.05 nats a day of the
                particle filter's (exact for the QE law up to Monte Carlo), and closer to it
                than the Kalman filter's
  protocol      a host well away from zero stays with the Kalman filter; a host at the
                boundary switches to the cf filter, with the particle filter confirming

    python test_protocol.py
"""

import math
import sys
import time

import numpy as np

import filters.fourier as ff
import filters.protocol as proto
import models.rough_heston as rh

PASS, FAIL = [], []
DT, M = 1.0 / 252, 78


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def synth_rv(p, n, seed, H=0.12, substeps=16):
    """Daily RV from the QE step with 16 substeps a day, plus R and RV's sampling error."""
    w, x = rh.lift_nodes(H)
    st = rh.LiftedAffineStep(w, x, 0.04, p["kappa"], p["theta"], p["xi"], DT / substeps)
    rng = np.random.default_rng(seed)
    y = st.y_star[:, None].copy()
    for _ in range(40 * substeps):
        y, _ = st.step_y(y, rng)
    out = np.empty(n)
    for t in range(n):
        I = 0.0
        for _ in range(substeps):
            y, V, dI, dZ, fix = st.step_y(y, rng, with_log_price=True)
            I += float(dI[0])
        iv = I / DT
        out[t] = iv + math.sqrt(p["R"] + 2.0 / M * iv * iv) * rng.standard_normal()
    return out


def test_transform():
    print("\nThe joint transform of (V_h, int V): running term against an ODE, one factor")
    from scipy.integrate import solve_ivp
    kap, th, xi, v0, h = 3.0, 0.04, 0.5, 0.04, DT

    def ref(z, zeta):
        def f(t, s):
            b, a = s[0] + 1j * s[1], s[2] + 1j * s[3]
            db = -kap * b + 0.5 * xi * xi * b * b + 1j * zeta
            da = kap * (th - v0) * b + v0 * 0.5 * xi * xi * b * b + 1j * zeta * v0
            return [db.real, db.imag, da.real, da.imag]
        r = solve_ivp(f, (0, h), [0.0, z, 0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14)
        return r.y[2, -1] + 1j * r.y[3, -1], r.y[0, -1] + 1j * r.y[1, -1]
    worst = 0.0
    for z in (0.0, 5.0, 50.0):
        for zeta in (0.0, 1e2, 1e4, 1e6):
            a, b = ff.variance_cf_coefficients(np.array([z]), h, np.array([1.0]), np.array([0.0]), v0, kap, th, xi,
                                               zeta=np.array([zeta]))
            ar, br = ref(z, zeta)
            worst = max(worst, max(abs(a[0] - ar), abs(b[0, 0] - br)) / max(1.0, abs(br)))
    check("Heston (one factor, x = 0): a and b match an 8th-order ODE solve to 1e-7", worst < 1e-7, f"worst {worst:.1e}")


def test_panels():
    print("\nFrequency panels: meshes sized by the transform's stability edge")
    import warnings
    w, x = rh.lift_nodes(0.12)
    kap, th, xi, v0 = 3.0, 0.04, 0.5, 0.04
    U = np.random.default_rng(0).normal(0.0, 0.01, len(w))
    zeta = np.array([1e7])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a0, b0 = ff.variance_cf_coefficients(np.array([0.0]), DT, w, x, v0, kap, th, xi, M=240, zeta=zeta)
    flt = ff.FourierFilter(DT, observation="rv", xi_cap=xi, kappa_cap=kap)
    Mp = flt._panel_steps(1e7)
    a1, b1 = ff.variance_cf_coefficients(np.array([0.0]), DT, w, x, v0, kap, th, xi, M=Mp, grade=2.0, zeta=zeta)
    a2, b2 = ff.variance_cf_coefficients(np.array([0.0]), DT, w, x, v0, kap, th, xi, M=4 * Mp, grade=2.0, zeta=zeta)
    l0, l1, l2 = a0[0] + U @ b0[:, 0], a1[0] + U @ b1[:, 0], a2[0] + U @ b2[:, 0]
    check("at zeta = 1e7 the spot filter's 240-step graded mesh does not survive", not np.isfinite(l0),
          f"{l0}")
    check("...the stability-sized panel mesh does, within 1e-4 of a 4x finer one", abs(l1 - l2) < 1e-4,
          f"{Mp} steps; |diff| {abs(l1 - l2):.1e}; log-transform {l2:.4f}")


def test_rv_filter_and_protocol():
    print("\nThe RV filters and the protocol")
    away = dict(kappa=3.0, theta=0.04, xi=0.08, R=1e-8)
    near = dict(kappa=3.0, theta=0.04, xi=0.25, R=1e-8)
    t0 = time.perf_counter()
    y_away = synth_rv(away, 150, seed=7)
    out = proto.run(away, y_away, confirm=False)
    check("a host well away from zero: the boundary does not bind and the state estimate stays Kalman",
          not out["boundary"]["binds"] and out["state_estimate"] == "kalman" and "cf" not in out,
          f"boundary share {100 * out['boundary']['share']:.1f}% ({time.perf_counter() - t0:.0f}s)")

    t0 = time.perf_counter()
    y_near = synth_rv(near, 150, seed=7)
    out = proto.run(near, y_near, confirm=True, n_particles=1500)
    c = out.get("confirmation", {})
    check("a host near zero: the boundary binds and the cf filter takes the state estimate",
          out["boundary"]["binds"] and out["state_estimate"] == "cf",
          f"boundary share {100 * out['boundary']['share']:.1f}%")
    check("...with no day falling back to the Gaussian update and none beyond the frequency cap",
          out["cf"]["fallback_days"] == 0 and out["cf"]["capped_days"] == 0,
          f"fallback {out['cf']['fallback_days']}, capped {out['cf']['capped_days']}, {out['cf']['panels']} panels")
    check("the cf log-likelihood is within 0.05 nats a day of the particle filter's",
          abs(c.get("cf_minus_particle_per_day", 1.0)) < 0.05,
          f"cf - particle {c.get('cf_minus_particle', float('nan')):+.2f} over 150 days; "
          f"Kalman - particle {c.get('kalman_minus_particle', float('nan')):+.2f}")
    check("...and closer to it than the Kalman filter's",
          abs(c.get("cf_minus_particle", 1e9)) < abs(c.get("kalman_minus_particle", 0.0)))
    check("the protocol's confirmation holds (seeds agree, likelihood and filtered RV agree)",
          c.get("confirmed", False), f"{c.get('checks')} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    print("=" * 74)
    print("Session I -- zero-boundary filtering protocol")
    print("=" * 74)
    t0 = time.perf_counter()
    test_transform()
    test_panels()
    test_rv_filter_and_protocol()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter()-t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
