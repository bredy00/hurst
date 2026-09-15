"""
Session H -- non-Gaussian filtering of rough variance at the zero boundary.

  pieces        sorted systematic resampling keeps every count within one of P w_i; the
                particle filter's fixed-shape noise reproduces the QE step draw for draw
  densities     the adapted particle filter's predictive density (QE law convolved with
                the observation noise, atom included) and the characteristic-function
                filter's Fourier-inverted density each integrate to one
  the cf        the variance cf's Riccati against the exact affine conditional mean and
                variance -- two independent derivations -- and |cf| <= 1
  limits        away from zero both filters agree with the exact-moment Kalman filter;
                near zero both gain tens of nats over it and filter V more accurately
  estimation    common random numbers make the particle likelihood reproducible and
                smooth enough in kappa to profile

    python test_nongaussian.py
"""

import math
import sys
import time

import numpy as np

import filters.fourier as ff
import filters.kalman as kf
import filters.particle as pfm
import models.rough_heston as rh

PASS, FAIL = [], []
DT = 1.0 / 252
NEAR = dict(kappa=3.0, theta=0.04, xi=0.3, R=0.01 ** 2)
AWAY = dict(kappa=3.0, theta=0.09, xi=0.08, R=0.01 ** 2)


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def data(p, T, seed=61):
    m = kf.LiftedRoughModel(DT, H=0.12, v0=p["theta"])
    V, _ = kf.simulate_rough(m, p, T, seed=seed)
    return m, V, V + np.random.default_rng(600 + seed).normal(0.0, math.sqrt(p["R"]), T)


# --- pieces -------------------------------------------------------------------------
def test_pieces():
    print("\nResampling and the particle filter's noise")
    rng = np.random.default_rng(1)
    P = 1000
    logw = rng.normal(0.0, 2.0, P)
    key = rng.random(P)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    worst = 0.0
    for u in (0.0, 0.37, 0.999):
        counts = np.bincount(pfm.sorted_systematic(key, logw, u), minlength=P)
        worst = max(worst, float(np.max(np.abs(counts - P * w))))
    check("systematic resampling: every particle's count within one of P w_i", worst <= 1.0 + 1e-9, f"worst {worst:.3f}")
    idx = pfm.sorted_systematic(key, np.zeros(P), 0.5)
    check("equal weights: each particle exactly once", np.array_equal(np.sort(idx), np.arange(P)))
    # Sorting is what makes the likelihood smooth in the parameters: when a small change
    # in the weights flips an ancestor, the replacement is the NEIGHBOUR in V, not a
    # random particle. Measured as how far the resampled keys move.
    bump = logw + 0.001 * rng.standard_normal(P)
    moved = lambda i, j: float(np.mean(np.abs(np.sort(key[i]) - np.sort(key[j]))))
    srt = moved(pfm.sorted_systematic(key, logw, 0.3), pfm.sorted_systematic(key, bump, 0.3))
    perm = rng.permutation(P)
    uns = moved(perm[pfm.sorted_systematic(np.arange(P), logw[perm], 0.3)],
                perm[pfm.sorted_systematic(np.arange(P), bump[perm], 0.3)])
    check("a small change in the weights moves the resampled set 10x less when sorted by the key",
          srt * 10 < uns, f"mean key shift {srt:.2e} sorted vs {uns:.2e} in random order")

    w24, x24 = rh.lift_nodes(0.12, 24)
    st = rh.LiftedAffineStep(w24, x24, 0.04, 3.0, 0.04, 0.3, DT / 4)
    n = 5000
    Y = np.repeat(st.y_star[:, None], n, axis=1) * (1.0 + 0.3 * np.random.default_rng(2).standard_normal((1, n)))
    r = st.L_perp.shape[1]
    g = np.random.default_rng(5)
    z, u, zp = g.standard_normal(n), g.random(n), g.standard_normal((r, n))
    full = np.zeros((len(w24), n))
    full[:r] = zp
    y1, V1, _ = st.step_given(Y, z, u, full)
    y2, V2 = st.step_y(Y, np.random.default_rng(5))
    check("step_given with the same draws is the QE step exactly", float(np.max(np.abs(V1 - V2))) == 0.0
          and float(np.max(np.abs(y1 - y2))) == 0.0)


# --- densities ------------------------------------------------------------------------
def test_densities():
    print("\nPredictive densities integrate to one")
    ys = np.linspace(-0.12, 0.40, 521)
    dy = ys[1] - ys[0]
    for label, p in (("near zero", NEAR), ("away from zero", AWAY)):
        apf = pfm.AdaptedParticleFilter(DT, H=0.12, v0=p["theta"], n_particles=200, burn_days=5, seed=3)
        dens = np.array([math.exp(apf.loglik(p, [y])) for y in ys])
        mass = float(np.sum(dens) * dy)
        check(f"{label}: adapted particle filter, one day ahead (QE law * noise, atom included)", abs(mass - 1) < 2e-3,
              f"mass {mass:.5f}")
        fil = ff.FourierFilter(DT, H=0.12, v0=p["theta"])
        dens = np.array([math.exp(fil.loglik(p, [y])) for y in ys])
        mass = float(np.sum(dens) * dy)
        check(f"{label}: characteristic-function filter, Fourier-inverted density", abs(mass - 1) < 2e-3,
              f"mass {mass:.5f}, min {dens.min():.1e}")


# --- the cf ---------------------------------------------------------------------------------
def test_cf():
    print("\nThe variance cf against the exact affine moments")
    w24, x24 = rh.lift_nodes(0.12, 24)
    worst = 0.0
    for kap, th, xi in ((3.0, 0.04, 0.3), (3.0, 0.09, 0.08), (1.0, 0.02, 0.6)):
        st = rh.LiftedAffineStep(w24, x24, th, kap, th, xi, DT)
        A, b = st.transition_U()
        Us = st.U_from_y(st.y_star[:, None])[:, 0]
        for U in (Us, Us + 0.002 * np.random.default_rng(0).standard_normal(24)):
            dz = 1e-3
            z = np.array([-dz, 0.0, dz])
            a, B = ff.variance_cf_coefficients(z, DT, w24, x24, th, kap, th, xi)
            L = 1j * z * th + a + B.T @ U
            mean_cf = ((L[2] - L[0]) / (2 * dz) / 1j).real
            var_cf = -((L[2] - 2 * L[1] + L[0]) / dz ** 2).real
            mean_ex, var_ex = th + float(w24 @ (A @ U + b)), float(w24 @ st.cov_U(U) @ w24)
            worst = max(worst, abs(mean_cf / mean_ex - 1), abs(var_cf / var_ex - 1))
    check("mean and variance of V one day ahead: Riccati vs eigen-coordinate moments, 3 parameter sets x 2 states",
          worst < 1e-6, f"worst relative gap {worst:.1e}")
    st = rh.LiftedAffineStep(w24, x24, 0.04, 3.0, 0.04, 0.3, DT)
    U = st.U_from_y(st.y_star[:, None])[:, 0]
    z = np.linspace(0.0, 900.0, 500)
    a, B = ff.variance_cf_coefficients(z, DT, w24, x24, 0.04, 3.0, 0.04, 0.3)
    mod = np.abs(np.exp(1j * z * 0.04 + a + B.T @ U))
    check("|cf| <= 1 on real z up to the filter's cut-off (9 / sqrt(R) = 900)", float(mod.max()) <= 1.0 + 1e-9,
          f"max {mod.max():.12f}; cf(0) = {mod[0]:.12f}")


# --- limits -----------------------------------------------------------------------------------
def test_limits():
    print("\nAgainst the Kalman filter, away from zero and near it (400 days)")
    T = 400
    out = {}
    for label, p in (("away", AWAY), ("near", NEAR)):
        m, V, y = data(p, T)
        t0 = time.perf_counter()
        rk = kf.kalman_filter(m, p, y)
        rf = ff.FourierFilter(DT, H=0.12, v0=p["theta"]).run(p, y, keep_path=True)
        rp = pfm.AdaptedParticleFilter(DT, H=0.12, v0=p["theta"], n_particles=500, seed=1).run(p, y, keep_path=True)
        rmse = lambda e: float(np.sqrt(np.mean((e - V) ** 2)))
        out[label] = dict(zero=float(np.mean(V < 1e-12)), llk=rk["loglik"], llf=rf["loglik"], llp=rp["loglik"],
                          ek=rmse(rk["estimate"]), ef=rmse(rf["mean_V"]), ep=rmse(rp["mean_V"]),
                          cover=float(np.mean((V >= rp["q05"]) & (V <= rp["q95"]))), secs=time.perf_counter() - t0)
    a, n = out["away"], out["near"]
    check("away from zero: both filters within 3 nats of Kalman", abs(a["llf"] - a["llk"]) < 3 and abs(a["llp"] - a["llk"]) < 3,
          f"CF {a['llf'] - a['llk']:+.2f}, particle {a['llp'] - a['llk']:+.2f}")
    check("away from zero: filtered-variance RMSE within 2% of Kalman's", max(abs(a["ef"] / a["ek"] - 1), abs(a["ep"] / a["ek"] - 1)) < 0.02,
          f"Kalman {1e3 * a['ek']:.3f}, CF {1e3 * a['ef']:.3f}, particle {1e3 * a['ep']:.3f} (x 1e-3)")
    check(f"near zero (V = 0 on {100 * n['zero']:.0f}% of days): particle filter gains > 20 nats over Kalman",
          n["llp"] - n["llk"] > 20, f"{n['llp'] - n['llk']:+.1f}")
    check("near zero: the CF filter's likelihood within 6 nats of the particle filter's", abs(n["llf"] - n["llp"]) < 6,
          f"{n['llf'] - n['llp']:+.2f}")
    check("near zero: both non-Gaussian filters track V > 5% better than Kalman",
          n["ef"] < 0.95 * n["ek"] and n["ep"] < 0.95 * n["ek"],
          f"Kalman {1e3 * n['ek']:.3f}, CF {1e3 * n['ef']:.3f}, particle {1e3 * n['ep']:.3f} (x 1e-3)")
    check("near zero: the particle filter's 90% band covers V 80-97% of days", 0.80 <= n["cover"] <= 0.97, f"{n['cover']:.3f}")


def test_exponential_branch():
    print("\nThe exponential branch's density, beta from 10 to 1e9")
    R = 1e-4
    worst = 0.0
    for beta in (10.0, 1e3, 5e4, 1e6):
        v = np.linspace(0.0, 40.0 / beta, 200001)
        for y in (-0.01, 0.0, 0.004, 0.03):
            f = beta * np.exp(-beta * v) * np.exp(-0.5 * (y - v) ** 2 / R) / math.sqrt(2 * math.pi * R)
            ref = float(np.trapezoid(f, v))
            worst = max(worst, abs(float(pfm.exp_gauss_density(np.array([beta]), y, R)[0]) / ref - 1))
    check("against direct integration, including where exp(beta^2 R / 2) overflows", worst < 1e-5, f"worst {worst:.1e}")
    lim = pfm.exp_gauss_density(np.array([1e9, 1e12]), 0.004, R)
    gauss = math.exp(-0.5 * 0.004 ** 2 / R) / math.sqrt(2 * math.pi * R)
    check("tends to the observation density N(y; 0, R) as beta -> infinity (a point mass at zero)",
          bool(np.all(np.abs(lim / gauss - 1) < 1e-6)), f"{lim / gauss - 1}")


# --- estimation ----------------------------------------------------------------------------------
def test_common_random_numbers():
    print("\nCommon random numbers")
    m, V, y = data(NEAR, 300, seed=63)
    apf = pfm.AdaptedParticleFilter(DT, H=0.12, v0=0.04, n_particles=300, seed=7)
    l1, l2 = apf.loglik(NEAR, y), apf.loglik(NEAR, y)
    check("the particle likelihood is reproducible for a fixed seed", l1 == l2, f"{l1:.6f}")
    ks = np.linspace(2.9, 3.1, 5)
    ll = np.array([apf.loglik(dict(NEAR, kappa=k), y) for k in ks])
    rough = float(np.max(np.abs(np.diff(ll, 2))))
    check("second differences in kappa (step 0.05) below 0.5 nats: smooth enough to profile", rough < 0.5,
          f"{np.round(ll - ll[2], 3)}")


if __name__ == "__main__":
    print("=" * 74)
    print("Session H -- non-Gaussian filters at the zero boundary")
    print("=" * 74)
    t0 = time.perf_counter()
    test_pieces()
    test_densities()
    test_cf()
    test_limits()
    test_exponential_branch()
    test_common_random_numbers()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.perf_counter() - t0:.0f}s)")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
