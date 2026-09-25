"""
Should the lift go from 40 nodes to 44? (Session L, 25 September 2026)

Akin's decision on 25 September: adopt 44 nodes, and "compare and overcome" -- measure
what the four extra nodes buy, measure what they cost, and pay for them. Candidates: the
default since Session I (N = 40), the proposal (N = 44), and a bracket (N = 48), all with
the fastest node at 1e8/y. `rh.using_lift` swaps the lift for the whole stack, so the
numbers are what adoption would actually do.

  K  the review's criterion: the worst kernel error over the whole H box [0.02, 0.5]
  A  prices against the TRUE rough Heston (fractional Adams, M and 2M steps for its own
     error) at H = 0.02, 0.05, 0.12 and 1, 7, 30 days
  B  the kernel at H = 0.05 and 0.12: sup-norm error, Ito isometry, variogram exponent
     (study_finer_lift.section_b, unchanged)
  C  cost through the production code (study_finer_lift.section_c, unchanged)
  D1 the calibrated H from a surface priced by the true model (study_finer_lift, 40 and 44)
  E  the lift against EXACT fractional Gaussian noise -- the part that does not need a
     pricer. The lifted driver started in the infinite past, Y_t = int K_N(t - s) dW_s,
     has stationary increments; with the exact kernel those increments are
     Mandelbrot-Van Ness fBm's, fGn with variance V_H h^{2H}, V_H = 1 / (Gamma(2H+1)
     sin(pi H)). So the lift can be graded against fGn directly:
     E1 its variogram against V_H h^{2H}, lag by lag, and its local exponent against 2H;
     E2 how much data a likelihood-ratio test would need to tell the lift's increments
        from exact fGn, sampled every second to every day. Both spectral densities are
        closed form: fGn's through the Hurwitz zeta function, the lift's as a sum of
        AR(1) spectra, one per node. The Whittle KL divergence over n Fourier
        frequencies is the expected log-likelihood ratio; 5.41 is where a 5%-size test
        reaches 95% power;
     E3 Monte Carlo by Shevchenko's circulant embedding (models/fbm.py): exact fGn, and
        the lift's increments drawn by the SAME algorithm from their own covariance.
        Shevchenko's quadratic-variation estimator reads H from both, and the Whittle
        log-likelihood ratio on the draws is checked against E2's closed form.
  F  what E implies: F1 the data needed to tell the 40-node lift from the 44-node one;
     F2 the lever E points at, the fastest node (1e8 to 1e14/y at the same node density)

    python study_lift_44.py K A B E F    (~25 minutes)
    python study_lift_44.py C            (~5 minutes; nothing else loading the CPU)
    python study_lift_44.py D1           (~1 hour)
    -> captures/lift_44.json (sections merge), captures/lift_44.log
"""

import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy import integrate, special

import models.fbm as fbm
import models.rough_heston as rh
import models.rough_heston_adams as ad
import pricing.fourier as fo
import study_finer_lift as sfl

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "lift_44.json"
LIFTS = {"N=40, 1e8": (40, 1e8), "N=44, 1e8": (44, 1e8), "N=48, 1e8": (48, 1e8)}
YEAR_S = 365.0 * 24 * 3600
SAMPLING = {"1 s": 1.0, "1 min": 60.0, "5 min": 300.0, "1 h": 3600.0, "1 d": 86400.0}
POWER_KL = 0.5 * (2 * 1.6448536269514722) ** 2      # 5% size, 95% power, Gaussian LLR
_NODES = rh._LIFT_NODES


def say(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ K: the box
def section_k():
    Hs = np.linspace(0.02, 0.49, 48)
    out = {}
    for name, (N, eta) in LIFTS.items():
        e_d = [rh.kernel_error(H, N, eta_N=eta)[0] for H in Hs]
        e_h = [rh.kernel_error(H, N, t_lo=1 / (365 * 24), eta_N=eta)[0] for H in Hs]
        i = int(np.argmax(e_d))
        out[name] = {"worst_1d_2y": float(max(e_d)), "at_H": float(Hs[i]), "worst_1h_2y": float(max(e_h)),
                     "headroom_to_1pct": float(1 - max(e_d) / 0.01)}
        say(f"K  {name}: worst kernel error on [1 d, 2 y] {100 * max(e_d):.3f}% at H = {Hs[i]:.2f}, "
            f"on [1 h, 2 y] {100 * max(e_h):.3f}%; headroom to the 1% criterion {100 * out[name]['headroom_to_1pct']:.0f}%")
    return out


# ------------------------------------------------------------------ A: prices
def section_a():
    out = {}
    for H in (0.02, 0.05, 0.12):
        P = rh.RoughHestonParams(0.04, 2.0, 0.045, 0.5, -0.7, H)
        for d_, M in ((1, 1500), (7, 2000), (30, 3000)):
            tau = d_ / 365
            s = math.sqrt(P.v0 * tau)
            ks = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0]) * s
            um = 1.3 * rh.u_max_for(P, tau)
            ref1 = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, M), um), ks, tau)
            ref = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(ks, lambda z: ad.adams_log_cf(z, tau, P, 2 * M), um), ks, tau)
            row = {"adams_self_error_vp": float(100 * np.max(np.abs(ref - ref1)))}
            for name, (N, eta) in LIFTS.items():
                iv = fo.implied_vols_from_calls(ad.lewis_prices_from_logcf(
                    ks, lambda z: rh.log_char_func(z, tau, P, N=N, eta_N=eta, steps_mult=2.0), um), ks, tau)
                row[name] = {"worst_vp": float(100 * np.max(np.abs(iv - ref))), "atm_vp": float(100 * (iv[3] - ref[3]))}
            out[f"H={H} {d_}d"] = row
            say(f"A  H={H} {d_:2d}d (Adams own error {row['adams_self_error_vp']:.1e} vp): " + "; ".join(
                f"{n}: worst {r['worst_vp']:.4f} vp, ATM {r['atm_vp']:+.4f}" for n, r in row.items() if isinstance(r, dict)))
    return out


# ------------------------------------------------------------------ E: fGn
def v_h(H):
    """Mandelbrot-Van Ness variance for the kernel t^(H-1/2)/Gamma(H+1/2): E(B_1)^2."""
    return 1.0 / (special.gamma(2 * H + 1) * math.sin(math.pi * H))


def v_h_quadrature(H):
    """The same constant by quadrature, as a check on the closed form."""
    a = H + 0.5
    f = lambda s: ((1 + s) ** (a - 1) - s ** (a - 1)) ** 2
    tail = integrate.quad(f, 0, 1, limit=200)[0] + integrate.quad(f, 1, np.inf, limit=200)[0]
    return (1 / (2 * H) + tail) / special.gamma(a) ** 2


def lift_parts(H, N, eta):
    """(c, x): the stationary lifted driver's autocovariance is r(h) = sum_i c_i e^{-x_i h}."""
    w, x = _NODES(H, N, eta_N=eta)
    c = w * (w[None, :] / (x[:, None] + x[None, :])).sum(axis=1)
    return c, x


def variogram_lift(h, c, x):
    """E(Y_{t+h} - Y_t)^2 = 2 sum_i c_i (1 - e^{-x_i h})."""
    h = np.atleast_1d(np.asarray(h, float))
    return 2.0 * (c[None, :] * -np.expm1(-np.outer(h, x))).sum(axis=1)


def acov_lift(k, c, x, dt):
    """Autocovariance of the lift's increments at step dt, lag k (no cancellation)."""
    k = np.abs(np.atleast_1d(np.asarray(k, float)))
    a = x * dt
    out = -(c[None, :] * np.exp(-np.outer(np.maximum(k - 1, 0), a)) * np.expm1(-a)[None, :] ** 2).sum(axis=1)
    out[k == 0] = 2.0 * float((c * -np.expm1(-a)).sum())
    return out


def acov_fgn(k, H, dt):
    return v_h(H) * dt ** (2 * H) * fbm.autocovariance(np.atleast_1d(k), H)


def spec_fgn(lam, H, dt):
    """Spectral density of fGn with variance V_H dt^{2H}, rho(k) = (1/2pi) int f e^{ik lam}."""
    s = 2 * H + 1
    lam = np.asarray(lam, float)
    alias = lam ** (-s) + (2 * math.pi) ** (-s) * (special.zeta(s, 1 + lam / (2 * math.pi))
                                                   + special.zeta(s, 1 - lam / (2 * math.pi)))
    return 4.0 * np.sin(lam / 2) ** 2 * dt ** (2 * H) * alias


def spec_lift(lam, c, x, dt, chunk=4096):
    """Spectral density of the lift's increments: 4 sin^2(lam/2) sum_i c_i (1-q_i^2)/|1-q_i e^{-i lam}|^2."""
    lam = np.asarray(lam, float)
    a = x * dt
    q = np.exp(-a)
    num = -np.expm1(-2 * a)
    out = np.empty_like(lam)
    for i in range(0, lam.size, chunk):
        s2 = np.sin(lam[i:i + chunk] / 2) ** 2
        den = np.expm1(-a)[None, :] ** 2 + 4.0 * q[None, :] * s2[:, None]
        out[i:i + chunk] = 4.0 * s2 * (c[None, :] * num[None, :] / den).sum(axis=1)
    return out


def kl_whittle(n, H, c, x, dt, grid=40000):
    """
    Expected log-likelihood ratio, exact fGn against the lift, for n observations at step
    dt: the Whittle sum of r - 1 - log r, r = f_exact / f_lift, over the Fourier frequencies
    2 pi j / n, j = 1..(n-1)/2, as (n / 2 pi) int over [pi/n, pi] on a log grid.
    """
    lam = np.exp(np.linspace(math.log(math.pi / n), math.log(math.pi), grid))
    r = spec_fgn(lam, H, dt) / spec_lift(lam, c, x, dt)
    D = r - 1.0 - np.log(r)
    return float(n / (2 * math.pi) * integrate.trapezoid(D * lam, np.log(lam)))


def kl_whittle_sum(n, H, c, x, dt, reverse=False):
    """The same, summed exactly over the Fourier frequencies (for E3's check)."""
    lam = 2 * math.pi * np.arange(1, (n - 1) // 2 + 1) / n
    r = spec_fgn(lam, H, dt) / spec_lift(lam, c, x, dt)
    if reverse:
        r = 1.0 / r
    return float(np.sum(r - 1.0 - np.log(r)))


N_FLOOR = 64.0          # below this many observations the Whittle approximation says nothing


def n_for_power(H, c, x, dt, target=POWER_KL, n_max=1e13, kl=None):
    """
    Observations at step dt until the expected LLR reaches `target` (bisection in log n);
    N_FLOOR when even that few suffice. `kl(n)` overrides the divergence (section F).
    """
    kl = kl or (lambda n: kl_whittle(n, H, c, x, dt))
    lo, hi = N_FLOOR, N_FLOOR
    if kl(hi) >= target:
        return N_FLOOR
    while kl(hi) < target:
        lo, hi = hi, hi * 4
        if hi > n_max:
            return float("inf")
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if kl(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def kl_whittle_lifts(n, parts_a, parts_b, dt, grid=40000):
    """The same divergence between two lifts' increments (data from a, model b)."""
    lam = np.exp(np.linspace(math.log(math.pi / n), math.log(math.pi), grid))
    r = spec_lift(lam, *parts_a, dt) / spec_lift(lam, *parts_b, dt)
    D = r - 1.0 - np.log(r)
    return float(n / (2 * math.pi) * integrate.trapezoid(D * lam, np.log(lam)))


def years(n_star, sec):
    return "never (> 1e13 obs)" if not math.isfinite(n_star) else (
        f"< {N_FLOOR:.0f} obs" if n_star <= N_FLOOR else f"{n_star * sec / YEAR_S:.3g} y")


def section_f():
    """
    F1: how much data tells the 40-node lift from the 44-node one (and 44 from 48).
    F2: the lever E points at -- the fastest node. Lifts at the 44-node density
        (4.9 nodes per decade from 0.1/y) with the top node at 1e8, 1e10, 1e12, 1e14/y.
    """
    out = {"F1": {}, "F2": {}}
    for H in (0.02, 0.05, 0.12):
        for a_, b_ in ((40, 44), (44, 48)):
            pa, pb = lift_parts(H, a_, 1e8), lift_parts(H, b_, 1e8)
            row = {}
            for lab, sec in SAMPLING.items():
                dt = sec / YEAR_S
                n_star = n_for_power(H, None, None, dt, kl=lambda n: kl_whittle_lifts(n, pb, pa, dt))
                row[lab] = {"kl_one_year": kl_whittle_lifts(YEAR_S / sec, pb, pa, dt), "n_for_95pct_power": n_star}
            out["F1"][f"H={H} N={b_} vs N={a_}"] = row
            say(f"F1 H={H} data from N={b_}, model N={a_}: expected LLR from one year / data for 95% power: " + "; ".join(
                f"{lab} {r['kl_one_year']:.3g} / {years(r['n_for_95pct_power'], SAMPLING[lab])}" for lab, r in row.items()))
    hs = np.exp(np.linspace(math.log(1.0 / YEAR_S), math.log(2.0), 1200))
    for H in (0.02, 0.05, 0.12):
        exact = v_h(H) * hs ** (2 * H)
        for top in (1e8, 1e10, 1e12, 1e14):
            N = int(round(44 * math.log10(top / 0.1) / 9.0))
            c, x = lift_parts(H, N, top)
            g = variogram_lift(hs, c, x)
            rel = g / exact - 1
            band = lambda lo: float(np.max(np.abs(rel[hs >= lo])))
            qv = {lab: float(0.5 * math.log2(variogram_lift(np.array([2 * s / YEAR_S]), c, x)[0]
                                             / variogram_lift(np.array([s / YEAR_S]), c, x)[0]))
                  for lab, s in (("5 min", 300.0), ("1 d", 86400.0))}
            power = {lab: n_for_power(H, c, x, SAMPLING[lab] / YEAR_S) for lab in ("5 min", "1 d")}
            row = {"N": N, "kernel_err_1d_2y": rh.kernel_error(H, N, eta_N=top)[0],
                   "variogram_err_1min_2y": band(60 / YEAR_S), "variogram_err_1h_2y": band(3600 / YEAR_S),
                   "variogram_err_1d_2y": band(86400 / YEAR_S), "qv_H_5min": qv["5 min"], "qv_H_1d": qv["1 d"],
                   "n_for_power_5min": power["5 min"], "n_for_power_1d": power["1 d"]}
            out["F2"][f"H={H} top={top:.0e}"] = row
            say(f"F2 H={H} top node {top:.0e}/y (N = {N}): kernel err [1 d, 2 y] {100 * row['kernel_err_1d_2y']:.2f}%; "
                f"variogram err [1 min / 1 h / 1 d, 2 y] {100 * row['variogram_err_1min_2y']:.1f} / "
                f"{100 * row['variogram_err_1h_2y']:.1f} / {100 * row['variogram_err_1d_2y']:.1f}%; "
                f"QV estimator reads H = {qv['5 min']:.4f} at 5 min, {qv['1 d']:.4f} at 1 d (truth {H}); "
                f"data for 95% power {years(power['5 min'], 300.0)} at 5 min, {years(power['1 d'], 86400.0)} at 1 d")
    return out


def section_e():
    out = {"checks": {}}
    # closed forms checked: V_H against quadrature, both spectral densities against rho(0)
    for H in (0.02, 0.12, 0.45):
        c, x = lift_parts(H, 44, 1e8)
        dt = 300.0 / YEAR_S
        lam = np.concatenate([np.exp(np.linspace(math.log(1e-9), math.log(1e-3), 4000)),
                              np.linspace(1e-3, math.pi, 200001)[1:]])
        int_e = integrate.trapezoid(spec_fgn(lam, H, dt), lam) / math.pi
        int_l = integrate.trapezoid(spec_lift(lam, c, x, dt), lam) / math.pi
        out["checks"][f"H={H}"] = {
            "V_H_closed": v_h(H), "V_H_quadrature": v_h_quadrature(H),
            "fgn_spec_vs_rho0": float(int_e / acov_fgn(0, H, dt)[0] - 1),
            "lift_spec_vs_rho0": float(int_l / acov_lift(0, c, x, dt)[0] - 1)}
        say(f"E0 H={H}: V_H closed {v_h(H):.8f}, quadrature {v_h_quadrature(H):.8f}; (1/2pi) int f / rho(0) - 1: "
            f"fGn {out['checks'][f'H={H}']['fgn_spec_vs_rho0']:+.1e}, lift {out['checks'][f'H={H}']['lift_spec_vs_rho0']:+.1e}")

    # E1: the variogram, lag by lag
    hs = np.exp(np.linspace(math.log(1.0 / YEAR_S), math.log(2.0), 1200))
    out["E1"] = {}
    for H in (0.02, 0.05, 0.12, 0.25):
        exact = v_h(H) * hs ** (2 * H)
        for name, (N, eta) in LIFTS.items():
            c, x = lift_parts(H, N, eta)
            g = variogram_lift(hs, c, x)
            rel = g / exact - 1
            slope = np.gradient(np.log(g), np.log(hs))
            band = lambda lo: float(np.max(np.abs(rel[hs >= lo])))
            at = lambda t: float(np.interp(math.log(t), np.log(hs), slope))
            row = {"max_rel_err_1min_2y": band(60 / YEAR_S), "max_rel_err_1h_2y": band(3600 / YEAR_S),
                   "max_rel_err_1d_2y": band(86400 / YEAR_S),
                   "exponent_1min": at(60 / YEAR_S), "exponent_1h": at(3600 / YEAR_S), "exponent_1d": at(86400 / YEAR_S),
                   "exponent_exact": 2 * H}
            out["E1"][f"H={H} {name}"] = row
            say(f"E1 H={H} {name}: variogram vs V_H h^2H, worst on [1 min / 1 h / 1 d, 2 y] "
                f"{100 * row['max_rel_err_1min_2y']:.2f} / {100 * row['max_rel_err_1h_2y']:.2f} / {100 * row['max_rel_err_1d_2y']:.2f}%; "
                f"local exponent at 1 min / 1 h / 1 d {row['exponent_1min']:.4f} / {row['exponent_1h']:.4f} / "
                f"{row['exponent_1d']:.4f} (exact {2 * H:.4f})")

    # E2: data needed to tell the lift's increments from exact fGn
    out["E2"] = {}
    for H in (0.02, 0.05, 0.12):
        for name, (N, eta) in LIFTS.items():
            c, x = lift_parts(H, N, eta)
            row = {}
            for lab, sec in SAMPLING.items():
                dt = sec / YEAR_S
                n_year = YEAR_S / sec
                kl_year = kl_whittle(n_year, H, c, x, dt)
                n_star = n_for_power(H, c, x, dt)
                row[lab] = {"kl_one_year": kl_year, "n_for_95pct_power": n_star, "years_for_95pct_power": n_star / n_year}
            out["E2"][f"H={H} {name}"] = row
            say(f"E2 H={H} {name}: expected LLR from one year of perfectly observed driver increments / "
                "years for a 5%-size test to reach 95% power: " + "; ".join(
                    f"{lab} {r['kl_one_year']:.3g} / {years(r['n_for_95pct_power'], SAMPLING[lab])}" for lab, r in row.items()))

    # E3: Monte Carlo by Shevchenko's circulant embedding
    out["E3"] = {}
    n, paths, dt = 2 ** 16, 400, 300.0 / YEAR_S
    rng = np.random.default_rng(20260925)
    for H in (0.02, 0.12):
        g = fbm.fgn(n, H, rng=rng, size=paths) * math.sqrt(v_h(H) * dt ** (2 * H))
        h_exact = np.array([fbm.quadratic_variation_hurst(np.cumsum(p)) for p in g])
        lam = 2 * math.pi * np.arange(1, (n - 1) // 2 + 1) / n
        fe = spec_fgn(lam, H, dt)
        per_e = np.abs(np.fft.fft(g, axis=1)[:, 1:(n - 1) // 2 + 1]) ** 2 / n
        for name, (N, eta) in LIFTS.items():
            if N == 48:
                continue
            c, x = lift_parts(H, N, eta)
            y = fbm.stationary_gaussian(lambda k: acov_lift(k, c, x, dt), n, rng=rng, size=paths)
            h_lift = np.array([fbm.quadratic_variation_hurst(np.cumsum(p)) for p in y])
            fl = spec_lift(lam, c, x, dt)
            per_l = np.abs(np.fft.fft(y, axis=1)[:, 1:(n - 1) // 2 + 1]) ** 2 / n
            # Whittle log-likelihood ratio, exact model over lift model, on each kind of draw
            llr = lambda I: np.sum(np.log(fl / fe) + I / fl - I / fe, axis=1)
            vg = variogram_lift(np.array([dt, 2 * dt]), c, x)
            row = {"H_hat_exact_fgn": float(h_exact.mean()), "H_hat_exact_se": float(h_exact.std(ddof=1) / math.sqrt(paths)),
                   "H_hat_sd_one_path": float(h_exact.std(ddof=1)),
                   "H_hat_lift": float(h_lift.mean()), "H_hat_lift_se": float(h_lift.std(ddof=1) / math.sqrt(paths)),
                   "H_hat_lift_predicted": float(0.5 * math.log2(vg[1] / vg[0])),
                   "llr_on_exact_mean": float(llr(per_e).mean()), "llr_on_exact_se": float(llr(per_e).std(ddof=1) / math.sqrt(paths)),
                   "kl_exact_vs_lift_predicted": kl_whittle_sum(n, H, c, x, dt),
                   "llr_on_lift_mean": float(llr(per_l).mean()), "llr_on_lift_se": float(llr(per_l).std(ddof=1) / math.sqrt(paths)),
                   "kl_lift_vs_exact_predicted": -kl_whittle_sum(n, H, c, x, dt, reverse=True)}
            out["E3"][f"H={H} {name}"] = row
            say(f"E3 H={H} {name} ({paths} paths of {n} five-minute steps): Shevchenko QV estimate, exact fGn "
                f"{row['H_hat_exact_fgn']:.5f} +/- {row['H_hat_exact_se']:.5f} (one path sd {row['H_hat_sd_one_path']:.4f}); "
                f"lift {row['H_hat_lift']:.5f} +/- {row['H_hat_lift_se']:.5f} (predicted {row['H_hat_lift_predicted']:.5f}); "
                f"Whittle LLR on exact draws {row['llr_on_exact_mean']:.3f} +/- {row['llr_on_exact_se']:.3f} "
                f"(predicted {row['kl_exact_vs_lift_predicted']:.3f}), on lift draws {row['llr_on_lift_mean']:.3f} "
                f"+/- {row['llr_on_lift_se']:.3f} (predicted {row['kl_lift_vs_exact_predicted']:.3f})")
    return out


def section_b():
    sfl.LIFTS = LIFTS
    return sfl.section_b()


def section_c():
    sfl.LIFTS = LIFTS
    return sfl.section_c()


def section_d1():
    sfl.LIFTS = {k: v for k, v in LIFTS.items() if v[0] != 48}
    return sfl.section_d1()


def main(argv):
    wanted = [a.upper() for a in argv] or ["K", "A", "B", "E", "F"]
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    funcs = {"K": section_k, "A": section_a, "B": section_b, "C": section_c, "D1": section_d1, "E": section_e,
             "F": section_f}
    for key in wanted:
        t0 = time.perf_counter()
        res[key] = funcs[key]()
        say(f"[{key} done in {time.perf_counter() - t0:.0f}s]")
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
