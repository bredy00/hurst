"""
Tests for volsurf_core -- the pure maths, against known answers.

The two that matter most are test_grid_uniformity and test_vega_uniformity:
they are the numerical proof that the sigma-normalised grid removes the
"wide and narrow at the same time" problem rather than hiding it.

    python test_core.py
"""

import datetime
import math
import sys

import numpy as np

import volsurf_core as c


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def close(a, b, tol=1e-8):
    return abs(a - b) <= tol


# --- Black-Scholes ----------------------------------------------------------
def test_no_scipy_normal():
    """
    volsurf_core drops scipy so the startup path stays fast. Its own normal
    pdf/cdf must therefore match scipy's to full double precision -- checked
    here, where importing scipy costs nothing.
    """
    print("\nLocal normal pdf/cdf vs scipy (scipy imported only in the test)")
    from scipy.stats import norm as sp_norm

    xs = np.concatenate([np.linspace(-8, 8, 401), [-40.0, -1e-9, 0.0, 1e-9, 40.0]])
    dp = float(np.max(np.abs(c.norm_pdf(xs) - sp_norm.pdf(xs))))
    dc = float(np.max(np.abs(c.norm_cdf(xs) - sp_norm.cdf(xs))))
    check("norm_pdf matches scipy to 1e-15", dp < 1e-15, f"max abs diff {dp:.3e}")
    check("norm_cdf matches scipy to 1e-15", dc < 1e-15, f"max abs diff {dc:.3e}")
    check("core imports no scipy",
          not any(m.startswith('scipy') for m in _core_imports()),
          f"{sorted(m for m in _core_imports() if m.startswith('scipy'))}")

    # The Newton/bisection inversion must match what brentq would have found
    from scipy.optimize import brentq
    F, K, tau, sig = 650.0, 620.0, 0.09, 0.23
    px = float(c.bs_price(F, K, sig, tau, right='P'))
    mine = c.implied_vol(px, F, K, tau, right='P')
    theirs = brentq(lambda s: float(c.bs_price(F, K, s, tau, right='P')) - px,
                    1e-4, 5.0, xtol=1e-12)
    check("Newton solver agrees with brentq", abs(mine - theirs) < 1e-9,
          f"{mine:.12f} vs {theirs:.12f}")


def _core_imports():
    """Modules pulled in by importing volsurf_core alone, in a clean subprocess."""
    import json
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, json, volsurf_core; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True, text=True, cwd=None)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return []


def test_black_scholes():
    print("\nBlack-Scholes: d1/d2, vega, price")
    F, K, sig, tau = 650.0, 650.0, 0.20, 0.25

    d1, d2 = c.d1_d2(F, K, sig, tau)
    # ATM forward: d1 = +sigma*sqrt(tau)/2, d2 = -sigma*sqrt(tau)/2
    half = 0.5 * sig * math.sqrt(tau)
    check("ATM-forward d1 = +sigma*sqrt(tau)/2", close(float(d1), half), f"{float(d1):.10f}")
    check("ATM-forward d2 = -sigma*sqrt(tau)/2", close(float(d2), -half), f"{float(d2):.10f}")
    check("d2 = d1 - sigma*sqrt(tau)", close(float(d2), float(d1) - sig * math.sqrt(tau)))

    # The drift term must enter exactly as the user's formula says
    u = 0.03
    d1u, _ = c.d1_d2(F, K, sig, tau, drift=u)
    want = (math.log(F / K) + (u + 0.5 * sig * sig) * tau) / (sig * math.sqrt(tau))
    check("drift enters as (u + sigma^2/2)*tau", close(float(d1u), want))

    # Put-call parity on the model itself
    call = c.bs_price(F, 640.0, sig, tau, right='C')
    put = c.bs_price(F, 640.0, sig, tau, right='P')
    check("put-call parity C - P = F - K", close(call - put, F - 640.0, 1e-9),
          f"{call - put:.10f} vs {F - 640.0:.10f}")

    # Vega against a central difference
    h = 1e-6
    fd = (c.bs_price(F, 660.0, sig + h, tau) - c.bs_price(F, 660.0, sig - h, tau)) / (2 * h)
    an = float(c.bs_vega(F, 660.0, sig, tau))
    check("vega matches central difference", abs(fd - an) < 1e-4, f"fd={fd:.6f} analytic={an:.6f}")

    # Vega peaks where d1 = 0, which is k = sigma^2*tau/2 -- NOT at k = 0.
    # It is symmetric about that point, not about the forward.
    k_peak = 0.5 * sig * sig * tau
    v_peak = float(c.bs_vega(F, F * math.exp(k_peak), sig, tau))
    v_up = float(c.bs_vega(F, F * math.exp(0.1), sig, tau))
    v_dn = float(c.bs_vega(F, F * math.exp(-0.1), sig, tau))
    check("vega peaks at k = sigma^2*tau/2 (where d1 = 0)",
          v_peak > v_up and v_peak > v_dn, f"peak at k={k_peak:.5f}")
    check("vega asymmetric about k = 0 (it must be)", not close(v_up, v_dn, 1e-6),
          f"k=+0.1 -> {v_up:.4f}, k=-0.1 -> {v_dn:.4f}")
    lo = float(c.bs_vega(F, F * math.exp(k_peak - 0.1), sig, tau))
    hi = float(c.bs_vega(F, F * math.exp(k_peak + 0.1), sig, tau))
    check("vega symmetric about k = sigma^2*tau/2", close(lo, hi, 1e-9), f"{lo:.8f} vs {hi:.8f}")


def test_implied_vol():
    print("\nImplied vol inversion")
    F, tau = 650.0, 0.08
    for K, sig in [(600.0, 0.28), (650.0, 0.18), (700.0, 0.16)]:
        right = c.otm_right(K, F)
        px = c.bs_price(F, K, sig, tau, right=right)
        back = c.implied_vol(px, F, K, tau, right=right)
        check(f"round-trip K={K:.0f} sigma={sig}", back is not None and close(back, sig, 1e-6),
              f"recovered {back}")

    # Below intrinsic means an ITM contract priced under F-K; an OTM put at
    # K < F has zero intrinsic, so a tiny price there is legitimate.
    check("ITM call priced below intrinsic -> None",
          c.implied_vol(10.0, F, 600.0, tau, right='C') is None,
          f"intrinsic is {F-600.0:.0f}")
    check("tiny OTM put price is legitimate, not rejected",
          c.implied_vol(1e-9, F, 600.0, tau, right='P') is not None)
    check("price above upper bound -> None",
          c.implied_vol(1e9, F, 650.0, tau, right='C') is None)
    check("zero price -> None", c.implied_vol(0.0, F, 650.0, tau) is None)
    check("negative tau -> None", c.implied_vol(10.0, F, 650.0, -1.0) is None)


# --- the forward ------------------------------------------------------------
def test_forward_parity():
    print("\nForward implied from put-call parity")
    F_true, df_true, tau, sig = 648.35, 0.9985, 0.12, 0.19
    Ks = np.arange(600.0, 701.0, 5.0)
    calls = [c.bs_price(F_true, K, sig, tau, df_true, 'C') for K in Ks]
    puts = [c.bs_price(F_true, K, sig, tau, df_true, 'P') for K in Ks]

    F, df, r2 = c.forward_from_parity(Ks, calls, puts)
    check("recovers F on clean data", close(F, F_true, 1e-6), f"{F:.8f} vs {F_true}")
    check("recovers discount factor", close(df, df_true, 1e-9), f"{df:.8f}")
    check("r2 = 1 on clean data", close(r2, 1.0, 1e-9), f"{r2:.10f}")

    # The forward must be recovered WITHOUT being told the dividend
    S = F_true / (df_true ** -1 * math.exp(0.0))
    check("F != spot (carry is real and recovered)", not close(F, S, 1e-3))

    # Noise should degrade r2 but keep F close
    rng = np.random.default_rng(3)
    nc = [x + rng.normal(0, 0.01) for x in calls]
    npu = [x + rng.normal(0, 0.01) for x in puts]
    Fn, dfn, r2n = c.forward_from_parity(Ks, nc, npu)
    check("noisy data: F still within 0.1", abs(Fn - F_true) < 0.1, f"{Fn:.4f}")
    check("noisy data: r2 < 1 (quality score reacts)", r2n < 1.0, f"r2={r2n:.6f}")

    check("too few points -> None", c.forward_from_parity([1.0], [1.0], [1.0])[0] is None)


# --- THE POINT: uniformity --------------------------------------------------
EXPIRY_DAYS = [0.5, 3.0, 5.0, 7.0, 10.0, 12.0]


def atm_vol(tau):
    return 0.115 + 0.09 * (1.0 - math.exp(-tau / 0.06))


def test_grid_uniformity():
    """
    The complaint: a fixed +/-2% band is 10.9 sigma wide at half a day and
    0.8 sigma wide at twelve days -- too wide and too narrow at once.
    A sigma-normalised band must be the same width at every expiry, by
    construction.
    """
    print("\nGrid uniformity: fixed % band vs sigma-normalised band")
    F = 650.0
    all_strikes = [float(s) for s in range(500, 801)]

    pct_edges, sig_edges = [], []
    for days in EXPIRY_DAYS:
        tau = days / 365.0
        s_atm = atm_vol(tau)

        # the alpha's rule
        pct = [K for K in all_strikes if 0.98 * F <= K <= 1.02 * F]
        z_pct = max(abs(c.normalised_moneyness(K, F, s_atm, tau)) for K in pct)
        pct_edges.append(z_pct)

        # the fix
        sel = c.select_strikes_by_sigma(all_strikes, F, s_atm, tau, n_sigma=3.0)
        z_sig = max(abs(c.normalised_moneyness(K, F, s_atm, tau)) for K in sel)
        sig_edges.append(z_sig)

    spread_pct = max(pct_edges) / min(pct_edges)
    spread_sig = max(sig_edges) / min(sig_edges)
    print(f"      fixed 2%  band edge in sigma: "
          f"{', '.join(f'{z:.2f}' for z in pct_edges)}")
    print(f"      3-sigma   band edge in sigma: "
          f"{', '.join(f'{z:.2f}' for z in sig_edges)}")

    check("fixed 2% band varies wildly across expiries (>5x)",
          spread_pct > 5.0, f"{min(pct_edges):.2f}..{max(pct_edges):.2f} sigma, {spread_pct:.1f}x")
    check("sigma band is uniform across expiries (<1.2x)",
          spread_sig < 1.2, f"{min(sig_edges):.2f}..{max(sig_edges):.2f} sigma, {spread_sig:.2f}x")
    check("every sigma-band edge is within 3 sigma", max(sig_edges) <= 3.0 + 1e-9)
    check("sigma band widens with tau (it must)",
          c.select_strikes_by_sigma(all_strikes, F, atm_vol(12 / 365), 12 / 365, 3.0).__len__()
          > c.select_strikes_by_sigma(all_strikes, F, atm_vol(0.5 / 365), 0.5 / 365, 3.0).__len__())


def test_vega_uniformity():
    """
    Uniform z means comparable vega, which means comparable IV uncertainty.
    That is the actual cure for 'small variance here, huge variance there'.
    """
    print("\nVega / IV-uncertainty uniformity")
    F, half_spread = 650.0, 0.005
    all_strikes = [float(s) for s in range(500, 801)]

    worst_pct, worst_sig = [], []
    for days in EXPIRY_DAYS:
        tau = days / 365.0
        s_atm = atm_vol(tau)

        pct = [K for K in all_strikes if 0.98 * F <= K <= 1.02 * F]
        v = c.bs_vega(F, np.array(pct), s_atm, tau)
        worst_pct.append(float(np.max(c.iv_uncertainty(half_spread, v))))

        sel = c.select_strikes_by_sigma(all_strikes, F, s_atm, tau, n_sigma=3.0)
        v = c.bs_vega(F, np.array(sel), s_atm, tau)
        worst_sig.append(float(np.max(c.iv_uncertainty(half_spread, v))))

    print(f"      worst IV error, fixed 2% band : "
          f"{', '.join(f'{e*100:8.2f}' for e in worst_pct)}  (vol points)")
    print(f"      worst IV error, 3-sigma band  : "
          f"{', '.join(f'{e*100:8.2f}' for e in worst_sig)}  (vol points)")

    check("fixed band: front expiry IV error is absurd (>100 vol points)",
          worst_pct[0] > 1.0, f"{worst_pct[0]*100:.1f} vol points at 0.5d")
    check("sigma band: worst-case IV error bounded (<5 vol points everywhere)",
          max(worst_sig) < 0.05, f"max {max(worst_sig)*100:.2f} vol points")
    check("sigma band collapses the spread of uncertainty across expiries",
          (max(worst_sig) / min(worst_sig)) < (max(worst_pct) / min(worst_pct)) / 100.0,
          f"sigma {max(worst_sig)/min(worst_sig):.1f}x vs pct {max(worst_pct)/min(worst_pct):.0f}x")

    check("quote_weight is zero for a worthless quote",
          float(c.quote_weight(half_spread, 0.0)) == 0.0)
    check("quote_weight rises with vega",
          float(c.quote_weight(half_spread, 40.0)) > float(c.quote_weight(half_spread, 4.0)))


# --- total variance ---------------------------------------------------------
def test_total_variance():
    print("\nTotal variance")
    check("w = sigma^2 tau", close(float(c.total_variance(0.2, 0.25)), 0.01))
    check("round-trip sigma", close(float(c.sigma_from_total_variance(0.01, 0.25)), 0.2))

    # A flat-vol surface is automatically calendar-arbitrage free in w
    taus = np.array([0.01, 0.05, 0.1, 0.25])
    w = c.total_variance(0.2, taus)
    check("flat vol -> w monotone in tau", len(c.calendar_violations(taus, w)) == 0)
    check("decreasing w flagged as calendar arb",
          len(c.calendar_violations(taus, w[::-1])) > 0)

    ks = np.linspace(-0.2, 0.2, 9)
    convex = 0.01 + 0.5 * ks ** 2
    check("convex slice -> no butterfly violation", len(c.butterfly_violations(ks, convex)) == 0)
    # A downward dent at i makes its NEIGHBOURS concave, not the dent itself
    # (the dent is a local minimum, so its own second difference is positive).
    dented = convex.copy()
    dented[4] -= 0.004
    bad = c.butterfly_violations(ks, dented)
    check("dented slice -> butterfly violation detected", len(bad) > 0, f"indices {bad}")
    check("violation lands on the dent's neighbours", set(bad) == {3, 5}, f"indices {bad}")


# --- roughness --------------------------------------------------------------
def test_roughness():
    print("\nRoughness estimator")
    H_true = 0.12
    taus = np.array([1, 3, 7, 14, 30, 60, 90]) / 365.0
    skews = -1.3 * (7 / 365.0 / taus) ** (0.5 - H_true)

    H, H_err, _, r2 = c.estimate_hurst(taus, skews)
    check("recovers planted H exactly on clean data", close(H, H_true, 1e-9), f"H={H:.6f}")
    check("r2 = 1 on clean data", close(r2, 1.0, 1e-9))

    rng = np.random.default_rng(11)
    noisy = skews * np.exp(rng.normal(0, 0.05, size=skews.shape))
    Hn, _, _, r2n = c.estimate_hurst(taus, noisy)
    check("H within 0.03 under 5% multiplicative noise", abs(Hn - H_true) < 0.03, f"H={Hn:.4f}")

    # A classical diffusion has a flat short-end skew -> H = 1/2
    flat = np.full_like(taus, -0.5)
    Hf, _, _, _ = c.estimate_hurst(taus, flat)
    check("flat skew -> H = 0.5 (classical, not rough)", close(Hf, 0.5, 1e-9), f"H={Hf:.6f}")

    check("too few points -> None", c.estimate_hurst([0.1], [0.2])[0] is None)

    # Recover the skew from a synthetic slice
    F, tau, s_atm = 650.0, 7 / 365.0, 0.14
    ks = np.linspace(-0.05, 0.05, 21)
    slope_true = -1.3
    sig = s_atm + slope_true * ks + 6.0 * ks ** 2
    got, se, n = c.atm_skew_from_slice(ks, sig, zs=ks / (s_atm * math.sqrt(tau)), window=3.0)
    check("ATM skew recovered from a slice", close(got, slope_true, 1e-6), f"{got:.8f}")
    check("skew fit reports a standard error", se is not None and se >= 0, f"se={se}")
    check("skew fit reports n used", n > 3, f"n={n}")


# --- time -------------------------------------------------------------------
def test_time():
    print("\nTime increments (tau = T - t0)")
    t0 = datetime.datetime(2026, 9, 4, 14, 30)
    check("same-day expiry is a real fraction, not zero",
          0 < c.tau_years(t0, datetime.date(2026, 9, 4)) < 1 / 365.0,
          f"{c.tau_years(t0, datetime.date(2026, 9, 4))*365*24:.2f} hours")
    check("7-day expiry ~ 7/365",
          abs(c.tau_years(t0, datetime.date(2026, 9, 11)) - 7 / 365.0) < 0.002)
    check("tau is strictly positive for a past date",
          c.tau_years(t0, datetime.date(2026, 9, 1)) > 0)
    check("tau grows with expiry",
          c.tau_years(t0, datetime.date(2026, 9, 11)) > c.tau_years(t0, datetime.date(2026, 9, 7)))
    check("parse_ib_date", c.parse_ib_date("20260911") == datetime.date(2026, 9, 11))

    # Session G: the live caller passed local wall-clock time as if it were UTC.
    import zoneinfo
    ny = zoneinfo.ZoneInfo("America/New_York")
    days = [datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(0, 1461, 3)]
    wrong = [d for d in days
             if c.us_close_utc(d) != datetime.datetime.combine(d, datetime.time(16, 0), ny)
             .astimezone(datetime.timezone.utc).replace(tzinfo=None)]
    check("US close in UTC matches zoneinfo on 487 dates over four years (DST both ways)",
          not wrong, f"{len(wrong)} wrong, first {wrong[:2]}")
    check("summer close is 20:00 UTC, winter 21:00",
          c.us_close_utc(datetime.date(2026, 9, 15)).hour == 20
          and c.us_close_utc(datetime.date(2026, 12, 15)).hour == 21)
    ist = datetime.timezone(datetime.timedelta(hours=3))
    t_local = datetime.datetime(2026, 9, 15, 17, 0, tzinfo=ist)          # 10:00 New York
    got = c.tau_years(t_local, datetime.date(2026, 9, 16)) * 365 * 24
    check("aware local time: 17:00 Istanbul to the next day's close is 30 hours",
          abs(got - 30.0) < 1e-9, f"{got:.4f} h")
    naive_local = c.tau_years(datetime.datetime(2026, 9, 15, 17, 0),
                              datetime.date(2026, 9, 16)) * 365 * 24
    check("the old bug, reproduced: a naive local clock loses the UTC offset",
          abs(naive_local - 27.0) < 1e-9, f"{naive_local:.1f} h instead of 30")
    check("New York date of 02:00 UTC in September is the previous day",
          c.new_york_date(datetime.datetime(2026, 9, 16, 2, 0)) == datetime.date(2026, 9, 15))


def test_local_skew_window():
    """
    The reason the window exists: a global quadratic across the whole slice is
    a regression, not a derivative. On a smile with real (cubic) structure the
    bias must fall as the window narrows.
    """
    print("\nLocal ATM-skew estimator: window controls the wing bias")
    F, tau, s_atm = 650.0, 3.0 / 365.0, 0.128
    slope_true = -1.736
    ks = np.linspace(-3, 3, 21) * s_atm * math.sqrt(tau)
    zs = ks / (s_atm * math.sqrt(tau))

    quad = s_atm + slope_true * ks + 6.0 * ks ** 2
    got, _, _ = c.atm_skew_from_slice(ks, quad, zs=zs, window=3.0)
    check("exact on a purely quadratic smile", close(got, slope_true, 1e-6), f"{got:.6f}")

    cubic = quad - 40.0 * ks ** 3
    errs = {}
    for win in (3.0, 2.0, 1.5, 1.0):
        g, _, n = c.atm_skew_from_slice(ks, cubic, zs=zs, window=win)
        errs[win] = abs(g - slope_true)
        print(f"      window |z|<={win:<4} n={n:2d}  slope={g:+.5f}  err={errs[win]:.5f}")
    check("bias falls monotonically as the window narrows",
          errs[3.0] > errs[2.0] > errs[1.5] > errs[1.0],
          f"{errs[3.0]:.5f} -> {errs[1.0]:.5f}")
    check("wide window is materially biased on a real smile", errs[3.0] > 1e-2)
    check("default window (1.5) keeps the bias small", errs[1.5] < 1e-2)

    _, se, n = c.atm_skew_from_slice(ks, cubic, zs=zs, window=1.5)
    check("returns a standard error", se is not None and se > 0, f"se={se:.3e}")
    check("too few points -> (None, None, 0)",
          c.atm_skew_from_slice([0.0, 0.1], [0.2, 0.2])[0] is None)

    # Weighted H fit must beat the unweighted one when one skew is badly measured
    taus = np.array([3, 5, 7, 14, 30, 60]) / 365.0
    H_true = 0.12
    sk = -1.3 * ((7 / 365.0) / taus) ** (0.5 - H_true)
    sk_bad = sk.copy()
    sk_bad[0] *= 1.35                      # front expiry mismeasured
    good_err = np.abs(sk) * 0.01
    bad_err = good_err.copy()
    bad_err[0] = abs(sk[0]) * 0.60         # and known to be uncertain
    H_un, _, _, _ = c.estimate_hurst(taus, sk_bad)
    H_w, _, _, _ = c.estimate_hurst(taus, sk_bad, bad_err)
    check("weighting by skew stderr beats equal weighting",
          abs(H_w - H_true) < abs(H_un - H_true),
          f"weighted {H_w:.4f} vs unweighted {H_un:.4f} (true {H_true})")


def test_structure_function():
    """H from the realised log-vol path, independent of the option surface."""
    print("\nStructure function -- the second, independent route to H")
    rng = np.random.default_rng(5)

    # Brownian log-vol: increments ~ Delta^(1/2), so zeta(q) = q/2 and H = 0.5
    n = 60000
    bm = np.cumsum(rng.normal(0, 1.0, n))
    deltas = np.unique(np.round(np.logspace(0, 2, 15)).astype(int))
    m1 = c.structure_function(bm, deltas, q=1.0)
    zeta, _, r2, se = c.zeta_regression(deltas, m1)
    check("Brownian path -> zeta(1) ~ 0.5", abs(zeta - 0.5) < 0.02, f"zeta={zeta:.4f}")
    check("scaling regression r2 high", r2 > 0.99, f"r2={r2:.5f}")
    check("zeta carries a standard error", se is not None and se > 0)

    res = c.hurst_from_structure(bm, deltas)
    check("monofractal H ~ 0.5 from zeta(q)=qH", abs(res['H'] - 0.5) < 0.03,
          f"H={res['H']:.4f}")
    check("zeta(q) is linear in q for a monofractal", res['linearity_r2'] > 0.99,
          f"r2={res['linearity_r2']:.5f}")
    for q in (1.0, 2.0):
        z = res['per_q'][q]['zeta']
        check(f"zeta({q}) ~ {q/2}", abs(z - q * 0.5) < 0.04, f"{z:.4f}")

    # White noise log-vol is maximally rough: increments do not grow, zeta ~ 0
    wn = rng.normal(0, 1.0, n)
    z_wn, _, _, _ = c.zeta_regression(deltas, c.structure_function(wn, deltas, 1.0))
    check("white noise -> zeta(1) ~ 0 (H ~ 0, maximally rough)",
          abs(z_wn) < 0.02, f"zeta={z_wn:.4f}")
    check("rough < Brownian in the estimator's ordering", z_wn < zeta)

    check("delta >= series length -> NaN",
          not np.isfinite(c.structure_function([1.0, 2.0], [5], 1.0)[0]))


def test_finite_differences():
    print("\nFinite differences and Breeden-Litzenberger")
    x = np.linspace(0.5, 3.0, 60)
    xn = np.sort(np.concatenate([np.linspace(0.5, 1.5, 12), np.linspace(1.6, 3.0, 30)]))

    # Exactness class: both stencils are exact for quadratics on ANY grid.
    for name, g in (("uniform", x), ("non-uniform", xn)):
        check(f"fd_first exact on a quadratic, {name} grid",
              np.nanmax(np.abs(c.fd_first(g, g ** 2)[1:-1] - 2 * g[1:-1])) < 1e-12)
        check(f"fd_second exact on a quadratic, {name} grid",
              np.nanmax(np.abs(c.fd_second(g, g ** 2)[1:-1] - 2.0)) < 1e-10)

    # fd_second gains cubic exactness only when the spacing is uniform.
    check("fd_second exact on a cubic when the grid IS uniform",
          np.nanmax(np.abs(c.fd_second(x, x ** 3)[1:-1] - 6 * x[1:-1])) < 1e-10)
    check("fd_second loses that on a NON-uniform grid (documented order loss)",
          np.nanmax(np.abs(c.fd_second(xn, xn ** 3)[1:-1] - 6 * xn[1:-1])) > 1e-3,
          "matters for Breeden-Litzenberger on real strike grids")

    # fd_first on a cubic is second-order: halving h must quarter the error.
    errs = []
    for n in (60, 120, 240, 480):
        g = np.linspace(0.5, 3.0, n)
        errs.append(np.nanmax(np.abs(c.fd_first(g, g ** 3)[1:-1] - 3 * g[1:-1] ** 2)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    check("fd_first is second-order accurate (error ratio ~4 per halving)",
          all(3.7 < r < 4.4 for r in ratios),
          "ratios " + ", ".join(f"{r:.2f}" for r in ratios))

    check("ends are NaN for the second derivative",
          not np.isfinite(c.fd_second(x, x ** 3)[0])
          and not np.isfinite(c.fd_second(x, x ** 3)[-1]))

    # Breeden-Litzenberger against the analytic Black-76 density
    F, tau, sig, df = 650.0, 0.25, 0.20, 0.995
    Ks = np.linspace(300.0, 1400.0, 1401)   # wide enough to hold the tails
    Cs = np.array([float(c.bs_price(F, K, sig, tau, df, 'C')) for K in Ks])
    Kb, dens = c.breeden_litzenberger(Ks, Cs, df)
    _, d2b = c.d1_d2(F, Kb, sig, tau)
    analytic = c.norm_pdf(d2b) / (Kb * sig * math.sqrt(tau))
    m = np.isfinite(dens)
    err = float(np.max(np.abs(dens[m] - analytic[m])))
    check("RND matches the analytic lognormal density", err < 1e-6, f"max abs err {err:.2e}")
    check("RND is non-negative on an arbitrage-free surface",
          np.nanmin(dens) >= -1e-12, f"min {np.nanmin(dens):.3e}")
    check("RND integrates to 1 over a range covering the tails",
          abs(np.trapezoid(dens[m], Kb[m]) - 1.0) < 1e-6,
          f"{np.trapezoid(dens[m], Kb[m]):.8f}")

    # Truncating the range loses real mass -- the estimator is fine, the window is not
    Kt = np.linspace(450.0, 900.0, 601)
    Ct = np.array([float(c.bs_price(F, K, sig, tau, df, 'C')) for K in Kt])
    Kb2, d2t = c.breeden_litzenberger(Kt, Ct, df)
    m2 = np.isfinite(d2t)
    check("a truncated range under-integrates (truncation, not error)",
          np.trapezoid(d2t[m2], Kb2[m2]) < 0.9995,
          f"{np.trapezoid(d2t[m2], Kb2[m2]):.6f} over [450,900]")

    # A dented call curve must produce a NEGATIVE density -- the arbitrage alarm
    bad = Cs.copy()
    bad[700] += 0.5
    _, dbad = c.breeden_litzenberger(Ks, bad, df)
    check("dented call curve -> negative density (arbitrage caught)",
          np.nanmin(dbad) < 0, f"min {np.nanmin(dbad):.4f}")


def test_second_derivative_on_real_grids():
    """
    fd_second_poly is the fix for the documented order loss on non-uniform
    grids. It must be exact for quartics on ANY grid, hold second order on a
    kinked grid where the 3-point stencil is first order, and the density
    built from it must beat the old one on a uniform grid too.
    """
    print("\n-- second derivative on real strike grids (Fornberg 5-point)")
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.5, 3.0, 60))
    y = 1 + 2 * x - 0.5 * x ** 2 + 0.25 * x ** 3 - 0.1 * x ** 4
    d_exact = -1.0 + 1.5 * x - 1.2 * x ** 2
    err = float(np.nanmax(np.abs(c.fd_second_poly(x, y)[1:-1] - d_exact[1:-1])))
    check("5-point stencil is exact on a quartic over a RANDOM grid", err < 1e-8,
          f"max err {err:.2e}")
    check("...and the 3-point stencil is not",
          float(np.nanmax(np.abs(c.fd_second(x, y)[1:-1] - d_exact[1:-1]))) > 1e-3)

    # SPY-like grid: $1 spacing then $5 spacing. Order measured by doubling.
    def kink(n):
        a = np.arange(0.40, 1.0001, 0.6 / (n * 0.7))
        b = np.arange(a[-1] + 5 * (a[1] - a[0]), 1.8, 5 * (a[1] - a[0]))
        return np.concatenate([a, b])

    def dens(K):
        sig, tau = 0.25, 0.25
        return (np.exp(-(np.log(K) + 0.5 * sig * sig * tau) ** 2 / (2 * sig * sig * tau))
                / (K * sig * math.sqrt(2 * math.pi * tau)))

    def err_of(f, n):
        K = kink(n)
        C = np.array([float(c.bs_price(1.0, k, 0.25, 0.25, 1.0, 'C')) for k in K])
        d = f(K, C)
        ok = np.isfinite(d)
        return float(np.max(np.abs(d[ok] - dens(K)[ok])) / dens(K).max())

    e5 = [err_of(c.fd_second_poly, n) for n in (101, 201, 401)]
    e3 = [err_of(c.fd_second, n) for n in (101, 201, 401)]
    o5 = math.log2(e5[0] / e5[2]) / 2
    o3 = math.log2(e3[0] / e3[2]) / 2
    check("5-point holds at least second order across a $1/$5 kink", o5 > 2.0,
          f"observed order {o5:.2f}")
    check("3-point drops to first order there (the documented loss)", o3 < 1.5,
          f"observed order {o3:.2f}")
    check("5-point is at least 10x more accurate at every resolution",
          all(a < 0.1 * b for a, b in zip(e5, e3)),
          f"{e5[0]:.1e}/{e3[0]:.1e} at n=101")

    # Uniform grid: the fast path must agree with the general one, and the
    # density must be better than the 3-point version was.
    K = np.linspace(300.0, 1400.0, 1401)
    C = np.array([float(c.bs_price(650.0, k, 0.2, 0.25, 0.995, 'C')) for k in K])
    Kb, q5 = c.breeden_litzenberger(K, C, 0.995)
    _, q3 = c.breeden_litzenberger(K, C, 0.995, stencil="3pt")
    _, d2 = c.d1_d2(650.0, Kb, 0.2, 0.25)
    an = c.norm_pdf(d2) / (Kb * 0.2 * 0.5)
    m = np.isfinite(q5)
    e5u = float(np.max(np.abs(q5[m] - an[m])))
    e3u = float(np.max(np.abs(q3[m] - an[m])))
    check("uniform-grid density: 5-point beats 3-point by 1000x", e5u < 1e-3 * e3u,
          f"{e5u:.1e} vs {e3u:.1e}")
    slow = c.fd_second_poly(K + 0.0, C)
    Kj = K.copy()
    Kj[1] += 1e-9          # break exact uniformity to force the general path
    gen = c.fd_second_poly(Kj, C)
    ok = np.isfinite(slow) & np.isfinite(gen)
    check("uniform fast path agrees with the general Fornberg path",
          float(np.max(np.abs(slow[ok] - gen[ok]))) < 1e-6 * float(np.max(np.abs(slow[ok]))))

    # Endpoint convention preserved
    check("endpoints are NaN, as documented",
          not np.isfinite(q5[0]) and not np.isfinite(q5[-1]))


def test_density_for_sampling():
    """
    The floored, renormalised copy for downstream sampling. It must never touch
    the raw density (which is the diagnostic), must be >= eps everywhere, must
    integrate to exactly one, and must report what it removed.
    """
    print("\n-- density_for_sampling")
    K = np.linspace(300.0, 1400.0, 1401)
    C = np.array([float(c.bs_price(650.0, k, 0.2, 0.25, 0.995, 'C')) for k in K])
    Kb, q = c.breeden_litzenberger(K, C, 0.995)
    raw_min = float(np.nanmin(q))
    Ks, qs, info = c.density_for_sampling(Kb, q)
    check("raw density is untouched", float(np.nanmin(q)) == raw_min)
    check("sampling density is >= machine epsilon everywhere",
          float(qs.min()) >= np.finfo(float).eps, f"min {qs.min():.3e}")
    check("sampling density integrates to one to 1e-14",
          abs(float(np.trapezoid(qs, Ks)) - 1.0) < 1e-14,
          f"{float(np.trapezoid(qs, Ks)):.16f}")
    check("no NaN survives", np.all(np.isfinite(qs)))
    check("reports the negative mass it removed and the renormalisation",
          all(k in info for k in ("negative_mass", "renorm_factor", "n_floored", "raw_mass")),
          f"{info}")
    check("on a clean slice the correction is at the noise floor",
          abs(info['renorm_factor'] - 1.0) < 1e-9 and abs(info['negative_mass']) < 1e-9,
          f"renorm {info['renorm_factor']-1:+.1e}, negative mass {info['negative_mass']:.1e}")

    # A dented (arbitrage) slice: the raw density goes negative, the sampling
    # copy hides it -- and SAYS so.
    bad = C.copy()
    bad[700] += 0.5
    _, qb = c.breeden_litzenberger(K, bad, 0.995)
    _, qbs, ib = c.density_for_sampling(K, qb)
    check("arbitrage slice: raw density negative, sampling copy is not",
          float(np.nanmin(qb)) < 0 and float(qbs.min()) > 0)
    check("...and the removed negative mass is reported as material",
          ib['negative_mass'] < -1e-4 and ib['n_floored'] > 0,
          f"negative mass {ib['negative_mass']:.4f} over {ib['n_floored']} points")


if __name__ == "__main__":
    print("=" * 74)
    print("volsurf_core -- maths verification")
    print("=" * 74)
    test_no_scipy_normal()
    test_black_scholes()
    test_implied_vol()
    test_forward_parity()
    test_grid_uniformity()
    test_vega_uniformity()
    test_total_variance()
    test_roughness()
    test_time()
    test_local_skew_window()
    test_structure_function()
    test_finite_differences()
    test_second_derivative_on_real_grids()
    test_density_for_sampling()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
