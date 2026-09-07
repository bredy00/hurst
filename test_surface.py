"""
Surface-assembly tests: the layer between the pure maths and the live socket.

volsurf_core is covered by test_core.py and the IBKR callbacks by test_fixes.py.
This file covers what sits between them -- the arbitrage audit, SVI slice
fitting, the risk-neutral density panel, and the replay source.

    python test_surface.py
"""

import math
import sys

import numpy as np

import volsurf_core as vc
import volatility_surface_3 as v3


PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class Ctx:
    """Minimal stand-in for ExpiryContext."""
    def __init__(self, tau, forward=650.0, sigma_atm=0.20, discount=1.0):
        self.tau = tau
        self.forward = forward
        self.sigma_atm = sigma_atm
        self.discount = discount
        self.parity_r2 = 1.0
        self.strikes = []


def slice_rows(tau, F=650.0, s_atm=0.20, skew=-1.0, curv=6.0, n=15, span=0.10):
    """A clean synthetic slice in the shape surface_points() produces."""
    ks = np.linspace(-span, span, n)
    rows = []
    for k in ks:
        sig = s_atm + skew * k + curv * k * k
        rows.append({'k': float(k),
                     'z': float(k / (s_atm * math.sqrt(tau))),
                     'w': float(sig * sig * tau),
                     'iv': float(sig),
                     'weight': 1.0,
                     'strike': float(F * math.exp(k))})
    return rows


# --- A1: arbitrage audit ----------------------------------------------------
def test_audit_butterfly():
    print("\nA1 -- butterfly audit")
    ctxs = {"E1": Ctx(0.02)}
    pts = {"E1": slice_rows(0.02)}

    flags = v3.audit_surface(pts, ctxs)
    check("clean convex slice raises no butterfly flag",
          flags["E1"]["butterfly"] == [], f"{flags['E1']['butterfly']}")

    dented = {"E1": [dict(r) for r in slice_rows(0.02)]}
    dented["E1"][7]['w'] -= 0.0008
    flags = v3.audit_surface(dented, ctxs)
    check("a dented slice is flagged", len(flags["E1"]["butterfly"]) > 0,
          f"indices {flags['E1']['butterfly']}")
    check("the flag lands adjacent to the dent",
          all(abs(i - 7) <= 1 for i in flags["E1"]["butterfly"]),
          f"indices {flags['E1']['butterfly']}")

    check("a 2-point slice cannot be audited, and does not crash",
          v3.audit_surface({"E1": slice_rows(0.02, n=2)}, ctxs)["E1"]["butterfly"] == [])


def test_audit_calendar():
    print("\nA1 -- calendar audit")
    taus = [7 / 365, 30 / 365, 90 / 365]
    names = ["E1", "E2", "E3"]
    ctxs = {n: Ctx(t) for n, t in zip(names, taus)}

    # Flat vol -> w = sigma^2 tau is strictly increasing -> no calendar arb
    pts = {n: slice_rows(t, skew=0.0, curv=0.0) for n, t in zip(names, taus)}
    flags = v3.audit_surface(pts, ctxs)
    check("increasing total variance raises no calendar flag",
          all(not flags[n]["calendar"] for n in names),
          f"{ {n: flags[n]['calendar'] for n in names} }")

    # Make the middle expiry's variance actually fall BELOW the short one.
    # w = sigma^2 tau, so at flat vol w = [7.7e-4, 3.3e-3, 9.9e-3]; scaling E2
    # by 0.25 still leaves it above E1. It has to go below to be an arbitrage.
    bad = {n: [dict(r) for r in slice_rows(t, skew=0.0, curv=0.0)]
           for n, t in zip(names, taus)}
    for r in bad["E2"]:
        r['w'] *= 0.10
    seq = [bad[n][7]['w'] for n in names]
    check("the planted violation really is non-monotone in tau",
          seq[1] < seq[0], f"w = {[f'{x:.2e}' for x in seq]}")
    flags = v3.audit_surface(bad, ctxs)
    check("a dip in w across tau is flagged as calendar arb",
          len(flags["E2"]["calendar"]) > 0, f"{flags['E2']['calendar']}")

    # The probe must not extrapolate: a slice that does not cover k0 is skipped
    narrow = {n: slice_rows(t, span=0.005) for n, t in zip(names, taus)}
    flags = v3.audit_surface(narrow, ctxs, k_probes=(0.5,))
    check("a probe outside every slice produces no flags",
          all(not flags[n]["calendar"] for n in names))

    # Fixed-k, not fixed-z. Probe an OFF-the-money k: at k = 0 every z is 0 by
    # construction, so it cannot show the difference. At k != 0, z = k/(s*sqrt(tau))
    # shrinks as tau grows, so a common z is a different strike at each maturity.
    idx = 11
    k_at = slice_rows(taus[0], skew=0.0, curv=0.0)[idx]['k']
    zs_at_k = [slice_rows(t, skew=0.0, curv=0.0)[idx]['z'] for t in taus]
    check("z at a fixed non-zero k differs across tau (why the probe uses k)",
          max(zs_at_k) - min(zs_at_k) > 0.5,
          f"k={k_at:.4f} -> z = {[round(z, 3) for z in zs_at_k]}")


# --- A2: SVI ----------------------------------------------------------------
def test_svi():
    print("\nA2 -- SVI slice fitting")
    import fit.svi as svi

    true = dict(a=0.004, b=0.10, rho=-0.7, m=0.01, sigma=0.12)
    k = np.linspace(-0.4, 0.4, 41)
    w = svi.raw_svi(k, **true)

    check("raw_svi is positive across the slice", np.all(w > 0))
    check("raw_svi is convex in k",
          len(vc.butterfly_violations(k, w)) == 0)

    got = svi.fit_slice(k, w, tau=0.25)
    errs = {key: abs(got[key] - true[key]) for key in true}
    check("recovers planted SVI parameters", max(errs.values()) < 1e-3,
          "max err " + f"{max(errs.values()):.2e} ({max(errs, key=errs.get)})")

    fitted = svi.raw_svi(k, **{p: got[p] for p in true})
    check("fitted slice reproduces w to 1e-6",
          float(np.max(np.abs(fitted - w))) < 1e-6)

    # Noise: parameters wobble but the CURVE must stay close
    rng = np.random.default_rng(4)
    wn = w * (1.0 + rng.normal(0, 0.01, size=w.shape))
    gotn = svi.fit_slice(k, wn, tau=0.25)
    curve = svi.raw_svi(k, **{p: gotn[p] for p in true})
    check("under 1% noise the fitted curve stays within 1% of truth",
          float(np.max(np.abs(curve - w) / w)) < 0.01,
          f"max rel {float(np.max(np.abs(curve - w) / w)):.4f}")

    check("Durrleman conditions hold on the fit", svi.durrleman_ok(got, tau=0.25))
    bad = dict(a=-0.5, b=0.10, rho=-0.7, m=0.0, sigma=0.01)
    check("Durrleman rejects a parameter set with negative minimum variance",
          not svi.durrleman_ok(bad, tau=0.25))

    check("too few points -> None",
          svi.fit_slice(np.array([0.0, 0.1]), np.array([0.01, 0.01]), tau=0.25) is None)
    check("a scalar weight is broadcast, not a crash",
          svi.fit_slice(k, w, weights=2.0, tau=0.25) is not None)

    # scipy must not reach the live startup path
    check("importing fit.svi pulls in no scipy", _no_scipy_on_import("fit.svi"))


def _no_scipy_on_import(module):
    import json
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys, json, {module}; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True, text=True)
    try:
        mods = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return False
    return not any(m.startswith("scipy") for m in mods)


# --- A3: risk-neutral density ----------------------------------------------
def test_rnd_panel():
    print("\nA3 -- risk-neutral density from the fitted slice")
    import fit.svi as svi

    tau, F, df = 0.25, 650.0, 0.995
    params = dict(a=0.006, b=0.12, rho=-0.6, m=0.0, sigma=0.15)

    K, dens = v3.slice_density(params, tau, F, df, n=1201, n_sigma=6.0)
    m = np.isfinite(dens)
    check("density is non-negative on an arbitrage-free SVI slice",
          float(np.nanmin(dens)) >= -1e-9, f"min {float(np.nanmin(dens)):.3e}")
    mass = float(np.trapezoid(dens[m], K[m]))
    check("density integrates to ~1", abs(mass - 1.0) < 5e-3, f"{mass:.6f}")

    mean = float(np.trapezoid(K[m] * dens[m], K[m]))
    check("density mean is the forward", abs(mean - F) / F < 5e-3,
          f"E[K] = {mean:.2f} vs F = {F:.2f}")

    check("the strike grid is uniform (fd_second needs it)",
          float(np.ptp(np.diff(K))) < 1e-9,
          "non-uniform spacing would cost an order of accuracy")

    # A butterfly-violating parameter set must show up as negative density
    bad = dict(a=0.006, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
    _, dbad = v3.slice_density(bad, tau, F, df, n=1201, n_sigma=6.0)
    check("a Durrleman-violating slice yields a negative density",
          float(np.nanmin(dbad)) < 0, f"min {float(np.nanmin(dbad)):.4e}")
    check("...and Durrleman flags the same parameter set",
          not svi.durrleman_ok(bad, tau=tau))


# --- A4: second H estimator -------------------------------------------------
def test_dual_hurst():
    print("\nA4 -- two independent H estimates")
    rng = np.random.default_rng(11)

    # Exact fBm at several roughnesses -- the estimator must track all of them,
    # not just the one value the surface happens to carry.
    for H_true in (0.10, 0.12, 0.30, 0.50):
        fbm = _fbm(2 ** 16, H_true, rng)
        res = vc.hurst_from_structure(fbm)
        check(f"structure-function H recovers planted H={H_true}",
              abs(res['H'] - H_true) < 0.02,
              f"got {res['H']:.4f}, linearity r2={res['linearity_r2']:.5f}")

    fbm = _fbm(2 ** 16, 0.12, rng)
    check("classical (H=0.5) and rough (H=0.12) are distinguishable",
          vc.hurst_from_structure(_fbm(2 ** 16, 0.50, rng))['H']
          - vc.hurst_from_structure(fbm)['H'] > 0.30)

    status = v3.hurst_status(fbm, min_samples=500)
    check("hurst_status reports a value when there is enough history",
          status['ready'] and status['H'] is not None,
          f"H={status['H']:.4f} from n={status['n']}")

    short = v3.hurst_status(fbm[:50], min_samples=500)
    check("hurst_status refuses to guess on a short history",
          not short['ready'] and short['H'] is None, f"{short}")
    check("...and says how many more samples it needs",
          short['needed'] == 450, f"needed={short['needed']}")

    check("agreement flag is set when the two H's are close",
          v3.hurst_agreement(0.12, 0.13)['agree'])
    check("agreement flag clears when they diverge",
          not v3.hurst_agreement(0.12, 0.35)['agree'])
    check("divergence reports the gap",
          abs(v3.hurst_agreement(0.12, 0.35)['gap'] - 0.23) < 1e-9)


def _fbm(n, H, rng):
    """
    Exact fractional Brownian motion by Davies-Harte circulant embedding.

    An approximate generator is worthless here: the whole point of the test is
    that H comes back, so the planted H has to be exactly right. A cheap
    convolution helper was tried first and returned the wrong exponent, which
    would have looked like an estimator bug rather than a generator bug.
    """
    k = np.arange(0, n)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([g, [0.0], g[:0:-1]])
    lam = np.fft.fft(c).real
    lam = np.maximum(lam, 0.0)
    m = len(c)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    fgn = np.fft.fft(np.sqrt(lam / (2 * m)) * z).real[:n]
    return np.cumsum(fgn)


# --- A5: replay -------------------------------------------------------------
def test_replay():
    print("\nA5 -- record and replay")
    import tempfile
    import pathlib
    import sources.replay as replay

    # Four expiries, so roughness() actually produces a number and the
    # "same H after replay" check is not vacuously None == None.
    ctxs = {"20260911": Ctx(7 / 365, forward=650.4, sigma_atm=0.14),
            "20260916": Ctx(12 / 365, forward=650.6, sigma_atm=0.15),
            "20260921": Ctx(17 / 365, forward=650.9, sigma_atm=0.16),
            "20261002": Ctx(28 / 365, forward=651.5, sigma_atm=0.18)}
    app = _fake_app(ctxs)

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "snap.json"
        replay.record(app, ctxs, path, symbol="SPY")
        check("record writes a file", path.exists() and path.stat().st_size > 0)

        app2, ctxs2 = replay.load(path)
        check("symbol survives", app2.symbol == "SPY")
        check("spot survives", abs(app2.spot_price - app.spot_price) < 1e-12)
        check("quote count survives", len(app2.quotes) == len(app.quotes))
        check("expiry contexts survive",
              set(ctxs2) == set(ctxs)
              and all(abs(ctxs2[e].forward - ctxs[e].forward) < 1e-12 for e in ctxs))

        a = v3.surface_points(app, ctxs, max_age=1e9)
        b = v3.surface_points(app2, ctxs2, max_age=1e9)
        check("replayed surface points are identical to the recorded ones",
              _points_equal(a, b),
              f"{sum(len(v) for v in a.values())} vs {sum(len(v) for v in b.values())} points")

        res_a = v3.roughness(a, ctxs)[0]
        res_b = v3.roughness(b, ctxs2)[0]
        check("roughness actually produced an H (not a vacuous None)",
              res_a is not None, f"{res_a}")
        check("the same H comes out of the replay, bit for bit",
              res_a is not None and res_b is not None
              and abs(res_a[0] - res_b[0]) < 1e-12,
              f"H {res_a[0]:.10f} vs {res_b[0]:.10f}" if res_a and res_b else "None")

        flags_a = v3.audit_surface(a, ctxs)
        flags_b = v3.audit_surface(b, ctxs2)
        check("the arbitrage audit agrees across the replay",
              {e: flags_a[e]['butterfly'] for e in flags_a}
              == {e: flags_b[e]['butterfly'] for e in flags_b})


def _fake_app(ctxs):
    app = v3.LiveSurfaceApp()
    app.spot_price = 650.0
    rid = 3000
    for exp, ctx in ctxs.items():
        for K in [630.0, 640.0, 650.0, 660.0, 670.0]:
            right = vc.otm_right(K, ctx.forward)
            sig = ctx.sigma_atm - 1.0 * math.log(K / ctx.forward)
            q = v3.OptionQuote(expiry=exp, strike=K, right=right)
            q.iv = sig
            q.vega = float(vc.bs_vega(ctx.forward, K, sig, ctx.tau))
            q.bid, q.ask = 1.00, 1.02
            q.model_price = 1.01
            q.und_price = 650.0
            q.ts = 1000.0
            app.id_map[rid] = (exp, K, right)
            app.quotes[rid] = q
            rid += 1
    return app


def _points_equal(a, b):
    if set(a) != set(b):
        return False
    for e in a:
        if len(a[e]) != len(b[e]):
            return False
        for ra, rb in zip(a[e], b[e]):
            for key in ('z', 'k', 'w', 'iv', 'strike'):
                if abs(ra[key] - rb[key]) > 1e-12:
                    return False
    return True


if __name__ == "__main__":
    print("=" * 74)
    print("Session A -- surface assembly")
    print("=" * 74)
    test_audit_butterfly()
    test_audit_calendar()
    test_svi()
    test_rnd_panel()
    test_dual_hurst()
    test_replay()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
