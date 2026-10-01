"""
Session O -- eSSVI, the surface that cannot be arbitraged (fit/essvi.py).

  slices      each eSSVI slice is exactly a raw SVI slice, and Gatheral-Jacquier's condition
              keeps Durrleman's g positive -- checked on random slices, not assumed
  calendar    the two cheap conditions are necessary but NOT sufficient: the counterexample
              the random search found passes both and crosses, and the guard rejects it
  fit         a planted surface is recovered; a noisy rough-Heston surface comes out with no
              arbitrage, where raw SVI fitted slice by slice to the same quotes crosses

    python test_essvi.py
"""

import math
import sys

import numpy as np

import fit.essvi as es
import fit.svi as svi

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_slices():
    print("\none slice")
    rng = np.random.default_rng(1)
    k = np.linspace(-2, 2, 801)
    conv, g_min, n = 0.0, np.inf, 0
    while n < 2000:
        th, r, psi = rng.uniform(5e-4, 0.3), rng.uniform(-0.99, 0.99), rng.uniform(1e-3, 3.0)
        if not es.butterfly_ok(th, r, psi):
            continue
        n += 1
        raw = es.to_raw_svi(th, r, psi)
        conv = max(conv, float(np.max(np.abs(svi.raw_svi(k, **raw) - es.essvi_w(k, th, r, psi)))))
        g_min = min(g_min, svi.durrleman_min(raw, 1.0, 3.0))
    check("an eSSVI slice is exactly a raw SVI slice", conv < 1e-14, f"worst difference {conv:.1e}")
    check("Gatheral-Jacquier's condition keeps Durrleman's g positive on 2000 random slices", g_min > 0,
          f"smallest g {g_min:+.3f}")
    check("...and the condition refuses a slice that breaks it", not es.butterfly_ok(0.01, 0.5, 3.0))


def test_calendar():
    print("\nconsecutive slices")
    prev = {"theta": 0.19751342800280938, "rho": -0.06694123118980044, "psi": 0.006656019081863228}
    th, r, psi = 0.3717105066840333, 0.9856673053322245, 1.3014524582640832
    necessary = th >= prev["theta"] and (psi - prev["psi"]) >= abs(r * psi - prev["rho"] * prev["psi"])
    k = np.linspace(-3, 3, 6001)
    cross = float((es.essvi_w(k, th, r, psi) - es.essvi_w(k, **prev)).min())
    check("the counterexample passes both cheap conditions and still crosses", necessary and cross < -0.15,
          f"worst crossing {cross:+.3f} in total variance")
    check("...and the dense-grid guard rejects it", not es.calendar_ok(prev, th, r, psi))
    rng = np.random.default_rng(0)
    far = np.geomspace(10.0, 1e4, 3000)
    k = np.concatenate([-far[::-1], np.linspace(-10, 10, 8001), far])   # ground truth, far past any quote
    agree, n = 0, 0
    while n < 3000:
        t1 = rng.uniform(1e-3, 0.2)
        p1 = dict(theta=t1, rho=rng.uniform(-0.99, 0.99), psi=rng.uniform(1e-3, 1.0))
        t2, r2, s2 = t1 * rng.uniform(1.0, 3.0), rng.uniform(-0.99, 0.99), rng.uniform(1e-3, 2.0)
        if not (s2 - p1["psi"] >= abs(r2 * s2 - p1["rho"] * p1["psi"])):
            continue
        n += 1
        crosses = bool((es.essvi_w(k, t2, r2, s2) - es.essvi_w(k, **p1)).min() < -1e-12)
        agree += crosses == (not es.calendar_ok(p1, t2, r2, s2))
    check("on 3000 random pairs the guard rejects exactly those that cross, out to |k| = 10^4",
          agree == n, f"{agree} of {n} agree")
    prev = {"theta": 0.19004835999064354, "rho": 0.9341225583406774, "psi": 0.046705749706769356}
    th, r, psi = 0.4858756539480947, -0.4894176414986581, 0.1870504030340544
    kk = np.linspace(8.0, 23.0, 1501)
    far_cross = float((es.essvi_w(kk, th, r, psi) - es.essvi_w(kk, **prev)).min())
    check("a pair that crosses only far out (k = 8 to 23) is rejected too", far_cross < 0
          and not es.calendar_ok(prev, th, r, psi), f"worst crossing {far_cross:+.4f} there")


def test_fit():
    print("\nfitting")
    taus = [d / 365 for d in (2, 7, 14, 30, 90)]
    planted, prev, sl = [], None, []
    for t in taus:
        th = 0.04 * t + 0.002 * math.sqrt(t)
        f = dict(theta=th, rho=-0.7 + 0.2 * t, psi=0.35 * math.sqrt(th))
        assert es.feasible(prev, f["theta"], f["rho"], f["psi"])
        planted.append(f)
        prev = f
        kk = np.linspace(-3, 3, 13) * math.sqrt(th)
        sl.append((t, kk, np.sqrt(es.essvi_w(kk, **f) / t)))
    got = es.fit_surface(sl)
    err = max(abs(g[p] - f[p]) / abs(f[p]) for g, f in zip(got, planted) for p in ("theta", "rho", "psi"))
    check("a planted surface is recovered", err < 1e-3, f"worst relative parameter error {err:.1e}")
    a = es.audit(got)
    check("...with no arbitrage in the audit", a["durrleman_min"] > 0 and a["worst_crossing"] >= 0, f"{a}")

    import models.rough_heston as rh
    import sources.synthetic as syn
    iv_fn = syn.model_iv_fn(rh.RoughHestonParams(0.025, 2.0, 0.035, 0.4, -0.7, 0.10))
    rng = np.random.default_rng(7)
    noisy = []
    for t in taus:
        atm = iv_fn(0.0, t)
        kk = np.linspace(-2.5, 2.5, 11) * atm * math.sqrt(t)
        noisy.append((t, kk, np.array([iv_fn(x, t) for x in kk]) + 0.003 * rng.standard_normal(11)))
    ef = es.fit_surface(noisy)
    ea = es.audit(ef)
    raw = [svi.fit_slice(kk, iv ** 2 * t, tau=t) for t, kk, iv in noisy]
    ra = es.audit_raw_svi(raw, taus)
    check("0.3 vp of noise on rough Heston: eSSVI comes out with no arbitrage",
          ea["durrleman_min"] > 0 and ea["worst_crossing"] >= 0, f"{ea}")
    check("...where raw SVI, slice by slice, crosses on the same quotes", ra["crossing_pairs"] > 0,
          f"{ra['crossing_pairs']} crossing pair(s), worst {ra['worst_crossing']:+.1e}")
    check("a slice with fewer quotes than parameters is refused, not fitted",
          es.fit_slice([0.0, 0.1], [0.2, 0.21], 0.1) is None)


if __name__ == "__main__":
    print("=" * 74)
    print("Session O -- eSSVI")
    print("=" * 74)
    test_slices()
    test_calendar()
    test_fit()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
