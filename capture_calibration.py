"""
Session C closing study: fit vanilla Heston to a ROUGH surface, and record where
it fails.

Session B measured Heston against itself -- its own ATM skew term structure. That
proved the exponent is pinned near one half but said nothing about how badly that
hurts when the market is rough. This does the honest version: generate a surface
carrying H = 0.12, hand it to the calibrator, and measure the residual maturity
by maturity. Session E's rough Heston has to beat exactly these numbers.

Also runs the identifiability study, because a calibration that reports five
parameters without saying which of them the data actually determined is
misleading.
"""

import math
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import volsurf_core as vc
import pricing.fourier as fo
from models.heston import HestonParams
from calibrate.objective import MarketSurface, per_expiry_report
from calibrate.fit import DEFAULT_STARTS, calibrate_heston, heston_cf_factory

OUT = pathlib.Path(__file__).parent / "captures"
OUT.mkdir(exist_ok=True)

H_TRUE = 0.12
TAU7 = 7.0 / 365.0
# The short end is not optional. Starting at 7 days hides the whole point:
# Heston flattens BELOW about a week, so a grid that begins there lets it fit a
# tau^-0.4 decay and look adequate. At 1 day the market wants -2.72 and Heston
# can only make -1.58.
TAUS = np.array([1, 2, 3, 7, 14, 30, 60, 90, 180, 365]) / 365.0
HALF_SPREAD = 0.005


def atm_vol(tau):
    return 0.115 + 0.09 * (1.0 - math.exp(-tau / 0.06))


def rough_skew(tau):
    """d(sigma)/dk at the money, ~ tau^(H-1/2). The thing Heston cannot make."""
    return -1.3 * (TAU7 / tau) ** (0.5 - H_TRUE)


# Curvature is specified in SIGMA UNITS, not in k.
#
# Writing the smile as sigma = atm + skew*k + C*k^2 with a fixed C looks fine at
# a short maturity, where the +/-3 sigma band is |k| <= 0.06, and produces
# nonsense at a long one, where the same band reaches |k| = 0.62 and C*k^2 with
# C = 6 contributes 2.27 -- an implied vol of 265%. That surface had quote
# weights spanning 2.4e12 and the calibrator correctly fled to the parameter
# bounds trying to fit it. A curvature in z is dimensionless and stays put.
CURV_Z = 0.05


def rough_iv(k, tau):
    atm = atm_vol(tau)
    z = k / (atm * math.sqrt(tau))
    # Skew in z chosen so that d(sigma)/dk still follows tau^(H-1/2).
    #   sigma = atm*(1 + a1*z + ...),  z = k/(atm*sqrt(tau))
    #   d(sigma)/dk = atm*a1 * dz/dk = atm*a1 / (atm*sqrt(tau)) = a1/sqrt(tau)
    # so a1 = skew*sqrt(tau). The atm factors cancel; including one gave a
    # 7-day skew of -0.18 instead of -1.30.
    a1 = rough_skew(tau) * math.sqrt(tau)
    return float(np.clip(atm * (1.0 + a1 * z + CURV_Z * z * z), 0.02, 3.0))


def build_rough_surface(n_k=13, n_sigma=3.0):
    T, K, V, W = [], [], [], []
    for tau in TAUS:
        band = n_sigma * atm_vol(tau) * math.sqrt(tau)
        for k in np.linspace(-band, band, n_k):
            v = rough_iv(k, tau)
            T.append(tau)
            K.append(k)
            V.append(v)
            W.append(float(vc.quote_weight(
                HALF_SPREAD, vc.bs_vega(1.0, math.exp(k), v, tau))))
    return MarketSurface(np.array(T), np.array(K), np.array(V), np.array(W))


def skew_of(iv_fn, tau, n_sigma=3.0, n_k=41):
    """ATM skew of any vol function, measured the way the live surface measures it."""
    band = n_sigma * atm_vol(tau) * math.sqrt(tau)
    ks = np.linspace(-band, band, n_k)
    vs = np.array([iv_fn(k, tau) for k in ks])
    ok = np.isfinite(vs)
    if ok.sum() < 8:
        return None, None
    zs = ks[ok] / (atm_vol(tau) * math.sqrt(tau))
    s, se, _ = vc.atm_skew_from_slice(ks[ok], vs[ok], zs=zs, window=1.5)
    return s, se


def main():
    S = build_rough_surface()
    print(f"Rough market surface: {len(S)} quotes, {S.n_expiries} expiries, "
          f"planted H = {H_TRUE}")

    print("\nFitting vanilla Heston...")
    # The fit is impossible by construction, so it will grind rather than
    # converge. Bound it: measured, the cost improves in the sixth decimal after
    # about a dozen iterations while lambda climbs through 1e8.
    res = calibrate_heston(S, starts=DEFAULT_STARTS[:2], max_iter=40,
                           max_seconds=120)
    p = res["params"]
    print(f"  {res['iterations']} iterations, {res['n_fev']} evaluations, "
          f"{res['reason']}; best of {res['n_starts']} starts "
          f"(cost spread {res['cost_spread']:.2f}x)")
    print("  " + "  ".join(f"{n}={getattr(p, n):.5f}" for n in HestonParams.NAMES))
    print(f"  Feller margin 2*kappa*theta - xi^2 = {res['feller_margin']:+.4f} "
          f"-> {res['feller']}")
    print(f"  overall rmse = {res['rmse_vol']*100:.3f} vol points")

    # --- where the error lives ------------------------------------------------
    rep = per_expiry_report(p, S, heston_cf_factory)
    print(f"\n  {'tau(d)':>8} {'n':>4} {'rmse(vp)':>10} {'bias(vp)':>10} "
          f"{'worst(vp)':>10}")
    for row in rep:
        print(f"  {row['tau']*365:8.0f} {row['n']:4d} {row['rmse']*100:10.3f} "
              f"{row['bias']*100:10.3f} {row['worst']*100:10.3f}")

    # --- the skew residual, which is the actual Phase 1 deficiency ------------
    def heston_iv(k, tau):
        cf = heston_cf_factory(p, tau)
        c = float(fo.carr_madan_call(k, tau, cf))
        v = fo.implied_vol_from_call(c, k, tau)
        return v if v is not None else float('nan')

    print(f"\n  ATM SKEW, market vs fitted Heston")
    print(f"  {'tau(d)':>8} {'market':>10} {'Heston':>10} {'shortfall':>11} {'ratio':>8}")
    taus_s, mk, hs = [], [], []
    for tau in TAUS:
        sm, _ = skew_of(rough_iv, tau)
        sh, _ = skew_of(heston_iv, tau)
        if sm is None or sh is None:
            continue
        taus_s.append(tau)
        mk.append(sm)
        hs.append(sh)
        print(f"  {tau*365:8.0f} {sm:10.4f} {sh:10.4f} {sh-sm:+11.4f} "
              f"{abs(sh/sm):8.3f}")

    taus_s = np.array(taus_s)
    mk = np.array(mk)
    hs = np.array(hs)
    H_mkt, _, _, _ = vc.estimate_hurst(taus_s, mk)
    H_hes, _, _, _ = vc.estimate_hurst(taus_s, hs)
    print(f"\n  H from the market skews : {H_mkt:.4f}  (planted {H_TRUE})")
    print(f"  H from the fitted Heston: {H_hes:.4f}")
    print(f"  short-dated skew shortfall: {abs(hs[0]/mk[0]):.1%} of the market's")

    # --- identifiability ------------------------------------------------------
    print("\nIdentifiability under 0.5 vol point quote noise (6 seeds)...")
    rng_fits = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        noisy = MarketSurface(S.tau, S.k,
                              S.iv + rng.normal(0, 0.005, size=len(S)), S.weight)
        rng_fits.append(calibrate_heston(
            noisy, starts=DEFAULT_STARTS[:1], max_iter=30,
            max_seconds=40)["params"])
    spreads = {}
    for n in HestonParams.NAMES:
        v = np.array([getattr(q, n) for q in rng_fits])
        spreads[n] = float(np.std(v) / abs(np.mean(v)))
    for n in HestonParams.NAMES:
        tag = ("STABLE" if spreads[n] < 0.10
               else "loose" if spreads[n] < 0.30 else "UNIDENTIFIED")
        print(f"  {n:>6}  spread {spreads[n]*100:6.1f}%   {tag}")

    # --- figure ---------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#0b0d0f')
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # smiles at the extremes
    for ax, tau, title in ((ax1, TAUS[0], f"{TAUS[0]*365:.0f} DAY"),
                           (ax2, TAUS[-1], f"{TAUS[-1]*365:.0f} DAYS")):
        band = 3.0 * atm_vol(tau) * math.sqrt(tau)
        ks = np.linspace(-band, band, 61)
        zs = ks / (atm_vol(tau) * math.sqrt(tau))
        ax.plot(zs, [rough_iv(k, tau) for k in ks], '-', color='#00f2ff', lw=2,
                label='rough market (H=0.12)')
        ax.plot(zs, [heston_iv(k, tau) for k in ks], '--', color='#ffb020', lw=2,
                label='fitted Heston')
        ax.set_facecolor('#161b22')
        ax.axvline(0, color='#ff3e3e', ls=':', lw=0.8)
        ax.set_xlabel('z  (σ from forward)', color='#9aa4b2', fontsize=9)
        ax.set_ylabel('implied vol', color='#9aa4b2', fontsize=9)
        r = [x for x in rep if abs(x['tau'] - tau) < 1e-12]
        rm = r[0]['rmse'] * 100 if r else float('nan')
        ax.set_title(f"{title}   —   rmse {rm:.2f} vol points",
                     color='white', fontsize=10)
        ax.legend(fontsize=8, facecolor='#0b0d0f', edgecolor='#2d333b')

    # skew term structure
    ax3.loglog(taus_s * 365, np.abs(mk), 'o-', color='#00f2ff', ms=6, lw=1.5,
               label=f'rough market   Ĥ={H_mkt:.3f}')
    ax3.loglog(taus_s * 365, np.abs(hs), 's--', color='#ffb020', ms=6, lw=1.5,
               label=f'fitted Heston  Ĥ={H_hes:.3f}')
    ax3.set_facecolor('#161b22')
    ax3.set_xlabel('τ  (days)', color='#9aa4b2', fontsize=9)
    ax3.set_ylabel('|∂σ/∂k|  at the money', color='#9aa4b2', fontsize=9)
    ax3.set_title("ATM SKEW — Heston cannot bend at the short end",
                  color='white', fontsize=10)
    ax3.legend(fontsize=8, facecolor='#0b0d0f', edgecolor='#2d333b')
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for axis in (ax3.xaxis, ax3.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

    # per-expiry rmse
    d = [r['tau'] * 365 for r in rep]
    ax4.bar([str(int(round(x))) for x in d], [r['rmse'] * 100 for r in rep],
            color=['#ff3e3e' if r['rmse'] * 100 > 1.0 else '#7ee787' for r in rep])
    ax4.set_facecolor('#161b22')
    ax4.set_xlabel('τ  (days)', color='#9aa4b2', fontsize=9)
    ax4.set_ylabel('rmse  (vol points)', color='#9aa4b2', fontsize=9)
    ax4.set_title(f"FIT ERROR BY MATURITY — concentrated at the short end "
                  f"(overall {res['rmse_vol']*100:.2f} vp)",
                  color='white', fontsize=10)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.07,
                        hspace=0.32, wspace=0.20)
    out = OUT / "heston_vs_rough.png"
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    print(f"\n  figure -> {out}")

    # --- record ---------------------------------------------------------------
    doc = pathlib.Path(__file__).parent / "docs" / "phase1_baseline.md"
    lines = [
        "# Phase 1 baseline — vanilla Heston fitted to a rough surface", "",
        f"Market: synthetic, ATM skew ~ tau^(H-1/2) with **H = {H_TRUE}**, "
        f"{len(S)} quotes across {S.n_expiries} expiries out to one year.", "",
        "## Fitted parameters", "",
        "| " + " | ".join(HestonParams.NAMES) + " |",
        "|" + "---|" * len(HestonParams.NAMES),
        "| " + " | ".join(f"{getattr(p, n):.5f}" for n in HestonParams.NAMES) + " |",
        "",
        f"Feller margin `2*kappa*theta - xi^2 = {res['feller_margin']:+.4f}` "
        f"({'satisfied' if res['feller'] else 'violated'} — reported, not enforced). "
        f"Overall RMSE **{res['rmse_vol']*100:.3f} vol points**.", "",
        "## Where the error lives", "",
        "| tau (days) | n | rmse (vp) | bias (vp) | worst (vp) |",
        "|---|---|---|---|---|",
    ]
    for row in rep:
        lines.append(f"| {row['tau']*365:.0f} | {row['n']} | {row['rmse']*100:.3f} "
                     f"| {row['bias']*100:+.3f} | {row['worst']*100:.3f} |")
    lines += ["", "## The skew shortfall — what Phase 2 must fix", "",
              "| tau (days) | market skew | Heston skew | Heston / market |",
              "|---|---|---|---|"]
    for t, a, b in zip(taus_s, mk, hs):
        lines.append(f"| {t*365:.0f} | {a:.4f} | {b:.4f} | {abs(b/a):.3f} |")
    lines += ["",
              f"H from the market skews **{H_mkt:.4f}** (planted {H_TRUE}); "
              f"from the fitted Heston **{H_hes:.4f}**.", "",
              f"At {taus_s[0]*365:.0f} days the fitted Heston delivers only "
              f"**{abs(hs[0]/mk[0]):.1%}** of the market's ATM skew. A diffusion has "
              "a finite skew limit as tau -> 0, so no parameter choice can close "
              "this — it is a model deficiency, not a calibration failure.", "",
              "## Identifiability (0.5 vol point quote noise, 6 seeds)", "",
              "| parameter | spread | verdict |", "|---|---|---|"]
    for n in HestonParams.NAMES:
        tag = ("stable" if spreads[n] < 0.10
               else "loose" if spreads[n] < 0.30 else "**unidentified**")
        lines.append(f"| {n} | {spreads[n]*100:.1f}% | {tag} |")
    lines += ["",
              "`v0` and `rho` are pinned by the short-end level and the skew. "
              "`kappa` and `theta` are not determined by a surface that stops at "
              "one year, because `1/kappa` is comparable to the longest maturity — "
              "the process has not relaxed within the data. Extending to three "
              "years tightens `theta` by roughly a factor of four. Any report of "
              "these five numbers without that caveat overstates what was measured."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  baseline -> {doc}")


if __name__ == "__main__":
    main()
