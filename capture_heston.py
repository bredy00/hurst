"""
Session B closing figure: what Heston's ATM skew does as tau -> 0.

This is the Phase 1 baseline. The market's ATM skew explodes like tau^(H-1/2)
with H ~ 0.1; a classical diffusion cannot do that, and the point of Phase 2 is
to fix it. Rather than assert that, measure it here with the same estimator the
live surface uses, so Session E's comparison has a number to beat.
"""

import math
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import volsurf_core as vc
import pricing.fourier as fo
from models.heston import HestonParams, char_func

OUT = pathlib.Path(__file__).parent / "captures"
OUT.mkdir(exist_ok=True)

P = HestonParams(v0=0.04, kappa=2.0, theta=0.045, xi=0.5, rho=-0.7)
TAUS = np.array([1, 2, 5, 10, 21, 42, 63, 126, 252, 504]) / 252.0
H_MARKET = 0.12


def heston_smile(tau, n_sigma=3.0, n_k=41):
    """Implied vols across +/- n_sigma, in the same z coordinate as the surface."""
    cf = lambda u: char_func(u, tau, P)
    atm = math.sqrt(P.v0)
    band = n_sigma * atm * math.sqrt(tau)
    ks = np.linspace(-band, band, n_k)
    vols = fo.smile(ks, tau, cf)
    ok = [(k, v) for k, v in zip(ks, vols) if v is not None and np.isfinite(v)]
    ks = np.array([a for a, _ in ok])
    vs = np.array([b for _, b in ok])
    return ks, vs, atm


def main():
    print("Heston ATM skew term structure (Phase 1 baseline)")
    print(f"  params: v0={P.v0} kappa={P.kappa} theta={P.theta} "
          f"xi={P.xi} rho={P.rho}   Feller={P.feller}")
    print(f"\n  {'tau(d)':>8} {'ATM vol':>9} {'ATM skew':>11} {'stderr':>10} {'n':>4}")

    taus, skews, errs, smiles = [], [], [], {}
    for tau in TAUS:
        ks, vs, atm = heston_smile(tau)
        if len(ks) < 8:
            continue
        zs = ks / (atm * math.sqrt(tau))
        s, se, n = vc.atm_skew_from_slice(ks, vs, zs=zs, window=1.5)
        if s is None:
            continue
        taus.append(tau)
        skews.append(s)
        errs.append(se)
        smiles[tau] = (zs, vs)
        print(f"  {tau*252:8.0f} {vs[len(vs)//2]:9.4f} {s:11.4f} "
              f"{se if se else float('nan'):10.2e} {n:4d}")

    taus = np.array(taus)
    skews = np.array(skews)

    H, H_err, icpt, r2 = vc.estimate_hurst(taus, skews, errs)
    print(f"\n  Heston   H = {H:.4f} +/- {H_err:.4f}   (r2 = {r2:.4f})")

    # The short end alone, which is where the difference lives
    short = taus <= 21 / 252
    Hs, Hs_err, icpt_s, r2_s = vc.estimate_hurst(taus[short], skews[short],
                                                 [errs[i] for i in np.where(short)[0]])
    print(f"  short end (<= 21d):  H = {Hs:.4f} +/- {Hs_err:.4f}")
    print(f"  market is around H ~ {H_MARKET}; a classical diffusion cannot reach it")
    print(f"  at 1 day Heston's skew is {abs(skews[0]):.3f}; a market with "
          f"H={H_MARKET} would be near "
          f"{abs(skews[-1]) * (taus[-1]/taus[0])**(0.5-H_MARKET):.3f}")

    # --- figure ---------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('#0b0d0f')
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    cmap = plt.get_cmap('viridis')
    for i, tau in enumerate(sorted(smiles)):
        zs, vs = smiles[tau]
        ax1.plot(zs, vs, '-', lw=1.4, color=cmap(i / max(len(smiles) - 1, 1)),
                 label=f"{tau*252:.0f}d")
    ax1.set_facecolor('#161b22')
    ax1.set_xlabel('z  =  ln(K/F) / (σ√τ)', color='#9aa4b2', fontsize=9)
    ax1.set_ylabel('implied vol', color='#9aa4b2', fontsize=9)
    ax1.set_title('HESTON SMILES  —  note the short-dated slices FLATTEN',
                  color='white', fontsize=10)
    ax1.legend(fontsize=7, ncol=2, facecolor='#0b0d0f', edgecolor='#2d333b')
    ax1.axvline(0, color='#ff3e3e', ls='--', lw=0.8)

    d = taus * 252
    ax2.loglog(d, np.abs(skews), 'o', color='#ffb020', ms=6, label='Heston')
    xs = np.linspace(taus.min(), taus.max(), 100)
    ax2.plot(xs * 252, np.exp(icpt) * xs ** (H - 0.5), '-', color='#ffb020',
             lw=1.4, alpha=0.8, label=f'fit: H = {H:.3f}')
    anchor = abs(skews[-1]) * taus[-1] ** (0.5 - H_MARKET)
    ax2.plot(xs * 252, anchor * xs ** (H_MARKET - 0.5), '--', color='#00f2ff',
             lw=1.4, label=f'market-like H = {H_MARKET}')
    ax2.set_facecolor('#161b22')
    ax2.set_xlabel('τ  (trading days)', color='#9aa4b2', fontsize=9)
    ax2.set_ylabel('|∂σ/∂k|  at the money', color='#9aa4b2', fontsize=9)
    ax2.set_title(f'ATM SKEW ~ τ^(H-½)      Heston Ĥ = {H:.3f}   vs market ≈ {H_MARKET}',
                  color='white', fontsize=10)
    ax2.text(0.03, 0.06, 'the short-end gap is what Phase 2 must close',
             transform=ax2.transAxes, color='#6b7684', fontsize=8)
    ax2.legend(fontsize=8, facecolor='#0b0d0f', edgecolor='#2d333b')
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for axis in (ax2.xaxis, ax2.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.22)
    out = OUT / "heston_skew_baseline.png"
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    print(f"\n  figure -> {out}")

    doc = pathlib.Path(__file__).parent / "docs" / "phase1_baseline.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 1 baseline — vanilla Heston ATM skew", "",
             f"Parameters: v0={P.v0}, kappa={P.kappa}, theta={P.theta}, "
             f"xi={P.xi}, rho={P.rho} (Feller {P.feller})", "",
             "| tau (days) | ATM skew | stderr |", "|---|---|---|"]
    for t, s, e in zip(taus, skews, errs):
        lines.append(f"| {t*252:.0f} | {s:.4f} | {e:.2e} |")
    lines += ["",
              f"Fitted **H = {H:.4f} ± {H_err:.4f}** (r² = {r2:.4f}); "
              f"short end alone **H = {Hs:.4f}**.", "",
              f"The market sits near H ≈ {H_MARKET}. Heston's exponent is pinned "
              "close to 1/2 because a classical diffusion has a finite ATM skew "
              "limit as tau -> 0. This is the number Phase 2 has to beat."]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  baseline -> {doc}")


if __name__ == "__main__":
    main()
