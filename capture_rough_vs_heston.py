"""
Session E closing study -- the headline result of the project.

Fit vanilla Heston (Phase 1) and lifted rough Heston (Phase 2) to the SAME
rough market surface, then compare the ATM skew term structure of the three:
market, Heston, rough Heston. The market carries d(sigma)/dk ~ tau^(H - 1/2)
with H = 0.12. A diffusion has a finite skew limit as tau -> 0, so vanilla
Heston's log-log slope must flatten towards 0 at the short end; the rough
model's must follow the market's H - 1/2.

Numbers go to docs/phase2_rough_vs_heston.md, the figure to
captures/rough_vs_heston.png. The rough fit is minutes, not seconds: about
1.5 s per objective evaluation on this 130-quote surface.

    python capture_rough_vs_heston.py
"""

import math
import pathlib
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import volsurf_core as vc
import pricing.fourier as fo
import models.rough_heston as rh
from models.heston import HestonParams
from calibrate.objective import per_expiry_report
from calibrate.fit import (DEFAULT_STARTS, DEFAULT_ROUGH_STARTS, calibrate_heston,
                           calibrate_rough_heston, heston_cf_factory)
from capture_calibration import (H_TRUE, TAUS, atm_vol, build_rough_surface, rough_iv,
                                 skew_of)

OUT = pathlib.Path(__file__).parent / "captures"
OUT.mkdir(exist_ok=True)
DOC = pathlib.Path(__file__).parent / "docs" / "phase2_rough_vs_heston.md"


def model_iv_fn(kind, p):
    """An implied-vol function sigma(k, tau) for either model."""
    if kind == "heston":
        def f(k, tau):
            c = float(fo.carr_madan_call(k, tau, heston_cf_factory(p, tau)))
            v = fo.implied_vol_from_call(c, k, tau)
            return v if v is not None else float("nan")
        return f

    def g(k, tau):
        c, _ = rh.call_prices(np.array([k]), tau, p)
        v = fo.implied_vol_from_call(float(c[0]), k, tau)
        return v if v is not None else float("nan")
    return g


def skew_curve(iv_fn):
    taus, skews = [], []
    for tau in TAUS:
        s, _ = skew_of(iv_fn, tau)
        if s is not None and np.isfinite(s):
            taus.append(tau)
            skews.append(s)
    return np.array(taus), np.array(skews)


def slope_short_end(taus, skews, max_days=14):
    """log-log slope of |skew| vs tau over the short end only."""
    m = taus * 365 <= max_days
    if m.sum() < 3:
        return float("nan")
    x, y = np.log(taus[m]), np.log(np.abs(skews[m]))
    return float(np.polyfit(x, y, 1)[0])


def main():
    S = build_rough_surface()
    print(f"Rough market surface: {len(S)} quotes, {S.n_expiries} expiries, planted H = {H_TRUE}")

    print("\nPhase 1: vanilla Heston ...")
    t0 = time.perf_counter()
    r_h = calibrate_heston(S, starts=DEFAULT_STARTS[:2], max_iter=40, max_seconds=120)
    ph = r_h["params"]
    print(f"  {time.perf_counter()-t0:.0f}s, rmse {r_h['rmse_vol']*100:.3f} vp, "
          + "  ".join(f"{n}={getattr(ph, n):.4f}" for n in HestonParams.NAMES))

    print("\nPhase 2: lifted rough Heston ...")
    t0 = time.perf_counter()
    r_r = calibrate_rough_heston(S, starts=DEFAULT_ROUGH_STARTS[:1], max_iter=40,
                                 max_seconds=1500,
                                 callback=lambda it, x, cost, lam:
                                 print(f"   it {it:2d} cost {cost:.4e} lam {lam:.1e}", flush=True))
    pr = r_r["params"]
    print(f"  {time.perf_counter()-t0:.0f}s, {r_r['iterations']} iters, {r_r['reason']}, "
          f"rmse {r_r['rmse_vol']*100:.3f} vp")
    print("  " + "  ".join(f"{n}={getattr(pr, n):.4f}" for n in rh.RoughHestonParams.NAMES))
    print(f"  kernel error at fitted H: {r_r['kernel_error']:.2%}; pricer {r_r['pricer_stats']}")

    # --- per-expiry errors --------------------------------------------------
    rep_h = per_expiry_report(ph, S, heston_cf_factory)
    pricer = rh.RoughPricer(tol=1e-9)
    rep_r = per_expiry_report(pr, S, lambda p, t: rh.cf_factory(p, t), pricer=pricer)
    print(f"\n  {'tau(d)':>7} {'Heston rmse':>12} {'rough rmse':>11}   (vol points)")
    for a, b in zip(rep_h, rep_r):
        print(f"  {a['tau']*365:7.0f} {a['rmse']*100:12.3f} {b['rmse']*100:11.3f}")

    # --- the skew term structures --------------------------------------------
    print("\n  ATM skew term structure")
    t_m, s_m = skew_curve(rough_iv)
    t_h, s_h = skew_curve(model_iv_fn("heston", ph))
    t_r, s_r = skew_curve(model_iv_fn("rough", pr))
    H_m = vc.estimate_hurst(t_m, s_m)[0]
    H_h = vc.estimate_hurst(t_h, s_h)[0]
    H_r = vc.estimate_hurst(t_r, s_r)[0]
    sl_m, sl_h, sl_r = (slope_short_end(t_m, s_m), slope_short_end(t_h, s_h),
                        slope_short_end(t_r, s_r))
    print(f"  {'tau(d)':>7} {'market':>9} {'Heston':>9} {'rough':>9} {'H/mkt':>7} {'R/mkt':>7}")
    for i, tau in enumerate(t_m):
        jh = np.argmin(np.abs(t_h - tau))
        jr = np.argmin(np.abs(t_r - tau))
        print(f"  {tau*365:7.0f} {s_m[i]:9.4f} {s_h[jh]:9.4f} {s_r[jr]:9.4f} "
              f"{abs(s_h[jh]/s_m[i]):7.3f} {abs(s_r[jr]/s_m[i]):7.3f}")
    print(f"\n  H from skews  : market {H_m:.4f}   Heston {H_h:.4f}   rough {H_r:.4f}   (planted {H_TRUE})")
    print(f"  short-end log-log slope (<= 14 d): market {sl_m:+.3f}  Heston {sl_h:+.3f}  "
          f"rough {sl_r:+.3f}   (H - 1/2 = {H_TRUE-0.5:+.3f})")

    # --- figure ---------------------------------------------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#0b0d0f")
    ax1, ax2 = plt.subplot(2, 2, 1), plt.subplot(2, 2, 2)
    ax3, ax4 = plt.subplot(2, 2, 3), plt.subplot(2, 2, 4)
    hi, ri = model_iv_fn("heston", ph), model_iv_fn("rough", pr)
    for ax, tau, title in ((ax1, TAUS[0], f"{TAUS[0]*365:.0f} DAY"),
                           (ax2, TAUS[3], f"{TAUS[3]*365:.0f} DAYS")):
        band = 3.0 * atm_vol(tau) * math.sqrt(tau)
        ks = np.linspace(-band, band, 41)
        zs = ks / (atm_vol(tau) * math.sqrt(tau))
        ax.plot(zs, [rough_iv(k, tau) for k in ks], "-", color="#00f2ff", lw=2,
                label=f"rough market (H={H_TRUE})")
        ax.plot(zs, [hi(k, tau) for k in ks], "--", color="#ffb020", lw=2, label="vanilla Heston")
        ax.plot(zs, [ri(k, tau) for k in ks], "-.", color="#7ee787", lw=2,
                label=f"rough Heston (Ĥ={pr.H:.3f})")
        ax.set_facecolor("#161b22")
        ax.axvline(0, color="#ff3e3e", ls=":", lw=0.8)
        ax.set_xlabel("z  (σ from forward)", color="#9aa4b2", fontsize=9)
        ax.set_ylabel("implied vol", color="#9aa4b2", fontsize=9)
        ax.set_title(title, color="white", fontsize=10)
        ax.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")

    ax3.loglog(t_m * 365, np.abs(s_m), "o-", color="#00f2ff", ms=6, lw=1.5,
               label=f"rough market   Ĥ={H_m:.3f}")
    ax3.loglog(t_h * 365, np.abs(s_h), "s--", color="#ffb020", ms=6, lw=1.5,
               label=f"vanilla Heston  Ĥ={H_h:.3f}")
    ax3.loglog(t_r * 365, np.abs(s_r), "^-.", color="#7ee787", ms=6, lw=1.5,
               label=f"rough Heston   Ĥ={H_r:.3f}")
    ax3.set_facecolor("#161b22")
    ax3.set_xlabel("τ  (days)", color="#9aa4b2", fontsize=9)
    ax3.set_ylabel("|∂σ/∂k|  at the money", color="#9aa4b2", fontsize=9)
    ax3.set_title(f"ATM SKEW — short-end slope: market {sl_m:+.2f}, Heston {sl_h:+.2f}, "
                  f"rough {sl_r:+.2f}", color="white", fontsize=10)
    ax3.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for axis in (ax3.xaxis, ax3.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

    d = [str(int(round(r["tau"] * 365))) for r in rep_h]
    xs = np.arange(len(d))
    ax4.bar(xs - 0.2, [r["rmse"] * 100 for r in rep_h], width=0.4, color="#ffb020",
            label=f"Heston  ({r_h['rmse_vol']*100:.2f} vp overall)")
    ax4.bar(xs + 0.2, [r["rmse"] * 100 for r in rep_r], width=0.4, color="#7ee787",
            label=f"rough Heston  ({r_r['rmse_vol']*100:.2f} vp overall)")
    ax4.set_xticks(xs)
    ax4.set_xticklabels(d)
    ax4.set_facecolor("#161b22")
    ax4.set_xlabel("τ  (days)", color="#9aa4b2", fontsize=9)
    ax4.set_ylabel("rmse  (vol points)", color="#9aa4b2", fontsize=9)
    ax4.set_title("FIT ERROR BY MATURITY", color="white", fontsize=10)
    ax4.legend(fontsize=8, facecolor="#0b0d0f", edgecolor="#2d333b")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.07, hspace=0.32, wspace=0.20)
    out = OUT / "rough_vs_heston.png"
    fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    print(f"\n  figure -> {out}")

    # --- record ---------------------------------------------------------------
    lines = [
        "# Phase 2 — rough Heston vs vanilla Heston on the same rough surface", "",
        f"Market: synthetic, ATM skew ~ tau^(H-1/2) with **H = {H_TRUE}**, {len(S)} quotes "
        f"across {S.n_expiries} expiries out to one year. Both models fitted by the same "
        "Levenberg-Marquardt driver in implied vol.", "",
        "## Fitted parameters", "",
        "| model | " + " | ".join(rh.RoughHestonParams.NAMES) + " | rmse (vp) |",
        "|---|" + "---|" * (len(rh.RoughHestonParams.NAMES) + 1),
        "| vanilla Heston | " + " | ".join(f"{getattr(ph, n):.4f}" for n in HestonParams.NAMES)
        + " | — | " + f"{r_h['rmse_vol']*100:.3f} |",
        "| rough Heston | " + " | ".join(f"{getattr(pr, n):.4f}" for n in rh.RoughHestonParams.NAMES)
        + f" | {r_r['rmse_vol']*100:.3f} |", "",
        f"Rough fit: {r_r['iterations']} iterations, stopped on `{r_r['reason']}`; lift with "
        f"N = {r_r['N']} nodes, kernel error {r_r['kernel_error']:.2%} at the fitted H.", "",
        "## Error by maturity (vol points)", "",
        "| tau (days) | Heston | rough Heston |", "|---|---|---|",
    ]
    for a, b in zip(rep_h, rep_r):
        lines.append(f"| {a['tau']*365:.0f} | {a['rmse']*100:.3f} | {b['rmse']*100:.3f} |")
    lines += ["", "## ATM skew term structure", "",
              "| tau (days) | market | Heston | rough Heston | Heston/market | rough/market |",
              "|---|---|---|---|---|---|"]
    for i, tau in enumerate(t_m):
        jh = np.argmin(np.abs(t_h - tau))
        jr = np.argmin(np.abs(t_r - tau))
        lines.append(f"| {tau*365:.0f} | {s_m[i]:.4f} | {s_h[jh]:.4f} | {s_r[jr]:.4f} | "
                     f"{abs(s_h[jh]/s_m[i]):.3f} | {abs(s_r[jr]/s_m[i]):.3f} |")
    lines += ["",
              f"H read off the skews: market **{H_m:.4f}**, Heston **{H_h:.4f}**, rough Heston "
              f"**{H_r:.4f}** (planted {H_TRUE}; the rough model's own H parameter fitted to "
              f"**{pr.H:.4f}**).", "",
              f"Short-end log-log slope of |skew| against tau (maturities up to 14 days): market "
              f"**{sl_m:+.3f}**, Heston **{sl_h:+.3f}**, rough Heston **{sl_r:+.3f}**; the planted "
              f"H − ½ is {H_TRUE-0.5:+.3f}. A diffusion's slope must go to 0 as tau -> 0; the "
              "fractional kernel's does not, and that is the whole difference.", ""]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  record -> {DOC}")


if __name__ == "__main__":
    main()
