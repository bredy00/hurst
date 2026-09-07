"""
Stand-in feed for volatility_surface_3, and an end-to-end validation.

This is not just a picture. The generator plants two things the pipeline is
never told:

    the forward   F = S * exp((r - q) * tau)   with r = 4.2%, q = 1.3%
    the roughness ATM skew ~ tau^(H - 1/2)     with H = 0.12

The pipeline sees only bid/ask quotes. If put-call parity recovers F without a
rate or dividend input, and the log-log skew fit recovers H, the coordinate
machinery is doing what it claims.
"""

import datetime
import math
import pathlib
import sys
import threading
import time
import types

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import volsurf_core as vc
import volatility_surface_3 as v3

OUT = pathlib.Path(__file__).parent / "captures"
FRAMES = OUT / "frames_v3"
FRAMES.mkdir(parents=True, exist_ok=True)

SPOT = 650.0
R, Q = 0.042, 0.013           # planted carry -- the pipeline is never told these
H_TRUE = 0.12                 # planted roughness
T0 = datetime.datetime(2026, 9, 4, 14, 30)
EXPIRIES = ["20260907", "20260909", "20260911", "20260914", "20260916",
            "20260921", "20261002", "20261016"]
HALF_SPREAD = 0.005
TICK = 0.01


def true_forward(tau):
    return SPOT * math.exp((R - Q) * tau)


def atm_vol(tau):
    return 0.115 + 0.09 * (1.0 - math.exp(-tau / 0.06))


def true_skew(tau):
    """d(sigma)/dk at the money, ~ tau^(H-1/2). Slope on log-log is H - 1/2."""
    tau7 = 7.0 / 365.0
    return -1.3 * (tau7 / tau) ** (0.5 - H_TRUE)


def true_iv(K, tau):
    F = true_forward(tau)
    k = math.log(K / F)
    return float(np.clip(atm_vol(tau) + true_skew(tau) * k + 6.0 * k * k, 0.02, 3.0))


def quote_for(K, tau, right, rng):
    """A realistic two-sided market: BS mid, penny-wide, rounded to the tick."""
    F, sig = true_forward(tau), true_iv(K, tau)
    df = math.exp(-R * tau)
    mid = float(vc.bs_price(F, K, sig, tau, df, right))
    if mid < 0.02:
        return None
    jitter = rng.normal(0.0, 0.25 * TICK)
    bid = max(round((mid - HALF_SPREAD + jitter) / TICK) * TICK, 0.01)
    ask = max(round((mid + HALF_SPREAD + jitter) / TICK) * TICK, bid + TICK)
    return bid, ask, sig, float(vc.bs_vega(F, K, sig, tau, df))


def make_app():
    app = v3.LiveSurfaceApp()
    app._symbol = "SPY"
    app.trading_class = "SPY"
    app.spot_price = SPOT
    app.strikes = [float(s) for s in range(450, 861, 1)]
    app.expirations = list(EXPIRIES)
    app.reqMktData = types.MethodType(lambda s, *a, **k: None, app)
    app.cancelMktData = types.MethodType(lambda s, rid: None, app)
    app.disconnect = types.MethodType(lambda s: None, app)
    return app


def push(app, rid, exp, K, right, tau, rng):
    q = quote_for(K, tau, right, rng)
    if q is None:
        return False
    bid, ask, sig, vega = q
    app.id_map[rid] = (exp, K, right)
    oq = v3.OptionQuote(expiry=exp, strike=K, right=right)
    oq.bid, oq.ask = bid, ask
    # What IBKR's model tick would report: the IV of the mid, plus its vega
    mid = 0.5 * (bid + ask)
    F, df = true_forward(tau), math.exp(-R * tau)
    iv = vc.implied_vol(mid, F, K, tau, df, right)
    oq.iv = iv if iv is not None else float('nan')
    oq.vega = vega
    oq.model_price = mid
    oq.und_price = SPOT
    oq.ts = time.monotonic()
    app.quotes[rid] = oq
    return True


def main():
    rng = np.random.default_rng(7)
    app = make_app()

    # --- seed pass: both rights near the money, for put-call parity ----------
    ctxs = v3.build_expiry_contexts(app, EXPIRIES, T0, SPOT)
    rid = 1000
    for exp, ctx in ctxs.items():
        for K in ctx.strikes:
            for right in ('C', 'P'):
                if push(app, rid, exp, K, right, ctx.tau, rng):
                    rid += 1
    v3.resolve_forwards(app, ctxs, SPOT)

    print("Put-call parity recovery (the pipeline is never told r or q):")
    print(f"  {'expiry':>10} {'tau(d)':>7} {'F true':>9} {'F found':>9} "
          f"{'err':>8} {'r2':>8} {'sig_atm':>8}")
    ferrs = []
    for exp in sorted(ctxs, key=lambda e: ctxs[e].tau):
        c = ctxs[exp]
        ft = true_forward(c.tau)
        err = c.forward - ft
        ferrs.append(abs(err))
        print(f"  {exp:>10} {c.tau*365:7.2f} {ft:9.3f} {c.forward:9.3f} "
              f"{err:+8.4f} {c.parity_r2:8.5f} {c.sigma_atm:8.4f}")
    print(f"  worst |F error| = {max(ferrs):.4f}  "
          f"({100*max(ferrs)/SPOT:.4f}% of spot)   spot-as-forward would be off by "
          f"{max(abs(true_forward(ctxs[e].tau) - SPOT) for e in ctxs):.3f}")

    # --- grid pass: sigma-normalised band, OTM only --------------------------
    app.quotes.clear()
    app.id_map.clear()
    rid = 2000
    per_exp = {}
    for exp, ctx in ctxs.items():
        sel = vc.select_strikes_by_sigma(app.strikes, ctx.forward, ctx.sigma_atm,
                                         ctx.tau, v3.N_SIGMA_BAND, 21)
        ctx.strikes = sel
        n = 0
        for K in sel:
            right = vc.otm_right(K, ctx.forward)
            if push(app, rid, exp, K, right, ctx.tau, rng):
                rid += 1
                n += 1
        per_exp[exp] = (len(sel), n)

    print("\nGrid pass (+/-3 sigma, OTM only):")
    print(f"  {'expiry':>10} {'tau(d)':>7} {'band $':>16} {'strikes':>8} "
          f"{'quoted':>7} {'dead':>5}")
    for exp in sorted(ctxs, key=lambda e: ctxs[e].tau):
        c = ctxs[exp]
        req, got = per_exp[exp]
        lo, hi = (min(c.strikes), max(c.strikes)) if c.strikes else (0, 0)
        pct = 100.0 * (hi - lo) / (2 * c.forward)
        print(f"  {exp:>10} {c.tau*365:7.2f} {lo:7.0f}-{hi:<6.0f} "
              f"(+/-{pct:4.1f}%) {req:8d} {got:7d} {req-got:5d}")

    pts = v3.surface_points(app, ctxs, max_age=1e9)
    total = sum(len(v) for v in pts.values())
    print(f"\n  usable quotes after the vega filter: {total}")

    res, rt, rs, re = v3.roughness(pts, ctxs)
    if res:
        H, H_err, r2 = res
        e = f" +/- {H_err:.4f}" if H_err else ""
        print(f"  planted H = {H_TRUE:.3f}   recovered H = {H:.4f}{e}  "
              f"(error {abs(H-H_TRUE):.4f}, r2 = {r2:.4f})")

    # --- render --------------------------------------------------------------
    n_frames, state = 16, {"i": 0}

    def capture_pause(interval):
        fig = plt.gcf()
        fig.savefig(FRAMES / f"frame_{state['i']:03d}.png", dpi=110,
                    facecolor=fig.get_facecolor())
        state["i"] += 1
        if state["i"] >= n_frames:
            raise KeyboardInterrupt
        now = time.monotonic()
        for q in app.quotes.values():
            q.ts = now

    # live_desktop_plot imports pyplot lazily, but `import matplotlib.pyplot`
    # yields the same module object we already hold, so patching pause here
    # survives that import.
    plt.pause = capture_pause
    print(f"\nRendering {n_frames} frames ...")
    v3.live_desktop_plot(app, ctxs, max_age=1e9)

    files = sorted(FRAMES.glob("frame_*.png"))
    print(f"Captured {len(files)} frames -> {FRAMES}")
    if files:
        from PIL import Image
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        imgs[0].save(OUT / "v3_surface.gif", save_all=True,
                     append_images=imgs[1:], duration=450, loop=0)
        print(f"GIF -> {OUT / 'v3_surface.gif'}")


if __name__ == "__main__":
    main()
