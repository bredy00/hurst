"""
Same stand-in feed as alpha/capture_alpha.py, driving volatility_surface_2.py,
so the two renderings can be compared like for like. Also emits a side-by-side.
"""

import datetime
import pathlib
import sys
import time
import types

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "alpha"))

import volatility_surface_2 as v2
from capture_alpha import (SPOT, EXPIRIES, year_fraction, true_iv,
                           bs_vega_and_price, HALF_SPREAD, MIN_TRADEABLE_PRICE)

OUT = HERE / "captures"
FRAMES = OUT / "frames_v2"
FRAMES.mkdir(parents=True, exist_ok=True)

TODAY = datetime.date(2026, 9, 4)


def build_feed(app, rng):
    app.spot_price = SPOT
    # Fix 6 in action: v2's own selector, not `e >= today`
    exps, skipped = v2.select_expiries(EXPIRIES, TODAY, min_days_to_expiry=1, n_expiries=6)
    all_strikes = [float(s) for s in range(int(SPOT * 0.90), int(SPOT * 1.10) + 1)]
    strikes = v2.select_strikes(all_strikes, SPOT, 0.02)

    print(f"  expiries kept    : {exps}")
    print(f"  expiries skipped : {skipped}  <- 0DTE, excluded by Fix 6")
    print(f"  strikes          : {strikes[0]:.0f}-{strikes[-1]:.0f} ({len(strikes)})")

    req_id, live = 1000, []
    for exp in exps:
        T = year_fraction(exp)
        for strike in strikes:
            app.id_map[req_id] = (exp, strike)
            sig = true_iv(strike, exp, SPOT)
            _, price = bs_vega_and_price(SPOT, strike, T, sig)
            if price >= MIN_TRADEABLE_PRICE:
                live.append(req_id)
            req_id += 1
    print(f"  contracts        : {req_id - 1000} requested, {len(live)} with a market")
    return live


def tick(app, live, rng, drift):
    spot = SPOT * (1.0 + drift)
    app.spot_price = spot
    now = time.monotonic()
    for rid in live:
        exp, strike = app.id_map[rid]
        T = year_fraction(exp)
        sig = true_iv(strike, exp, spot)
        vega, _ = bs_vega_and_price(spot, strike, T, sig)
        noise = rng.normal(0.0, HALF_SPREAD / vega)
        # v2 stores (iv, timestamp) so quotes can be aged out
        app.iv_dict[rid] = (float(np.clip(sig + noise, 0.01, 3.0)), now)


def main():
    rng = np.random.default_rng(7)
    app = v2.LiveSurfaceApp.__new__(v2.LiveSurfaceApp)
    v2.LiveSurfaceApp.__init__.__wrapped__ if False else None
    # Initialise state without opening a socket
    import threading
    app._lock = threading.RLock()
    app.iv_dict, app.id_map = {}, {}
    app.expirations, app.strikes = [], []
    app.trading_class = "SPY"
    app.spot_price, app._spot_quotes, app.underlying_conId = 0.0, {}, 0
    app._symbol = "SPY"
    for ev in ("resolved", "chain_resolved", "spot_ready", "connect_failed"):
        setattr(app, ev, threading.Event())
    app.disconnect = types.MethodType(lambda self: None, app)
    app.cancelMktData = types.MethodType(lambda self, rid: None, app)

    print("Stand-in feed through volatility_surface_2:")
    live = build_feed(app, rng)

    n_frames, state = 24, {"i": 0}

    def capture_pause(interval):
        fig = plt.gcf()
        fig.savefig(FRAMES / f"frame_{state['i']:03d}.png", dpi=110,
                    facecolor=fig.get_facecolor())
        state["i"] += 1
        if state["i"] >= n_frames:
            raise KeyboardInterrupt
        tick(app, live, rng, drift=0.0012 * np.sin(state["i"] / 3.0))

    v2.plt.pause = capture_pause
    tick(app, live, rng, drift=0.0)

    print(f"\nRendering {n_frames} frames through v2.live_desktop_plot() ...")
    v2.live_desktop_plot(app)
    files = sorted(FRAMES.glob("frame_*.png"))
    print(f"Captured {len(files)} frames -> {FRAMES}")

    # Side-by-side against the alpha
    from PIL import Image
    a = OUT / "frames" / "frame_012.png"
    b = FRAMES / "frame_012.png"
    if a.exists() and b.exists():
        ia, ib = Image.open(a), Image.open(b)
        w, h = ia.width, ia.height + ib.height
        canvas = Image.new("RGB", (w, h), (11, 13, 15))
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (0, ia.height))
        out = OUT / "alpha_vs_v2.png"
        canvas.save(out)
        print(f"Side-by-side -> {out}   (top = alpha, bottom = v2)")

    if files:
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        gif = OUT / "v2_surface.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=400, loop=0)
        print(f"GIF -> {gif}")


if __name__ == "__main__":
    main()
