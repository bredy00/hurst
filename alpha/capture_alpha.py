"""
Capture harness for the ALPHA volatility surface.

There is no TWS on this machine, so the real IBKR path cannot produce a frame.
This harness drives volatility_surface_1_ALPHA.py's OWN plotting code
(live_desktop_plot) with a stand-in feed, so what you see on screen is
genuinely the alpha's rendering logic -- the pivot, the interpolate/bfill/ffill,
the meshgrid, the plot_surface call, the skew panel. Nothing in the alpha file
is modified.

The stand-in surface is deliberately realistic for SPY:
  - upward-sloping ATM term structure
  - put skew that steepens at the short end like T**(-0.38)
    (i.e. a rough-vol signature with H ~ 0.12 baked in)
  - heteroskedastic quote noise, larger where vega is small
  - ~12% of contracts never tick, so the ffill/bfill behaviour is visible
"""

import importlib.util
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).parent
FRAMES = HERE.parent / "captures" / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)

# Headless: we save frames rather than needing a window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- load the alpha module without running its __main__ block -----------------
def load_alpha():
    path = HERE / "volatility_surface_1_ALPHA.py"
    spec = importlib.util.spec_from_file_location("alpha_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alpha_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- the stand-in surface -----------------------------------------------------
SPOT = 650.0
TODAY = "20260904"

# SPY-like expiries: today (0DTE) plus the next five M/W/F dates.
EXPIRIES = ["20260904", "20260907", "20260909", "20260911", "20260914", "20260916"]


def year_fraction(exp: str) -> float:
    """ACT/365 from TODAY. Today's expiry gets half a session, not zero."""
    import datetime as dt
    d0 = dt.date(2026, 9, 4)
    d1 = dt.date(int(exp[:4]), int(exp[4:6]), int(exp[6:]))
    days = (d1 - d0).days
    return max(days, 0.5) / 365.0


def atm_vol(T: float) -> float:
    """Upward-sloping ATM term structure."""
    return 0.115 + 0.09 * (1.0 - np.exp(-T / 0.06))


def atm_skew(T: float) -> float:
    """
    d(sigma)/d(k) at the money, normalised to -1.3 at 7 days and scaling as
    T**(-0.38). That exponent is H - 1/2 with H = 0.12 -- a rough surface.
    """
    T7 = 7.0 / 365.0
    return -1.3 * (T7 / max(T, 1e-4)) ** 0.38


def true_iv(strike: float, exp: str, spot: float) -> float:
    T = year_fraction(exp)
    k = np.log(strike / spot)          # alpha has no forward, so moneyness vs spot
    sig = atm_vol(T) + atm_skew(T) * k + 6.0 * k * k
    return float(np.clip(sig, 0.02, 3.0))


def bs_vega_and_price(spot, strike, T, sigma):
    """Black-Scholes vega (per 1.0 of vol) and OTM option price. r = q = 0."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 1e-12, 0.0
    srt = sigma * np.sqrt(T)
    d1 = (np.log(spot / strike) + 0.5 * srt * srt) / srt
    d2 = d1 - srt
    vega = spot * np.sqrt(T) * norm.pdf(d1)
    if strike >= spot:                                   # OTM call
        price = spot * norm.cdf(d1) - strike * norm.cdf(d2)
    else:                                                # OTM put
        price = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return max(vega, 1e-12), max(price, 0.0)


# SPY options are penny-wide on the liquid strikes.
HALF_SPREAD = 0.005
MIN_TRADEABLE_PRICE = 0.02   # below this there is no real two-sided market


def build_feed(app, alpha, rng):
    """
    Populate app.id_map exactly the way alpha.start_app would, using the alpha's
    own selection rules: next 6 expiries, strikes within +/-2% of spot.
    """
    app.spot_price = SPOT
    target_exps = [e for e in EXPIRIES if e >= TODAY][:6]
    # SPY strikes are $1 apart
    all_strikes = [float(s) for s in range(int(SPOT * 0.90), int(SPOT * 1.10) + 1)]
    target_strikes = [s for s in all_strikes if SPOT * 0.98 <= s <= SPOT * 1.02]

    req_id = 1000
    live = []
    dead_by_exp = {}
    for exp in target_exps:
        T = year_fraction(exp)
        dead = 0
        for strike in target_strikes:
            app.id_map[req_id] = (exp, strike)
            sig = true_iv(strike, exp, SPOT)
            _, price = bs_vega_and_price(SPOT, strike, T, sig)
            # A contract only quotes if it is worth at least a couple of ticks.
            if price >= MIN_TRADEABLE_PRICE:
                live.append(req_id)
            else:
                dead += 1
            req_id += 1
        dead_by_exp[exp] = dead
    print(f"  contracts requested : {req_id - 1000}")
    print(f"  contracts that tick : {len(live)}")
    print("  no market (price < $0.02), by expiry:")
    for e, d in dead_by_exp.items():
        T = year_fraction(e)
        print(f"    {e}  T={T*365:5.1f}d   {d:2d}/{len(target_strikes)} dead")
    print(f"  strike band         : {target_strikes[0]:.0f} - {target_strikes[-1]:.0f} "
          f"({len(target_strikes)} strikes)")
    print(f"  expiries            : {', '.join(target_exps)}")
    return live


def tick(app, live, rng, drift):
    """One refresh of the stand-in feed, with realistic per-quote noise."""
    spot = SPOT * (1.0 + drift)
    app.spot_price = spot
    for rid in live:
        exp, strike = app.id_map[rid]
        T = year_fraction(exp)
        sig = true_iv(strike, exp, spot)
        # IV uncertainty = half-spread / vega. This is the real thing, not a proxy:
        # vega collapses as an option goes far out in sigma-units, so the implied
        # vol of a wing contract is enormously less certain than an ATM one.
        vega, _ = bs_vega_and_price(spot, strike, T, sig)
        noise = rng.normal(0.0, HALF_SPREAD / vega)
        app.iv_dict[rid] = float(np.clip(sig + noise, 0.01, 3.0))


def main():
    alpha = load_alpha()
    rng = np.random.default_rng(7)

    app = alpha.LiveSurfaceApp.__new__(alpha.LiveSurfaceApp)
    app.iv_dict, app.id_map = {}, {}
    app.expirations, app.strikes = [], []
    app.spot_price, app.underlying_conId = 0, 0
    app.disconnect = types.MethodType(lambda self: None, app)

    print("Stand-in feed (alpha's own selection rules):")
    live = build_feed(app, alpha, rng)

    n_frames = 24
    state = {"i": 0}
    real_pause = plt.pause

    def capture_pause(interval):
        fig = plt.gcf()
        out = FRAMES / f"frame_{state['i']:03d}.png"
        fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
        state["i"] += 1
        if state["i"] >= n_frames:
            raise KeyboardInterrupt
        tick(app, live, rng, drift=0.0012 * np.sin(state["i"] / 3.0))

    alpha.plt.pause = capture_pause
    tick(app, live, rng, drift=0.0)

    print(f"\nRendering {n_frames} frames through alpha.live_desktop_plot() ...")
    alpha.live_desktop_plot(app)

    files = sorted(FRAMES.glob("frame_*.png"))
    print(f"Captured {len(files)} frames -> {FRAMES}")

    if files:
        from PIL import Image
        imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        gif = FRAMES.parent / "alpha_surface.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=400, loop=0)
        print(f"GIF -> {gif}")


if __name__ == "__main__":
    main()
