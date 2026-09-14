"""
Record real IBKR option chains through the replay source (Session G).

Every cycle is self-consistent. A seed pass (both rights near the money) gives
this cycle's forward for each expiry; the OTM grid is laid on a sigma band
around THOSE forwards; one full sweep fills it; sources.replay.record() writes
it. Nothing from an earlier cycle survives into a later one: quotes are
cleared and request ids move on, so a late tick for a cancelled request can
never land on another contract.

    .venv/Scripts/python.exe record_chains.py --check                    # ~15 s: is it wired?
    .venv/Scripts/python.exe record_chains.py --every 15 --until 15:45   # a session (New York time)
    .venv/Scripts/python.exe record_chains.py --once --data-type frozen  # after the close

Walkthrough: docs/tutorial-ibkr-recording.md
"""

import argparse
import collections
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

import sources.replay as replay
import volatility_surface_3 as v3
import volsurf_core as vc

# One expiry per target maturity. Dailies crowd the front of a SPY chain, so
# "the first n expiries" would leave no term structure to identify H from.
TARGET_DAYS = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)
DATA_TYPES = {"live": 1, "frozen": 2, "delayed": 3, "delayed-frozen": 4}
PORTS = {"tws-live": 7496, "tws-paper": 7497, "gateway-live": 4001, "gateway-paper": 4002}

HINTS = {
    162: "historical/market data service message (often pacing or no data)",
    200: "no security definition (a strike or expiry that does not exist; harmless in small numbers)",
    321: "request rejected as invalid (check the symbol / contract)",
    326: "client id already in use -- pass a different --client-id",
    354: "not subscribed -- add OPRA Top of Book in Client Portal, or use --data-type delayed",
    10089: "subscription required for these ticks; delayed data is available",
    10090: "part of the requested market data is not subscribed",
    10167: "IBKR is displaying delayed data, not live",
    10197: "competing live session -- log out of IBKR on your phone/web, then retry",
    1100: "connectivity between TWS/Gateway and IBKR lost",
    2110: "connectivity between TWS/Gateway and IBKR servers is broken",
}


def new_york_now():
    """Wall-clock New York time as a naive datetime (display and scheduling only)."""
    u = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return u - datetime.timedelta(hours=4 if vc.us_dst(u.date()) else 5)


def regular_hours(ny):
    """Is `ny` (New York wall clock) inside 09:30-16:00 on a weekday? Holidays ignored."""
    return ny.weekday() < 5 and datetime.time(9, 30) <= ny.time() <= datetime.time(16, 0)


def explain_errors(counts):
    lines = []
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if code in v3.BENIGN_CODES:
            continue
        lines.append(f"    {code} x{n}: {HINTS.get(code, 'see IBKR API message codes')}")
    return lines


def cycle_stats(app, ctxs, t0, errors):
    quotes = list(app.quotes.values())
    by_exp = {}
    for q in quotes:
        row = by_exp.setdefault(q.expiry, {"quotes": 0, "with_iv": 0, "usable": 0})
        row["quotes"] += 1
        row["with_iv"] += int(bool(np.isfinite(q.iv)))
        row["usable"] += int(bool(q.usable()))
    exps = {}
    for exp, ctx in sorted(ctxs.items(), key=lambda kv: kv[1].tau):
        exps[exp] = {"tau_days": ctx.tau * 365.0, "forward": ctx.forward,
                     "parity_r2": ctx.parity_r2, "sigma_atm": ctx.sigma_atm,
                     **by_exp.get(exp, {"quotes": 0, "with_iv": 0, "usable": 0})}
    n_iv = sum(e["with_iv"] for e in exps.values())
    return {
        "t0_utc": t0.isoformat(timespec="seconds"),
        "market_data_type": v3.MARKET_DATA_TYPES.get(getattr(app, "market_data_type", 1)),
        "spot": float(app.spot_price),
        "quotes": len(quotes),
        "with_iv": n_iv,
        "usable": sum(e["usable"] for e in exps.values()),
        "expiries": exps,
        "errors": {str(k): v for k, v in errors.items()},
    }


def record_cycle(app, symbol, exps, out_dir, cycle_no, n_sigma=3.0, max_strikes=21,
                 seed_timeout=5.0):
    """One seed pass, one grid sweep, one snapshot file. Returns (path, stats)."""
    t0 = datetime.datetime.now(datetime.timezone.utc)
    with app._lock:
        app.quotes.clear()
        app.id_map.clear()
    before = collections.Counter(app.error_counts)
    id_base = 10_000 + 10_000 * cycle_no
    ctxs, _ = v3.seed_and_grid(app, symbol, exps, t0, n_sigma=n_sigma,
                               max_strikes=max_strikes, seed_timeout=seed_timeout,
                               id_base=id_base)
    stats = cycle_stats(app, ctxs, t0, app.error_counts - before)
    ny = t0.replace(tzinfo=None) - datetime.timedelta(hours=4 if vc.us_dst(t0.date()) else 5)
    day_dir = pathlib.Path(out_dir) / symbol / ny.strftime("%Y-%m-%d")
    path = day_dir / f"{symbol}_{ny.strftime('%Y-%m-%d_%H-%M-%S')}ET.json"
    replay.record(app, ctxs, path, symbol=symbol, meta=stats)
    with open(day_dir / "session.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"file": path.name, **{k: v for k, v in stats.items()
                                                     if k != "expiries"}}) + "\n")
    return path, stats


def print_cycle(path, stats, elapsed):
    print(f"\n[{new_york_now():%H:%M:%S} ET] wrote {path}  ({elapsed:.0f} s)")
    print(f"  spot {stats['spot']:.2f}  {stats['market_data_type']} data  "
          f"{stats['quotes']} quotes, {stats['with_iv']} with IV, {stats['usable']} usable")
    weak = [e for e, r in stats["expiries"].items() if r["parity_r2"] < 0.99]
    if weak:
        print(f"  parity r2 < 0.99 on {len(weak)} expiries: {', '.join(weak)}"
              " (forward fell back to spot where r2 = 0)")
    if stats["quotes"] and stats["with_iv"] < 0.5 * stats["quotes"]:
        print("  WARNING: fewer than half the quotes carry an implied vol")
    for line in explain_errors({int(k): v for k, v in stats["errors"].items()}):
        print(line)


def check(app, symbol):
    """Connection smoke test: one near-the-money weekly call, end to end."""
    t0 = datetime.datetime.now(datetime.timezone.utc)
    today = vc.new_york_date(t0)
    print(f"server version {app.serverVersion()}, connection time {app.twsConnectionTime()}")
    exps = v3.select_expiries_by_target(app.expirations, today, (7,))
    if not exps:
        print("no expiry 1+ days out")
        return False
    K = min(app.strikes, key=lambda s: abs(s - app.spot_price))
    rid = 5000
    app.id_map[rid] = (exps[0], K, "C")
    contract = v3.make_option(symbol, exps[0], K, "C", app.trading_class)
    v3.ChainSweeper(app, [(rid, contract)], quote_timeout=12.0).sweep()
    q = app.quotes.get(rid)
    ok = q is not None and np.isfinite(q.iv)
    if q is None:
        print(f"{symbol} {exps[0]} {K:g}C: nothing arrived in 12 s")
    else:
        print(f"{symbol} {exps[0]} {K:g}C: bid {q.bid} ask {q.ask} iv {q.iv:.4f} "
              f"delta {q.delta:.3f} vega {q.vega:.3f} iv_error {q.iv_error:.4f}")
        if ok and np.isfinite(q.vega) and np.isfinite(q.und_price):
            tau = vc.tau_years(t0, vc.parse_ib_date(exps[0]))
            ratio = q.vega / float(vc.bs_vega(q.und_price, K, q.iv, tau))
            units_ok = 0.8 < ratio < 1.25
            print(f"vega units: IBKR x {v3.IB_VEGA_PER_UNIT_VOL:g} / Black-Scholes = {ratio:.3f} "
                  + ("(consistent)" if units_ok else "-- UNITS DISAGREE, tell Claude before recording"))
            ok = ok and units_ok
    for line in explain_errors(app.error_counts):
        print(line)
    print("CHECK PASSED -- ready to record" if ok else
          "CHECK FAILED -- no implied vol arrived; see the error codes above and the tutorial")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="gateway-live",
                    help="a number, or one of " + ", ".join(PORTS))
    ap.add_argument("--client-id", type=int, default=37)
    ap.add_argument("--data-type", default="live", choices=list(DATA_TYPES))
    ap.add_argument("--every", type=float, default=15.0, help="minutes between cycles")
    ap.add_argument("--until", default="15:45", help="New York time HH:MM to stop")
    ap.add_argument("--once", action="store_true", help="record a single cycle and exit")
    ap.add_argument("--check", action="store_true", help="connection smoke test, no files")
    ap.add_argument("--targets", default=",".join(str(d) for d in TARGET_DAYS),
                    help="target maturities in days")
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--max-strikes", type=int, default=21)
    ap.add_argument("--min-dte", type=int, default=1)
    ap.add_argument("--out", default="captures/real")
    args = ap.parse_args(argv)

    port = PORTS.get(args.port) or int(args.port)
    ny = new_york_now()
    print(f"New York time {ny:%a %H:%M}; regular hours: {'yes' if regular_hours(ny) else 'NO'}")
    if not regular_hours(ny) and args.data_type == "live" and not args.check:
        print("  Outside 09:30-16:00 New York the live book is thin or closed. After the close,\n"
              "  use --data-type frozen for the closing quotes.")
    try:
        app = v3.connect_app(args.symbol, args.host, port, args.client_id,
                             market_data_type=DATA_TYPES[args.data_type])
    except v3.ConnectionError_ as exc:
        print(f"\nCould not connect: {exc}")
        print("See docs/tutorial-ibkr-recording.md, section 'Troubleshooting'.")
        return 1

    try:
        if args.check:
            return 0 if check(app, args.symbol) else 2

        targets = [int(t) for t in args.targets.split(",") if t.strip()]
        today = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
        exps = v3.select_expiries_by_target(app.expirations, today, targets, args.min_dte)
        print(f"{len(exps)} expiries for targets {targets}: {', '.join(exps)}")
        stop_at = datetime.datetime.combine(ny.date(), datetime.time.fromisoformat(args.until))
        cycle_no = 0
        while True:
            started = time.monotonic()
            path, stats = record_cycle(app, args.symbol, exps, args.out, cycle_no,
                                       args.n_sigma, args.max_strikes)
            print_cycle(path, stats, time.monotonic() - started)
            cycle_no += 1
            if args.once:
                break
            wake = started + 60.0 * args.every
            nxt = new_york_now() + datetime.timedelta(seconds=max(0.0, wake - time.monotonic()))
            if nxt > stop_at:
                print(f"\nNext cycle would start after {args.until} New York; stopping.")
                break
            print(f"  next cycle at {nxt:%H:%M:%S} ET  (Ctrl+C to stop; completed files are kept)")
            while time.monotonic() < wake:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped. Every file written so far is complete; the cycle in flight was not saved.")
    finally:
        try:
            app.cancelMktData(app.SPOT_REQ_ID)
        except Exception:
            pass
        app.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
