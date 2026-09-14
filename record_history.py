"""
Pull the daily history the real-data filters need (Session G). Works at any
hour: historical data does not need the market to be open.

    .venv/Scripts/python.exe record_history.py                 # 5 y daily, 12 months of 5-min bars
    .venv/Scripts/python.exe record_history.py --rv-months 3   # quicker (~3 min)

Writes captures/real/history_<SYMBOL>_<YYYY-MM-DD>.json (format volsurf-history/1):

  daily   daily OHLCV, regular hours           -> Garman-Klass variance
  iv30    IBKR's 30-day implied volatility     -> what the market expects
  hv30    IBKR's 30-day historical volatility
  bars5m  5-minute bars, last N months         -> realised variance per day
  vix     Cboe VIX daily, if your account has the index permission

IBKR serves 5-minute bars at most one week per request and allows about 60
historical requests per ten minutes, so the 5-minute pull walks back one week
at a time with a pause between requests. Twelve months takes ~10 minutes.

Walkthrough: docs/tutorial-ibkr-recording.md
"""

import argparse
import datetime
import json
import pathlib
import sys
import threading
import time

from ibapi.contract import Contract

import record_chains as rc
import sources.history as hist
import volatility_surface_3 as v3
import volsurf_core as vc


class HistoryApp(v3.LiveSurfaceApp):
    """The v3 app (its arity-safe error handler included) plus historical bars."""

    def __init__(self):
        super().__init__()
        self.bars, self.done, self.req_errors = {}, {}, {}

    def historicalData(self, reqId, bar):
        self.bars.setdefault(reqId, []).append({
            "date": str(bar.date), "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume)})

    def historicalDataEnd(self, reqId, start, end):
        ev = self.done.get(reqId)
        if ev is not None:
            ev.set()

    def error(self, *args):
        super().error(*args)
        if len(args) >= 3:
            req_id = args[0]
            code, msg = (args[2], args[3]) if len(args) >= 5 else (args[1], args[2])
            if req_id in self.done and code not in v3.BENIGN_CODES:
                self.req_errors[req_id] = (code, msg)
                self.done[req_id].set()


def connect(host, port, client_id):
    if not v3.probe_tws(host, port):
        raise v3.ConnectionError_(f"Nothing is listening on {host}:{port}.")
    app = HistoryApp()
    app.connect(host, port, clientId=client_id)
    threading.Thread(target=app.run, daemon=True).start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not app.isConnected():
        if app.connect_failed.is_set():
            raise v3.ConnectionError_(f"TWS refused the API connection on {host}:{port}.")
        time.sleep(0.05)
    if not app.isConnected():
        raise v3.ConnectionError_(f"Timed out connecting to {host}:{port}.")
    return app


def stock(symbol):
    c = Contract()
    c.symbol, c.secType, c.exchange, c.currency = symbol, "STK", "SMART", "USD"
    return c


def vix_index():
    c = Contract()
    c.symbol, c.secType, c.exchange, c.currency = "VIX", "IND", "CBOE", "USD"
    return c


def fetch(app, rid, contract, end, duration, bar_size, what, use_rth=1, timeout=90.0):
    """One historical request. Returns (bars, error or None)."""
    app.bars[rid] = []
    app.done[rid] = threading.Event()
    app.reqHistoricalData(rid, contract, end, duration, bar_size, what, use_rth, 1, False, [])
    if not app.done[rid].wait(timeout):
        return app.bars[rid], ("timeout", f"no end-of-data after {timeout:.0f} s")
    return app.bars[rid], app.req_errors.get(rid)


def pull(app, symbol, years=5, rv_months=12, pause=10.5, with_vix=True, log=print):
    out = {"format": hist.FORMAT, "symbol": symbol,
           "pulled_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    rid = 100
    for key, what in (("daily", "TRADES"), ("iv30", "OPTION_IMPLIED_VOLATILITY"),
                      ("hv30", "HISTORICAL_VOLATILITY")):
        bars, err = fetch(app, rid, stock(symbol), "", f"{years} Y", "1 day", what)
        out[key] = bars
        log(f"  {key:6s} {len(bars):5d} daily bars" + (f"   error {err[0]}: {err[1]}" if err else ""))
        rid += 1
        time.sleep(pause)
    if with_vix:
        bars, err = fetch(app, rid, vix_index(), "", f"{years} Y", "1 day", "TRADES")
        out["vix"] = bars
        log(f"  vix    {len(bars):5d} daily bars" + (f"   error {err[0]}: {err[1]} (skipped)" if err else ""))
        rid += 1
        time.sleep(pause)

    chunks, weeks = [], int(round(rv_months * 52 / 12))
    end_day = vc.new_york_date(datetime.datetime.now(datetime.timezone.utc))
    for w in range(weeks):
        end = f"{end_day - datetime.timedelta(days=7 * w):%Y%m%d} 16:00:00 US/Eastern"
        bars, err = fetch(app, rid, stock(symbol), end, "1 W", "5 mins", "TRADES")
        chunks.append(bars)
        log(f"  5-min week {w + 1:2d}/{weeks} ending {end[:8]}: {len(bars):4d} bars"
            + (f"   error {err[0]}: {err[1]}" if err else ""))
        rid += 1
        if w + 1 < weeks:
            time.sleep(pause)
    out["bars5m"] = hist.merge_bars(chunks)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="gateway-live", help="a number, or one of " + ", ".join(rc.PORTS))
    ap.add_argument("--client-id", type=int, default=38)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--rv-months", type=int, default=12)
    ap.add_argument("--no-vix", action="store_true")
    ap.add_argument("--out", default="captures/real")
    args = ap.parse_args(argv)

    port = rc.PORTS.get(args.port) or int(args.port)
    try:
        app = connect(args.host, port, args.client_id)
    except v3.ConnectionError_ as exc:
        print(f"Could not connect: {exc}\nSee docs/tutorial-ibkr-recording.md, 'Troubleshooting'.")
        return 1
    t0 = time.monotonic()
    try:
        data = pull(app, args.symbol, args.years, args.rv_months, with_vix=not args.no_vix)
    except KeyboardInterrupt:
        print("\nStopped before the pull finished; nothing written.")
        return 1
    finally:
        app.disconnect()
    path = pathlib.Path(args.out) / f"history_{args.symbol}_{datetime.date.today():%Y-%m-%d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    rv = hist.realised_variance(data["bars5m"])
    print(f"\nwrote {path} in {time.monotonic() - t0:.0f} s: {len(data['daily'])} daily bars, "
          f"{len(data['iv30'])} IV days, {len(data['bars5m'])} 5-min bars -> {len(rv['date'])} RV days")
    for line in rc.explain_errors(app.error_counts):
        print(line)
    return 0 if data["daily"] else 2


if __name__ == "__main__":
    sys.exit(main())
