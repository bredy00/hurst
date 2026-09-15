"""
Record a chain snapshot to JSON and play it back.

This is the piece that makes everything above it developable. Without it, every
change to the maths needs a live TWS, a market session, and the same quotes
twice -- which is not a thing you get. With it, one recorded SPY chain becomes a
fixture that any test, any refactor and any model comparison can be run against,
deterministically, offline.

It is also the honest baseline for the model work coming next: Phase 1 and
Phase 2 have to be calibrated to the SAME surface for their skew residuals to
mean anything, and "the same surface" only exists if it was recorded.

Format is JSON rather than pickle so a snapshot stays readable and diffable, and
survives a refactor of the classes it came from.
"""

import datetime
import json
import math
import pathlib

import volatility_surface_3 as v3

FORMAT = "volsurf-replay/1"

_QUOTE_FIELDS = ("expiry", "strike", "right", "iv", "delta", "gamma", "vega",
                 "theta", "model_price", "und_price", "bid", "ask", "ts")
_CTX_FIELDS = ("tau", "forward", "discount", "parity_r2", "sigma_atm")


def _clean(x):
    """JSON has no NaN or Infinity. Write null, read it back as NaN."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _restore(x):
    return float('nan') if x is None else x


def record(app, ctxs, path, symbol=None, meta=None):
    """Write the current chain state to `path`. Returns the path.

    `meta` is an optional dict stored verbatim (data type, cycle stats, error
    counts). load() ignores it, so older readers are unaffected.
    """
    path = pathlib.Path(path)
    snap = {
        "format": FORMAT,
        "symbol": symbol or getattr(app, "_symbol", None) or "UNKNOWN",
        # UTC with its offset (Session H). The bare now() written before was local
        # wall-clock time with no offset -- on this UTC+3 machine three hours from
        # the instant the taus were measured at. recorded_utc() handles both.
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "spot": float(app.spot_price),
        "trading_class": getattr(app, "trading_class", None),
        "expiries": {
            exp: {f: _clean(getattr(ctx, f)) for f in _CTX_FIELDS}
            | {"strikes": [float(s) for s in getattr(ctx, "strikes", [])]}
            for exp, ctx in ctxs.items()
        },
        "quotes": [],
    }
    if meta:
        snap["meta"] = meta
    for req_id, q in sorted(app.quotes.items()):
        row = {"req_id": int(req_id)}
        for f in _QUOTE_FIELDS:
            v = getattr(q, f)
            row[f] = _clean(float(v)) if isinstance(v, (int, float)) else v
        snap["quotes"].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    return path


def load(path):
    """
    Read a snapshot back. Returns (app, ctxs) ready for surface_points().

    The app is a real LiveSurfaceApp with no socket, so every function that
    works on a live chain works on a replayed one unchanged -- which is the
    whole point, and is asserted in test_surface.py.
    """
    snap = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if snap.get("format") != FORMAT:
        raise ValueError(f"unexpected snapshot format {snap.get('format')!r}")

    app = v3.LiveSurfaceApp()
    app._symbol = snap["symbol"]
    app.symbol = snap["symbol"]
    app.spot_price = float(snap["spot"])
    app.trading_class = snap.get("trading_class")
    app.recorded_at = snap.get("recorded_at")
    app.meta = snap.get("meta", {})

    ctxs = {}
    for exp, c in snap["expiries"].items():
        ctx = v3.ExpiryContext(expiry=exp, tau=_restore(c["tau"]))
        for f in _CTX_FIELDS:
            if f != "tau":
                setattr(ctx, f, _restore(c[f]))
        ctx.strikes = list(c.get("strikes", []))
        ctxs[exp] = ctx

    for row in snap["quotes"]:
        req_id = int(row["req_id"])
        q = v3.OptionQuote(expiry=row["expiry"], strike=float(row["strike"]),
                           right=row["right"])
        for f in _QUOTE_FIELDS:
            if f in ("expiry", "strike", "right"):
                continue
            setattr(q, f, _restore(row[f]))
        app.quotes[req_id] = q
        app.id_map[req_id] = (q.expiry, q.strike, q.right)

    return app, ctxs


def recorded_utc(app, ctxs=None):
    """
    The instant a snapshot's taus were measured at, as a naive UTC datetime.

    Every expiry's tau in a cycle is tau_years(t0, expiry) for ONE t0 on the UTC
    clock, so t0 = expiry close (UTC) - tau is exact and needs no timestamp at all;
    it is what this returns when the expiries are given (median over expiries, a
    guard against a hand-edited file). Without them, only a timestamp that carries
    its offset is trusted: snapshots written before Session H stored local time
    with no offset, and those raise rather than silently shift every short tau.
    """
    if ctxs:
        import numpy as np
        import volsurf_core as vc
        stamps = sorted(vc.us_close_utc(vc.parse_ib_date(e)) - datetime.timedelta(seconds=c.tau * vc.SECONDS_PER_YEAR)
                        for e, c in ctxs.items() if c.tau is not None and np.isfinite(c.tau) and c.tau > 2e-6)
        if stamps:
            return stamps[len(stamps) // 2]
    raw = getattr(app, "recorded_at", None)
    t = datetime.datetime.fromisoformat(raw) if raw else None
    if t is None or t.tzinfo is None:
        raise ValueError(f"snapshot timestamp {raw!r} has no UTC offset and there are no taus to recover it from")
    return t.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def summary(path):
    """One-line description of a snapshot, without loading the whole thing."""
    snap = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return (f"{snap['symbol']} @ {snap['spot']:.2f}  "
            f"{len(snap['expiries'])} expiries, {len(snap['quotes'])} quotes, "
            f"recorded {snap['recorded_at']}")
