"""
Daily history for the real-data filters (Session G).

record_history.py writes one JSON file; this module reads it and turns it into
variance observations. There are three noisy views of the same latent spot
variance, and they are NOT interchangeable:

  garman_klass       from each day's open/high/low/close. Unbiased for the day's
                     integrated variance under a driftless diffusion, ~7.4x the
                     efficiency of a squared close-to-close return. Blind to the
                     overnight gap.
  realised_variance  sum of squared 5-minute log returns inside the session.
                     The standard estimator; also blind to the overnight gap.
  iv30_variance      IBKR's 30-day implied vol, squared. A risk-neutral
                     expectation of AVERAGE variance over the next 30 days, so it
                     is smoothed and carries the variance risk premium. A filter
                     that reads it as spot variance will mistake the premium for
                     a level and the smoothing for slow mean reversion.

Everything is annualised with 252 trading days.
"""

import json
import math
import pathlib

import numpy as np

FORMAT = "volsurf-history/1"
TRADING_DAYS = 252.0


def load(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"unexpected history format {data.get('format')!r}")
    return data


def _day(date_str):
    """'20260914', '20260914  09:35:00' or '20260914 09:35:00 US/Eastern' -> '20260914'."""
    return str(date_str).strip()[:8]


def merge_bars(chunks):
    """Stitch overlapping request chunks: one bar per timestamp, ascending."""
    seen = {}
    for chunk in chunks:
        for bar in chunk:
            seen[str(bar["date"]).strip()] = bar
    return [seen[k] for k in sorted(seen)]


def daily_series(bars):
    """OHLC arrays keyed by trading date, ascending."""
    bars = sorted(bars, key=lambda b: _day(b["date"]))
    return {
        "date": [_day(b["date"]) for b in bars],
        "open": np.array([b["open"] for b in bars], float),
        "high": np.array([b["high"] for b in bars], float),
        "low": np.array([b["low"] for b in bars], float),
        "close": np.array([b["close"] for b in bars], float),
    }


def garman_klass(o, h, l, c):
    """
    Garman-Klass (1980) daily variance, annualised:

        0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2
    """
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    hl = np.log(h / l)
    co = np.log(c / o)
    return TRADING_DAYS * (0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co)


def realised_variance(bars5m, min_bars=60):
    """
    Annualised realised variance per session from intraday bars.

    Within a day: the first bar contributes ln(close/open), every later bar
    ln(close/previous close). The overnight gap is excluded by construction.
    Days with fewer than `min_bars` bars (half days, gaps in the pull) are
    dropped rather than scaled up.
    """
    days = {}
    for b in sorted(bars5m, key=lambda b: str(b["date"])):
        days.setdefault(_day(b["date"]), []).append(b)
    dates, rv, counts = [], [], []
    for d in sorted(days):
        rows = days[d]
        if len(rows) < min_bars:
            continue
        closes = np.array([r["close"] for r in rows], float)
        r = np.empty(len(rows))
        r[0] = math.log(rows[0]["close"] / rows[0]["open"])
        r[1:] = np.diff(np.log(closes))
        dates.append(d)
        rv.append(TRADING_DAYS * float(np.sum(r * r)))
        counts.append(len(rows))
    return {"date": dates, "rv": np.array(rv), "bars": np.array(counts)}


def iv30_variance(bars):
    s = daily_series(bars)
    return {"date": s["date"], "v": s["close"] ** 2}


def align(*series):
    """Intersect several {'date': [...], key: array} series on their dates."""
    common = set(series[0]["date"])
    for s in series[1:]:
        common &= set(s["date"])
    dates = sorted(common)
    out = []
    for s in series:
        idx = {d: i for i, d in enumerate(s["date"])}
        take = [idx[d] for d in dates]
        out.append({k: (np.asarray(v)[take] if k != "date" else dates)
                    for k, v in s.items()})
    return dates, out
