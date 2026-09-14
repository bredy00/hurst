# Recording real IBKR data — step by step

**For:** Akin. **Written:** 2026-09-15, Session G.
**Why you, not Claude:** logging in to IBKR needs your credentials and your phone
for two-factor authentication. Claude must never handle either. Everything after the
login (connecting, recording, analysing) runs from this project and needs nothing
else from you.

**Time budget:** 20 minutes once (install and settings), then 1 minute a day.

---

## 0. What gets recorded, and when it can be recorded

| What | Command | When | Takes |
|---|---|---|---|
| Connection check | `record_chains.py --check` | any time you are logged in | ~15 s |
| **History** (5 y daily, IV index, 12 months of 5-min bars) | `record_history.py` | **any time**, even at night | ~10–12 min |
| **Option chains** (16 maturities, OTM grid, bid/ask/IV/greeks) | `record_chains.py` | **US market hours** | ~1–2 min per snapshot |
| Closing chain (last quotes of the day) | `record_chains.py --once --data-type frozen` | after the close | ~2 min |

**US market hours in Istanbul time.** Turkey is UTC+3 all year; New York moves its clocks.

| Period | New York session | Istanbul |
|---|---|---|
| now until Sat 31 Oct 2026 | 09:30–16:00 EDT | **16:30–23:00** |
| from Sun 1 Nov 2026 | 09:30–16:00 EST | **17:30–00:00** |

The first 15 minutes after the open have wide spreads. The recorder's default stops a
cycle from starting after 15:45 New York time.

---

## 1. Install IB Gateway (once)

IB Gateway is the lighter of IBKR's two programs: no charts, just the API connection.
If you already use **TWS** (Trader Workstation), that works too; only the port differs
(section 3).

1. On IBKR's website go to **Trading → Platforms → IB Gateway** and download the
   **Stable** version for Windows.
2. Run the installer with the defaults. It installs to `C:\Jts\ibgateway\...`.
3. Start **IB Gateway** from the Start menu.

## 2. Log in (you, every day)

1. In the login window choose **IB API** (not FIX CTCI).
2. Choose **Live Trading** (recommended, see below) or **Paper Trading**.
3. Enter your username and password and approve the IBKR Mobile notification.

**Live or paper?** Market data subscriptions belong to the **live** account. A paper
login only sees them if **Client Portal → Settings → Paper Trading Account → Share
real-time market data subscriptions with paper trading account** is on, and that can
take up to 24 hours to apply. For recording data, a **live login with a read-only API**
(next section) is the simplest: the API then cannot place orders at all.

> **One login at a time gets market data.** If IBKR is also open on your phone or in the
> web portal with live data, the API session receives error **10197** and no quotes.
> Log out of the other sessions while recording.

## 3. API settings (once)

In IB Gateway: **Configure → Settings → API → Settings**
(in TWS: **File → Global Configuration → API → Settings**).

| Setting | Value | Why |
|---|---|---|
| Enable ActiveX and Socket Clients | **on** (TWS only; Gateway always has it) | lets Python connect |
| **Read-Only API** | **on** | the recorder only reads; this makes orders impossible |
| Socket port | leave the default (table below) | |
| Allow connections from localhost only | **on** | nothing outside this PC can connect |
| Master API client ID | empty | |

Then **Apply** and **OK**.

| You logged in to | Port | Pass to the scripts as |
|---|---|---|
| IB Gateway, live | 4001 | `--port gateway-live` (the default) |
| IB Gateway, paper | 4002 | `--port gateway-paper` |
| TWS, live | 7496 | `--port tws-live` |
| TWS, paper | 7497 | `--port tws-paper` |

**Stay logged in across days (optional).** IB Gateway: **Configure → Settings → Lock and
Exit → Auto restart**. It then restarts itself each night without asking you to log in
again, until the weekly re-authentication (Sundays).

## 4. Market data subscriptions (once, check what you have)

In **Client Portal → Settings → Market Data Subscriptions**:

- **US options quotes:** *OPRA Top of Book (L1)*, or a US equity-and-options streaming
  bundle that includes OPRA. Without it you can still record **delayed** (15 min) data
  with `--data-type delayed`; quotes are real, just late.
- **SPY itself:** a US equities feed (e.g. NYSE / Network A, or the US securities
  snapshot bundle).
- **VIX (optional):** Cboe index data. Without it the history pull skips VIX and
  carries on.

Prices and bundle names change; the page shows them. As a private individual, answer
the **non-professional** questionnaire truthfully if prompted.

---

## 5. Connection check (every session, 15 seconds)

Open a PowerShell terminal in VS Code:

```powershell
cd C:\Projects\volatility-surface-tuning
.\.venv\Scripts\python.exe record_chains.py --check
```

(add `--port gateway-paper`, `--port tws-live`, etc. if you are not on Gateway live).

**What success looks like:**

```
New York time Tue 10:02; regular hours: yes
Spot SPY = 651.23  (live data)
Trading class 'SPY': 58 expiries, 412 strikes
server version 157, connection time 20260915 10:02:11 US/Eastern
SPY 20260922 651C: bid 7.61 ask 7.64 iv 0.1432 delta 0.512 vega 37.215 iv_error 0.0004
vega units: IBKR x 100 / Black-Scholes = 1.003 (consistent)
CHECK PASSED -- ready to record
```

Two lines matter most:

- **`(live data)`**: it may say `delayed`; that is fine but tell me.
- **`vega units ... (consistent)`**: this verifies, on a real quote, a unit conversion
  I added today. If it says **UNITS DISAGREE**, stop and paste the output to me before
  recording anything.

## 6. Pull the history (any time, ~10–12 minutes)

```powershell
.\.venv\Scripts\python.exe record_history.py
```

It walks back one week at a time through the 5-minute bars (IBKR serves 5-minute bars at
most one week per request), pausing ~10 s between requests to stay inside IBKR's pacing
limit. Leave it running. The last lines look like:

```
wrote captures\real\history_SPY_2026-09-15.json in 611 s: 1257 daily bars, 1257 IV days, 19250 5-min bars -> 250 RV days
```

For a quick first run: `--rv-months 3` (~3 minutes).

## 7. Record chains (during market hours)

Recommended for today: start at **17:00 Istanbul** (10:00 New York), one snapshot every
15 minutes until the close:

```powershell
.\.venv\Scripts\python.exe record_chains.py --every 15 --until 15:45
```

Each cycle prints one block:

```
[10:00:41 ET] wrote captures\real\SPY\2026-09-15\SPY_2026-09-15_09-59-02ET.json  (99 s)
  spot 651.40  live data  612 quotes, 598 with IV, 571 usable
```

- Leave the terminal open. The PC must not sleep: in Windows **Settings → System →
  Power**, set sleep to *Never* while plugged in, for today.
- **Ctrl+C** stops it. Every file already written is complete; the cycle in flight is
  discarded, never half-saved.
- A line `parity r2 < 0.99 on N expiries` means put-call parity could not pin that
  expiry's forward (usually a thin far-dated expiry), so it fell back to spot for it.
  A few are fine; most of them is not. Tell me.
- If the connection drops mid-session, just run the same command again. New files get
  new timestamps, nothing is overwritten.

**After the close (optional):** one snapshot of the closing quotes:

```powershell
.\.venv\Scripts\python.exe record_chains.py --once --data-type frozen
```

## 8. Hand it over

Tell me **"recorded"** in chat. Everything lands in `captures\real\`. From there I run the
calibration (Heston and rough Heston, with the new short-end weighting) on the
snapshots, and the filters (including H learned inside the filter) on the history.

---

## Troubleshooting

| You see | Meaning | Do this |
|---|---|---|
| `Nothing is listening on 127.0.0.1:4001` | Gateway/TWS not running, not logged in, or a different port | Log in; check the port table in section 3; pass the matching `--port` |
| `TWS refused the API connection` | API not enabled, or connection not from localhost | Section 3 settings, then restart Gateway |
| `326 ... client id already in use` | another program uses client id 37 | add `--client-id 41` |
| `354` / `10089` not subscribed | no live subscription for these quotes | `--data-type delayed`, or add OPRA (section 4) |
| `10167` displaying delayed data | IBKR is sending delayed quotes | fine for testing; the snapshot records the data type |
| `10197` competing live session | IBKR is open with live data elsewhere | log out of phone / web sessions |
| `200` no security definition (a few) | a listed strike doesn't trade for that expiry | harmless |
| `162` pacing violation in the history pull | too many requests too fast | wait 10 minutes, rerun; or `--rv-months 3` |
| `1100` / `2110` connectivity lost | IBKR's side | wait for `1102` (restored); rerun the command |
| `No usable price for SPY` | neither live nor delayed spot arrived | check the subscription; try `--data-type delayed` |
| `CHECK FAILED -- no implied vol arrived` | quotes came without greeks | outside market hours use `--data-type frozen`; otherwise paste me the output |
| connect succeeds then errors about an unsupported version | the installed Python API (ibapi 9.81) is older than your Gateway accepts | install IBKR's **TWS API** for Windows from IBKR's website, then `.\.venv\Scripts\python.exe -m pip install "C:\TWS API\source\pythonclient"`, and tell me (the handlers already accept the 10.x signatures) |

## What the recorder does and does not touch

- It **reads** quotes, greeks and historical bars. With Read-Only API on, IBKR rejects
  any order or account change, and the code sends none.
- It requests no account, position or balance data. Snapshots contain market data
  only, but they are **licensed to you**: keep them in private storage (the GitHub
  backup is private for this reason).
- Never paste your password or 2FA codes into chat. Claude will never ask for them.
