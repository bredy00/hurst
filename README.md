# volatility-surface-tuning

IBKR implied-volatility surface. `C:\Projects\volatility-surface-tuning`.

```
alpha/volatility_surface_1_ALPHA.py   frozen original, md5 5134eff57fedbe6ac0698b7d94975e22
alpha/capture_alpha.py                stand-in feed -> alpha's own plot code
volatility_surface_2.py               six runtime fixes, maths untouched
volsurf_core.py                       pure maths: no IO, no scipy, no matplotlib
volatility_surface_3.py               v2's fixes + correct coordinates
capture_v2.py / capture_v3.py         stand-in feeds + comparison images
test_fixes.py                         47 checks   (runtime behaviour)
test_core.py                          55 checks   (the maths)
captures/                             frames, GIFs, alpha_vs_v2.png, alpha_vs_v3.png
.venv/                                python 3.12.3
```

```bash
.venv/Scripts/python.exe test_core.py     # 55 passed
.venv/Scripts/python.exe test_fixes.py    # 47 passed
.venv/Scripts/python.exe capture_v3.py    # re-render + end-to-end validation
```

## Environment

No TWS or IB Gateway on this machine; nothing listens on 7496/7497/4001/4002.
The capture harnesses drive the real plotting code with a stand-in feed. They do
not modify any surface script.

`ibapi 9.81.1.post1` (PyPI serves nothing newer — 10.x ships only via IBKR's own
installer, and the two differ in callback signatures, which is why the `error()`
handler dispatches on arity). numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1,
scipy 1.18.1, pillow 12.3.0.

Packets to a closed port on 127.0.0.1 here are **dropped, not refused**, so a TCP
probe raises `TimeoutError` rather than `ConnectionRefusedError` and always burns
its full timeout. Probably the firewall. 250 ms is used.

## v2 -- the six runtime fixes

| # | Was | Now |
|---|---|---|
| 1 | `error()` pinned to one signature | `error(*args)`, arity dispatch across `<=9.81` / `10.10+` / `10.19+`; 502 releases all waiters |
| 2 | `while spot==0: sleep(0.1)` | bounded waits; LAST / BID+ASK mid / CLOSE + delayed 66/67/68/75; `reqMarketDataType(3)` fallback |
| 3 | 162 lines at ~100 msg/s, never cancelled | `ChainSweeper`: <=90 lines, 40 msg/s token bucket, cancel-on-arrival, 30 s staleness |
| 4 | no class filter | SMART + multiplier 100 + `tradingClass == symbol`, pinned per contract |
| 5 | `"106"` on OPT contracts | empty generic list (106 yields tick 24, the *underlying's* 30-day IV) |
| 6 | `e >= today` | `select_expiries(min_days_to_expiry=1)` |

## v3 -- the four that mattered more

**A. Startup.** A TCP probe before `EClient.connect()`. But profiling showed the
10 s was never the socket: **imports were 9.1 s of it, `scipy.stats` alone 5.8 s**.
So `volsurf_core` has no scipy at all — `norm_pdf`/`norm_cdf` are three lines each
(checked against scipy to 1e-15 in `test_core.py`) and implied vol uses Newton on
vega with a bisection guard instead of `brentq`. matplotlib is imported lazily
inside the plot function. **9.6 s -> 1.97 s.**

**B. The greeks are kept.** IBKR sends delta/gamma/vega/theta and the model price
on every `tickOptionComputation`; v1 and v2 both bound them and threw them away,
including the vega the weighting needs. `OptionQuote` keeps all of it and exposes
`iv_error = half_spread / vega`.

**C. Coordinates.** Grid on `z = ln(K/F) / (sigma_atm * sqrt(tau))`, surface plots
`w = sigma^2 * tau`, `tau = T - t0` as a real time increment, and the forward comes
from a put-call parity regression per expiry (`C - P = df*(F - K)`), so no rate or
dividend is assumed and the regression r2 doubles as a chain-quality score. Calls
and puts split at the **forward**, not spot.

**D. No `bfill().ffill()`.** Outside the observed z range the grid is NaN and stays
NaN; matplotlib draws it as a hole. A gap in the market renders as a gap.

### Why C is not cosmetic

A fixed +/-2% band has a width in standard deviations of `0.02 / (sigma*sqrt(tau))`,
which **diverges as tau -> 0**. Measured on this chain:

| tau | band edge, fixed 2% | band edge, 3-sigma | worst IV error, 2% | worst IV error, 3-sigma |
|---|---|---|---|---|
| 0.5 d | 4.66 sigma | 2.86 sigma | **2786 vol points** | 3.12 vol points |
| 3 d | 1.76 | 2.90 | 0.10 | 1.41 |
| 7 d | 1.04 | 2.95 | 0.02 | 1.10 |
| 12 d | 0.73 | 2.98 | 0.01 | 0.86 |

Same band, 6.4x variation in width; the sigma band varies 1.05x. Vega decays like
`exp(-d1^2/2)` while `d1` itself scales like `1/sqrt(tau)`, so uncertainty
compounds twice — 198906x spread across expiries under the fixed band, 3.6x under
the sigma band. That is the "wide and narrow at the same time" problem, and the
sigma grid removes it by construction rather than masking it.

## End-to-end validation

`capture_v3.py` plants a forward `F = S*exp((r-q)*tau)` with r=4.2%, q=1.3%, and a
roughness `H = 0.12`, then tells the pipeline neither.

- worst forward error **0.0036** (0.0006% of spot); using spot as the forward
  would be off by 2.187 at 42 days
- parity r2 = 1.00000 on every expiry
- **recovered H = 0.1205** against a planted 0.120, r2 = 1.0000
- sigma band widens from +/-3.6% at 3 days to +/-19.6% at 42 days, as it must

## Still open

SVI / eSSVI slice fitting; butterfly and calendar audits wired into the live view
(the detectors exist in `volsurf_core`, nothing calls them yet); Kalman smoothing
of the fitted parameters; replay source; rough Bergomi and the Heston jump-diffusion
with a Hawkes-style self-exciting intensity rather than constant-lambda Poisson.
