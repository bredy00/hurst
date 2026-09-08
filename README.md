# volatility-surface-tuning

IBKR implied-volatility surface. `C:\Projects\volatility-surface-tuning`.

```
alpha/volatility_surface_1_ALPHA.py   frozen original, md5 5134eff57fedbe6ac0698b7d94975e22
alpha/capture_alpha.py                stand-in feed -> alpha's own plot code
volatility_surface_2.py               six runtime fixes, maths untouched
volsurf_core.py                       pure maths: no IO, no scipy, no matplotlib
volatility_surface_3.py               v2's fixes + correct coordinates
fit/svi.py                            raw SVI slice fitting, scipy-free
sources/replay.py                     JSON record / replay of a chain snapshot
models/heston.py                      Heston cf (Albrecher branch) + MC
pricing/fourier.py                    Lewis + Carr-Madan, adaptive composite quadrature
capture_v2.py / capture_v3.py         stand-in feeds + comparison images
test_core.py                          87 checks   (the maths)
test_fixes.py                         47 checks   (IBKR runtime behaviour)
test_surface.py                       45 checks   (audit, SVI, density, replay)
test_models.py                        20 checks   (Heston cf)
test_pricing.py                       29 checks   (Fourier pricers)
docs/superpowers/plans/               the six-session rough Heston plan
captures/                             frames, GIFs, comparisons, snapshot.json
.venv/                                python 3.12.3
```

```bash
.venv/Scripts/python.exe test_core.py     # 87 passed
.venv/Scripts/python.exe test_fixes.py    # 47 passed
.venv/Scripts/python.exe test_surface.py  # 45 passed
.venv/Scripts/python.exe test_models.py   # 20 passed
.venv/Scripts/python.exe test_pricing.py  # 29 passed
.venv/Scripts/python.exe capture_heston.py # Phase 1 skew baseline
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
- **recovered H = 0.1199 +/- 0.0009** against a planted 0.120, r2 = 1.0000
- sigma band widens from +/-3.6% at 3 days to +/-19.6% at 42 days, as it must

## Session A (2026-09-07) -- the surface is finished

| | |
|---|---|
| **A1** | Butterfly + calendar audit wired into the live view. Calendar probes fixed **k**, not fixed z -- the condition is `dw/dtau >= 0` at `k = ln(K/F_tau)`, and at one fixed k the z values across three maturities were 2.06 / 1.00 / 0.58, i.e. entirely different moneyness |
| **A2** | `fit/svi.py`: raw SVI, **scipy-free** via the Zeliade reduction (fix `(m, sigma)` and the rest is linear). Recovers planted parameters to **1.4e-08** |
| **A3** | Risk-neutral density panel off the **fitted** slice on a dense **uniform** grid. Mass 0.9997, mean = forward to 0.01 |
| **A4** | Second, independent H from the realised log-vol path via the structure function, shown beside the skew estimate with a divergence flag |
| **A5** | `sources/replay.py`: JSON record/replay. Replayed surface points and H are bit-identical to the recorded ones |

Three bugs found and fixed while building it, each caught by a test rather than
assumed away:

- **Durrleman was judging extrapolation.** Evaluating `g(k)` over a fixed +/-2
  scored a 42-day slice fitted on `|k| <= 0.196` at **-216092**; over its own
  data range it is **+0.113**. Now data-aware, with the wings reported separately.
- **The density range ran away.** An arbitrage-violating slice is not a density,
  so its mass never reaches 1 and a mass-driven loop widened to `K = 0..5.9e8`
  with `dK = 33282`, hiding the negative region the panel exists to show. Capped.
- **Sizing the density grid from the wing volatility does not work.** SVI total
  variance grows linearly in `|k|`, so `half = n_sigma*sigma(half)*sqrt(tau)` is
  a fixed point iteration settling near `n_sigma^2 * b(1+|rho|)`.

`butterfly_violations` is now documented as a raw-grid convexity **heuristic** --
useful for catching a single bad print, but not the arbitrage condition. The
rigorous test is Durrleman `g(k) >= 0` on the fit, or a negative density.

End-to-end on the synthetic surface: planted `H = 0.120` recovered as
**0.1199 +/- 0.0009** from the skew and **0.1178** from the path (gap 0.0022,
agree), forward recovered to 0.0036, density mass 0.99967.

**179 tests green**: `test_core.py` 87, `test_fixes.py` 47, `test_surface.py` 45.

## Session B (2026-09-08) -- Heston cf and the Fourier pricers

**B1.** `models/heston.py`. The Albrecher form `g = (a-d)/(a+d)`, never the
reciprocal. Demonstrated instead of asserted: `char_func_trap` keeps the broken
version so the suite can measure it. At u = 4 the max/median step across a
4000-point tau grid is **5.2** for the shipped form and **5652** for the
reciprocal -- a real branch-cut crossing, not folklore.

Documented limitation: as `xi -> 0`, `kappa*theta/xi^2` and `(a-d)/xi^2` both
diverge while their combination stays finite, so accuracy against the exact
Black-Scholes cf degrades -- 1e-10 at `xi = 1e-4`, 3.5e-06 at `xi = 1e-6`. Never
an issue at calibration-realistic `xi`, but the degeneracy test has to run at a
moderate one.

**B2.** `pricing/fourier.py`. Lewis and Carr-Madan, sharing only the cf. Both
match the exact Black-Scholes cf to ~1e-15 and each other to 1e-8 on Heston.

Two findings worth keeping:

- **`leggauss` is O(n^2) and cost 8.4 SECONDS at n = 4000.** A calibrator prices
  a surface per objective evaluation, thousands of times, so a single large rule
  is unusable. Quadrature is now composite -- one cached 64-point rule over as
  many panels as needed -- and resolution is linear in the panel count.
- **No fixed `(u_max, n_nodes)` serves the whole surface.** The transform is
  still 2.9e-06 at u = 200 for a 1-day option but 7e-21 for a 1-year one, and
  short maturities need ~10 nodes per unit of u to resolve the decay cliff.
  Under-provisioning cost 1.7e-07 where 7.3e-15 was available. Both are now
  derived from the cf itself.

Carr-Madan turns out to be the better-conditioned of the two (2.6e-16 vs Lewis's
1.6e-07 at one day under identical automatic settings), because its damping
denominator decays like `1/v^2` on top of the transform's own decay. The module
docstring said the opposite before this was measured.

### Phase 1 baseline -- the number Phase 2 has to beat

| tau | 1d | 21d | 252d | 504d |
|---|---|---|---|---|
| Heston ATM skew | -0.4367 | -0.4164 | -0.2069 | -0.1340 |

Fitted **H = 0.483 +/- 0.015**; the short end alone gives **H = 0.495** -- the
classical one half, as theory requires, since a diffusion has a finite ATM skew
limit as `tau -> 0`. A market at `H = 0.12` would carry a 1-day skew near
**1.426** against Heston's **0.437**, a 3.3x shortfall. See
`docs/phase1_baseline.md` and `captures/heston_skew_baseline.png`.

**228 tests green**: core 87, fixes 47, surface 45, models 20, pricing 29.

## Still open

eSSVI (a shared parameterisation that is calendar-arbitrage-free by construction;
`essvi_calendar_ok` currently measures the violation rather than preventing it);
Kalman smoothing of the fitted parameters; then Sessions B-F in
`docs/superpowers/plans/2026-09-07-rough-heston-pipeline.md`.
