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
models/rough_heston.py                lifted rough Heston: kernel, lift, Riccati cf, Lewis pricer
models/hawkes.py                      Hawkes process: exact simulation, closed forms, MLE, time rescaling
models/jump_hosts.py                  Hawkes jumps on OU / Heston / rough Heston, exact expectations
filters/kalman.py                     Kalman on OU / CIR / lifted rough; adaptive, robust, dual, recursive MLE
calibrate/objective.py                implied-vol loss; model failure is charged, never zeroed
calibrate/fit.py                      Levenberg-Marquardt, generic over the model; Tikhonov prior
capture_v2.py / capture_v3.py         stand-in feeds + comparison images
capture_rough_vs_heston.py            Phase 2 headline: both models, in-model + stylised market
study_h_identifiability.py            Gauss-Newton standard error of H by surface design
capture_session_f.py                  Session F figure: intensity, shock responses, kurtosis, filters
debug_fbm_helper.py                   the Session A fBm fixture bug, reproduced
debug_fd_methods.py                   which 2nd-derivative stencil holds order on real grids
test_core.py                         103 checks   (the maths)
test_fixes.py                         47 checks   (IBKR runtime behaviour)
test_surface.py                       45 checks   (audit, SVI, density, replay)
test_models.py                        25 checks   (Heston cf)
test_pricing.py                       29 checks   (Fourier pricers)
test_calibrate.py                     65 checks   (objective, LM, identifiability, prior)
test_rough.py                         58 checks   (Session D: kernel, lift, cf, robustness)
test_rough_calibration.py             27 checks   (Session E: calibration, H, skew)
test_hawkes.py                        54 checks   (Session F1: Hawkes on three hosts)
test_filters.py                       34 checks   (Session F2: Kalman on three hosts)
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
.venv/Scripts/python.exe test_calibrate.py # 65 passed
.venv/Scripts/python.exe test_rough.py     # 58 passed
.venv/Scripts/python.exe test_rough_calibration.py  # 27 passed, ~4 min
.venv/Scripts/python.exe capture_rough_vs_heston.py # Phase 2 headline, ~6 min
.venv/Scripts/python.exe test_hawkes.py    # 54 passed, ~15 s
.venv/Scripts/python.exe test_filters.py   # 34 passed, ~2.5 min
.venv/Scripts/python.exe capture_session_f.py # Session F figure
.venv/Scripts/python.exe capture_heston.py      # Heston's own skew term structure
.venv/Scripts/python.exe capture_calibration.py # Phase 1 baseline vs a rough surface
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
inside the plot function. **9.6 s -> 0.76 s** (measured 2026-09-10; an earlier 1.97 s figure predated the 250 ms probe timeout).

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

## Session C (2026-09-10) -- calibration

**C1** `calibrate/objective.py`. The loss is in **implied vol**, not price:
price errors are dominated by at-the-money contracts worth a hundred times a
wing contract, so a price-space fit is nearly blind to the skew. Weights are
`volsurf_core.quote_weight` -- the same inverse-variance notion the live surface
already uses, not a second one invented for the fit.

**C2** `calibrate/fit.py`. Levenberg-Marquardt with Marquardt (diag) damping,
forward-difference Jacobian, multi-start, and a **time budget**. Generic over the
model: a `Transform` plus a characteristic-function factory is all it needs, so
Session E's rough Heston is calibrated by this identical code.

On clean data every parameter comes back to **0.000%**.

### Four bugs, all caught by a test or a crash

- **`math.tanh(40)` is exactly 1.0.** A bare tanh map for rho could hand the
  model `rho = 1`, where `sqrt(1 - rho^2) = 0` and the simulator degenerates.
- **A log map runs away.** Fitting a rough surface -- a shape Heston cannot make
  -- pushed `kappa` outward until `math.exp` overflowed past x ~ 709. Worse, well
  before that the characteristic function stops decaying, `_auto_u_max` runs to
  its cap, and one objective evaluation needs half a million quadrature nodes:
  the fit does not fail, it just never finishes. Both parameters now go through
  a bounded logistic.
- **The synthetic rough surface was nonsense.** A smile written as
  `sigma = atm + skew*k + 6*k^2` is fine at 7 days where the +/-3 sigma band is
  `|k| <= 0.06`, and produces a **265% implied vol** at one year where the same
  band reaches `|k| = 0.62`. Quote weights spanned 2.4e12 and the calibrator
  correctly fled to the parameter bounds. Curvature is now specified in sigma
  units, which is dimensionless and stays put.
- **A factor of `atm` in the skew scaling** gave a 7-day skew of -0.18 instead
  of -1.30. `d(sigma)/dk = a1/sqrt(tau)`, so `a1 = skew*sqrt(tau)`; the atm
  factors cancel.

Two performance findings: the implied-vol inversion was **half** the cost of an
objective evaluation (1.8 ms per strike, nearly all numpy call overhead), now
vectorised across the strip; and `Re[e^{-iuk} z]` is computed with cos/sin on
real arrays rather than a complex exponential that computes `exp(0) = 1` and
discards it. Together **470 ms -> 225 ms** per evaluation.

### What is actually identifiable

Under 0.5 vol point quote noise the surface is still recovered to about the noise
level while the parameters are not:

| | v0 | kappa | theta | xi | rho |
|---|---|---|---|---|---|
| spread across noise seeds | 2.1% | **37.7%** | 17.3% | 13.4% | 2.1% |

`v0` and `rho` are pinned by the short-end level and the skew. `kappa` is not,
and the mechanism is provable rather than asserted: extending the maturities from
one year to three -- past the relaxation time `1/kappa = 0.5 y` -- tightens
`theta` from 33.0% to 5.0%. You cannot read a long-run level off a surface that
stops before the process has relaxed. The plan's original "recover to 5% under
noise" was simply not achievable, and asserting it would have been asserting
something false.

### Phase 1 baseline -- Heston fitted to a rough surface

Overall RMSE **5.21 vol points**, and the error is **U-shaped**: Heston sacrifices
both ends to fit the middle.

| tau | 1d | 2d | 7d | 30d | 90d | 365d |
|---|---|---|---|---|---|---|
| rmse (vp) | **7.52** | 6.93 | 4.95 | 2.76 | 4.05 | **5.42** |
| market skew | -2.723 | -2.093 | -1.300 | -0.748 | -0.493 | -0.289 |
| Heston skew | -1.575 | -1.591 | -1.568 | -0.996 | -0.576 | -0.299 |
| Heston / market | **0.58** | 0.76 | 1.21 | 1.33 | 1.17 | 1.03 |

At one day Heston delivers **58%** of the market's ATM skew, and its skew goes
flat below about a week while the market keeps climbing. Fitted `xi = 2.02` with
Feller violated by -3.58 -- the model visibly contorted trying. **H from the
market 0.120, from the fitted Heston 0.205**; it cannot get below roughly 0.2.

Note the short end is not optional: a grid starting at 7 days lets Heston fit a
`tau^-0.4` decay and look adequate. See `docs/phase1_baseline.md` and
`captures/heston_vs_rough.png`.

**274 tests green**: core 87, fixes 47, surface 45, models 20, pricing 29,
calibrate 46.

## Analytics review (2026-09-12)

| item | done |
|---|---|
| Heston cf as ξ → 0 | `(a−d)(a+d) = −ξ²(u²+iu)` makes `(a−d)/ξ²` exact; the log term is `log1p` of an O(ξ²) quantity, Taylor series below ξ = 1e-4. Machine precision (2e-16) at ξ = 1e-8…1e-12 where the literal form is 19%–100% wrong; `char_func_naive` kept to prove it |
| `fd_second` on non-uniform grids | the coordinate-map proposal was implemented and **measured** (`debug_fd_methods.py`): order **0.01** on a $1/$5 kinked grid, negative on random spacing, because its differenced metric is O(1) wrong at a kink; on smooth grids the plain stencil is already O(h²). Shipped instead: Fornberg 5-point, order **3.06** on the kink. Breeden-Litzenberger error 7.9e-8 → 5.3e-12. The 2.08e-11 on the old check was rounding (floor 5.1e-11), not truncation |
| density non-negativity | `density_for_sampling`: floor at ε, renormalise to one, report the removed mass. The raw density stays raw — its sign is the arbitrage signal |
| κ identifiability | threshold not tightened. `calibrate(..., prior=, prior_weight=)` Tikhonov rows, reported as `prior_cost` separately from `data_cost`; κ spread 34.7% → 0.07%; `kappa_from_variance_swap` for the external pin |
| trend monitoring | `healthcheck.py --trend`: history per run, mean / std / drift z per check, Jacobian condition median < 100 and spike > 1e4 rules |
| fBm fixture bug | debugged once, `docs/fbm_helper_bug.md`: the point-sampled kernel capped the singular `(t−s)^(H−½)` at 1 for the newest shock; lag-1 autocorrelation −0.18 instead of −0.41; exact treatment of the first cell (hybrid scheme κ = 1) recovers 0.1207 |

## Session D (2026-09-12) -- the Markovian lift

`models/rough_heston.py`. Rough Heston written with Heston's own parameters, so
H = ½ **is** vanilla Heston.

| | |
|---|---|
| **D1** | `K(t) = t^(α−1)/Γ(α)` and its measure `μ(dx) = x^(−α)/(Γ(α)Γ(1−α))dx`; representation verified to 6.7e-10 |
| **D2** | N = 24 nodes by cell mass / cell mean of μ, first cell from **zero** (keeps the long memory), geometric to 1e5: **0.87%** on [1 hour, 2 years]. The published `r_n = 1 + 10n^−0.9` rule measured 27%. The plan's Laplace criterion (<1% on z ∈ [0.1, 100]) is **not met**, 3.5%: z = 0.1 probes ten years of memory |
| **D3** | the lifted recursion equals the convolution with the same kernel to 5e-14 (a semigroup identity, checked); against the true-kernel Volterra path from the same Brownian increments, **0.80%** in L² |
| **D4** | cf from the N-factor Riccati system. N = 1 at x = 0 reproduces closed-form Heston to 6e-11; the full lift converges to Heston **linearly in ½ − H** (1.6e-3, 1.6e-4, 1.6e-5 at H = 0.49, 0.499, 0.4999); lifted Monte Carlo agrees with the lifted cf at \|z\| < 0.4 |

Two time-steppers, ETDRK4 (order 4.0 measured) and an implicit exponential
trapezoidal rule whose implicit equation is a scalar quadratic per u.

## Session E (2026-09-12/13) -- calibration, and the comparison that matters

**E1.** The same Levenberg-Marquardt driver fits (v0, κ, θ, ξ, ρ, H). Clean
data, start 25–100% off in every parameter: all six recovered to **0.0015%**,
H = 0.120001, 30 s.

**The first E run failed 6 of 16 checks, and the failures were real bugs.**

- **Moment explosion.** Carr-Madan at α = 1.5 needs E[S^2.5] < ∞. The fit
  walked to (ξ, ρ, κ) = (0.63, −0.56, 0.19), where the Heston discriminant for
  p = 2.5 is −0.36, and the pricer returned **1.9e18** and **2.9e28**. The rough
  pricer now uses the **Lewis** contour, which needs only E[S^½] — finite for
  every martingale model — and makes `|φ(u − i/2)| ≤ 1` a hard check on every
  solve.
- **The objective rewarded failure.** Unpriceable quotes contributed exactly
  zero, so blowing up two maturities *lowered* the cost: that is how the fit
  ran to H = 0.02, θ = 0.88. A model failure is now charged at the inversion
  ceiling (500% vol), which no priceable outcome can exceed.
- **The implicit scheme is not unconditionally stable.** Session D's docstring
  said it was, on tests that only probed each maturity's pricing range. At
  u = 1200, H = 0.12, 90 days, 120 steps return |φ| = **1.2e36**. Stability
  edges measured for both schemes (z ≈ 2.65 and 15–25); step counts sized from
  them; an M-vs-2M error estimate refines a solve that is unstable.
- **The capture aliased missing maturities.** Model curves were aligned to the
  market by nearest neighbour, so a maturity the model failed on printed as a
  copy of its neighbour (−1.2574 at 14, 30, 60, 90 days), and `zip` dropped
  rows from a shorter report. Everything is keyed by maturity now; missing is a
  dash.

Speed, after all of that: OpenBLAS spent ~1.3 ms per ODE step starting threads
for 24×n matrix products — einsum below 1000 nodes, one pre-stacked real @
complex product for the update — and the u-grid became doubling panels
(2176 → 640 nodes at one day). **Ten-expiry objective 6.3 s → 1.06 s** with every
solve checked; worst IV error against a brute-force reference 1.3e-3 vol points.

**Identifiability of H**, measured (`study_h_identifiability.py`), not assumed:

| surface | weighting | SE(H) |
|---|---|---|
| 4 expiries × 5 strikes | vega² from a price half-spread, 0.5 vp noise | **0.202** (2.3 effective quotes of 20) |
| same | equal in vol | 0.047 |
| 8 × 7 with a real short end | vega², consistent noise, realistic spread | 0.020 |

Under vega² weights the short end — where H lives — carries no weight, and
three noisy seeds gave H = 0.50 / 0.25 / 0.12 with the optimiser beating the
truth's cost in all three: the data did not contain H. **corr(H, ξ) = +0.98**,
the rough analogue of Session C's κ/θ. With equal weights: H = 0.114, 0.100,
0.125.

**E2 — the comparison, on a market generated by rough Heston (H = 0.12)**, 8
maturities × 7 strikes, both models fitted by the same driver with equal weights
(`capture_rough_vs_heston.py`; `test_rough_calibration.py` asserts the same on a
30-quote surface):

| | market | rough Heston | vanilla Heston |
|---|---|---|---|
| fit rmse | — | **0.000 vp** (all 6 parameters exact) | 0.76 vp (κ = 35.6, ξ = 3.93) |
| fitted H | 0.12 | **0.1200** | — |
| one-day ATM skew | −4.84 | −4.84 (100%) | −3.66 (76%) |
| log-log slope, 1–3 days | −0.441 | −0.441 | **−0.123** |
| log-log slope, 7–14 days | −0.512 | −0.512 | −0.525 |

Vanilla Heston matches the market at 7–14 days — by pushing 1/κ to about ten
days so its transition from flat to decaying sits inside the window — and
**cannot follow below a week**, where a diffusion's skew must go flat. The slope
over the whole 1–14 day window hides this (−0.32 vs −0.47); the bend does not.
Note the market's own slope is −0.47, not H − ½ = −0.38: the τ^(H−½) law is
asymptotic, and at ξ = 0.5 the higher-order vol-of-vol term steepens it (−0.42
at ξ = 0.05).

**The stylised Phase 1 surface does not discriminate.** Fitted with equal
weights, rough Heston goes to **H = 0.5** — vanilla Heston — from all four
starts tried (H = 0.08, 0.12, 0.20, 0.25; costs equal to 3e-4), and both models
sit at 2.10 vp. That surface has too
little smile curvature for its skew to be any stochastic-volatility smile.
Phase 1's "Heston delivers 58% of the one-day skew" was a consequence of the
vega² weighting; with equal weights it delivers 201%. What survives is the
misfit, not its sign. See `docs/phase1_baseline.md` and
`docs/phase2_rough_vs_heston.md`.

**399 tests green**: core 103, fixes 47, surface 45, models 25,
pricing 29, calibrate 65, rough 58, rough calibration 27.
Health checks 72 of 72.

## Session F (2026-09-13) -- Hawkes arrivals and Kalman filtering, on three hosts

Scope as corrected on 2026-09-12: neither component is tied to the OU process.
Each is built once and attached to an OU log-variance, classical Heston and the
lifted rough Heston.

### F1 -- Hawkes jumps (`models/hawkes.py`, `models/jump_hosts.py`)

Exact Ogata simulation, vectorised across paths, checked against closed forms:
the mean rate (12.4998 vs μ/(1−n) = 12.5), the count variance of Hawkes (1971),
and the expected intensity after planted shocks, where an extra event's excess
decays at β − α rather than β. Maximum likelihood recovers (μ, α, β) within 3 SE;
the time-rescaling test passes for the Hawkes fit and rejects a Poisson fit to
the same events (p ≈ 0).

**Your second-spike claim, made exact.** By linearity,

    E[increment after shock 2] − E[increment after shock 1] = r(d + W) − r(d)

with r the single-shock mean response (verified to 1e-15). So the second spike is
larger exactly when r is still *rising* over the window.

| host | property |
|---|---|
| any host, **Poisson** | **never**: r only decays. The *level* after the second shock is still higher, by superposition, so a level comparison tests nothing |
| OU, Heston with Hawkes | at short gaps iff **α > κ** (r′(0) = η(α − κ)), confirmed: on at 1.03κ, off at 0.97κ. At n = 0.6 it holds for gaps up to ~5 days |
| rough Heston, jumps in the **Volterra driver** | **0 of 21** (gap, window) pairs at n = 0.6: each jump's effect decays like t^(H−½) faster than excitation builds. 14 of 21 at n = 0.95: it needs near-critical clustering, in line with El Euch–Fukasawa–Rosenbaum |
| rough Heston, jumps **added to V directly** | holds at short gaps |

Which rough variant is right is a modelling decision; both are implemented.

**Kurtosis.** With constant variance and zero-mean Gaussian jumps, daily returns are a
scale mixture, and excess kurtosis is exactly 3s⁴Var[N]/(vw + E[N]s²)². So the
Hawkes/Poisson ratio is the daily Fano factor, 1.92; simulation matches both closed
forms. On every stochastic-variance host Hawkes exceeds Poisson by 12–14 SE.

### F2 -- Kalman filtering (`filters/kalman.py`)

One filter, three state spaces: your lecture's OU (F, B, u, Q, K exactly), CIR with
state-dependent Q, and **the 24 lifted rough factors as the state**. Their drift is
linear and their process noise is rank one; that is what makes a rough model
filterable.

- Matches the lecture's four-line recursion to 1.7e-16. The gain converges to the
  analytic steady state, and the RMSE gain over raw quotes (45.6%) is what theory
  predicts (45.2%). R↑ ⇒ K↓: small R tracks the data, large R the model.
- **The lecture's AR(1) calibration is biased by quote noise.** It puts κ at 61–65
  for a true 5, because noise attenuates the regression slope. Maximum likelihood
  through the filter recovers it within 1 SE.
- **Strict vs adaptive** (the lecture's two panels, 40 seeds):

  | host | strict | adaptive | robust (persistence-gated) |
  |---|---|---|---|
  | OU | 18 steps, bad print 2.3× | 0 steps, **17×** | **1 step, 1.7×** |
  | Heston | 2 steps, 3.2× | 0 steps, **10×** | **1 step, 1.9×** |
  | rough | 0 steps, 2.9× | 0 steps, 6.2× | 0 steps, 3.0× |

  On rough Heston a shock relaxes like a power law and leaves no regime to lag behind.
- **Dual Kalman.** The plan's "within 10% over 2000 steps" is not attainable:
  the MLE's own SE at 2000 steps is 20% of κ. Wan–Nelson's dual EKF and the joint EKF
  agree to 0.14%, but both carry **Ljung's (1979) bias**: they ignore how the gain
  depends on the parameters, and land up to 2 SE from the MLE. `recursive_mle` carries
  the full sensitivity equations and stays within 1 SE on every host-seed (total
  deviation 2.5 SE vs 4.6 SE).
- **Rough host findings.**
  - QML κ is biased 2–4 SE when simulated variance spends ~13% of days below zero, and
    unbiased (z +0.4, +0.4) where it never does. The clamp is a non-linearity the linear
    filter cannot represent.
  - A CIR filter fitted by MLE to rough variance forecasts it at 5 days within
    −0.6% to +2.8% of the *true* lifted model, by pushing κ to 43–54. That is the
    Session E phenomenon again; the lift's edge is the correct state space, not
    short-horizon forecasting.

See `captures/session_f.png`.

**487 tests green**: core 103, fixes 47, surface 45, models 25, pricing 29,
calibrate 65, rough 58, rough calibration 27, hawkes 54, filters 34.
Health checks 90 of 90.

## Still open

- **After Session F**: learning H itself in the filter (H defines the lifted
  factors, so a parameter filter would change what the state means), a
  positivity-preserving scheme for rough variance (the floor biases QML),
  Hawkes estimation from real event data, and the nearly-unstable Hawkes →
  rough volatility link as a model rather than a citation.
- eSSVI (calendar-arbitrage-free by construction; `essvi_calendar_ok` measures
  but does not prevent).
- The live IBKR path has still never run against a real TWS.
- `volatility_surface_3.py` (963 lines) should be split; no pytest / CI.
- A rough fit is ~1 s per ten-expiry objective evaluation: fine for a study,
  slow for a live recalibration loop.
