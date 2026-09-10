---
title: "Volatility Surface & Rough Heston Pipeline"
subtitle: "Progress report — Sessions A to C"
author: "Akin Akandere"
date: "10 September 2026"
---

# Status

Three of six planned sessions are complete. The live implied-volatility surface
is finished; the Heston characteristic function and two independent Fourier
pricers are built and cross-validated; the calibrator runs and the Phase 1
baseline is recorded. Session D — the Volterra kernel and the Markovian lift —
is next and has not been started.

| | |
|---|---|
| Test suites | 6 |
| Tests passing | **274 / 274** |
| Analytical health checks | **50 / 50** (16 s) |
| Production lines | 2,268 |
| Test lines | 1,641 |
| Commits | 15 |
| Python / numpy | 3.12.3 / 2.5.2 |

The repository is local-only. No remote, nothing pushed.

# Numerical methods

## In use

**Black–Scholes / Black-76 in forward measure.** Everything is expressed in
log-moneyness `k = ln(K/F)` and total variance `w = σ²τ`, with `τ = T − t₀` as a
time increment rather than absolute time. The normal CDF and PDF are computed
locally — `scipy.stats` alone costs 5.8 s to import and this module sits on the
startup path — and are verified against scipy to 2.2e-16.

**Implied volatility by Newton on vega with a bisection guard.** Price is
monotone in σ so the bracket can never be lost, and vega is exactly the
derivative required and already computed. Solved across a whole strike strip at
once, which matters: the one-at-a-time version was measured at half the cost of
an entire calibration objective evaluation.

**Put–call parity regression for the forward.** Regressing `C − P` on `K` gives
slope `−df` and intercept `df·F`, so both the discount factor and the forward
come out of the market rather than from an assumed rate and dividend. The
regression R² doubles as a chain-quality score. Recovered to 1.1e-13 on clean
data.

**Raw SVI slice fitting, scipy-free.** Holding `(m, σ)` fixed makes the problem
linear in the remaining three parameters, so the inner solve is one `lstsq` and
only a two-dimensional search remains — the Zeliade quasi-explicit reduction,
handled by a coarse grid plus a compass search. Recovers planted parameters to
1.4e-08.

**Heston characteristic function, Albrecher branch.** `g = (a−d)/(a+d)`, never
the reciprocal. The broken form is kept in the codebase so the suite can measure
the difference rather than assert it: at `u = 4` the max/median step across τ is
5.2 for the shipped form and 5652 for the reciprocal.

**Fourier pricing, two independent routes.** Lewis (no damping parameter) and
Carr–Madan (damped, plus an FFT variant for a whole strike strip). They share
only the characteristic function, so their agreement — 5.3e-15 — is evidence
rather than restatement. Quadrature is **composite** Gauss–Legendre: one cached
64-point rule over as many panels as needed, because `leggauss` is O(n²) and cost
8.4 seconds at n = 4000. Truncation and panel count are derived from the
characteristic function itself, since no fixed pair serves both a one-day and a
one-year option.

**Levenberg–Marquardt with Marquardt (diagonal) damping.** Forward-difference
Jacobian, bounded-logistic parameter transforms, multi-start, and a time budget.
Generic over the model: a transform plus a characteristic-function factory is all
it needs, so Session E's rough Heston will be calibrated by identical code.

**Breeden–Litzenberger risk-neutral density.** Differentiated on a dense
*uniform* strike grid off the *fitted* slice — the three-point second difference
loses an order of accuracy on non-uniform spacing, and real strike grids are $1
near the money and $5 in the wings.

**Structure function** `m(q,Δ) = E|X(t+Δ) − X(t)|^q` with `ζ(q) = qH`. A second,
independent route to the Hurst exponent, reading the realised log-volatility path
rather than the option surface. The two routes agree to 0.0029 on a common
planted truth.

**Monte Carlo, full-truncation Euler.** Used only to validate the Fourier pricers
against something that shares no machinery with them.

**Davies–Harte circulant embedding** for exact fractional Brownian motion in the
test fixtures. An approximate generator was tried first and returned the wrong
exponent, which would have looked like an estimator bug rather than a fixture bug.

## Planned

- **Fractional Volterra kernel** `K(t) = t^(α−1)/Γ(α)`, `α = H + ½`, and its
  completely-monotone representation `K(t) = ∫ e^(−xt) μ(dx)` with
  `μ(dx) = x^(−α)/(Γ(α)Γ(1−α)) dx`. Already verified numerically to 6.7e-10.
- **Markovian lift** by a sum of exponentials, `K(t) ≈ Σ wᵢ e^(−xᵢt)`, with nodes
  and weights from a geometric partition of `μ` (Abi Jaber–El Euch). Each factor
  `Uᵢ(t) = ∫ e^(−xᵢ(t−s)) dZ(s)` obeys `dUᵢ = −xᵢUᵢ dt + dZ(t)`, so the
  non-Markovian process becomes a finite Markovian system.
- **Lifted fractional Riccati system**, solved by RK4 — an N-dimensional set of
  ordinary Riccati ODEs, one per factor.
- **Hawkes self-exciting jump intensity** `λ(t) = μ + Σ α e^(−β(t−tᵢ))` over
  past events, in place of constant-λ Poisson.
- **Kalman and dual Kalman filtering** on an Ornstein–Uhlenbeck state space, with
  adaptive measurement noise across regime changes.

# Literature and resources

## Used so far

- Gatheral, Jaisson & Rosenbaum, *Volatility is Rough* (2014) — the structure
  function, `ζ(q) = qH`, and the SPX scaling regression (`ζ̂(1) ≈ 0.1555`,
  R² ≈ 0.93) as the empirical anchor for H ≈ 0.1.
- Heston (1993) for the model; Albrecher et al. (2007) for the branch-cut
  correction, the "little Heston trap".
- Carr & Madan (1999) for the damped Fourier transform; Lewis (2001) for the
  fundamental-transform form.
- Gatheral's SVI parameterisation and the Durrleman condition `g(k) ≥ 0` for the
  no-butterfly test; Gatheral–Jacquier for the necessary parameter conditions.
- Breeden & Litzenberger (1978) for the risk-neutral density.
- Lord et al. — full-truncation Euler for the Heston variance process.
- Davies & Harte — circulant embedding for exact fractional Brownian motion.
- Zeliade Systems' quasi-explicit SVI calibration note.
- Interactive Brokers TWS API reference — tick-type semantics and pacing limits,
  verified directly against the installed `ibapi` rather than from memory.
- Your own lecture notes (65 slides): model specification versus
  parameterisation, the Kalman gain and OU state space, the structure function,
  fBm and fractional OU, Hawkes versus Poisson, and your handwritten derivation
  of the Markovian lift.

## To be used

- Abi Jaber & El Euch, *Multifactor approximation of rough volatility models* —
  the geometric node and weight construction for the lift.
- El Euch & Rosenbaum, *The characteristic function of rough Heston models* — the
  fractional Riccati equation.
- Bayer, Friz & Gatheral, *Pricing under rough volatility* — rough Bergomi and
  the short-maturity skew.
- Shevchenko on fractional Brownian motion and Markovian lifting (not yet
  retrieved).
- Hawkes (1971) and Bacry–Muzy for self-exciting point processes in finance.

# Health and sanity checks

Run with `python healthcheck.py`. These are distinct from the test suites: the
suites assert that behaviour is correct, this reports the numbers behind those
assertions, so drift stays visible even while everything still passes.

**Foundations**

| check | measured | threshold | verdict |
|---|---|---|---|
| norm_pdf vs scipy | `0` | `1.00e-15` | PASS |
| norm_cdf vs scipy | `2.22e-16` | `1.00e-15` | PASS |
| put-call parity C-P = F-K | `2.84e-14` | `1.00e-09` | PASS |
| vega vs central difference | `4.23e-09` | `1.00e-04` | PASS |
| implied-vol round trip | `8.05e-16` | `1.00e-06` | PASS |
| fd_first exact on a quadratic | `7.46e-14` | `1.00e-12` | PASS |
| fd_second exact on a quadratic | `2.08e-11` | `1.00e-10` | PASS |

**Heston cf**

| check | measured | threshold | verdict |
|---|---|---|---|
| phi(0) = 1 | `0` | `1.00e-12` | PASS |
| phi(-i) = 1 (martingale) | `0` | `1.00e-10` | PASS |
| phi(-u) = conj(phi(u)) | `0` | `1.00e-12` | PASS |
| |phi(u)| <= 1 | `0.9999` | `1` | PASS |
| branch continuity, shipped (max/median step)<br><sub>Albrecher g=(a-d)/(a+d)</sub> | `5.242` | `50` | PASS |
| branch discontinuity, reciprocal form<br><sub>the trap, kept to prove it is real</sub> | `2825` | `500` | PASS |
| trap / shipped separation | `538.9 x` | `100 x` | PASS |
| xi -> 0 degenerates to Black-Scholes<br><sub>at xi=1e-4; 3.5e-06 at xi=1e-6 by cancellation</sub> | `5.92e-09` | `1.00e-08` | PASS |

**Pricers**

| check | measured | threshold | verdict |
|---|---|---|---|
| Lewis vs exact Black-76 | `3.55e-15` | `1.00e-10` | PASS |
| Carr-Madan vs exact Black-76 | `1.94e-16` | `1.00e-10` | PASS |
| Lewis vs Carr-Madan on Heston<br><sub>independent routes, 5 maturities</sub> | `5.29e-15` | `1.00e-08` | PASS |
| Carr-Madan insensitive to damping alpha<br><sub>alpha 0.75 to 3.0</sub> | `6.11e-16` | `1.00e-08` | PASS |
| Fourier vs Monte Carlo (sigmas)<br><sub>residual is Euler O(dt) bias</sub> | `0.3618 sd` | `5 sd` | PASS |
| simulated E[S/F] = 1 | `1.94e-06` | `6.12e-04` | PASS |
| risk-neutral density mass | `1` | `1` | PASS |
| risk-neutral density mean = forward | `1` | `1` | PASS |
| density non-negativity (min/peak)<br><sub>tail dither scales as 1/dK^2</sub> | `-7.04e-10` | `-1.00e-07` | PASS |

**Estimators**

| check | measured | threshold | verdict |
|---|---|---|---|
| H from ATM skew (planted 0.12)<br><sub>r2 = 1.00000</sub> | `0.12` | `0.12` | PASS |
| H from structure function, worst of 4 planted<br><sub>H = 0.10, 0.12, 0.30, 0.50</sub> | `0.0062` | `0.03` | PASS |
| two independent H routes agree<br><sub>surface route vs path route</sub> | `0.002891` | `0.05` | PASS |
| forward from put-call parity<br><sub>r2 = 1.000000, no rate or dividend input</sub> | `1.14e-13` | `1.00e-06` | PASS |
| discount factor from parity | `4.44e-16` | `1.00e-09` | PASS |
| SVI parameter recovery | `1.43e-08` | `0.001` | PASS |
| SVI slice passes Durrleman<br><sub>g_min over data = 0.2297</sub> | `1` | `1` | PASS |
| butterfly audit: false positives on a convex slice | `0` | `0` | PASS |
| butterfly audit: catches a planted dent<br><sub>flags the dent's neighbours</sub> | `2` | `1` | PASS |
| calendar audit: clean on monotone w | `0` | `0` | PASS |
| calendar audit: catches a reversal | `3` | `1` | PASS |

**Calibration**

| check | measured | threshold | verdict |
|---|---|---|---|
| plant-and-recover, worst relative error<br><sub>7 iters, 1.0s</sub> | `3.13e-11 rel` | `0.01 rel` | PASS |
| fitted RMSE on clean data | `4.06e-09 vol pts` | `0.001 vol pts` | PASS |
| Jacobian condition number at the solution<br><sub>high = parameters trade off; see identifiability</sub> | `30.05` | `1.00e+08` | PASS |
| spread of v0 under 0.5vp noise<br><sub>stable</sub> | `3.775 %` | `10 %` | PASS |
| spread of kappa under 0.5vp noise<br><sub>unidentified</sub> | `34.73 %` | `50 %` | PASS |
| spread of theta under 0.5vp noise<br><sub>stable</sub> | `2.803 %` | `50 %` | PASS |
| spread of xi under 0.5vp noise<br><sub>loose</sub> | `19.13 %` | `50 %` | PASS |
| spread of rho under 0.5vp noise<br><sub>stable</sub> | `3.73 %` | `10 %` | PASS |

**Session D readiness**

| check | measured | threshold | verdict |
|---|---|---|---|
| K(t) = int e^-xt mu(dx) representation<br><sub>mu(dx) = x^-a / (Gamma(a)Gamma(1-a)) dx, a = H+1/2</sub> | `6.67e-10 rel` | `1.00e-08 rel` | PASS |

**Engineering**

| check | measured | threshold | verdict |
|---|---|---|---|
| scipy modules on the live import path<br><sub>scipy.stats alone costs 5.8 s</sub> | `0` | `0` | PASS |
| startup to actionable error, no TWS<br><sub>was 9.6 s before lazy imports</sub> | `0.6875 s` | `3 s` | PASS |
| calibration objective evaluation<br><sub>39 quotes; 470 ms before vectorising</sub> | `13.69 ms` | `200 ms` | PASS |
| Gauss-Legendre rules constructed<br><sub>2 reuses; leggauss is O(n^2)</sub> | `1` | `1` | PASS |
| quadrature tolerance 1e-12 vs 1e-14<br><sub>calibration runs at 1e-12, ~2x faster</sub> | `0` | `1.00e-12` | PASS |
| replay round-trip fidelity<br><sub>recorded vs replayed surface points</sub> | `0` | `1.00e-12` | PASS |

50 of 50 checks pass (16s).

# Key results so far

**End-to-end surface validation.** A synthetic chain is generated with a planted
forward `F = S·e^((r−q)τ)` (r = 4.2%, q = 1.3%) and a planted roughness H = 0.12,
and the pipeline is told neither. It recovers the forward to 0.0036 — 0.0006% of
spot, where using spot instead would be off by 2.187 at 42 days — and
H = 0.1199 ± 0.0009 from the surface, 0.1178 from the realised path.

**Why the coordinate change was not cosmetic.** A fixed ±2% strike band has a
width in standard deviations of `0.02/(σ√τ)`, which diverges as τ → 0: measured
at 4.66σ at half a day and 0.73σ at twelve days. Vega then decays like
`exp(−d₁²/2)` while `d₁` itself scales like `1/√τ`, so the worst implied-vol
uncertainty ran to **2786 vol points** at the front against 0.01 at the back — a
spread of 198,906×. On a σ-normalised grid that spread is 3.6×.

**Phase 1 baseline.** Vanilla Heston fitted to a rough (H = 0.12) surface reaches
5.21 vol points RMSE, and the error is U-shaped — 7.52 at one day, 2.76 at
thirty, 5.42 at a year. At one day it delivers **58%** of the market's ATM skew
and goes flat below about a week. H from the market 0.120, from the fitted Heston
0.205. Fitted ξ = 2.02 with Feller violated by −3.58: the model visibly contorted
trying. This is the number Phase 2 must beat.

**What the calibration actually determines.** Under 0.5 vol-point quote noise the
surface is recovered to about the noise level while the parameters are not:
`v0` at 3.8% and `ρ` at 3.7% are stable, `κ` at 34.7% is not. The mechanism is
proven rather than asserted — extending maturities from one year to three, past
the relaxation time `1/κ = 0.5 y`, tightens `θ` from 33.0% to 5.0%. The original
plan called for 5% parameter recovery under noise; that is not achievable, and
asserting it would have been asserting something false.

# Open flags

| | severity | item |
|---|---|---|
| 1 | **major** | The live IBKR path has never run against a real TWS. Seven callbacks are exercised only by stubs. There is no Gateway on this machine and nothing listens on 7496/7497/4001/4002. |
| 2 | **major** | eSSVI is not built. `essvi_calendar_ok` measures calendar violations across independently fitted slices but nothing prevents them; a shared parameterisation would be arbitrage-free by construction. |
| 3 | moderate | `volatility_surface_3.py` is 963 lines holding six classes — IBKR callbacks, the sweeper, surface assembly, the audit, the density and the plotting. It should be split. |
| 4 | moderate | The Heston characteristic function loses precision as ξ → 0 (3.5e-06 at ξ = 1e-6) through cancellation in `κθ/ξ²`. Irrelevant at calibration-realistic ξ, but it constrains how the degeneracy limit can be tested. |
| 5 | moderate | `fd_second` drops to first-order accuracy on non-uniform grids. Documented and worked around by differentiating fitted slices on uniform grids, but a density taken straight off raw strikes would be biased. |
| 6 | minor | The test runner is hand-rolled — no pytest, no CI, no coverage measurement. |
| 7 | minor | Session D task D4, the fractional Riccati solve, is the identified schedule risk. If it slips, Session F (Hawkes) is the branch to drop. |

# Next

Session D: the Volterra kernel and the Markovian lift. D1 and D2 are de-risked —
the completely-monotone representation is already verified to 6.7e-10. The
load-bearing test in D4 is that as H → ½ the lifted characteristic function
converges to vanilla Heston, which is what proves the lift is a generalisation
rather than a different model.
