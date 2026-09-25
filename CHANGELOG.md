# Changelog

## v1.1.0 (2026-09-25): Session L, forty-four nodes, a volatility backup, and a trial

Full accounts: `docs/study-lift-44.md`, `docs/study-egarch.md`, `docs/trial-bl-hurst.md`.

### Changed
- **The lift is 44 nodes** to the same 1e8/y top node (was 40). The worst kernel error over
  the H box falls from 0.908% to 0.763%, so the review's 1% criterion keeps a quarter of its
  headroom instead of a tenth, and implied vols come 16% closer to true rough Heston at
  every H and maturity measured. `rh.using_lift(40, 1e8)` and `run_real_data.py --lift 40:1e8`
  run the old one. Stability re-measured (`study_stability_constants.py 44 1e8`): the same
  edges, so `C_STAB` is unchanged.
- **The 44-node lift is faster than the 40-node one was** (ten-expiry objective 0.425 s
  against 0.470 s, best of three interleaved), through two exact changes to `log_char_func`:
  - the carried state drops the factors with no memory beyond one step (`DEAD_DECAY`):
    20 of 44 at one year, 5-8 at one day;
  - `rh.reuse_riccati()` reuses a solve when only v0 or theta changed -- they enter the log cf
    linearly, so two of a rough calibration's six Jacobian columns cost no solve. On inside
    `calibrate_rough_heston` only.

  Both verified against 72 saved reference cf and price arrays (worst 1.2e-14 and 2.2e-16)
  and by `test_rough.test_carried_state`.

### Added
- `study_lift_44.py` and `docs/study-lift-44.md`: what the nodes buy, and what they do not.
  The lifted driver's stationary increments are graded against **exact fGn** -- closed-form
  spectral densities both sides, the Whittle divergence as the expected log-likelihood ratio,
  and Monte Carlo through Shevchenko's circulant embedding. Four nodes change the increment
  law in the third decimal; **the fastest node, not the node count, is the lever** for
  intraday roughness, and 13,000 years of daily data would be needed to tell the two lifts
  apart.
- `models.fbm.stationary_gaussian`: Shevchenko's embedding for any stationary covariance with
  a nonnegative embedding, refusing rather than truncating a negative one.
- **`models/egarch.py`, `study_egarch.py`, `test_egarch.py` (17 checks), `docs/study-egarch.md`:**
  EGARCH(p,q) with Student-t innovations, Beta-t-EGARCH (Harvey's score-driven model, whose
  response to one extreme return is bounded) and GARCH, one interface, BIC scan and QLIKE
  forecast scoring. Scanned against this project's own rough Heston, where the true daily
  integrated variance is known: **BIC ranks the t models first and they forecast worst** --
  every t fit lands at nu ~ 2.2, matching the tail shape while setting the variance forecast
  six times too high. The best returns-only model carries 1.20x HAR-RV's QLIKE.
- **The trial** (`portfolio/`, `study_trial_bl_hurst.py`, `test_portfolio.py` (42 checks),
  `docs/trial-bl-hurst.md`): a Black-Litterman portfolio on a synthetic market with a SQLite
  database, FF4 attribution and deflation through **falsify**, and a dual Kalman filter
  calibrating (H, view confidence, risk aversion, equilibrium) to a target market exposure.
  Self-contained, and revertible by deleting those four paths.

  **Verdict: practical as machinery, not as stated.** The objective is monotone in H -- both
  the grid and the filter drive it to the bound -- so the cost-optimal H (0.02) is not the
  market's H (0.10), and a knob whose optimum is always at its boundary needs a bound, not a
  filter. The pre-registered target was also unreachable: the portfolio's true market exposure
  swings with sd 0.24 by design (Black-Litterman is volatility-managed), against a 0.05
  tolerance. The filter does control the exposure's mean, and with two measurement defects
  fixed it matches a 54-point grid at a fraction of the cost.

### Fixed
- **A bias judged in standard errors is not a stable test.** `test_rough.test_positivity_scheme`
  asked QE prices to match the cf within 4.5 SE at rho = -0.7; the SE shrinks with the path
  count while the documented skew bias does not, so the check fails once enough paths are
  thrown at it. Now bounded in vol points (under 0.35; measured 0.16-0.23 over four seeds).
- **The synthetic history's H is three times noisier than its reported SE.**
  `test_filters.test_learn_h` used 3 of the profile curvature's standard error. Over 21 seeds
  the estimator reads 0.135 +/- 0.079 against a median reported SE of 0.028. A 2x2 of data
  lift against filter lift shows the filter's lift changes nothing (three decimals) while the
  data's moves H by up to 0.09 between seeds. Judged now against the measured spread.
- Unused imports and dead locals across the library (pyflakes-clean), and the CI runner
  pinned to `ubuntu-24.04` before `ubuntu-latest` becomes Ubuntu 26 on 19 October 2026.
- Four standing health checks for the lift (128 in total): the carried state and the reused
  Riccati solve are exact, the kernel error over the H box, and the driver's variogram at one
  day against exact fGn's. The two exactness guards protect an optimisation whose failure mode
  is a wrong number, not a crash.

## Sessions J and K, discrete hedging and the checks that watch it

### Added
- `models/hedging.py`: rough-Heston paths carrying the exact forward variance and a
  variance swap, Black-Scholes greeks, and Hedged Monte Carlo (Potters, Bouchaud &
  Sestovic 2001) -- the discrete-time risk-minimising hedge, fitted by backward
  regression on realised hedged cash flows and applied out of sample, with the underlying
  alone or with the variance swap that completes the lifted model.
- `study_hedging.py`, `test_hedging.py` (14 checks) and `docs/study-hedging.md`: the
  Bertsimas-Kogan-Lo law reproduced in the control; the volatility-risk floor and its
  dependence on H; the floor removed by the variance swap (residual down 2.7-3.9x, the
  99% loss about 5x at rough H); and the approach to the continuous limit slowing from
  dt^0.97 to dt^0.75 as H falls to 0.05.
- Three standing health checks for that machinery (124 in total).

### Fixed
- **The throttle guard could not fire.** It compared a run against the MEDIAN of earlier
  runs, and those included throttled ones: on 21 September a machine at 58% of its own
  best speed read 0.97-1.09x, and its timings went into the trend. The baseline is now
  the fastest of the last 20 runs on that machine, which read 1.58x for the same run.
- **The cross-scheme check measured its own reference.** ETDRK4 against
  exptrap+Richardson sat at 4.1e-6 of a 5e-6 limit because the reference used 200 steps;
  at 800 the difference settles at 6.2e-7 (1600: 7.0e-7). Limit now 2e-6.
- **The QML-bias health check** simulates at 16 substeps, as the test suite has since
  Session I; at 4 the simulator under-resolves the 40-node lift (+1.06 SE against +0.35).

## v1.0.0 (2026-09-19): Session I, the decisions implemented

The configuration for real-data runs chosen in the analytics review of 18 September 2026,
the health-check thresholds set there, the last sanity checks, and the dashboard and
figure polish. Full account: `docs/session-i-report.pdf`.

### Changed
- **The finer lift is the default:** 40 nodes to 1e8/y (was 24 to 1e5).
  `rh.using_lift(24, 1e5)` and `run_real_data.py --lift 24:1e5` run the old one. The
  calibration's size-dependent refinement is gone: the default's kernel error is < 1% over
  [1 day, 2 y] for every H in [0.02, 0.5].
- **Rough jumps go through the Volterra driver only,** sized by integrated variance
  impact: `JumpSizes.state_mean` is the average extra variance over the first trading
  day. Direct mode is retired from the model and kept in `study_jump_modes.py`.
- **The pipeline's history stage** ends with the zero-boundary protocol. Its CIR fit is
  multi-start.
- **The live dashboard and the report figure** use one design system (`ui/theme.py`):
  - validated categorical order;
  - single-hue ramp for magnitude;
  - status colours reserved for state, with icon and label;
  - no dual axis.
- **The spot cf filter's Riccati mesh** is 480 steps (was 240; 1.1e-5 on the new lift,
  now 6.5e-7).

### Added
- `filters/protocol.py`: the Kalman filter, then a boundary diagnostic. Where the boundary
  binds, the cf filter takes the state estimate. A 16-substep particle filter confirms it,
  escalating to 64 substeps when they disagree.
- **Realised-variance observations in the cf filter:**
  - the joint transform of (V, ∫V) as a Riccati running term;
  - frequency panels sized by the transform's stability edge;
  - the posterior of the day's integral by Tweedie's formula;
  - a counted Gaussian fallback.
- `rv_noise` in the Kalman, particle and cf filters: RV's sampling error at the prior mean
  (for estimation) or at the observed RV (for state estimation at the boundary).
- `kf.fit_mle_multistart`.
- **Health checks:**
  - timing is measured first, on a cool machine (0.46 s for the rough objective);
  - a CPU reference workload flags a throttled machine; its timings are
    speed-normalised and kept out of the trend;
  - trends compare runs within one model configuration;
  - the 2025 clock split, asserted exactly (1622.5 / 3412.5 / 3725 h);
  - trend rules on the clock identities.
- `volatility_surface_3.py --demo` (no TWS) and `--frames N --save PNG`; the dashboard's
  L and S keys.
- `models/fbm.py`: one exact fGn/fBm generator. It replaces three copies with a
  variance-½ scaling slip and a wrong circulant middle entry (sent as
  `docs/comparison-fbm-methods.pdf`).
- `test_protocol.py`, `test_fbm.py`, `study_stability_constants.py`, `study_fbm_methods.py`.

### Fixed
- **QE simulator bias on the finer lift.** The factor noise orthogonal to V had a fixed
  size. Draws at V_h = 0 still moved the factors, 5.2% of steps started inadmissible, and
  E[∫V] came out +3.0%. The noise now scales with V_h / m: 0.6% inadmissible,
  E[∫V] +0.9%. The martingale correction and the exact slope of I on V_h follow.
- `step_given` and `step_y` agree bit for bit.
- **Test premises the finer lift made obsolete:**
  - sub-cell resolution;
  - step doubling, now judged on implied vols;
  - the EI reference quadrature, graded to h·1e-12;
  - QML κ on 4-substep data, now 16.

## Earlier sessions
- **H** (2026-09-15): particle and cf filters at the zero boundary, the NYSE trading clock,
  UTC snapshot instants, three decision briefs.
- **G** (2026-09-15): review fixes, the IBKR recorder, learning H in the filter,
  positivity-preserving QE step, stability guards, pytest and CI.
- **F** (2026-09-13): Hawkes arrivals on three hosts, Kalman filtering.
- **E** (2026-09-12/13): rough Heston calibration and the comparison with Heston.
- **D** (2026-09-12): the Markovian lift and its Riccati characteristic function.
- **C** (2026-09-10): calibration (Levenberg–Marquardt, identifiability, prior).
- **B** (2026-09-08): Heston cf and the Fourier pricers.
- **A** (2026-09-07): the surface, finished.
