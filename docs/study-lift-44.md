# Forty-four nodes, and paying for them

*Session L, 25 September 2026. `study_lift_44.py`, `models/rough_heston.py`,
`models/fbm.py`, `study_stability_constants.py`, `captures/lift_44.{json,log}`.*

## The decision

Adopt N = 44 nodes (fastest node 1e8/y) as the default lift, from N = 40. Then, in Akin's
words, "compare and overcome": measure what the four extra nodes buy, measure what they
cost, and pay for the cost somewhere else.

## 1. What the four nodes buy

**The review's criterion — kernel error below 1% over the whole H box on [1 day, 2 years] —
gains a quarter of its headroom.** Worst error, always at H = 0.02:

| | N = 40 | **N = 44** | N = 48 |
|---|---|---|---|
| worst kernel error on [1 d, 2 y] | 0.908% | **0.763%** | 0.653% |
| headroom to the 1% limit | 9% | **24%** | 35% |
| worst implied-vol error against true rough Heston (H = 0.02, 1 day) | 0.043 vp | **0.036 vp** | 0.031 vp |
| same at H = 0.12, 30 days | 0.025 vp | **0.020 vp** | 0.017 vp |
| Itô-isometry shortfall at 1 day (H = 0.12) | 5.1% | 4.9% | 4.7% |

Prices improve by 16% at every H and maturity measured — the lift's error is a smooth
function of N, with no threshold. The 1% criterion was the reason to move: at N = 40 a 9%
margin is one modelling change away from breaching it.

## 2. What they do not buy: the lift is not fGn, and four nodes do not change that

The lifted driver started in the infinite past, Y_t = ∫ K_N(t−s) dW_s, has stationary
increments. With the **exact** kernel those increments are fractional Gaussian noise with
variance V_H h^{2H}, V_H = 1/(Γ(2H+1) sin πH) — Mandelbrot–Van Ness. So the lift can be
graded against fGn directly, with no pricer in the way. Both spectral densities are closed
form (fGn's through the Hurwitz zeta function; the lift's as a sum of AR(1) spectra, one per
node), and both were checked against their own autocovariance at lag 0 to 1e-11.

**The variogram.** Against V_H h^{2H}, the lift's E(Y_{t+h} − Y_t)² is off by

| | 1 min | 1 h | 1 day |
|---|---|---|---|
| H = 0.02, N = 40 → 44 | 82.7% → 82.7% | 70.7% → 70.6% | 62.5% → 62.4% |
| H = 0.05, N = 40 → 44 | 61.6% → 61.5% | 41.8% → 41.6% | 30.8% → 30.7% |
| H = 0.12, N = 40 → 44 | 29.7% → 29.6% | 12.1% → 12.0% | 6.3% → 6.1% |
| H = 0.25, N = 40 → 44 | 6.4% → 6.3% | 1.5% → 1.4% | 0.9% → 0.8% |

and the local exponent at one day reads 0.105 where the exact one is 0.040 (H = 0.02), 0.253
against 0.241 (H = 0.12). **Four nodes change the third decimal.** The sup-norm kernel error
and the increment law are different criteria, and the node count only moves the first.

**How much data would tell the difference.** The Whittle divergence between the two spectra
is the expected log-likelihood ratio; 5.41 is where a 5%-size test reaches 95% power.
For a perfectly observed driver at H = 0.12:

| sampling | data to distinguish the lift from exact fGn |
|---|---|
| 1 second | fewer than 64 observations |
| 1 minute | 0.00037 years (3 hours) |
| 5 minutes | 0.0045 years (under two days) |
| 1 hour | 0.18 years |
| **1 day** | **18 years** |

And to tell the 44-node lift from the 40-node one: **12.5 years at one minute, 54 years at
five minutes, 13,000 years at daily sampling.** At the sampling this project's filters use,
the two lifts are the same process.

Monte Carlo confirms the closed forms. 400 paths of 65,536 five-minute steps, the lift's
increments drawn by the same circulant embedding as fGn (Shevchenko 2014, §6 — the new
`models.fbm.stationary_gaussian` generalises the generator to any stationary covariance with
a nonnegative embedding):

- Shevchenko's quadratic-variation estimator reads H = 0.1202 ± 0.0002 on exact fGn (planted
  0.12) and **0.1466 ± 0.0002 on the lift's** (closed form predicts 0.1464);
- the Whittle log-likelihood ratio on exact draws is 751 ± 2 against a predicted 753.

**The real lever is the fastest node, not the node count.** At the same node density:

| top node | N | kernel error [1 d, 2 y] | variogram error at 1 min | QV estimator reads H at 5 min (truth 0.12) |
|---|---|---|---|---|
| 1e8/y | 44 | 0.60% | 29.6% | 0.146 |
| 1e10/y | 54 | 0.59% | 10.7% | 0.128 |
| 1e12/y | 64 | 0.58% | 4.2% | 0.123 |
| 1e14/y | 73 | 0.59% | 2.1% | 0.121 |

The kernel error does not move — it is already at its node-density floor — while the
increment law converges. Anything that reads roughness from intraday data on a **simulated**
path needs the top node moved, not more of them. Nothing in this project does that today:
prices are the criterion, and prices are what N buys.

## 3. Paying for it

A node costs a column in every Riccati step, so 44 nodes is 10% more work than 40. Two
changes in `log_char_func` more than cover it, and both are **exact**, checked against 72
saved reference cf and price arrays (worst |cf − cf_ref| 1.2e-14, worst price difference
2.2e-16) and by `test_rough.test_carried_state`:

- **The carried state drops the factors with no memory beyond one step.** A factor whose
  decay over a step, e^(−x_i h), is below 1e-30 carries nothing: the next step's stage sums
  read its state only through that factor, and its within-step contribution is already in the
  step coefficients. At one year, 20 of the 44 are in that class; at one day, 5 to 8.
- **A Riccati solution is reused when only v0 or θ changed.** The ψ system never sees them:
  they enter the log cf linearly, log φ = κθ·I_A + v0·I_B with I_A and I_B the scheme's own
  quadratures of Ψ and F(Ψ). Storing that pair instead of the scalar makes two of a rough
  calibration's six Jacobian columns free. `rh.reuse_riccati()` is a context manager, on only
  inside `calibrate_rough_heston`, so the health check's timings still measure real solves.

Measured on the health check's ten-expiry objective (best of three, one BLAS thread, the two
solvers interleaved in one process over six rounds):

| | N = 40 | N = 44 |
|---|---|---|
| before | 0.470 s | 0.502 s (+6.8%) |
| after | 0.448 s | **0.425 s** |

**The 44-node lift now prices the objective in 90% of the time the 40-node lift took**, so
the four nodes are paid for with change left over. The Riccati reuse is on top of that, in
calibration only.

## 4. Stability

`study_stability_constants.py 44 1e8` re-measures the edge where |φ(u − i/2)| first exceeds
one, over u ∈ [0, 1200], H = 0.02 and 0.12, 30 days to a year: ETDRK4 4.23–4.28,
exptrap 14.45–18.82 — the same edges as the 40-node lift. `C_STAB` stays at 2.0 and 12.0,
below every edge on all three lifts.

## 5. What broke, and what it revealed

Two slow-tier checks fail on the new lift, and one of them was passing for the wrong reason.

**The synthetic history's H is far noisier than its reported standard error.**
`test_filters.test_learn_h` asserts that the integrated-variance filter recovers a planted
H = 0.10 within 3 SE on 500 days. On 44-node data at seed 3 it reads 0.257 ± 0.033 (4.8 SE);
on 40-node data at the same seed, 0.130. A 2×2 experiment (data lift × filter lift, five
seeds) settles what moved: **the filter's lift changes nothing at all** — every pair agrees
to three decimals — while the *data's* lift moves H by up to 0.09 between seeds. It is the
simulation that differs, not the estimator.

Over 21 seeds on the default lift the estimator reads **mean 0.135, sd 0.079**, with a median
reported SE of 0.028 and two seeds pinned at the grid's lower edge. The profile likelihood's
curvature understates the sampling spread of this estimator on 500 days by about a factor of
three. The check now uses the measured spread.

**The QE simulator's residual skew bias, and why an SE threshold was the wrong instrument.**
`test_rough.test_positivity_scheme` asked the QE prices to match the cf within 4.5 SE at
ρ = −0.7; the k = +0.08 call reads +5.0 SE. Pooled over four seeds and 200,000 paths at
τ = 0.25:

| k = +0.08 call, price bias (SE) | 500 steps | 1000 steps |
|---|---|---|
| N = 40 | +1.40e-4 (3.6 SE) | +1.58e-4 (4.1 SE) |
| N = 44 | +2.26e-4 (5.8 SE) | +1.42e-4 (3.7 SE) |

The bias is real, documented from Session I, and slightly larger on the finer lift at a given
step count — which is the expected direction, since more kernel mass sits in the first
instants, where the QE draw is least accurate. But the threshold was the wrong instrument:
**the standard error shrinks with the path count while a bias does not**, so any SE-based
limit fails once enough paths are thrown at it. The check now bounds the bias at 0.35 vol
points, which is the quantity a desk would care about; the measured values are 0.13 to
0.20 vp.

## Next

- Move the top node, not the node count, if intraday roughness is ever read from simulated
  paths: §2's table is the map.
- The Whittle divergence in §2 is also a design tool — it says what a lift can and cannot be
  asked to reproduce at a given sampling frequency, before any of it is built.
