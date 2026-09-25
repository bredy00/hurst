# The trial: calibrating H with a dual Kalman filter

*Session L, 25 September 2026. `portfolio/` (5 modules), `study_trial_bl_hurst.py`,
`test_portfolio.py` (42 checks), `captures/trial_bl_hurst.{json,log,png}`,
`captures/trial/demo_market_*.sqlite`. Revert: `git rm -r portfolio study_trial_bl_hurst.py
test_portfolio.py docs/trial-bl-hurst.md` — nothing else imports it.*

## The idea, as Akin put it

> Implement Hurst parameter optimisation based on a default portfolio constructed using only
> the Black–Litterman model at all times, then run a Fama–French 4 on that boy, and with the
> results apply a dual Kalman filter to our Hurst parameter to calibrate it, adjust the
> jaggedness to match our desired market exposure to cut down from costs. The same framework
> could also be applied in the manner of the confidence matrix, and maybe even the market
> equilibrium matrix. I want to see if this idea is practical or not.

## The setup

**A demo market with a known answer** (`portfolio/demo_market.py`): twelve years, 40 assets,
MKT/SMB/HML/MOM, each factor's volatility rough — log-vol a fractional Ornstein–Uhlenbeck
process driven by **exact** fractional Gaussian noise (`models/fbm.py`) — and a daily
realised-variance proxy with 35% noise. **No asset carries alpha**, so any alpha a backtest
reports is noise or cost (asserted: every intercept within 4 SE of zero). Everything lands in
SQLite, and the trial runs on what it reads back.

**The portfolio** (`portfolio/backtest.py`): Black–Litterman at every weekly rebalance.
Π = δ_eq Σ̄ w_mkt from three years of covariance and live cap weights; two views (momentum
and value, top third minus bottom third) refreshed every 21 days; Ω = diag(P τΣ̄ P′)/c;
w = (δ(Σ_opt + M))⁻¹ μ_BL, uncapitalised remainder in cash, 10 bp a unit of turnover.

**Where H enters.** Σ_opt forecasts each factor's volatility with the rough-volatility
predictor of Gatheral, Jaisson & Rosenbaum (2018) at index H. Low H weights a long stretch of
the past and gives a smooth forecast; H → ½ collapses onto yesterday's noisy realised
variance and gives a jagged one. **H is exactly the jaggedness knob Akin described**, and the
test suite asserts it works: at H = 0.02 the exposure path's daily change has sd 0.16 and
turnover 28; at H = 0.45, sd 0.27 and turnover 48.

**The filter** (`portfolio/dual_kalman.py`), dual estimation in the sense of Wan & Nelson
(1997). A *state* filter carries the portfolio's exposures (α, b_MKT, b_SMB, b_HML, b_MOM),
measured every period by the FF4 regression **through falsify** with its Newey–West variances
as the noise. A *parameter* filter carries θ = (H, log c, log δ [, log δ_eq]) and is measured
by a pseudo-observation of what is *wanted*: z = (β*, 0) against h(θ) = (b_MKT, cost),
linearised with a Jacobian measured by **counterfactual portfolios** — the database holds
every return, so the portfolio built with θ + dθ is re-run over the same days from the same
holdings. Each filter uses the other's latest estimate.

**falsify** (`portfolio/factor_fit.py`) supplies the FF4 regression with Newey–West errors,
the deflated Sharpe ratio, and CSCV/PBO. It is found at `../falsify` or `FALSIFY_PATH`;
without it the package falls back to its own copies, and `test_portfolio` asserts the two
agree to rounding (they do, to 0.0e+00).

**Pre-registered**, written before the test years were scored: target FF4 market beta 0.8;
objective J = mean((b − 0.8)²)/0.05² + cost²/0.25², i.e. 0.05 of beta error weighs like
0.25%/yr of cost; train on years 5–8, test on 9–12; two markets, stationary (H = 0.10
throughout) and shift (0.08 → 0.25 at the train/test boundary, so the whole test period runs
at the new roughness). *The idea is
practical if the dual filter has the lowest test objective of the alternatives in both
markets, with a stable filter.*

## The result: no, not as stated

Test years, both markets (the full table is in `captures/trial_bl_hurst.log`):

| | stationary J | shift J | mean beta (stationary) | cost %/yr |
|---|---|---|---|---|
| S2 grid-best (54 fixed configurations) | **33.4** | **31.3** | 0.751 | 0.87 |
| S3 dual Kalman | 49.9 | 55.7 | 0.876 | 0.97 |
| S3e + equilibrium (δ_eq too) | 52.0 | 50.8 | 0.899 | 1.05 |
| S4 no-trade band at S3's cost | 53.5 | 43.8 | 0.851 | 1.23 |
| S1 statistical H (variogram) | 57.7 | 46.6 | 0.850 | 1.33 |
| S1r rolling H | 58.9 | 59.6 | 0.851 | 1.34 |
| S0 default (H = 0.10, c = 1, δ = 2.5) | 140.8 | 109.0 | 1.112 | 1.81 |

**The dual filter loses to a grid search in both markets.** By the pre-registered reading,
the idea as stated is not practical. Four findings say why, and three of them are worth more
than the verdict.

### 1. H has no interior optimum here; the objective is monotone in it

The grid picks H = 0.02 — the lowest value offered — in every market and every scenario
including a fresh seed, and the objective falls monotonically toward it:

| H | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 | 0.49 |
|---|---|---|---|---|---|---|
| best training J | 34.1 | 36.1 | 40.3 | 52.8 | 73.8 | 154.6 |
| cost %/yr | 0.78 | 0.84 | 0.96 | 1.25 | 1.61 | 2.53 |

The dual filter *agrees*: it drives H to its lower bound and holds it there in 89–100% of
periods. **Both methods find the same answer, and the answer is "as smooth as you will let
me".** A knob whose optimum is always at the boundary does not need a filter — it needs a
bound, chosen for a reason the objective does not contain.

**And the cost-optimal H is not the market's H.** The true H is 0.10 (or 0.08 → 0.25); the
best H for this objective is 0.02 in every case. Calibrating H to a cost-and-exposure
objective produces a **cost-control setting, not an estimate of roughness** — the two happen
to share a symbol. That is worth saying plainly: the statistically honest H (S1's variogram
estimate, 0.078 and 0.061, both close to the truth) *loses* to the deliberately wrong one.

### 2. The target was not achievable — by design, not by defect

Per-period FF4 beta has sd 0.27 across the test years. Decomposing it with the Newey–West
standard errors: the measurement noise is 0.09, so the **true exposure genuinely swings with
sd 0.24**. That is the Black–Litterman portfolio being volatility-managed: when forecast
volatility rises, (δΣ_opt)⁻¹μ shrinks the position. It is the feature, not a flaw.

The pre-registered tolerance was 0.05. **No setting of (H, c, δ) can hold a volatility-managed
portfolio's beta to ±0.05, because the strategy's whole mechanism is to move it.** The
objective is therefore dominated by a variance term nothing in θ controls, and the ranking is
decided by which strategy accidentally has the smoothest exposure path — which is the low-H
one, again. The instrument for a *constant* beta is a market overlay, not the risk model's
roughness.

Where the filter does win is the part the target actually governs: its **mean** beta is
0.876/0.906 against the grid's 0.751/0.715, and the post-hoc variant below lands at
0.795/0.800 against a target of 0.800. **The filter controls the level; nothing controls the
swing.**

### 3. The filter had two real defects, and fixing them closes most of the gap

Found by reading the filter's own diagnostics, not by tuning:

- **Precision weighting is biased on a volatility-managed portfolio.** Measuring each period
  with its own Newey–West variance underweights the periods with the most exposure, because
  those are the calm periods where factor variance is low and beta is least precisely
  measured: corr(b, se) = +0.55, and the precision-weighted mean beta is 0.73 where the
  simple mean is 0.88. Constant (running-median) measurement variances fix it.
- **The drift was four times too fast**, so the parameter filter chased period noise: the
  beta channel's normalised innovation has sd 1.5 where a consistent filter gives 1.

With both fixed (**S3c**, marked post-hoc in the study because it was chosen after seeing the
results — so it is itself a search, and is re-run on a fresh market for confirmation):

| | stationary | shift | confirmation (new seed) |
|---|---|---|---|
| S2 grid-best | 33.4 | 31.3 | 26.8 |
| S3 dual Kalman (pre-registered) | 49.9 | 55.7 | 33.6 |
| **S3c constant R, slow drift** | **37.4** | **36.3** | **26.0** |

It closes about three quarters of the gap and beats the grid on the fresh market. **The
machinery works; the thing it was pointed at does not have an interior answer.**

### 4. falsify says the grid's own win is mostly search

`falsify` deflated the grid's best training Sharpe by the 72 configurations in the ledger:
**DSR 0.387**, and PBO (CSCV, 16 blocks, ArgMax) **0.45** with performance degradation +0.17.
A PBO near 0.5 is a coin flip. The grid's *objective*-selected configuration does land at the
top of the out-of-sample distribution — rank 0 of 54 in both markets — but that is the
objective being monotone in H again, not a discovery. No strategy here has an out-of-sample
alpha distinguishable from zero (|t| ≤ 1.03), which is correct: the demo market has none.

## The verdict

**Practical as machinery, not as stated.** The dual Kalman filter runs, its Jacobians have
the signs the model implies, its parameters stay in bounds, and once two measurement defects
are fixed it matches or beats a 54-point grid at a fraction of the evaluations (48 periods ×
4 counterfactuals against 54 full backtests). Use it to calibrate **the mean level of an
exposure**, where it works and a grid is expensive.

Do not use it to "optimise H for costs". In this setup — and the mechanism is general, not an
artefact of the demo market — a smoother risk model always trades less, so the cost-optimal H
is whatever the bound allows, and calibrating it produces a number that is not the market's
roughness. Estimate H statistically (S1's variogram, which recovers 0.078 and 0.061 against
truths of 0.10 and 0.08), and control costs with the instrument built for it: S4's no-trade
band delivered the filter's cost with a one-parameter bisection.

The confidence c and the equilibrium δ_eq behaved as the theory says (c fell to weight the
views down; adding δ_eq changed the split between δ and δ_eq, not the portfolio — Π and δ are
not separately identified from a portfolio's returns, which the filter discovered by moving
both in opposite directions to no effect). Nothing here recommends letting a filter move the
equilibrium: it is, as Akin said, pre-determined by the market.

## What to do next, if this is taken further

- **Re-run with an achievable target.** A market overlay in the instrument set, or a target
  on the *mean* exposure over a year rather than each period, would test the filter on
  something it can hit.
- **Cost as a constraint, not a penalty.** The J used here trades beta against cost at a rate
  chosen by hand; a turnover budget makes the question well posed without an exchange rate.
- **The identifiability of (Π, δ_eq) deserves its own note** — it is a clean, small result and
  the filter found it unprompted.
