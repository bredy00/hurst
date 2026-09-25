# The T-EGARCH scan

*Session L, 25 September 2026. `models/egarch.py`, `study_egarch.py`, `test_egarch.py`
(17 checks), `captures/egarch.{json,log}`.*

Akin asked for a T-EGARCH scan to keep in the codebase as a backup for realised-volatility
work. What a backup has to show is how close it gets to the thing it backs up, so the scan
is run against this project's own rough Heston, where the answer is known: ten years of
daily returns simulated from the lifted model (H = 0.10, ξ = 0.4, ρ = −0.7, 78 five-minute
bars a day), with the **true integrated variance of every day** kept as the target.

## What is in the codebase

`models/egarch.py` — three families behind one interface (`Spec`, `fit`, `loglik`,
`variance_path`, `forecast`, `simulate`, `scan`):

- **EGARCH(p, q)** (Nelson 1991), normal or standardised Student-t innovations;
- **Beta-t-EGARCH(1,1)** (Harvey & Chakravarty 2008; Harvey 2013) — the log scale moves with
  the *score* of the Student-t, u_t = (ν+1)r²/(νe^{2λ}+r²) − 1 ∈ [−1, ν], so one extreme
  return moves it by a **bounded** amount where Nelson's |z| term is unbounded;
- **GARCH(p, q)** (Bollerslev 1986) as the benchmark every volatility model is scored against.

Estimation is maximum likelihood through maps that make every admissible parameter reachable
and nothing else: the log-variance autoregression's stationarity through its partial
autocorrelations (Barndorff-Nielsen & Schou 1973; Monahan 1984), GARCH's positivity and
persistence through a softmax, ν > 2.05 through an exponential. Standard errors come from
the numerical Hessian. `scan` fits a list of specs, ranks them by BIC, and — given a split —
scores their one-day forecasts by QLIKE (Patton 2011).

## The control: the machinery is right

On 2,520 days simulated **from** EGARCH(1,1)-t with ν = 7, BIC picks EGARCH(1,1)-t and it is
also the best out of sample (QLIKE 0.0040 against 0.0046 for the normal, 0.0255 for GARCH).
Parameter recovery on 4,000 days is within 3 SE for all three families
(`test_egarch.test_recovery`).

## The result: BIC and forecasting disagree, and BIC is wrong

On rough-Heston returns (kurtosis 21.5):

| model | log-likelihood | k | BIC | ν̂ | QLIKE vs true IV |
|---|---|---|---|---|---|
| **Beta-t-EGARCH(1,1)** | 6314.0 | 5 | **−12590.6** | 2.4 | 0.932 |
| EGARCH(1,1)-t | 6300.3 | 5 | −12563.1 | 2.2 | 1.049 |
| GARCH(1,1)-t | 6295.9 | 4 | −12561.9 | 3.1 | 0.798 |
| GARCH(1,1)-normal | 5899.0 | 3 | −11775.5 | — | **0.735** |
| EGARCH(1,1)-normal | 5819.2 | 4 | −11608.6 | — | 0.820 |
| EWMA (λ = 0.94) | — | — | — | — | 0.876 |
| **HAR-RV (uses realised variance)** | — | — | — | — | **0.611** |

**BIC ranks the t models first and they forecast worst.** Every Student-t fit lands at
ν ≈ 2.2–2.4, just above the bound where the variance does not exist. That is the likelihood
doing its job: rough-Heston returns really do have a heavy conditional tail, and a t with
ν → 2 matches the *shape* better than anything else on offer. But the conditional variance
is the scale times ν/(ν−2), so a tail index chosen to fit the density sets the variance
forecast — and at ν = 2.4 that multiplier is 6. **A density fit is not a variance forecaster,
and BIC cannot see the difference.** `models.egarch.fit` records `nu_near_2` for this reason.

With jumps added (12 a year, N(−2%, 3%)) the effect is violent: Beta-t-EGARCH's QLIKE goes
to 11.8 against GARCH-normal's 1.20, and the t models occupy the bottom five places. The
bounded score response makes Beta-t robust to a jump in *estimation* — the parameters barely
move — and that same boundedness is what stops it from raising its variance forecast when
the jump was real information.

**The honest summary for a backup model:** on this project's own process, the best
returns-only forecast (GARCH(1,1)-normal) carries **1.20× the QLIKE of HAR-RV**, which uses
realised variance directly. The 20% gap is what intraday data is worth here. Pick the family
by out-of-sample QLIKE against a variance proxy, never by in-sample BIC.

## What is kept, and why

Beta-t-EGARCH stays in the codebase despite losing every forecast contest here, because the
property it was chosen for is real and measured: replacing one return with a −1000σ outlier
moves its log-variance by 0.70 (its bound is 1.26 = 2(ν+1)(κ + 2|κ_s|)) where Nelson's moves
by 69. On data whose tails are contaminated rather than informative — a bad print, a stale
quote — that is the model to reach for. `test_egarch.test_robustness` asserts both.

## Next

- Score these against realised variance on the real recording when IBKR data arrives, with
  the same QLIKE-and-HAR comparison; the numbers above say what to expect.
- A Student-t with ν pinned (to a realised-kurtosis estimate) rather than fitted, which
  would separate the tail-shape fit from the variance forecast.
- HAR-RV belongs in the codebase in its own right, not as a benchmark inside a study script.
