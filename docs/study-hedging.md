# Discrete hedging under rough Heston

*Sessions J and K, 21–23 September 2026. `models/hedging.py`, `study_hedging.py`,
`test_hedging.py`, and three standing checks in `healthcheck.py`.*

## The question

A writer sells a one-month at-the-money call at the model's price and hedges it,
rebalancing every Δt. In the continuous-time textbook limit the Black–Scholes delta
replicates the option exactly. What is left when volatility is rough, how much does
rebalancing more often help, what does a hedge derived in discrete time do better, and
what happens when the writer is allowed a second instrument?

## Method

**The discrete-time hedge.** It is *Hedged Monte Carlo* (Potters, Bouchaud & Sestovic,
2001), the physicists' derivation of option hedging (Bouchaud & Potters, 2003). At each
rebalancing date it chooses the value C_k and the hedge ratios φ_k that minimise the
variance of the writer's wealth balance over the next interval,

    min E[ (C_{k+1} − C_k(z) − Σ_i φ_k^i(z) ΔX^i)² ],

solved backwards from the payoff by regression on simulated paths. This is Föllmer–
Schweizer local risk minimisation. With Gaussian returns and Δt → 0 it gives the
Black–Scholes delta; with stochastic volatility it also uses the correlation between
price and volatility.

- **State.** z = (S, σ̂), where σ̂² is the model's expected average variance to maturity,
  affine in the lifted factors and exact.
- **Regression targets.** The realised hedged cash flows from the next date on, so
  regression errors do not compound.
- **Out of sample.** Rules are fitted on 20 000 paths and scored on another 20 000.

**Instruments.** Either the underlying alone, or the underlying and a variance swap,
M_t = ∫_0^t V ds + E[∫_t^T V ds | F_t] — the price process of the claim paying the
option's own realised variance, a martingale and affine in the lifted factors. With both,
the lifted model is complete: one Brownian motion drives the price, one the variance. In
continuous time the hedge then replicates, so what remains at a finite Δt is the
discretisation alone, with no floor to hide it.

**The paths.** Lifted rough Heston (40 nodes, the default when this study ran; 44 since
Session L, which does not change the increment law — `docs/study-lift-44.md` §2), QE step,
512 steps a month (1.4 hours),
v0 = θ = 0.04, κ = 2, ξ = 0.3, ρ = −0.7, and H ∈ {0.05, 0.12, 0.25, 0.45}. A control run
uses ξ = 0.01, nearly deterministic variance.

## Results

Residual = the standard deviation of the terminal hedging error, as a fraction of the
option price, at 1.4-hour rebalancing.

| | cf price | HMC price | BS fixed | BS re-marked | HMC, S only | **HMC + var swap** | 99% loss, BS / HMC / +swap |
|---|---|---|---|---|---|---|---|
| H = 0.05 | 0.02116 | 0.02130 ± 0.00004 | 0.327 | 0.338 | 0.295 | **0.076** | 1.03 / 0.85 / 0.21 |
| H = 0.12 | 0.02159 | 0.02157 ± 0.00004 | 0.283 | 0.290 | 0.254 | **0.058** | 0.87 / 0.71 / 0.15 |
| H = 0.25 | 0.02224 | 0.02220 ± 0.00003 | 0.210 | 0.213 | 0.189 | **0.046** | 0.62 / 0.50 / 0.11 |
| H = 0.45 | 0.02277 | 0.02278 ± 0.00002 | 0.125 | 0.125 | 0.111 | **0.041** | 0.35 / 0.28 / 0.09 |
| control, ξ = 0.01 | 0.02302 | 0.02302 ± 0.00001 | 0.040 | 0.040 | 0.041 | 0.041 | 0.11 / 0.11 / 0.11 |

1. **The discrete-physics law holds where it should.** In the control, the Black–Scholes
   delta's residual follows the Bertsimas–Kogan–Lo leading order,
   Var = (Δt/2)·E∫Γ²S⁴σ⁴dt: 0.0403 against 0.0385 at 1.4 hours, 0.1086 against 0.1088 at
   11.4 hours, 0.1511 against 0.1539 daily. The risk-minimising hedge ratio there *is* the
   Black–Scholes delta (0.508 against 0.512), and the error scales as Δt^0.485.
2. **With the underlying alone, rough volatility leaves a floor** that rebalancing cannot
   remove, and it rises as H falls: 11% of the price at H = 0.45, 30% at H = 0.05.
3. **The discrete-time hedge takes 10–15% off Black–Scholes delta** and 17–20% off its
   99% loss. Re-marking the volatility inside a Black–Scholes delta does not help
   (0.338 against 0.327 at H = 0.05): the gain comes from the price–volatility
   correlation, not from a better volatility level.
4. **The floor is an instrument problem, not a law of the model.** Adding the variance
   swap drops the fitted floor from 0.10–0.28 to 0.000–0.013, which is what completeness
   predicts, and cuts the residual by 2.7× (H = 0.45) to 3.9× (H = 0.05). The 99% loss
   falls by about 5× at rough H. The writer still needs 1.4-hour rebalancing: at daily
   rebalancing the same hedge leaves 0.215 (H = 0.05) against 0.151 in the control.
5. **Roughness slows the approach to the continuous limit, and the complete hedge shows
   it cleanly.** Fitting residual² = floor² + a·Δt^γ over 1.4 hours to 3.8 days:

   | | BS fixed | BS re-marked | HMC, S only | HMC + var swap |
   |---|---|---|---|---|
   | H = 0.05 | 0.63 | 0.63 | 0.73 | **0.75 ± 0.02** |
   | H = 0.12 | 0.82 | 0.80 | 0.89 | **0.86 ± 0.01** |
   | H = 0.25 | 0.91 | 0.91 | 0.96 | **0.93 ± 0.01** |
   | H = 0.45 | 0.95 | 0.94 | 0.96 | **0.95 ± 0.00** |
   | control | 0.97 | 0.97 | 0.97 | 0.97 |

   With no floor in the way, the exponent still falls from 0.97 to 0.75 as H goes from
   ½ to 0.05, so the error decays as Δt^0.375 instead of the textbook Δt^0.5. Doubling
   the rebalancing frequency buys 19% at H = 0.05 where it buys 29% in the control.
6. **The simplest mechanism for that is wrong.** A rough-vanna term would make the
   correction scale as Δt^{2H} (0.1 at H = 0.05). Fitting
   residual² = floor² + a·Δt + b·Δt^{2H} with H known fits *worse* than one free exponent
   at every H (RSS 9.4e-5 against 3.5e-5 at H = 0.05), so the two-term form is rejected.
   The effective exponent is a description, not yet an explanation.
7. **Hedged Monte Carlo is a precise probe of simulator bias.** Its price carries 5–20×
   less Monte Carlo error than a plain average. At H = 0.05 it sits 3.5 SE above the cf
   price (+0.66%, about 0.12 vol points): the QE simulator's known residual skew bias at
   that roughness. At H ≥ 0.12 the two agree within 1.3 SE.

## Standing checks

`healthcheck.py` carries three of these as ongoing checks: the Hedged Monte Carlo price
against the cf price (in SE), the variance swap's martingale property, and the ratio of
the two-instrument residual to the one-instrument residual (< 0.6).

## Next steps

- **Where the roughness correction comes from.** The exponent is measured but not
  derived. The near-expiry region is the suspect: an option's sensitivity to the forward
  variance scales as τ^{H−1/2}, so the last few intervals dominate. Rebalancing on a
  non-uniform grid, finer near expiry, would test it directly.
- **Transaction costs**, which reverse the incentive to rebalance: the optimum is then
  interior, and the variance swap's own cost enters.
- **Deep hedging** (Buehler et al. 2019) on the same paths, with Hedged Monte Carlo as
  the baseline to beat rather than a neural net evaluated alone.
