# Discrete hedging under rough Heston

*Session J, 21 September 2026. `models/hedging.py`, `study_hedging.py`, `test_hedging.py`.*

## The question

A writer sells a one-month at-the-money call at the model's price and hedges it with
the underlying alone, rebalancing every Δt. In the continuous-time textbook limit the
Black–Scholes delta replicates the option exactly. What is left when volatility is rough,
how does rebalancing more often help, and what does a hedge derived in discrete time do
better?

## Method

**The discrete-time hedge.** It is *Hedged Monte Carlo* (Potters, Bouchaud & Sestovic,
2001), the physicists' derivation of option hedging (Bouchaud & Potters, 2003). At each
rebalancing date it chooses the value C_k and the hedge ratio φ_k that minimise the
variance of the writer's wealth balance over the next interval:

    min E[ (C_{k+1} − C_k(z) − φ_k(z) ΔS)² ],

solved backwards from the payoff by regression on simulated paths. This is Föllmer–
Schweizer local risk minimisation. With Gaussian returns and Δt → 0 it gives the
Black–Scholes delta. With stochastic volatility it also uses the correlation between
price and volatility.

- **State.** z = (S, σ̂), where σ̂² is the model's expected average variance to
  maturity. It is affine in the lifted factors and exact.
- **Regression targets.** They are the realised hedged cash flows from the next date on,
  so regression errors do not compound.
- **Out of sample.** The rules are fitted on 20 000 paths and scored on another 20 000.

**The paths.** Lifted rough Heston (40 nodes), QE step, 512 steps a month (1.4 hours),
v0 = θ = 0.04, κ = 2, ξ = 0.3, ρ = −0.7, and H ∈ {0.05, 0.12, 0.25, 0.45}. A control run
uses ξ = 0.01, nearly deterministic variance.

**The rules compared.**
- Black–Scholes delta at the at-the-money implied vol fixed at inception;
- Black–Scholes delta at σ̂, re-marked at every date;
- Hedged Monte Carlo.

## Results

Residual = the standard deviation of the terminal hedging error, as a fraction of the
option price.

| | cf price | HMC price | BS fixed, 1.4 h | BS re-marked | HMC | daily: BS / HMC | 99% loss, BS / HMC |
|---|---|---|---|---|---|---|---|
| H = 0.05 | 0.02116 | 0.02130 ± 0.00004 | 0.327 | 0.338 | **0.295** | 0.414 / 0.351 | 1.03 / 0.85 |
| H = 0.12 | 0.02159 | 0.02157 ± 0.00004 | 0.283 | 0.290 | **0.254** | 0.348 / 0.302 | 0.87 / 0.71 |
| H = 0.25 | 0.02224 | 0.02220 ± 0.00003 | 0.210 | 0.213 | **0.189** | 0.265 / 0.240 | 0.62 / 0.50 |
| H = 0.45 | 0.02277 | 0.02278 ± 0.00002 | 0.125 | 0.125 | **0.111** | 0.194 / 0.184 | 0.35 / 0.28 |
| control, ξ = 0.01 | 0.02302 | 0.02302 ± 0.00001 | 0.040 | 0.040 | 0.041 | 0.151 / 0.151 | 0.11 / 0.11 |

1. **The discrete-physics law holds where it should.** In the control, the Black–Scholes
   delta's residual follows the Bertsimas–Kogan–Lo leading order,
   Var = (Δt/2)·E∫Γ²S⁴σ⁴dt: 0.0403 against 0.0385 at 1.4 hours, 0.1086 against 0.1088 at
   11.4 hours, and 0.1511 against 0.1539 daily. The Hedged Monte Carlo hedge ratio is the
   Black–Scholes delta (0.508 against 0.512), and the error scales as Δt^0.485.
2. **Rough volatility leaves a floor, and it rises as H falls.** At 1.4-hour rebalancing,
   with the same vol-of-vol, the best residual goes from 11% of the price at H = 0.45 to
   30% at H = 0.05. Rebalancing cannot hedge volatility risk with the underlying.
3. **The discrete-time hedge takes 10–12% off Black–Scholes delta at fine rebalancing,**
   up to 15% daily at rough H, and cuts the 99% loss by 17–20% there. Re-marking the vol
   inside a Black–Scholes delta does not help (0.338 against 0.327 at H = 0.05). The gain
   comes from using the price–volatility correlation, not from a better volatility level.
4. **Roughness slows the approach to the continuous limit.** Fitting
   residual² = floor² + a·Δt^γ over 1.4 hours to 3.8 days gives the exponents below. They
   are the same for every rule, including Black–Scholes at fixed vol, whose hedge ratio
   ignores volatility.

   | | BS fixed | BS re-marked | HMC |
   |---|---|---|---|
   | H = 0.05 | 0.63 | 0.63 | 0.73 |
   | H = 0.12 | 0.82 | 0.80 | 0.89 |
   | H = 0.25 | 0.91 | 0.91 | 0.96 |
   | H = 0.45 | 0.95 | 0.94 | 0.96 |
   | control | 0.97 | 0.97 | 0.97 |

   The proposed mechanism: a short-dated option's true delta moves with spot variance,
   which is rough. A hedge held fixed for Δt then accrues a tracking error of order
   Δt^{2H} (vanna times rough volatility increments), which competes with the usual gamma
   term of order Δt. Two caveats:
   - At H = 0.05 a Δt^{0.1} term is nearly flat over two decades of Δt, so part of what
     the fit calls "floor" is this rough term. The floor is overstated at small H.
   - The exponents are effective values over the range fitted, not asymptotic ones.
5. **A side result: Hedged Monte Carlo is a precise probe of simulator bias.** Its price
   has 5–20× less Monte Carlo error than a plain average. At H = 0.05 it sits 3.5 SE above
   the cf price (+0.66%, about 0.12 vol points). That is the QE simulator's known residual
   skew bias, visible at this roughness. At H ≥ 0.12 the two agree within 1.3 SE.

## Next steps

- **Test the mechanism.** Fit residual² = floor² + a·Δt + b·Δt^{2H} with H known, and
  rebalance down to minutes on a finer grid. A clean separation of b from the floor would
  confirm the vanna term.
- **Hedge with a second instrument** (variance swap or a second option). The lifted model
  is complete with one, so the floor should vanish. Its hedging error's Δt exponent would
  then isolate the roughness effect.
- **Transaction costs, and a learned rule.** Deep hedging (Buehler et al. 2019) on the
  same paths, with Hedged Monte Carlo as the baseline.
