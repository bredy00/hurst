
**Engineering**

| check | measured | threshold | verdict |
|---|---|---|---|
| CPU reference workload<br><sub>best of 3; 1.00x this machine's median over 2 earlier runs</sub> | `0.4027 s` | `inf s` | PASS |

**Markovian lift**

| check | measured | threshold | verdict |
|---|---|---|---|
| rough objective evaluation, 10 expiries x 13 strikes<br><sub>best of 3 (all: 0.80, 0.78, 0.79); trend target: rolling mean <= 0.75 s</sub> | `0.7814 s` | `3 s` | PASS |

**Foundations**

| check | measured | threshold | verdict |
|---|---|---|---|
| norm_pdf vs scipy | `0` | `1.00e-15` | PASS |
| norm_cdf vs scipy | `2.22e-16` | `1.00e-15` | PASS |
| put-call parity C-P = F-K | `2.84e-14` | `1.00e-09` | PASS |
| vega vs central difference | `4.23e-09` | `1.00e-04` | PASS |
| implied-vol round trip | `8.05e-16` | `1.00e-06` | PASS |
| fd_first exact on a quadratic | `7.46e-14` | `1.00e-12` | PASS |
| fd_second on a quadratic vs its rounding floor<br><sub>2.08e-11 measured, floor 5.06e-11</sub> | `0.41 x floor` | `1 x floor` | PASS |
| shipped 2nd derivative: order on a $1/$5 kink grid<br><sub>Fornberg 5-point; mapped-coordinate idea measured at 0.01</sub> | `3.077` | `2` | PASS |
| 3-point stencil order on the same grid (kept for contrast)<br><sub>the documented first-order loss</sub> | `1.086` | `1.5` | PASS |

**Heston cf**

| check | measured | threshold | verdict |
|---|---|---|---|
| phi(0) = 1 | `0` | `1.00e-12` | PASS |
| phi(-i) = 1 (martingale) | `0` | `1.00e-10` | PASS |
| phi(-u) = conj(phi(u)) | `0` | `1.00e-12` | PASS |
| \|phi(u)\| <= 1 | `0.9999` | `1` | PASS |
| branch continuity, shipped (max/median step)<br><sub>Albrecher g=(a-d)/(a+d)</sub> | `5.242` | `50` | PASS |
| branch discontinuity, reciprocal form<br><sub>the trap, kept to prove it is real</sub> | `2825` | `500` | PASS |
| trap / shipped separation | `538.9 x` | `100 x` | PASS |
| xi -> 0 degenerates to Black-Scholes<br><sub>at xi=1e-8, cancellation-free form; was 5.9e-09 at xi=1e-4</sub> | `2.22e-16` | `1.00e-14` | PASS |
| naive-form cancellation at xi=1e-6 (kept to prove it)<br><sub>shipped form is 5.7e-13 at the same xi</sub> | `3.44e-05` | `1.00e-06` | PASS |

**Pricers**

| check | measured | threshold | verdict |
|---|---|---|---|
| Lewis vs exact Black-76 | `3.55e-15` | `1.00e-10` | PASS |
| Carr-Madan vs exact Black-76 | `1.94e-16` | `1.00e-10` | PASS |
| Lewis vs Carr-Madan on Heston<br><sub>independent routes, 5 maturities</sub> | `4.99e-15` | `1.00e-08` | PASS |
| Carr-Madan insensitive to damping alpha<br><sub>alpha 0.75 to 3.0</sub> | `6.11e-16` | `1.00e-08` | PASS |
| Fourier vs Monte Carlo (sigmas)<br><sub>residual is Euler O(dt) bias</sub> | `0.3618 sd` | `5 sd` | PASS |
| simulated E[S/F] = 1 | `1.94e-06` | `6.12e-04` | PASS |
| risk-neutral density mass | `1` | `1` | PASS |
| risk-neutral density mean = forward | `1` | `1` | PASS |
| density non-negativity (min/peak)<br><sub>tail dither scales as 1/dK^2</sub> | `-8.87e-10` | `-1.00e-07` | PASS |
| sampling density: min >= eps and mass = 1<br><sub>floored 622 pts, removed -5.1e-10 of mass</sub> | `0` | `1.00e-12` | PASS |

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
| plant-and-recover, worst relative error<br><sub>7 iters, 0.6s</sub> | `3.13e-11 rel` | `0.01 rel` | PASS |
| fitted RMSE on clean data | `4.06e-09 vol pts` | `0.001 vol pts` | PASS |
| Jacobian condition number at the solution<br><sub>high = parameters trade off; see identifiability</sub> | `30.05` | `1.00e+08` | PASS |
| spread of v0 under 0.5vp noise<br><sub>stable</sub> | `3.775 %` | `10 %` | PASS |
| spread of kappa under 0.5vp noise<br><sub>unidentified</sub> | `34.73 %` | `50 %` | PASS |
| spread of theta under 0.5vp noise<br><sub>stable</sub> | `2.803 %` | `50 %` | PASS |
| spread of xi under 0.5vp noise<br><sub>loose</sub> | `19.13 %` | `50 %` | PASS |
| spread of rho under 0.5vp noise<br><sub>stable</sub> | `3.73 %` | `10 %` | PASS |
| spread of kappa with a Tikhonov prior (weight 50)<br><sub>prior from a history or a variance swap</sub> | `0.07281 %` | `10 %` | PASS |

**Markovian lift**

| check | measured | threshold | verdict |
|---|---|---|---|
| K(t) = int e^-xt mu(dx) representation<br><sub>mu(dx) = x^-a / (Gamma(a)Gamma(1-a)) dx, a = H+1/2</sub> | `6.67e-10 rel` | `1.00e-08 rel` | PASS |
| kernel error of the default lift, t in [1 day, 2 y]<br><sub>max relative; H = 0.12; N = 40, top node 1e+08/y</sub> | `0.007027 rel` | `0.01 rel` | PASS |
| kernel error on [1 hour, 2 y]<br><sub>a one-day option lives here</sub> | `0.007027 rel` | `0.01 rel` | PASS |
| kernel error on [5 minutes, 2 y]<br><sub>the shipped N = 24 lift was 12% off here (Sessions D-H)</sub> | `0.007027 rel` | `0.01 rel` | PASS |
| kernel error of the default lift, worst over H in [0.02, 0.5], [1 day, 2 y]<br><sub>the calibration box; no refinement needed (Session I)</sub> | `0.009075 rel` | `0.01 rel` | PASS |
| kernel error of the default lift, worst over H in [0.02, 0.5], [1 hour, 2 y]<br><sub>the calibration box; no refinement needed (Session I)</sub> | `0.009075 rel` | `0.01 rel` | PASS |
| Laplace-domain error, z in [0.1, 100] (plan: 1%, see note)<br><sub>3.5% is the memory beyond 2 y; 1% is not reachable with a 2 y fit</sub> | `0.03349 rel` | `0.05 rel` | PASS |
| Laplace-domain error on the pricing band z in [1, 100]<br><sub>a year down to a few days</sub> | `0.006354 rel` | `0.03 rel` | PASS |
| lifted recursion == SOE convolution (identity) | `3.39e-14` | `1.00e-12` | PASS |
| lifted path vs true-kernel Volterra path (L2, same noise) | `0.006333 rel` | `0.01 rel` | PASS |
| N=1 at x=0 reproduces closed-form Heston (ETDRK4, 200 steps) | `7.09e-10` | `1.00e-09` | PASS |
| full N-node lift at H = 0.4999 vs Heston<br><sub>shrinks 10x per decade of (1/2 - H)</sub> | `1.65e-05` | `1.00e-04` | PASS |
| ETDRK4 vs implicit trapezoidal + Richardson (H=0.12)<br><sub>independent time-steppers, 1 d / 30 d / 1 y</sub> | `4.12e-06` | `5.00e-06` | PASS |

**Rough robustness**

| check | measured | threshold | verdict |
|---|---|---|---|
| exptrap at 120 steps, u to 1200, guards off (NOT unconditionally stable)<br><sub>\|phi(u - i/2)\| must be <= 1; kept to prove the constraint</sub> | `1.08e+36` | `1` | PASS |
| stability constants enforced: the same solve raises StabilityError<br><sub>hard precondition since Session G; \|phi\| <= 1 is a hard postcondition</sub> | `1` | `1` | PASS |
| stability-sized exptrap, same u range: max \|phi(u - i/2)\|<br><sub>383 steps</sub> | `0.9988` | `1` | PASS |
| runaway parameters: maturities priced in-band (of 4)<br><sub>Carr-Madan returned 1.9e18 / 2.9e28 at 30 / 90 d here</sub> | `4` | `4` | PASS |
| unpriceable quote: minimum charge (vol) vs 0 before<br><sub>exact fit costs 1.6e-22</sub> | `4.754` | `4.754` | PASS |

**Rough identifiability**

| check | measured | threshold | verdict |
|---|---|---|---|
| SE(H), 20 quotes, equal weights, 0.5 vp noise | `0.04652` | `0.06` | PASS |
| SE(H), same quotes, vega^2 weights<br><sub>H not identified: the weights discard the short end</sub> | `0.1967` | `0.15` | PASS |
| corr(H, xi) at the truth<br><sub>the rough analogue of kappa/theta</sub> | `0.9777` | `0.9` | PASS |

**Hawkes**

| check | measured | threshold | verdict |
|---|---|---|---|
| mean rate vs mu/(1-n), in SE<br><sub>12.457 vs 12.5</sub> | `0.5936 SE` | `3 SE` | PASS |
| count variance / Hawkes (1971) closed form, 0.25 y<br><sub>Fano 5.93</sub> | `1.019` | `0.15` | PASS |
| MLE plant-and-recover, worst \|z\| | `1.004` | `3` | PASS |
| time rescaling KS p, Hawkes fit (should pass) | `0.5838` | `0.01` | PASS |
| time rescaling KS p, Poisson fit to the same events<br><sub>clustering is detected, not assumed</sub> | `0` | `1.00e-06` | PASS |
| second spike: superposition identity, every host<br><sub>inc2 - inc1 = r(d+W) - r(d); OU, Heston, rough (the direct rough mode was retired in Session I)</sub> | `8.88e-16` | `1.00e-12` | PASS |
| rough jump: realised 1-day impact / its parameter, 5-minute steps<br><sub>ratio 0.9948; the parameter is the average extra variance over the first day</sub> | `0.005225` | `0.05` | PASS |
| Heston: property switches on at alpha = kappa (1.03 kappa)<br><sub>at 0.97 kappa: -2.1e-06</sub> | `1.38e-06` | `0` | PASS |
| rough (driver jumps): pairs with the property at n = 0.6<br><sub>14 of 21 at n = 0.95 -- needs near-critical clustering</sub> | `0` | `0` | PASS |
| kurtosis vs scale-mixture closed form, in SE<br><sub>5.73 vs 5.59; ratio to Poisson = daily Fano</sub> | `0.7597 SE` | `4 SE` | PASS |

**Kalman**

| check | measured | threshold | verdict |
|---|---|---|---|
| fast scalar path vs generic matrix filter | `0` | `1.00e-12` | PASS |
| RMSE improvement minus steady-state theory<br><sub>45.6% vs 45.2%</sub> | `0.3392 pts` | `2 pts` | PASS |
| lecture AR(1) kappa on noisy quotes / true kappa<br><sub>attenuation bias; MLE through the filter is unbiased</sub> | `12.6 x` | `2 x` | PASS |
| dual vs joint EKF agreement | `0.129 %` | `1 %` | PASS |
| recursive MLE (Ljung) distance from the MLE<br><sub>Wan-Nelson dual: 0.07 SE</sub> | `0.2287 SE` | `1 SE` | PASS |
| strict: steps to re-converge after the regime change<br><sub>bad-print excursion 2.0x normal error</sub> | `18` | `2` | PASS |
| adaptive: bad-print excursion (x normal error)<br><sub>re-converges in 0 steps</sub> | `14.92 x` | `6.044 x` | PASS |
| robust: steps to re-converge (and ignores the print)<br><sub>excursion 1.7x vs strict 2.0x</sub> | `1.5` | `2` | PASS |
| lifted rough filter: RMSE improvement over quotes<br><sub>the lifted factors (40 on the default lift) as the state, rank-one Q</sub> | `37.67 %` | `30 %` | PASS |

**Recording (real data)**

| check | measured | threshold | verdict |
|---|---|---|---|
| tau: 17:00 Istanbul to next close, hours<br><sub>a naive local clock gave 27 h (fixed in Session G)</sub> | `30 h` | `30 h` | PASS |
| US close in UTC follows daylight saving | `1` | `1` | PASS |
| delayed greeks (tick 83) recorded, vega per 1.00 vol<br><sub>IBKR sends vega per vol point</sub> | `1 x` | `1 x` | PASS |

**Positivity (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| QE step: mean vs exact conditional mean, in SE<br><sub>variance -0.29% vs exact; min V 2.9e-11</sub> | `0.009151 SE` | `4 SE` | PASS |
| quadrature of the integrated-variance moments | `1.11e-15 rel` | `1.00e-12 rel` | PASS |
| rough variance simulated 3000 days: minimum<br><sub>Euler at the same parameters: below zero on 13% of days</sub> | `-1.39e-17` | `-1.00e-12` | PASS |
| exact-moment filter: QML kappa bias away from zero, mean z<br><sub>near zero a Gaussian quasi-likelihood is unreliable (open flag)</sub> | `1.056 SE` | `2 SE` | PASS |

**Lift fidelity (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| 7-day IV, default lift vs true rough Heston (fractional Adams)<br><sub>N = 24 was 0.063 at 7 d, 0.148 at 1 d; N = 40 0.030 and 0.034 (study_finer_lift.py)</sub> | `0.02961 vp` | `0.05 vp` | PASS |
| Ito isometry of the lift at 1 day, shortfall vs true kernel<br><sub>21.6% on the N = 24 lift (its top node cut the kernel below 7 minutes); 5.1% now</sub> | `5.058 %` | `8 %` | PASS |

**H weighting (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| SE(H): inverse variance / vega^2<br><sub>0.0749 -> 0.0076</sub> | `0.1018 x` | `0.25 x` | PASS |
| worst-case RMS error in H: hybrid (anchors x25)<br><sub>inverse variance 0.087, vega^2 0.098</sub> | `0.03783` | `0.08654` | PASS |
| min \|corr(H, xi)\| over all schemes and designs<br><sub>structural: re-weighting cannot remove it</sub> | `0.9219` | `0.9` | PASS |

**Learning H (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| filter profile likelihood: \|H_hat - 0.12\| in SE<br><sub>H_hat 0.125 +/- 0.010, xi re-fitted at each H</sub> | `0.4698 SE` | `3 SE` | PASS |
| log-likelihood of H = 0.49 against the best H<br><sub>Heston-like roughness rejected with xi free</sub> | `-329.7` | `-20` | PASS |

**Zero boundary (Session H)**

| check | measured | threshold | verdict |
|---|---|---|---|
| variance cf (Riccati) vs exact affine mean and variance, 1 day<br><sub>two derivations: the CF filter's Riccati and the eigen-coordinate moments</sub> | `5.03e-07 rel` | `1.00e-06 rel` | PASS |
| away from zero, 400 days: both filters' log-likelihood vs Kalman<br><sub>CF +1.91, particle +0.91: nothing to gain where the Gaussian is right</sub> | `1.91 nats` | `3 nats` | PASS |
| near zero (V = 0 on 14% of days): particle log-likelihood gain over Kalman<br><sub>400 days; 257 nats per 1500 days over 6 seeds (study_zero_boundary.py)</sub> | `68.4 nats` | `20 nats` | PASS |
| near zero: CF filter vs particle filter log-likelihood<br><sub>they agree on the likelihood; the cf filter's low kappa on 4-substep data was a mismatch with the data's law, not the filter (study_zero_boundary_fine.py)</sub> | `1.174 nats` | `6 nats` | PASS |

**Trading clock (Session H)**

| check | measured | threshold | verdict |
|---|---|---|---|
| variance time at omega = 1 vs ACT/365 | `0 rel` | `1.00e-12 rel` | PASS |
| 2025 reference split: session / overnight / weekend-holiday hours<br><sub>1622.5 / 3412.5 / 3725 h (sum 8760 = 365 x 24); by rule: the one-off closure of 9 Jan 2025 is not modelled</sub> | `0 h` | `1.00e-09 h` | PASS |
| NYSE 2026 holidays by rule vs the published calendar<br><sub>days in one list only</sub> | `0` | `0` | PASS |
| Labor Day weekend: Friday close to Tuesday open | `89.5 h` | `89.5 h` | PASS |
| planted omega = 0.2 recovered from a noise-free chain | `0 rel` | `1.00e-04 rel` | PASS |
| 95% interval coverage, 20 chains with 2% noise<br><sub>profile GLS with an F(1, dof) cut</sub> | `0.95` | `0.85` | PASS |
| snapshot instant recovered from the taus (local timestamp ignored)<br><sub>snapshots before Session H stored local time with no offset</sub> | `0 s` | `0.001 s` | PASS |

**Engineering**

| check | measured | threshold | verdict |
|---|---|---|---|
| scipy modules on the live import path<br><sub>scipy.stats alone costs 5.8 s</sub> | `0` | `0` | PASS |
| startup to actionable error, no TWS<br><sub>was 9.6 s before lazy imports</sub> | `0.8307 s` | `3 s` | PASS |
| calibration objective evaluation<br><sub>39 quotes; 470 ms before vectorising</sub> | `15 ms` | `200 ms` | PASS |
| Gauss-Legendre rules constructed<br><sub>2 reuses; leggauss is O(n^2)</sub> | `1` | `1` | PASS |
| quadrature tolerance 1e-12 vs 1e-14<br><sub>calibration runs at 1e-12, ~2x faster</sub> | `0` | `1.00e-12` | PASS |
| replay round-trip fidelity<br><sub>recorded vs replayed surface points</sub> | `0` | `1.00e-12` | PASS |

121 of 121 checks pass (228s).

Trend over 12 recorded run(s); compared within this machine and configuration (LAPTOP-1VCRA7FA, lift 40:1e+08)
--------------------------------------------------------------------------
  check                                            n       mean        std       last  flags
  2025 reference split: session / overnight / we   3          0          0          0  
  3-point stencil order on the same grid (kept f   3      1.086          0      1.086  
  7-day IV, default lift vs true rough Heston (f   3    0.02961          0    0.02961  
  95% interval coverage, 20 chains with 2% noise   3       0.95   1.36e-16       0.95  
  CPU reference workload                           3     0.3387    0.08808     0.4027  
  Carr-Madan insensitive to damping alpha          3   6.11e-16          0   6.11e-16  
  Carr-Madan vs exact Black-76                     3   1.94e-16          0   1.94e-16  
  ETDRK4 vs implicit trapezoidal + Richardson (H   3   4.12e-06          0   4.12e-06  
  Fourier vs Monte Carlo (sigmas)                  3     0.3618   6.80e-17     0.3618  
  Gauss-Legendre rules constructed                 3          1          0          1  
  H from ATM skew (planted 0.12)                   3       0.12          0       0.12  
  H from structure function, worst of 4 planted    3     0.0062          0     0.0062  
  Heston: property switches on at alpha = kappa    3   1.38e-06          0   1.38e-06  
  Ito isometry of the lift at 1 day, shortfall v   3      5.058          0      5.058  
  Jacobian condition number at the solution        3      30.05   4.35e-15      30.05  
  K(t) = int e^-xt mu(dx) representation           3   6.67e-10          0   6.67e-10  
  Labor Day weekend: Friday close to Tuesday ope   3       89.5          0       89.5  
  Laplace-domain error on the pricing band z in    3   0.006354          0   0.006354  
  Laplace-domain error, z in [0.1, 100] (plan: 1   3    0.03349          0    0.03349  
  Lewis vs Carr-Madan on Heston                    3   4.99e-15          0   4.99e-15  
  Lewis vs exact Black-76                          3   3.55e-15          0   3.55e-15  
  MLE plant-and-recover, worst |z|                 3      1.004          0      1.004  
  N=1 at x=0 reproduces closed-form Heston (ETDR   3   7.09e-10   1.27e-25   7.09e-10  
  NYSE 2026 holidays by rule vs the published ca   3          0          0          0  
  QE step: mean vs exact conditional mean, in SE   3   0.009151          0   0.009151  
  RMSE improvement minus steady-state theory       3     0.3392          0     0.3392  
  SE(H), 20 quotes, equal weights, 0.5 vp noise    3    0.04652          0    0.04652  
  SE(H), same quotes, vega^2 weights               3     0.1967   3.40e-17     0.1967  
  SE(H): inverse variance / vega^2                 3     0.1018   1.70e-17     0.1018  
  SVI parameter recovery                           3   1.43e-08          0   1.43e-08  
  SVI slice passes Durrleman                       3          1          0          1  
  US close in UTC follows daylight saving          3          1          0          1  
  adaptive: bad-print excursion (x normal error)   3      14.92   2.18e-15      14.92  
  away from zero, 400 days: both filters' log-li   3       1.91          0       1.91  
  branch continuity, shipped (max/median step)     3      5.242          0      5.242  
  branch discontinuity, reciprocal form            3       2825          0       2825  
  butterfly audit: catches a planted dent          3          2          0          2  
  butterfly audit: false positives on a convex s   3          0          0          0  
  calendar audit: catches a reversal               3          3          0          3  
  calendar audit: clean on monotone w              3          0          0          0  
  calibration objective evaluation                 3      12.75      3.352         15  
  corr(H, xi) at the truth                         3     0.9777   1.36e-16     0.9777  
  count variance / Hawkes (1971) closed form, 0.   3      1.019          0      1.019  
  delayed greeks (tick 83) recorded, vega per 1.   3          1          0          1  
  density non-negativity (min/peak)                3  -8.87e-10          0  -8.87e-10  
  discount factor from parity                      3   4.44e-16          0   4.44e-16  
  dual vs joint EKF agreement                      3      0.129          0      0.129  
  exact-moment filter: QML kappa bias away from    3      1.056          0      1.056  
  exptrap at 120 steps, u to 1200, guards off (N   3   1.08e+36          0   1.08e+36  
  fast scalar path vs generic matrix filter        3          0          0          0  
  fd_first exact on a quadratic                    3   7.46e-14          0   7.46e-14  
  fd_second on a quadratic vs its rounding floor   3       0.41          0       0.41  
  filter profile likelihood: |H_hat - 0.12| in S   3     0.4698          0     0.4698  
  fitted RMSE on clean data                        3   4.06e-09          0   4.06e-09  
  forward from put-call parity                     3   1.14e-13          0   1.14e-13  
  full N-node lift at H = 0.4999 vs Heston         3   1.65e-05          0   1.65e-05  
  implied-vol round trip                           3   8.05e-16          0   8.05e-16  
  kernel error of the default lift, t in [1 day,   3   0.007027          0   0.007027  
  kernel error of the default lift, worst over H   3   0.009075          0   0.009075  
  kernel error of the default lift, worst over H   3   0.009075          0   0.009075  
  kernel error on [1 hour, 2 y]                    3   0.007027          0   0.007027  
  kernel error on [5 minutes, 2 y]                 3   0.007027          0   0.007027  
  kurtosis vs scale-mixture closed form, in SE     3     0.7597          0     0.7597  
  lecture AR(1) kappa on noisy quotes / true kap   3       12.6          0       12.6  
  lifted path vs true-kernel Volterra path (L2,    3   0.006333          0   0.006333  
  lifted recursion == SOE convolution (identity)   3   3.39e-14          0   3.39e-14  
  lifted rough filter: RMSE improvement over quo   3      37.67          0      37.67  
  log-likelihood of H = 0.49 against the best H    3     -329.7          0     -329.7  
  mean rate vs mu/(1-n), in SE                     3     0.5936          0     0.5936  
  min |corr(H, xi)| over all schemes and designs   3     0.9219          0     0.9219  
  naive-form cancellation at xi=1e-6 (kept to pr   3   3.44e-05          0   3.44e-05  
  near zero (V = 0 on 14% of days): particle log   3       68.4          0       68.4  
  near zero: CF filter vs particle filter log-li   3      1.174          0      1.174  
  norm_cdf vs scipy                                3   2.22e-16          0   2.22e-16  
  norm_pdf vs scipy                                3          0          0          0  
  phi(-i) = 1 (martingale)                         3          0          0          0  
  phi(-u) = conj(phi(u))                           3          0          0          0  
  phi(0) = 1                                       3          0          0          0  
  plant-and-recover, worst relative error          3   3.13e-11          0   3.13e-11  
  planted omega = 0.2 recovered from a noise-fre   3          0          0          0  
  put-call parity C-P = F-K                        3   2.84e-14          0   2.84e-14  
  quadrature of the integrated-variance moments    3   1.11e-15          0   1.11e-15  
  quadrature tolerance 1e-12 vs 1e-14              3          0          0          0  
  recursive MLE (Ljung) distance from the MLE      3     0.2287          0     0.2287  
  replay round-trip fidelity                       3          0          0          0  
  risk-neutral density mass                        3          1          0          1  
  risk-neutral density mean = forward              3          1          0          1  
  robust: steps to re-converge (and ignores the    3        1.5          0        1.5  
  rough (driver jumps): pairs with the property    3          0          0          0  
  rough jump: realised 1-day impact / its parame   3   0.005225          0   0.005225  
  rough objective evaluation, 10 expiries x 13 s   3     0.6697     0.1815     0.7814  
  rough variance simulated 3000 days: minimum      3  -1.39e-17          0  -1.39e-17  
  runaway parameters: maturities priced in-band    3          4          0          4  
  sampling density: min >= eps and mass = 1        3          0          0          0  
  scipy modules on the live import path            3          0          0          0  
  second spike: superposition identity, every ho   3   8.88e-16          0   8.88e-16  
  shipped 2nd derivative: order on a $1/$5 kink    3      3.077          0      3.077  
  simulated E[S/F] = 1                             3   1.94e-06          0   1.94e-06  
  snapshot instant recovered from the taus (loca   3          0          0          0  
  spread of kappa under 0.5vp noise                3      34.73          0      34.73  
  spread of kappa with a Tikhonov prior (weight    3    0.07281          0    0.07281  
  spread of rho under 0.5vp noise                  3       3.73          0       3.73  
  spread of theta under 0.5vp noise                3      2.803   5.44e-16      2.803  
  spread of v0 under 0.5vp noise                   3      3.775          0      3.775  
  spread of xi under 0.5vp noise                   3      19.13          0      19.13  
  stability constants enforced: the same solve r   3          1          0          1  
  stability-sized exptrap, same u range: max |ph   3     0.9988   1.36e-16     0.9988  
  startup to actionable error, no TWS              3     0.7195     0.1438     0.8307  
  strict: steps to re-converge after the regime    3         18          0         18  
  tau: 17:00 Istanbul to next close, hours         3         30          0         30  
  time rescaling KS p, Hawkes fit (should pass)    3     0.5838          0     0.5838  
  time rescaling KS p, Poisson fit to the same e   3          0          0          0  
  trap / shipped separation                        3      538.9          0      538.9  
  two independent H routes agree                   3   0.002891          0   0.002891  
  unpriceable quote: minimum charge (vol) vs 0 b   3      4.754          0      4.754  
  variance cf (Riccati) vs exact affine mean and   3   5.03e-07          0   5.03e-07  
  variance time at omega = 1 vs ACT/365            3          0          0          0  
  vega vs central difference                       3   4.23e-09          0   4.23e-09  
  worst-case RMS error in H: hybrid (anchors x25   3    0.03783          0    0.03783  
  xi -> 0 degenerates to Black-Scholes             3   2.22e-16          0   2.22e-16  
  |phi(u)| <= 1                                    3     0.9999          0     0.9999  
  0 flagged
