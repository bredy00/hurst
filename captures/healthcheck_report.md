
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
| kernel error, N=24, t in [1 day, 2 y]<br><sub>max relative; H = 0.12</sub> | `0.008697 rel` | `0.01 rel` | PASS |
| kernel error on [1 hour, 2 y]<br><sub>a one-day option lives here; 12% below 5 minutes</sub> | `0.008697 rel` | `0.01 rel` | PASS |
| kernel error, N=24, worst over H in [0.08, 0.5]<br><sub>H = 0.05: 0.0105, H = 0.02: 0.0113 -- above 1%</sub> | `0.00972 rel` | `0.01 rel` | PASS |
| kernel error, N=32, worst over the whole H box [0.02, 0.5]<br><sub>what calibration refits with when a fit lands below H = 0.08</sub> | `0.006656 rel` | `0.01 rel` | PASS |
| Laplace-domain error, z in [0.1, 100] (plan: 1%, see note)<br><sub>3.5% is the memory beyond 2 y; 1% is not reachable with a 2 y fit</sub> | `0.03508 rel` | `0.05 rel` | PASS |
| Laplace-domain error on the pricing band z in [1, 100]<br><sub>a year down to a few days</sub> | `0.01419 rel` | `0.03 rel` | PASS |
| lifted recursion == SOE convolution (identity) | `4.05e-14` | `1.00e-12` | PASS |
| lifted path vs true-kernel Volterra path (L2, same noise) | `0.008001 rel` | `0.01 rel` | PASS |
| N=1 at x=0 reproduces closed-form Heston (ETDRK4, 200 steps) | `7.09e-10` | `1.00e-09` | PASS |
| full N-node lift at H = 0.4999 vs Heston<br><sub>shrinks 10x per decade of (1/2 - H)</sub> | `1.64e-05` | `1.00e-04` | PASS |
| ETDRK4 vs implicit trapezoidal + Richardson (H=0.12)<br><sub>independent time-steppers, 1 d / 30 d / 1 y</sub> | `3.79e-06` | `5.00e-06` | PASS |
| rough objective evaluation, 10 expiries x 13 strikes<br><sub>best of 3 (all: 0.97, 0.62, 0.60); trend target: rolling mean <= 0.75 s</sub> | `0.6048 s` | `3 s` | PASS |

**Rough robustness**

| check | measured | threshold | verdict |
|---|---|---|---|
| exptrap at 120 steps, u to 1200, guards off (NOT unconditionally stable)<br><sub>\|phi(u - i/2)\| must be <= 1; kept to prove the constraint</sub> | `1.21e+36` | `1` | PASS |
| stability constants enforced: the same solve raises StabilityError<br><sub>hard precondition since Session G; \|phi\| <= 1 is a hard postcondition</sub> | `1` | `1` | PASS |
| stability-sized exptrap, same u range: max \|phi(u - i/2)\|<br><sub>383 steps</sub> | `0.9988` | `1` | PASS |
| runaway parameters: maturities priced in-band (of 4)<br><sub>Carr-Madan returned 1.9e18 / 2.9e28 at 30 / 90 d here</sub> | `4` | `4` | PASS |
| unpriceable quote: minimum charge (vol) vs 0 before<br><sub>exact fit costs 1.4e-22</sub> | `4.754` | `4.754` | PASS |

**Rough identifiability**

| check | measured | threshold | verdict |
|---|---|---|---|
| SE(H), 20 quotes, equal weights, 0.5 vp noise | `0.04711` | `0.06` | PASS |
| SE(H), same quotes, vega^2 weights<br><sub>H not identified: the weights discard the short end</sub> | `0.2024` | `0.15` | PASS |
| corr(H, xi) at the truth<br><sub>the rough analogue of kappa/theta</sub> | `0.9776` | `0.9` | PASS |

**Hawkes**

| check | measured | threshold | verdict |
|---|---|---|---|
| mean rate vs mu/(1-n), in SE<br><sub>12.457 vs 12.5</sub> | `0.5936 SE` | `3 SE` | PASS |
| count variance / Hawkes (1971) closed form, 0.25 y<br><sub>Fano 5.93</sub> | `1.019` | `0.15` | PASS |
| MLE plant-and-recover, worst \|z\| | `1.004` | `3` | PASS |
| time rescaling KS p, Hawkes fit (should pass) | `0.5838` | `0.01` | PASS |
| time rescaling KS p, Poisson fit to the same events<br><sub>clustering is detected, not assumed</sub> | `0` | `1.00e-06` | PASS |
| second spike: superposition identity, 4 hosts<br><sub>inc2 - inc1 = r(d+W) - r(d)</sub> | `8.88e-16` | `1.00e-12` | PASS |
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
| lifted rough filter: RMSE improvement over quotes<br><sub>24 lifted factors as the state, rank-one Q</sub> | `39.67 %` | `30 %` | PASS |

**Recording (real data)**

| check | measured | threshold | verdict |
|---|---|---|---|
| tau: 17:00 Istanbul to next close, hours<br><sub>a naive local clock gave 27 h (fixed in Session G)</sub> | `30 h` | `30 h` | PASS |
| US close in UTC follows daylight saving | `1` | `1` | PASS |
| delayed greeks (tick 83) recorded, vega per 1.00 vol<br><sub>IBKR sends vega per vol point</sub> | `1 x` | `1 x` | PASS |

**Positivity (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| QE step: mean vs exact conditional mean, in SE<br><sub>variance -0.27% vs exact; min V 2.7e-11</sub> | `0.02983 SE` | `4 SE` | PASS |
| quadrature of the integrated-variance moments | `1.07e-15 rel` | `1.00e-12 rel` | PASS |
| rough variance simulated 3000 days: minimum<br><sub>Euler at the same parameters: below zero on 13% of days</sub> | `-2.08e-17` | `-1.00e-12` | PASS |
| exact-moment filter: QML kappa bias away from zero, mean z<br><sub>near zero a Gaussian quasi-likelihood is unreliable (open flag)</sub> | `0.6433 SE` | `2 SE` | PASS |

**Lift fidelity (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| 7-day IV, lift N=24 vs true rough Heston (fractional Adams)<br><sub>1 day: 0.15 vp, skew -1.6% (study_lewis_isometry.py)</sub> | `0.0634 vp` | `0.1 vp` | PASS |
| Ito isometry of the lift at 1 day, shortfall vs true kernel<br><sub>93% of it from lags under 7 minutes; prices integrate it away</sub> | `21.6 %` | `25 %` | PASS |

**H weighting (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| SE(H): inverse variance / vega^2<br><sub>0.0749 -> 0.0076</sub> | `0.1018 x` | `0.25 x` | PASS |
| worst-case RMS error in H: hybrid (anchors x25)<br><sub>inverse variance 0.087, vega^2 0.098</sub> | `0.03783` | `0.08654` | PASS |
| min \|corr(H, xi)\| over all schemes and designs<br><sub>structural: re-weighting cannot remove it</sub> | `0.9219` | `0.9` | PASS |

**Learning H (Session G)**

| check | measured | threshold | verdict |
|---|---|---|---|
| filter profile likelihood: \|H_hat - 0.12\| in SE<br><sub>H_hat 0.133 +/- 0.012, xi re-fitted at each H</sub> | `1.051 SE` | `3 SE` | PASS |
| log-likelihood of H = 0.49 against the best H<br><sub>Heston-like roughness rejected with xi free</sub> | `-266` | `-20` | PASS |

**Engineering**

| check | measured | threshold | verdict |
|---|---|---|---|
| scipy modules on the live import path<br><sub>scipy.stats alone costs 5.8 s</sub> | `0` | `0` | PASS |
| startup to actionable error, no TWS<br><sub>was 9.6 s before lazy imports</sub> | `0.5423 s` | `3 s` | PASS |
| calibration objective evaluation<br><sub>39 quotes; 470 ms before vectorising</sub> | `8.565 ms` | `200 ms` | PASS |
| Gauss-Legendre rules constructed<br><sub>2 reuses; leggauss is O(n^2)</sub> | `1` | `1` | PASS |
| quadrature tolerance 1e-12 vs 1e-14<br><sub>calibration runs at 1e-12, ~2x faster</sub> | `0` | `1.00e-12` | PASS |
| replay round-trip fidelity<br><sub>recorded vs replayed surface points</sub> | `0` | `1.00e-12` | PASS |

107 of 107 checks pass (58s).

Trend over 8 recorded run(s)
--------------------------------------------------------------------------
  check                                            n       mean        std       last  flags
  3-point stencil order on the same grid (kept f   8      1.086          0      1.086  
  7-day IV, lift N=24 vs true rough Heston (frac   2     0.0634          0     0.0634  
  Carr-Madan insensitive to damping alpha          8   6.11e-16          0   6.11e-16  
  Carr-Madan vs exact Black-76                     8   1.94e-16          0   1.94e-16  
  ETDRK4 vs implicit trapezoidal + Richardson (H   8   3.79e-06   7.88e-18   3.79e-06  
  Fourier vs Monte Carlo (sigmas)                  8     0.3618          0     0.3618  
  Gauss-Legendre rules constructed                 8          1          0          1  
  H from ATM skew (planted 0.12)                   8       0.12          0       0.12  
  H from structure function, worst of 4 planted    8     0.0062          0     0.0062  
  Heston: property switches on at alpha = kappa    5   1.38e-06          0   1.38e-06  
  Ito isometry of the lift at 1 day, shortfall v   2       21.6          0       21.6  
  Jacobian condition number at the solution        8      30.05          0      30.05  
  K(t) = int e^-xt mu(dx) representation           8   6.67e-10          0   6.67e-10  
  Laplace-domain error on the pricing band z in    7    0.01419   3.75e-18    0.01419  
  Laplace-domain error, z in [0.1, 100] (plan as   1    0.03508          0    0.03508  
  Laplace-domain error, z in [0.1, 100] (plan: 1   7    0.03508          0    0.03508  
  Lewis vs Carr-Madan on Heston                    8   4.99e-15          0   4.99e-15  
  Lewis vs exact Black-76                          8   3.55e-15          0   3.55e-15  
  MLE plant-and-recover, worst |z|                 5      1.004          0      1.004  
  N=1 at x=0 reproduces closed-form Heston (ETDR   8   7.09e-10          0   7.09e-10  
  QE step: mean vs exact conditional mean, in SE   2    0.02983          0    0.02983  
  RMSE improvement minus steady-state theory       5     0.3392          0     0.3392  
  SE(H), 20 quotes, equal weights, 0.5 vp noise    6    0.04711          0    0.04711  
  SE(H), same quotes, vega^2 weights               6     0.2024          0     0.2024  
  SE(H): inverse variance / vega^2                 2     0.1018          0     0.1018  
  SVI parameter recovery                           8   1.43e-08          0   1.43e-08  
  SVI slice passes Durrleman                       8          1          0          1  
  US close in UTC follows daylight saving          2          1          0          1  
  adaptive: bad-print excursion (x normal error)   5      14.92          0      14.92  
  branch continuity, shipped (max/median step)     8      5.242          0      5.242  
  branch discontinuity, reciprocal form            8       2825          0       2825  
  butterfly audit: catches a planted dent          8          2          0          2  
  butterfly audit: false positives on a convex s   8          0          0          0  
  calendar audit: catches a reversal               8          3          0          3  
  calendar audit: clean on monotone w              8          0          0          0  
  calibration objective evaluation                 8      9.711      2.347      8.565  
  corr(H, xi) at the truth                         6     0.9776   1.22e-16     0.9776  
  count variance / Hawkes (1971) closed form, 0.   5      1.019          0      1.019  
  delayed greeks (tick 83) recorded, vega per 1.   2          1          0          1  
  density non-negativity (min/peak)                8  -8.87e-10          0  -8.87e-10  
  discount factor from parity                      8   4.44e-16          0   4.44e-16  
  dual vs joint EKF agreement                      5      0.129          0      0.129  
  exact-moment filter: QML kappa bias away from    2     0.6433          0     0.6433  
  exptrap at 120 steps, u to 1200 (NOT unconditi   4   1.21e+36          0   1.21e+36  
  exptrap at 120 steps, u to 1200, guards off (N   2   1.21e+36          0   1.21e+36  
  fast scalar path vs generic matrix filter        5          0          0          0  
  fd_first exact on a quadratic                    8   7.46e-14          0   7.46e-14  
  fd_second on a quadratic vs its rounding floor   8       0.41          0       0.41  
  filter profile likelihood: |H_hat - 0.12| in S   2      1.051          0      1.051  
  fitted RMSE on clean data                        8   4.06e-09          0   4.06e-09  
  forward from put-call parity                     8   1.14e-13          0   1.14e-13  
  full N-node lift at H = 0.4999 vs Heston         8   1.64e-05   2.33e-17   1.64e-05  
  implied-vol round trip                           8   8.05e-16          0   8.05e-16  
  kernel error on [1 hour, 2 y]                    8   0.008697          0   0.008697  
  kernel error, N=24, t in [1 day, 2 y]            8   0.008697          0   0.008697  
  kernel error, N=24, worst over H in [0.08, 0.5   2    0.00972          0    0.00972  
  kernel error, N=32, worst over the whole H box   2   0.006656          0   0.006656  
  kurtosis vs scale-mixture closed form, in SE     5     0.7597          0     0.7597  
  lecture AR(1) kappa on noisy quotes / true kap   5       12.6          0       12.6  
  lifted path vs true-kernel Volterra path (L2,    8   0.008001          0   0.008001  
  lifted recursion == SOE convolution (identity)   8   4.05e-14          0   4.05e-14  
  lifted rough filter: RMSE improvement over quo   5      44.31      4.242      39.67  
  log-likelihood of H = 0.49 against the best H    2       -266          0       -266  
  mean rate vs mu/(1-n), in SE                     5     0.5936          0     0.5936  
  min |corr(H, xi)| over all schemes and designs   2     0.9219          0     0.9219  
  naive-form cancellation at xi=1e-6 (kept to pr   8   3.44e-05          0   3.44e-05  
  norm_cdf vs scipy                                8   2.22e-16          0   2.22e-16  
  norm_pdf vs scipy                                8          0          0          0  
  phi(-i) = 1 (martingale)                         8          0          0          0  
  phi(-u) = conj(phi(u))                           8          0          0          0  
  phi(0) = 1                                       8          0          0          0  
  plant-and-recover, worst relative error          8   3.13e-11          0   3.13e-11  
  put-call parity C-P = F-K                        8   2.84e-14          0   2.84e-14  
  quadrature of the integrated-variance moments    2   1.07e-15          0   1.07e-15  
  quadrature tolerance 1e-12 vs 1e-14              8          0          0          0  
  recursive MLE (Ljung) distance from the MLE      5     0.2287          0     0.2287  
  replay round-trip fidelity                       8          0          0          0  
  risk-neutral density mass                        8          1          0          1  
  risk-neutral density mean = forward              8          1          0          1  
  robust: steps to re-converge (and ignores the    5        1.5          0        1.5  
  rough (driver jumps): pairs with the property    5          0          0          0  
  rough objective evaluation, 10 expiries x 13 s   8     0.7122       0.16     0.6048  
  rough variance simulated 3000 days: minimum      2  -2.08e-17          0  -2.08e-17  
  runaway parameters: maturities priced in-band    6          4          0          4  
  sampling density: min >= eps and mass = 1        8          0          0          0  
  scipy modules on the live import path            8          0          0          0  
  second spike: superposition identity, 4 hosts    5   8.88e-16          0   8.88e-16  
  shipped 2nd derivative: order on a $1/$5 kink    8      3.077          0      3.077  
  simulated E[S/F] = 1                             8   1.94e-06          0   1.94e-06  
  spread of kappa under 0.5vp noise                8      34.73          0      34.73  
  spread of kappa with a Tikhonov prior (weight    8    0.07281          0    0.07281  
  spread of rho under 0.5vp noise                  8       3.73          0       3.73  
  spread of theta under 0.5vp noise                8      2.803          0      2.803  
  spread of v0 under 0.5vp noise                   8      3.775          0      3.775  
  spread of xi under 0.5vp noise                   8      19.13          0      19.13  
  stability constants enforced: the same solve r   2          1          0          1  
  stability-sized exptrap, same u range: max |ph   6     0.9988          0     0.9988  
  startup to actionable error, no TWS              8      0.602     0.1361     0.5423  
  strict: steps to re-converge after the regime    5         18          0         18  
  tau: 17:00 Istanbul to next close, hours         2         30          0         30  
  time rescaling KS p, Hawkes fit (should pass)    5     0.5838          0     0.5838  
  time rescaling KS p, Poisson fit to the same e   5          0          0          0  
  trap / shipped separation                        8      538.9          0      538.9  
  two independent H routes agree                   8   0.002891          0   0.002891  
  unpriceable quote: minimum charge (vol) vs 0 b   6      4.754          0      4.754  
  vega vs central difference                       8   4.23e-09          0   4.23e-09  
  worst-case RMS error in H: hybrid (anchors x25   2    0.03783          0    0.03783  
  xi -> 0 degenerates to Black-Scholes             8   2.22e-16          0   2.22e-16  
  |phi(u)| <= 1                                    8     0.9999          0     0.9999  
  0 flagged
exit 0
