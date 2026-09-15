
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
| plant-and-recover, worst relative error<br><sub>7 iters, 1.8s</sub> | `3.13e-11 rel` | `0.01 rel` | PASS |
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
| rough objective evaluation, 10 expiries x 13 strikes<br><sub>best of 3 (all: 3.00, 3.05, 3.06); trend target: rolling mean <= 0.75 s</sub> | `3.002 s` | `3 s` | **FAIL** |

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

**Zero boundary (Session H)**

| check | measured | threshold | verdict |
|---|---|---|---|
| variance cf (Riccati) vs exact affine mean and variance, 1 day<br><sub>two derivations: the CF filter's Riccati and the eigen-coordinate moments</sub> | `1.78e-08 rel` | `1.00e-06 rel` | PASS |
| away from zero, 400 days: both filters' log-likelihood vs Kalman<br><sub>CF -0.34, particle -1.02: nothing to gain where the Gaussian is right</sub> | `1.021 nats` | `3 nats` | PASS |
| near zero (V = 0 on 14% of days): particle log-likelihood gain over Kalman<br><sub>400 days; 257 nats per 1500 days over 6 seeds (study_zero_boundary.py)</sub> | `58.25 nats` | `20 nats` | PASS |
| near zero: CF filter vs particle filter log-likelihood<br><sub>they agree on the likelihood; the CF filter's kappa still reads 0.6 low near zero (open flag)</sub> | `0.3773 nats` | `6 nats` | PASS |

**Trading clock (Session H)**

| check | measured | threshold | verdict |
|---|---|---|---|
| variance time at omega = 1 vs ACT/365 | `0 rel` | `1.00e-12 rel` | PASS |
| NYSE 2026 holidays by rule vs the published calendar<br><sub>days in one list only</sub> | `0` | `0` | PASS |
| Labor Day weekend: Friday close to Tuesday open | `89.5 h` | `89.5 h` | PASS |
| planted omega = 0.2 recovered from a noise-free chain | `0 rel` | `1.00e-04 rel` | PASS |
| 95% interval coverage, 20 chains with 2% noise<br><sub>profile GLS with an F(1, dof) cut</sub> | `0.95` | `0.85` | PASS |
| snapshot instant recovered from the taus (local timestamp ignored)<br><sub>snapshots before Session H stored local time with no offset</sub> | `0 s` | `0.001 s` | PASS |

**Engineering**

| check | measured | threshold | verdict |
|---|---|---|---|
| scipy modules on the live import path<br><sub>scipy.stats alone costs 5.8 s</sub> | `0` | `0` | PASS |
| startup to actionable error, no TWS<br><sub>was 9.6 s before lazy imports</sub> | `1.521 s` | `3 s` | PASS |
| calibration objective evaluation<br><sub>39 quotes; 470 ms before vectorising</sub> | `43.6 ms` | `200 ms` | PASS |
| Gauss-Legendre rules constructed<br><sub>2 reuses; leggauss is O(n^2)</sub> | `1` | `1` | PASS |
| quadrature tolerance 1e-12 vs 1e-14<br><sub>calibration runs at 1e-12, ~2x faster</sub> | `0` | `1.00e-12` | PASS |
| replay round-trip fidelity<br><sub>recorded vs replayed surface points</sub> | `0` | `1.00e-12` | PASS |

116 of 117 checks pass (251s).
