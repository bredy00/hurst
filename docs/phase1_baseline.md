# Phase 1 baseline — vanilla Heston ATM skew

Parameters: v0=0.04, kappa=2.0, theta=0.045, xi=0.5, rho=-0.7 (Feller False)

| tau (days) | ATM skew | stderr |
|---|---|---|
| 1 | -0.4367 | 5.17e-05 |
| 2 | -0.4359 | 1.05e-04 |
| 5 | -0.4333 | 2.80e-04 |
| 10 | -0.4285 | 6.18e-04 |
| 21 | -0.4164 | 1.52e-03 |
| 42 | -0.3883 | 3.07e-03 |
| 63 | -0.3593 | 3.80e-03 |
| 126 | -0.2884 | 3.65e-03 |
| 252 | -0.2069 | 2.24e-03 |
| 504 | -0.1340 | 9.72e-04 |

Fitted **H = 0.4832 ± 0.0151** (r² = 0.1350); short end alone **H = 0.4950**.

The market sits near H ≈ 0.12. Heston's exponent is pinned close to 1/2 because a classical diffusion has a finite ATM skew limit as tau -> 0. This is the number Phase 2 has to beat.
