# Phase 1 baseline — vanilla Heston fitted to a rough surface

Market: synthetic, ATM skew ~ tau^(H-1/2) with **H = 0.12**, 130 quotes across 10 expiries out to one year.

## Fitted parameters

| v0 | kappa | theta | xi | rho |
|---|---|---|---|---|
| 0.04534 | 2.03187 | 0.12239 | 2.01966 | -0.65689 |

Feller margin `2*kappa*theta - xi^2 = -3.5817` (violated — reported, not enforced). Overall RMSE **5.205 vol points**.

## Where the error lives

| tau (days) | n | rmse (vp) | bias (vp) | worst (vp) |
|---|---|---|---|---|
| 1 | 13 | 7.519 | +7.155 | 9.443 |
| 2 | 13 | 6.934 | +6.603 | 8.763 |
| 3 | 13 | 6.433 | +6.124 | 8.172 |
| 7 | 13 | 4.950 | +4.692 | 6.386 |
| 14 | 13 | 3.474 | +3.126 | 4.569 |
| 30 | 13 | 2.759 | +1.408 | 5.657 |
| 60 | 13 | 3.538 | +0.665 | 7.499 |
| 90 | 13 | 4.048 | +0.796 | 8.779 |
| 180 | 13 | 4.744 | +1.618 | 10.745 |
| 365 | 13 | 5.420 | +2.637 | 12.121 |

## The skew shortfall — what Phase 2 must fix

| tau (days) | market skew | Heston skew | Heston / market |
|---|---|---|---|
| 1 | -2.7232 | -1.5754 | 0.579 |
| 2 | -2.0926 | -1.5911 | 0.760 |
| 3 | -1.7938 | -1.6022 | 0.893 |
| 7 | -1.3000 | -1.5684 | 1.206 |
| 14 | -0.9990 | -1.3633 | 1.365 |
| 30 | -0.7478 | -0.9957 | 1.332 |
| 60 | -0.5746 | -0.7022 | 1.222 |
| 90 | -0.4926 | -0.5756 | 1.169 |
| 180 | -0.3785 | -0.4171 | 1.102 |
| 365 | -0.2893 | -0.2987 | 1.032 |

H from the market skews **0.1200** (planted 0.12); from the fitted Heston **0.2053**.

At 1 days the fitted Heston delivers only **57.9%** of the market's ATM skew. A diffusion has a finite skew limit as tau -> 0, so no parameter choice can close this — it is a model deficiency, not a calibration failure.

## Identifiability (0.5 vol point quote noise, 6 seeds)

| parameter | spread | verdict |
|---|---|---|
| v0 | 2.1% | stable |
| kappa | 37.7% | **unidentified** |
| theta | 17.3% | loose |
| xi | 13.4% | loose |
| rho | 2.1% | stable |

`v0` and `rho` are pinned by the short-end level and the skew. `kappa` and `theta` are not determined by a surface that stops at one year, because `1/kappa` is comparable to the longest maturity — the process has not relaxed within the data. Extending to three years tightens `theta` by roughly a factor of four. Any report of these five numbers without that caveat overstates what was measured.
