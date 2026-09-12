# Phase 2 — rough Heston vs vanilla Heston

Both models fitted by the same Levenberg-Marquardt driver, in implied vol, weighting **equal-in-vol** (see `study_h_identifiability.py` for why not vega²), to 8 maturities from 1 to 365 days × 7 strikes across ±3σ.

## IN-MODEL market (rough Heston, H = 0.12)

| model | v0 | kappa | theta | xi | rho | H | rmse (vp) |
|---|---|---|---|---|---|---|---|
| **truth** | 0.0400 | 2.0000 | 0.0450 | 0.5000 | -0.7000 | 0.1200 | — |
| vanilla Heston | 0.0385 | 35.6491 | 0.0437 | 3.9287 | -0.7169 | — | 0.760 |
| rough Heston | 0.0400 | 2.0001 | 0.0450 | 0.5000 | -0.7000 | 0.1200 | 0.000 |

Rough fit: 11 iterations, stopped on `xtol`; pricer {'solves': 752, 'refined': 0, 'extended': 0, 'failed_strips': 0}.

| tau (days) | Heston rmse (vp) | rough rmse (vp) |
|---|---|---|
| 1 | 1.054 | 0.000 |
| 2 | 0.419 | 0.000 |
| 3 | 0.204 | 0.000 |
| 7 | 0.676 | 0.000 |
| 14 | 0.894 | 0.000 |
| 30 | 0.799 | 0.000 |
| 90 | 0.396 | 0.000 |
| 365 | 1.116 | 0.000 |

| tau (days) | market skew | Heston | rough Heston | Heston / market | rough / market |
|---|---|---|---|---|---|
| 1 | -4.8414 | -3.6567 | -4.8414 | 0.755 | 1.000 |
| 2 | -3.5830 | -3.4768 | -3.5830 | 0.970 | 1.000 |
| 3 | -2.9839 | -3.1942 | -2.9839 | 1.070 | 1.000 |
| 5 | -2.3491 | -2.6854 | -2.3491 | 1.143 | 1.000 |
| 7 | -1.9944 | -2.3141 | -1.9944 | 1.160 | 1.000 |
| 10 | -1.6667 | -1.9336 | -1.6667 | 1.160 | 1.000 |
| 14 | -1.3989 | -1.6087 | -1.3989 | 1.150 | 1.000 |
| 30 | -0.9239 | -1.0282 | -0.9239 | 1.113 | 1.000 |
| 60 | -0.6268 | -0.6646 | -0.6268 | 1.060 | 1.000 |
| 90 | -0.4983 | -0.5047 | -0.4983 | 1.013 | 1.000 |
| 180 | -0.3311 | -0.2992 | -0.3311 | 0.904 | 1.000 |
| 365 | -0.2098 | -0.1641 | -0.2098 | 0.782 | 1.000 |

Log-log slope of |skew| against tau over 1–14 days: market **-0.470**, Heston **-0.319**, rough Heston **-0.470**; planted H − ½ = -0.380.

| local slope | market | Heston | rough Heston |
|---|---|---|---|
| 1–3 days | -0.441 | -0.123 | -0.441 |
| 7–14 days | -0.512 | -0.525 | -0.512 |

## STYLISED market (Phase 1 surface, H = 0.12)

| model | v0 | kappa | theta | xi | rho | H | rmse (vp) |
|---|---|---|---|---|---|---|---|
| vanilla Heston | 0.0203 | 10.8879 | 0.0941 | 3.7985 | -0.8176 | — | 2.104 |
| rough Heston | 0.0203 | 10.8928 | 0.0941 | 3.8001 | -0.8176 | 0.5000 | 2.104 |

Rough fit: 13 iterations, stopped on `ftol`; pricer {'solves': 914, 'refined': 0, 'extended': 10, 'failed_strips': 0}.

| tau (days) | Heston rmse (vp) | rough rmse (vp) |
|---|---|---|
| 1 | 1.745 | 1.745 |
| 2 | 1.593 | 1.593 |
| 3 | 1.527 | 1.527 |
| 7 | 1.528 | 1.528 |
| 14 | 1.780 | 1.780 |
| 30 | 2.211 | 2.211 |
| 90 | 2.174 | 2.174 |
| 365 | 3.517 | 3.517 |

| tau (days) | market skew | Heston | rough Heston | Heston / market | rough / market |
|---|---|---|---|---|---|
| 1 | -2.7232 | -5.4775 | -5.4796 | 2.011 | 2.012 |
| 2 | -2.0926 | -4.7554 | -4.7564 | 2.272 | 2.273 |
| 3 | -1.7938 | -4.1235 | -4.1241 | 2.299 | 2.299 |
| 5 | -1.4773 | -3.2889 | -3.2892 | 2.226 | 2.226 |
| 7 | -1.3000 | -2.7850 | -2.7851 | 2.142 | 2.142 |
| 10 | -1.1352 | -2.3191 | -2.3192 | 2.043 | 2.043 |
| 14 | -0.9990 | -1.9490 | -1.9491 | 1.951 | 1.951 |
| 30 | -0.7478 | -1.3301 | -1.3302 | 1.779 | 1.779 |
| 60 | -0.5746 | -0.9556 | -0.9557 | 1.663 | 1.663 |
| 90 | -0.4926 | -0.7828 | -0.7829 | 1.589 | 1.589 |
| 180 | -0.3785 | -0.5285 | -0.5285 | 1.396 | 1.396 |
| 365 | -0.2893 | -0.3228 | -0.3228 | 1.116 | 1.116 |

Log-log slope of |skew| against tau over 1–14 days: market **-0.380**, Heston **-0.404**, rough Heston **-0.404**; planted H − ½ = -0.380.

| local slope | market | Heston | rough Heston |
|---|---|---|---|
| 1–3 days | -0.380 | -0.258 | -0.259 |
| 7–14 days | -0.380 | -0.515 | -0.515 |

