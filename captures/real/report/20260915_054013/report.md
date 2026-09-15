# Real-data run 20260915_054013 (synthetic, known answer)

## Chain

- snapshot: `C:\Projects\volatility-surface-tuning\captures\real\synthetic\SPY\2026-09-14\SPY_2026-09-14_22-40-03ET.json`
- vega units: IBKR x 100 / Black-Scholes = 1.000 (consistent)
- IV from mid under our tau minus IBKR IV: median -0.00 vol points
- surface: 99 quotes, 5 expiries, 3 skew anchors, 70.9 effective quotes

## History

- 500 days of realised variance (5-minute bars), mean vol 0.246
- CIR MLE kappa 59.95 (se 14.29)
- lifted rough filter: H profile maximum 0.396 (se 0.041, 95% 0.316-0.486); bank posterior 0.367 +/- 0.046; log-likelihood over CIR -63.5
- structure function of log RV: H 0.15137405356443384 (r2 0.968521586804131)

## Four readings of H

| estimator | H | se |
|---|---|---|
| skew_term_structure | -0.028 | 0.008 |
| filter_profile | 0.396 | 0.041 |
| structure_function | 0.151 |  |

Truth (synthetic): H = 0.1
