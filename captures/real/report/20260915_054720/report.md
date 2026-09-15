# Real-data run 20260915_054720 (synthetic, known answer)

## Chain

- snapshot: `C:\Projects\volatility-surface-tuning\captures\real\synthetic\SPY\2026-09-14\SPY_2026-09-14_22-47-12ET.json`
- vega units: IBKR x 100 / Black-Scholes = 1.000 (consistent)
- IV from mid under our tau minus IBKR IV: median -0.00 vol points
- surface: 99 quotes, 5 expiries, 3 skew anchors, 70.9 effective quotes

## History

- 500 days of realised variance (5-minute bars), mean vol 0.246
- CIR MLE kappa 59.95 (se 14.29)
- lifted rough filter: H profile maximum 0.085 (se 0.025, 95% 0.036-0.132); bank posterior 0.080 +/- 0.026; log-likelihood over CIR -7.5
- structure function of log RV: H 0.15137405356443384 (r2 0.968521586804131)

## Four readings of H

| estimator | H | se |
|---|---|---|
| skew_term_structure | -0.027 | 0.008 |
| filter_profile | 0.085 | 0.025 |
| structure_function | 0.151 |  |

Truth (synthetic): H = 0.1
