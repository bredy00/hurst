# Real-data run 20260919_200137 (synthetic, known answer)

## Chain

- snapshot: `C:\Projects\volatility-surface-tuning\captures\real\synthetic\SPY\2026-09-19\SPY_2026-09-19_13-01-26ET.json`
- vega units: IBKR x 100 / Black-Scholes = 1.000 (consistent)
- IV from mid under our tau minus IBKR IV: median -0.00 vol points
- surface: 99 quotes, 5 expiries, 4 skew anchors, 71.1 effective quotes
- variance clock: not identified (no snapshot has weekend or holiday time inside the window)
- Heston: rmse 0.473 vp; rough Heston: rmse 0.002 vp, ok = True (kernel 0.0074, phi_max 0.999981)

| parameter | Heston | rough Heston |
|---|---|---|
| v0 | 0.0242 | 0.0250 |
| kappa | 50.0000 | 1.9986 |
| theta | 0.0293 | 0.0350 |
| xi | 3.6668 | 0.4000 |
| rho | -0.6828 | -0.6997 |
| H |  | 0.1000 |

## History

- 500 days of realised variance (5-minute bars), mean vol 0.214
- CIR MLE kappa 54.10 (se nan)
- lifted rough filter: H profile maximum 0.130 (se 0.024, 95% 0.082-0.175); bank posterior 0.126 +/- 0.025; log-likelihood over CIR +16.0
- structure function of log RV: H 0.14798998530266766 (r2 0.9867085746347618)
- zero boundary (at H = 0.12): predicted variance within 2 sd of zero on 100.0% of days -> state estimate: **cf**; cf filter log-likelihood 1381.2 vs Kalman 1050.0 (0 fallback days); particle-filter confirmation: **not confirmed** (16 substeps: cf - particle +4.700 nats/day, filtered RV median difference 0.6%; at 64 substeps: cf - particle +0.531 nats/day)

## Four readings of H

| estimator | H | se |
|---|---|---|
| skew_term_structure | -0.033 | 0.007 |
| rough_calibration | 0.100 |  |
| filter_profile | 0.130 | 0.024 |
| structure_function | 0.148 |  |

Truth (synthetic): H = 0.1
