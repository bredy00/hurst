"""
The Session L trial (25 September 2026): a Black-Litterman portfolio on a synthetic market,
Fama-French-Carhart attribution through falsify, and a dual Kalman filter that calibrates the
risk model's Hurst index, the views' confidence and the risk aversion to a target market
exposure at low cost.

Self-contained on purpose: nothing outside this package imports it, so the trial can be
reverted by removing it (docs/trial-bl-hurst.md has the commands).
"""
