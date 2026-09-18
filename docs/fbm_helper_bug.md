# The fBm fixture bug — reproduced, diagnosed, and left open for a second pass

Reproduce with `python debug_fbm_helper.py` (n = 2^17, seed 1).

## What it was

Session A's first fixture for a rough log-volatility path was a "cheap
Riemann–Liouville" convolution:

```python
kern = (np.arange(1, n + 1)) ** (H - 0.5)     # (j+1)^(H-1/2)  for  j = 0 .. n-1
out  = np.convolve(white, kern)[:n]           # X_t = sum_j kern_j * w_{t-j}
```

i.e. a Riemann sum for `X_t = ∫₀ᵗ (t−s)^(H−½) dW_s` with the kernel sampled at the
right-hand end of each unit cell. Fed to `hurst_from_structure` with a planted
`H = 0.12` it returned **`H = 0.26`**, which looked like an estimator bug. The
estimator was right; the generator was wrong, and it was replaced with exact
Davies–Harte circulant embedding.

## What was actually wrong

Measured, planted `H = 0.12`, structure function read at lags 1–60 as the
fixture does:

| generator | Ĥ | ρ(1) of increments |
|---|---|---|
| original helper, kernel `(j+1)^a` | **0.2599** | −0.179 |
| cell-mean kernel `∫ⱼʲ⁺¹ uᵃ du` | 0.1701 | −0.333 |
| BLP optimal points (hybrid, κ = 0) | 0.1701 | −0.333 |
| hybrid scheme, κ = 1 (first cell exact) | **0.1207** | −0.414 |
| Davies–Harte, exact fBm | 0.1195 | −0.410 |
| exact fGn at H = 0.12 | 0.12 | **−0.4095** |

Slope of `log m(1,Δ)` against `log Δ` by lag window:

| generator | Δ 1–8 | 8–64 | 64–512 | 512–4096 |
|---|---|---|---|---|
| original helper | **0.314** | 0.212 | 0.189 | 0.083 |
| hybrid κ = 1 | 0.117 | 0.121 | 0.122 | 0.205 |
| Davies–Harte | 0.119 | 0.120 | 0.119 | 0.115 |

Three things follow.

1. **The error is at the shortest lags, and that is where the estimator reads.**
   The exponent `a = H − ½ = −0.38` makes `(t−s)^a` *singular* as `s → t`. Sampling
   at `j + 1` caps the kernel at `1.0` for the most recent shock; the cell mean is
   `1.613` and the exact treatment is larger still. Under-weighting the newest
   shock is precisely the loss of roughness: the increments come out with lag-1
   autocorrelation −0.18 instead of the −0.41 fractional Gaussian noise requires,
   so the structure function rises too fast at small `Δ` and the fitted slope is
   too high. The helper is asymptotically right — the 64–512 window slope is
   already down to 0.19 — but the fixture reads lags 1–60.
2. **It is not the type-II (non-stationary start) issue.** Dropping the first
   20 000 samples moves Ĥ from 0.2599 to 0.2603.
3. **Fixing the cell means is only half the fix.** The cell-mean / BLP-optimal
   kernel gets ρ(1) to −0.33 and Ĥ to 0.17. What closes it is treating the
   singular first cell *exactly* — simulating `∫ₜ₋₁ᵗ (t−s)^a dW_s` jointly with
   `ΔW` as a Gaussian pair with `Var = 1/(2a+1)`, `Cov = 1/(a+1)`. That is the
   Bennedsen–Lunde–Pakkanen hybrid scheme with κ = 1, and it gives 0.1207.

## Why this matters beyond the fixture

The same discretisation question reappears in Session D3, where the lifted
system is checked against a *direct* Volterra convolution. A direct convolution
built the way the helper was would itself be biased, and the lift would be
graded against a wrong reference. The D3 reference therefore convolves with the
**same sum-of-exponentials kernel** the lift uses (which must agree to rounding,
since the lift is an exact evaluation of that convolution) and separately reports
the kernel-approximation error against `K(t)`; no point-sampled fractional kernel
appears anywhere in the tests.

## Left open for your pass

- The hybrid κ = 1 slope drifts to 0.205 in the 512–4096 window at n = 2^17. With
  32 independent blocks at lag 4096 that is mostly sampling noise, but the
  type-II construction does have non-stationary increments and a long-lag bias
  of order `Δ/t`; whether that is visible here has not been separated out.
- BLP optimal points and the cell mean give *identical* numbers above. That is
  expected — `b_j*` is defined so that `(b_j*)^a` equals the cell mean of `u^a` —
  and is a check that the two were coded consistently, not two independent fixes.

## Session I note: the exact generator itself

The "Davies–Harte, exact fBm" rows above came from the fixture copy of the time, which
had two defects found in Session I against Shevchenko (2014, Sec. 6): a zero in the
middle of the circulant row instead of ρ(n−1), which only matters from H ≈ 0.93, and a
√2 scale slip that gave the noise variance 1/2. Neither changes a number in this note:
at H = 0.12 that embedding was nonnegative, so the draws had exactly the right
correlations, and Ĥ and ρ(1) do not depend on the scale. All exact fBm in the project now
comes from `models/fbm.py`, tested in `test_fbm.py`; `docs/comparison-fbm-methods.pdf`
has the full comparison.
