# Reference papers

## Shevchenko (2014), *Fractional Brownian motion in a nutshell*

`shevchenko-2014-fbm-in-a-nutshell.pdf` — arXiv:1406.1956, 14 pages, retrieved
2026-09-10 from `https://arxiv.org/pdf/1406.1956`.
Extended lecture notes from a mini-course at the 7th Jagna International Workshop.

### What it is not

**It does not cover Markovian lifting.** There is no lifting, no sum-of-exponentials
approximation, no rough volatility and no finance beyond a passing mention in the
introduction. It is an introductory survey: definition and basic properties,
continuity, integral representations, Hurst estimation, simulation.

The Markovian lift is a different literature — Abi Jaber & El Euch for the
multifactor approximation, El Euch & Rosenbaum for the rough Heston characteristic
function, Harms & Stefanovits for affine representations of fractional processes.
Those still need retrieving before Session D's task D4.

### What it does give us — three things

**1. Theorem 4.3, the Volterra-type representation.** A kernel with *compact
support*, unlike Mandelbrot–van Ness (Thm 4.1, moving-average, support on the whole
line) or the harmonizable form (Thm 4.2). For `H > 1/2`:

```
k_t^V(x) = K_H^V x^(1/2-H) Int_x^t s^(H-1/2) (s-x)^(H-3/2) ds  ·  1_[0,t](x)
K_H^V = ( H(2H-1) / B(2-2H, H-1/2) )^(1/2) = K_H^MA
```

with a longer expression for `H < 1/2`.

**Note this is NOT the kernel we are lifting.** Ours is the Riemann–Liouville /
rough-Heston kernel `K(t) = t^(alpha-1)/Gamma(alpha)` with `alpha = H + 1/2`.
Shevchenko's is the Molchan–Golosov kernel, which reproduces fBm *exactly*
(same covariance). They share the `H - 1/2` power and the compact support, and
both are Volterra, but they are different objects and the distinction matters
when quoting a result. Ours is the simpler one and is what rough Heston uses.

**2. Section 5, a third and provably consistent H estimator.** We currently have
two routes — the ATM skew log-log slope and the structure function. This is a
third, from *filtered variations*:

- a filter `a(x) = sum a_k x^k` of order `r` has `a(1) = a'(1) = ... = a^(r-1)(1) = 0`
- filtered series `B_n^a = sum_k a_k B_(n+k)`, which is stationary
- dilated filter `a^m(x) = a(x^m)` gives `rho^(a^m)(0) = m^(2H) rho^a(0)`, hence
  `log rho^(a^m)(0) = 2H log m + log rho^a(0)`
- regress `log V_N^(a^m)` on `log m` across `m in M`; then `H_hat = slope / 2`
- the simplest instance, increments-1 with `M = {1,2}`:
  `H_hat = (1/2) log2( V_N^(d^2) / V_N^d )`

Three properties worth having:

- **Strong consistency** (Thm 5.1 + Cor 5.2), straight from the ergodic theorem.
- **Asymptotic normality iff `r > H + 1/4`** (Thm 5.5, via Breuer–Major). Our
  estimators are effectively `r = 1`, which is valid for `H < 3/4` — fine at our
  `H ~ 0.1`, and it means our error bars are legitimate. Above `H = 3/4` you need
  a second-order filter (Increments-2, `a(x) = (x-1)^2`) or the CLT fails.
- **Scale invariance** (Rmk 5.3). An unknown multiplicative constant `c` on the
  observations only adds `log c` to the intercept and does not move the slope.
  That matters for us: our realised log-vol series carries an arbitrary scale.

Popular filters he lists: Increments 1 `a(x) = x-1` (order 1), Daubechies 4
(order 1), Increments 2 `a(x) = (x-1)^2` (order 2).

**3. Section 6, the Wood–Chan circulant method** — the exact algorithm already
used in our test fixtures. Confirms the construction and adds two practical points:

- take `N = 2^q + 1` so `M = 2^(q+1)` is a power of two and the FFT is efficient;
- the eigenvalues should come out real and positive, so take the real part.

**This produced a real finding in our code.** He states the embedding for fBm is
positive definite, so every eigenvalue is positive. Checked directly: true across
our working range (minimum eigenvalue `+2.8e-05` at `H = 0.10`) but **false as
`H -> 1`** — at `H = 0.99` the minimum is `-0.76`. Our fixtures were flooring the
eigenvalues at zero with `np.maximum(lam, 0)`, which silently returns a path that
is not fBm. The generators now raise instead. Wood–Chan's own approximate fallback
would be needed to simulate that regime; we do not need it at `H ~ 0.1`.

### Bibliography he points to

Nourdin's lecture notes; Nualart's book chapter; Mishura (pathwise integration);
Biagini et al (white noise approach); Coeurjolly (statistical methods and
simulation).
