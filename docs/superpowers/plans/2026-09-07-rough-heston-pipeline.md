# Rough Heston Calibration Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the volatility-surface instrument, then build a calibration pipeline that fits vanilla Heston, lifted rough Heston, and a Hawkes jump-diffusion to the same surface and shows which one actually reproduces the short-maturity ATM skew.

**Architecture:** `volsurf_core.py` stays the pure-maths layer (no IO, no scipy, no matplotlib). Models go in a new `models/` package, each exposing one interface — `char_func(u, tau, params) -> complex` — so a single Fourier pricer and a single calibrator serve all three. Calibration is `surface -> params`; validation is always "plant known params, recover them."

**Tech Stack:** Python 3.12, numpy 2.5, matplotlib 3.11, ibapi 9.81.1. scipy is allowed inside `models/` and the calibrator (it is off the live startup path), but never in `volsurf_core.py`.

**Granularity note:** Session A is specified step-by-step. Sessions B–F are specified at task level with the exact mathematics, acceptance criteria and file paths, because in this project the maths is the hard part and the plumbing is not. Each session begins by expanding its tasks into steps.

---

## File Structure

```
volsurf_core.py                 (exists) pure maths, no scipy
volatility_surface_3.py         (exists) live app
sources/replay.py               NEW  record/replay a chain snapshot to JSON
fit/svi.py                      NEW  raw SVI + eSSVI slice fitting
models/__init__.py              NEW  CharFunc protocol
models/heston.py                NEW  vanilla Heston characteristic function
models/rough_heston.py          NEW  lifted rough Heston
models/hawkes.py                NEW  Hawkes jump diffusion
pricing/fourier.py              NEW  Carr-Madan + Lewis pricers
calibrate/objective.py          NEW  vega-weighted surface loss
calibrate/fit.py                NEW  Levenberg-Marquardt driver
filters/kalman.py               NEW  OU Kalman + dual (state & parameter) filter
test_svi.py, test_models.py, test_pricing.py, test_calibrate.py, test_filters.py
```

---

## Timeline

Six sessions across seven days. Each is a half-day to a day of focused work.

| Session | Day | Content | Done when |
|---|---|---|---|
| **A** | 1 | Finish the surface: arbitrage audit wired in, SVI slices, RND panel, second H estimator, replay source | Live view shows arb flags + RND; two independent H estimates agree on synthetic data |
| **B** | 2 | Heston characteristic function + Fourier pricer | Fourier price matches a Monte Carlo Heston price to 1e-3 |
| **C** | 3 | Calibrator; fit vanilla Heston to the surface | Planted Heston params recovered to <1% from a synthetic surface |
| **D** | 4–5 | Volterra kernel + Markovian lift; lifted rough Heston char. func. | Lifted kernel matches `t^(α-1)/Γ(α)` to <1% over 1d–2y; lifted char. func. → Heston as H→½ |
| **E** | 5–6 | Calibrate rough Heston; skew comparison vs Phase 1 | Rough Heston reproduces `τ^(H-½)` short-end skew; vanilla provably cannot |
| **F** | 7 | Hawkes branch + dual Kalman | Hawkes shows the second-spike-larger property; dual filter tracks a regime change |

Risk: Session D is the one that can overrun. The fractional Riccati solve is where rough Heston implementations usually break. If it slips, Session F is the one to drop — it is an optional branch by your own framing.

---

## Session A — Finish the surface

### Task A1: Wire the arbitrage audit into the live view

`butterfly_violations` and `calendar_violations` exist in `volsurf_core.py` and nothing calls them.

**Files:**
- Modify: `volatility_surface_3.py` (add `audit_surface`, call it in the plot loop)
- Test: `test_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_audit_flags_a_planted_violation():
    pts = {"E1": [{'z': z, 'k': 0.01 * z, 'w': 0.01 + 0.002 * z * z, 'iv': 0.2,
                   'weight': 1.0, 'strike': 650.0 + z} for z in range(-3, 4)]}
    pts["E1"][3]['w'] -= 0.004                      # dent the middle
    import volatility_surface_3 as v3
    flags = v3.audit_surface(pts, {"E1": type("C", (), {"tau": 0.02})()})
    assert flags["E1"]["butterfly"], "a dented slice must raise a butterfly flag"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python.exe -c "import test_core; test_core.test_audit_flags_a_planted_violation()"`
Expected: `AttributeError: module 'volatility_surface_3' has no attribute 'audit_surface'`

- [ ] **Step 3: Implement**

```python
def audit_surface(points, ctxs):
    """Butterfly per slice, calendar across slices at a common z. Flags, never fixes."""
    out = {}
    for exp, rows in points.items():
        ks = [r['k'] for r in rows]
        ws = [r['w'] for r in rows]
        out[exp] = {'butterfly': vc.butterfly_violations(ks, ws), 'calendar': []}
    exps = sorted(points, key=lambda e: ctxs[e].tau)
    for z0 in (-1.0, 0.0, 1.0):
        taus, ws = [], []
        for e in exps:
            rows = points[e]
            zs = np.array([r['z'] for r in rows])
            if len(zs) < 2 or z0 < zs.min() or z0 > zs.max():
                continue
            taus.append(ctxs[e].tau)
            ws.append(float(np.interp(z0, zs, [r['w'] for r in rows])))
        for i in vc.calendar_violations(taus, ws):
            out[exps[i]]['calendar'].append(z0)
    return out
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: same command. Expected: no assertion error.

- [ ] **Step 5: Render violations in the 3D title and as red markers on the front slice**

```python
        flags = audit_surface(pts, ctxs)
        n_bf = sum(len(f['butterfly']) for f in flags.values())
        n_cal = sum(len(f['calendar']) for f in flags.values())
        # append to the existing title string:
        #   f"  |  arb: {n_bf} butterfly, {n_cal} calendar"
        for i in flags[front]['butterfly']:
            ax_skew.plot(rows[i]['z'], rows[i]['iv'], 'x', color='#ff3e3e', ms=9, mew=2)
```

- [ ] **Step 6: Commit**

```bash
git add volatility_surface_3.py test_core.py
git commit -m "feat: wire butterfly and calendar arbitrage audit into the live surface"
```

### Task A2: SVI slice fitting

**Files:**
- Create: `fit/svi.py`, `test_svi.py`

Raw SVI in total variance:

$$w(k) = a + b\left[\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right]$$

Durrleman no-butterfly conditions to enforce: `b > 0`, `|ρ| < 1`, `a + b·σ·√(1−ρ²) ≥ 0`, `b(1+|ρ|) ≤ 4/τ`.

- [ ] **Step 1: Test — recover planted SVI parameters**

```python
def test_svi_round_trip():
    true = dict(a=0.004, b=0.10, rho=-0.7, m=0.01, sigma=0.12)
    k = np.linspace(-0.4, 0.4, 41)
    w = svi.raw_svi(k, **true)
    got = svi.fit_slice(k, w, weights=np.ones_like(k), tau=0.25)
    for key in true:
        assert abs(got[key] - true[key]) < 1e-4, f"{key}: {got[key]} vs {true[key]}"
```

- [ ] **Step 2: Run, confirm it fails (module missing).**
- [ ] **Step 3: Implement `raw_svi`, `fit_slice` (least squares with the Durrleman constraints as bounds), `essvi_fit` across slices.**
- [ ] **Step 4: Run, confirm pass.**
- [ ] **Step 5: Add a test that a fitted SVI slice has non-negative Breeden-Litzenberger density on a dense uniform grid.**
- [ ] **Step 6: Commit.**

### Task A3: RND panel

**Files:** Modify `volatility_surface_3.py`.

Differentiate the **fitted SVI slice** on a dense **uniform** strike grid, not the raw quotes — `fd_second` loses an order on non-uniform grids, which is documented in `volsurf_core.py`.

- [ ] Steps: test that a fitted slice yields a density integrating to 1.0 ± 1e-3 and non-negative everywhere; add the panel; commit.

### Task A4: Second H estimator on the panel

`hurst_from_structure` exists and nothing calls it. It reads the realised log-vol path; the skew estimator reads the option surface. Show both.

- [ ] Steps: feed a realised-vol series into the panel, display `Ĥ_skew` and `Ĥ_structure` side by side with a divergence warning when they differ by more than 0.05; commit.

### Task A5: Replay source

**Files:** Create `sources/replay.py`.

- [ ] Steps: `record(app, path)` dumps `ChainSnapshot` to JSON; `replay(path)` yields it back; round-trip test asserts byte-identical surface points; commit.

---

## Session B — Heston characteristic function and Fourier pricer

### Task B1: Heston characteristic function

**Files:** Create `models/heston.py`, `test_models.py`.

Model: `dS/S = √v dW₁`, `dv = κ(θ−v)dt + ξ√v dW₂`, `d⟨W₁,W₂⟩ = ρ dt`.

$$\phi(u) = \exp\Big(iu\log F + \frac{\kappa\theta}{\xi^2}\big[(\kappa-\rho\xi iu - d)\tau - 2\log\tfrac{1-g e^{-d\tau}}{1-g}\big] + \frac{v_0}{\xi^2}(\kappa-\rho\xi iu-d)\tfrac{1-e^{-d\tau}}{1-ge^{-d\tau}}\Big)$$

with `d = √((ρξiu−κ)² + ξ²(iu+u²))` and **`g = (κ−ρξiu−d)/(κ−ρξiu+d)`**.

**Critical:** use that form of `g`, not its reciprocal. The reciprocal form is the "little Heston trap" — it crosses the branch cut of the complex log and produces discontinuous prices for `τ` beyond roughly a year. This is the single most common bug in Heston implementations.

- [ ] Test: `φ(0) = 1` exactly for 20 random parameter sets.
- [ ] Test: `φ(-i) = F` (the martingale condition).
- [ ] Test: `φ` is continuous in `τ` on a 500-point grid to 2 years — `max|Δφ|` between neighbouring `τ` below 1e-6. **This is the test that catches the trap.**
- [ ] Test: as `ξ → 0`, the Heston price → Black-76 with `σ = √v₀` to 1e-6.
- [ ] Commit.

### Task B2: Fourier pricer

**Files:** Create `pricing/fourier.py`, `test_pricing.py`.

Carr-Madan with damping `α = 1.5`:

$$C(k) = \frac{e^{-\alpha k}}{\pi}\int_0^\infty \Re\Big[e^{-ivk}\frac{e^{-r\tau}\phi(v-(\alpha+1)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}\Big]dv$$

Also implement the Lewis form as an independent cross-check.

- [ ] Test: Carr-Madan and Lewis agree to 1e-8 on the same parameters.
- [ ] Test: Fourier Heston price matches a 400k-path Monte Carlo (full truncation Euler) inside its 95% MC confidence interval, across 5 strikes and 3 maturities.
- [ ] Test: put-call parity holds on the Fourier prices to 1e-9.
- [ ] Commit.

---

## Session C — Calibration

### Task C1: Objective

**Files:** Create `calibrate/objective.py`.

Fit in **implied vol**, weighted by `quote_weight` (already in `volsurf_core.py`), not in price — price errors are dominated by ATM contracts and ignore the wings entirely.

$$\text{loss}(\theta)=\sum_{i} w_i\big(\sigma^{\text{model}}_i(\theta)-\sigma^{\text{mkt}}_i\big)^2$$

- [ ] Test: loss is exactly 0 when the model params equal the params that generated the surface.
- [ ] Test: loss is strictly positive and increases monotonically as one parameter is perturbed away.
- [ ] Commit.

### Task C2: Levenberg-Marquardt driver

**Files:** Create `calibrate/fit.py`.

- [ ] Test: **plant-and-recover.** Generate a surface from `(v₀, κ, θ, ξ, ρ) = (0.04, 2.0, 0.045, 0.5, −0.7)`, calibrate from a deliberately poor start, assert every parameter recovered to <1%.
- [ ] Test: recovery still within 5% with 0.5 vol-point noise added to the surface.
- [ ] Test: Feller condition `2κθ > ξ²` reported (not enforced — real fits often violate it, and hiding that is worse than showing it).
- [ ] Commit.

### Task C3: Fit to the v3 surface, record the residual

- [ ] Fit vanilla Heston to the `capture_v3` synthetic surface, which has a planted `H = 0.12`.
- [ ] **Record the ATM skew residual by maturity.** This is the Phase 1 baseline and the whole point of Phase 2: Heston's ATM skew flattens as `τ → 0`, so the short end must misfit. Save the numbers to `docs/phase1_baseline.md`.
- [ ] Commit.

---

## Session D — Volterra kernel and the Markovian lift

### Task D1: Fractional kernel and its completely-monotone representation

**Files:** Create `models/rough_heston.py`.

$$K(t)=\frac{t^{\alpha-1}}{\Gamma(\alpha)},\qquad \alpha=H+\tfrac12$$

The lift rests on the Laplace representation

$$K(t)=\int_0^\infty e^{-xt}\,\mu(dx),\qquad \mu(dx)=\frac{x^{-\alpha}}{\Gamma(\alpha)\Gamma(1-\alpha)}dx$$

(check: `∫₀^∞ e^{-xt}x^{-α}dx = Γ(1-α)t^{α-1}`, so the constants cancel to `t^{α-1}/Γ(α)`).

- [ ] Test: numerically integrate `μ` and confirm it reproduces `K(t)` to 1e-8 for `t ∈ [1/365, 2]`.
- [ ] Commit.

### Task D2: Sum-of-exponentials approximation

$$K(t)\approx\sum_{i=1}^{N}w_i e^{-x_i t}$$

Choose nodes/weights by partitioning `μ` geometrically (Abi Jaber–El Euch): with breakpoints `η₀ < η₁ < … < η_N` spaced geometrically over `[1/T, 1/Δt]`,

$$w_i=\int_{\eta_{i-1}}^{\eta_i}\mu(dx),\qquad x_i=\frac{1}{w_i}\int_{\eta_{i-1}}^{\eta_i}x\,\mu(dx)$$

- [ ] Test: `N = 20` reproduces `K(t)` to <1% relative over `t ∈ [1/365, 2]`.
- [ ] Test: error decreases monotonically in `N` for `N ∈ {5, 10, 20, 40}`.
- [ ] Test: the Laplace transforms match — `Σ wᵢ/(z+xᵢ)` vs `z^{-α}` to <1% for `z ∈ [0.1, 100]`.
- [ ] Commit.

### Task D3: The lift itself

Exactly your derivation from `IMG_6122`/`IMG_6113`:

$$U_i(t)=\int_0^t e^{-x_i(t-s)}dZ_s \;\Longrightarrow\; dU_i=-x_iU_i\,dt+dZ_t,\qquad V_t=V_0+\sum_i w_iU_i(t)$$

and the semigroup step that makes each factor Markovian:

$$U_i(t+\Delta t)=e^{-x_i\Delta t}U_i(t)+\int_t^{t+\Delta t}e^{-x_i(t+\Delta t-s)}dZ_s$$

- [ ] Test: simulate the lifted system and the direct Volterra convolution from the same Brownian path; assert the paths agree to <1% in `L²` at `N = 20`.
- [ ] Test: `U_i` alone is an OU process — its autocorrelation matches `e^{-x_i Δ}`.
- [ ] Commit.

### Task D4: Lifted rough Heston characteristic function

The lift turns the fractional Riccati equation into an `N`-dimensional system of ordinary Riccati ODEs, one per factor, solved with RK4:

$$\partial_t\psi_i = -x_i\psi_i + \tfrac12(u^2+iu) + (\rho\xi iu-\kappa)\Psi + \tfrac{\xi^2}{2}\Psi^2,\qquad \Psi=\sum_i w_i\psi_i$$

- [ ] Test: **as `H → 0.5`, the lifted characteristic function converges to vanilla Heston** to 1e-4. This is the single most important test in the session — it proves the lift is a generalisation and not a different model.
- [ ] Test: `φ(0) = 1`, `φ(-i) = F`.
- [ ] Test: RK4 step-halving changes the price by <1e-6 (convergence).
- [ ] Commit.

---

## Session E — Calibrate rough Heston, and the comparison that matters

### Task E1: Calibrate

- [ ] Reuse `calibrate/fit.py` unchanged — this is why all models share `char_func`.
- [ ] Test: plant-and-recover on `(v₀, κ, θ, ξ, ρ, H)`; `H` recovered to ±0.02.
- [ ] Commit.

### Task E2: The Phase 1 vs Phase 2 comparison

- [ ] Fit both to the same surface.
- [ ] Plot ATM skew vs `τ` on log-log for: market, vanilla Heston, rough Heston.
- [ ] **Assert in a test** that rough Heston's short-end skew slope is within 0.05 of the planted `H − ½`, and that vanilla Heston's is not — it should flatten toward 0.
- [ ] Produce the figure. This is the headline result of the whole project.
- [ ] Commit.

---

## Session F — Hawkes branch and dual Kalman

### Task F1: Hawkes jump diffusion

**Files:** Create `models/hawkes.py`.

$$\lambda(t)=\mu+\sum_{t_i<t}\alpha e^{-\beta(t-t_i)}$$

- [ ] Test: unconditional intensity is `μ/(1−α/β)`; simulated mean rate matches to 2%.
- [ ] Test: stationarity requires `α < β`; the constructor rejects `α ≥ β`.
- [ ] Test: **your second-spike property.** Two clustered regime shocks must produce a second intensity peak strictly higher than the first, and a constant-λ Poisson process with the same average rate must not. This is the claim that motivates the model, so it gets asserted, not assumed.
- [ ] Test: excess kurtosis of Hawkes returns exceeds that of Poisson returns at equal average jump rate.
- [ ] Commit.

### Task F2: Dual Kalman filter

**Files:** Create `filters/kalman.py`.

State-space from the lecture notes:

`F = e^{−κΔt}`, `B = (1−e^{−κΔt})`, `u = θ`, `Q = (σ²/2κ)(1−e^{−2κΔt})`, `K = P/(P+R)`.

- [ ] Test: on a simulated OU path with known `(κ, θ, σ)`, the filter's RMSE beats the raw observation RMSE by >30%.
- [ ] Test: large `R` → small `K` → the filter tracks the model; small `R` → large `K` → it tracks the data. Assert both directions numerically.
- [ ] Test: **dual filter** — with parameters unknown, the parameter filter converges to the true `κ` within 10% over 2000 steps.
- [ ] Test: at a planted regime change, an adaptive-`R` filter re-converges in fewer steps than a fixed-`R` one.
- [ ] Commit.

---

## Self-Review

**Spec coverage.** Volterra kernels → D1–D2. Markovian lifting → D3. Laplace-transform node selection → D2. Discretisation of the non-Markovian process → D3–D4. Rough Heston → D4/E1. Hawkes mean-reverting jump diffusion → F1. Kalman and dual Kalman → F2. Model vs parameterisation → C2/E1 (plant-and-recover separates the two). Structure function and `ζ(q) = qH` → already shipped in `volsurf_core.py`, panel in A4. Phase 1/2/3 → Sessions B–C / D–E / F. Timeline → above.

**Placeholders.** None. Every task states its acceptance test in words precise enough to code directly; Session A carries literal code.

**Type consistency.** All three models expose `char_func(u, tau, params) -> complex`; `pricing/fourier.py` and `calibrate/fit.py` take that one interface. `quote_weight` and `atm_skew_from_slice` are used with the signatures currently in `volsurf_core.py` (the latter returns a 3-tuple).

**Known risk.** Session D, task D4. If the fractional Riccati solve misbehaves, drop Session F rather than compress D — F is the optional branch.
