"""
Discrete hedging on the simulated rough-Heston paths (Session J).

Hedged Monte Carlo -- Potters, Bouchaud & Sestovic (2001, Physica A 289), after the
physicists' treatment of option hedging in Bouchaud & Potters (Theory of Financial Risk
and Derivative Pricing, 2003): hedging is derived in discrete time, as the strategy
that minimises the variance of the hedger's wealth balance over each rebalancing
interval,

    min over C_k, phi_k of  E[ (C_{k+1} - C_k(z_k) - phi_k(z_k) (S_{k+1} - S_k))^2 ],

solved backwards from the payoff with C_k and phi_k expanded on basis functions of the
state z_k -- Follmer-Schweizer local risk minimisation, estimated by regression on
simulated paths. When returns are Gaussian and the interval shrinks, phi is the
Black-Scholes delta; with stochastic volatility the market is incomplete and the
minimal-variance hedge also carries the correlation between the price and the volatility
(for rough Heston, between S and the forward variance curve), and leaves a residual that
no rebalancing removes.

Here the state is z = (S, sigma_hat), sigma_hat = sqrt(E[int_t^T V ds | F_t] / (T - t)):
the model's expected average volatility to maturity, affine in the lifted factors and
exact (forward_variance_coefficients). Regression targets are the realised hedged cash
flows from the next date on (as in Longstaff-Schwartz, so regression errors do not
compound backwards), and the fitted rules are applied to an independent set of paths.
"""

import math

import numpy as np
from scipy.special import ndtr

import models.rough_heston as rh

SQRT_2PI = math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------ simulation
def forward_variance_coefficients(st, taus):
    """
    E[int_t^{t+tau} V ds | y_t] = A(tau) + B(tau) @ y_t, exactly, for each tau in taus:
    the integrated-variance moments of LiftedAffineStep (EI_const, EI_lin) at h = tau.
    The eigen-coordinates y do not depend on h, so one stepper serves every horizon.
    """
    lam, c = st.lam, st.c
    kt = st.kappa * (st.theta - st.v0)
    taus = np.asarray(taus, float)
    A = np.zeros(len(taus))
    B = np.zeros((len(taus), len(c)))
    for j, tau in enumerate(taus):
        if tau <= 0.0:
            continue
        p1, p2 = rh._phi12_nonpos(-lam * tau)
        A[j] = st.v0 * tau + kt * float((c * c) @ (tau * tau * p2))
        B[j] = c * tau * p1
    return A, B


def simulate_paths(p, T, n_steps, n_paths, seed=0):
    """
    Lifted rough Heston under the pricing measure, positivity-preserving QE step with the
    martingale correction: the forward S (S_0 = 1, zero rates), the spot variance V and the
    forward variance to maturity FV = E[int_t^T V ds | F_t] at every step, shape
    (n_steps + 1, n_paths).
    """
    rng = np.random.default_rng(seed)
    w, x = rh.lift_nodes(p.H)
    h = T / n_steps
    st = rh.LiftedAffineStep(w, x, p.v0, p.kappa, p.theta, p.xi, h, p.rho)
    taus = T - h * np.arange(n_steps + 1)
    A, B = forward_variance_coefficients(st, taus)
    y = np.zeros((len(w), n_paths))
    X = np.zeros(n_paths)
    srho = math.sqrt(1.0 - p.rho * p.rho)
    S = np.empty((n_steps + 1, n_paths))
    V = np.empty_like(S)
    FV = np.empty_like(S)
    S[0], V[0], FV[0] = 1.0, p.v0, A[0]
    for k in range(n_steps):
        y, Vk, dI, dZ, fix = st.step_y(y, rng, with_log_price=True)
        X += -0.5 * dI + p.rho * dZ + srho * np.sqrt(dI) * rng.standard_normal(n_paths) + fix
        S[k + 1] = np.exp(X)
        V[k + 1] = Vk
        FV[k + 1] = np.maximum(A[k + 1] + B[k + 1] @ y, 0.0)
    return {"t": h * np.arange(n_steps + 1), "tau": taus, "S": S, "V": V, "FV": FV, "T": T}


# ------------------------------------------------------------------ Black-Scholes, zero rates
def bs_call(S, K, sigma, tau):
    st = np.maximum(sigma * np.sqrt(np.maximum(tau, 0.0)), 1e-12)
    d1 = np.log(S / K) / st + 0.5 * st
    return S * ndtr(d1) - K * ndtr(d1 - st)


def bs_delta(S, K, sigma, tau):
    st = np.maximum(sigma * np.sqrt(np.maximum(tau, 0.0)), 1e-12)
    return ndtr(np.log(S / K) / st + 0.5 * st)


def model_sigma(paths, k):
    """sigma_hat at step k: the model's expected average volatility to maturity."""
    tau = paths["tau"][k]
    return np.sqrt(np.maximum(paths["FV"][k], 1e-12) / max(tau, 1e-12))


# ------------------------------------------------------------------ hedged Monte Carlo
def _features(S, sig, tau, K):
    """Value basis psi and hedge basis chi at one date, from the BS price and greeks at sigma_hat."""
    st = np.maximum(sig * math.sqrt(max(tau, 1e-12)), 1e-6)
    m = np.log(S / K) / st
    d1 = m + 0.5 * st
    pdf = np.exp(-0.5 * d1 * d1) / SQRT_2PI
    price = S * ndtr(d1) - K * ndtr(d1 - st)
    delta = ndtr(d1)
    vega = S * pdf * math.sqrt(max(tau, 1e-12))
    one = np.ones_like(S)
    psi = np.column_stack([one, price, delta, vega, vega * m, vega * m * m])
    chi = np.column_stack([one, delta, pdf, pdf * m, pdf * m * m, pdf * sig])
    return psi, chi


def _lstsq(X, y):
    """Least squares with column scaling; constant columns are dropped (all paths share z_0)."""
    sd = X.std(axis=0)
    keep = sd > 1e-12 * max(float(np.abs(X).max()), 1e-300)
    keep[0] = True                                   # the constant
    Xs = X[:, keep] / np.where(sd[keep] > 0, sd[keep], 1.0)
    coef_s, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    coef = np.zeros(X.shape[1])
    coef[keep] = coef_s / np.where(sd[keep] > 0, sd[keep], 1.0)
    return coef


def hmc_fit(paths, K, every):
    """
    Hedged Monte Carlo on training paths, rebalancing every `every` steps: coefficients of
    C_k and phi_k at each rebalancing date, and the price C_0.
    """
    n = len(paths["t"]) - 1
    dates = list(range(0, n, every))
    S = paths["S"]
    Y = np.maximum(S[n] - K, 0.0)                    # realised hedged cash flow from the next date on
    coefs = {}
    for k in reversed(dates):
        k2 = min(k + every, n)
        dS = S[k2] - S[k]
        psi, chi = _features(S[k], model_sigma(paths, k), paths["tau"][k], K)
        X = np.hstack([psi, chi * dS[:, None]])
        c = _lstsq(X, Y)
        a, b = c[:psi.shape[1]], c[psi.shape[1]:]
        phi = chi @ b
        coefs[k] = (a, b)
        Y = Y - phi * dS
    C0 = float(np.mean(Y))
    return {"dates": dates, "coefs": coefs, "C0": C0, "C0_se": float(np.std(Y) / math.sqrt(len(Y)))}


def hedge_error(paths, K, every, premium, rule, fit=None, sigma_fixed=None):
    """
    Terminal hedging error of the option writer: premium + sum_k phi_k dS_k - payoff, on
    `paths`, rebalancing every `every` steps. rule: "bs_fixed" (Black-Scholes delta at
    sigma_fixed), "bs_model" (BS delta at sigma_hat), "hmc" (the fitted risk-minimising rule).
    """
    n = len(paths["t"]) - 1
    S = paths["S"]
    gains = np.zeros(S.shape[1])
    for k in range(0, n, every):
        k2 = min(k + every, n)
        tau = paths["tau"][k]
        if rule == "bs_fixed":
            phi = bs_delta(S[k], K, sigma_fixed, tau)
        elif rule == "bs_model":
            phi = bs_delta(S[k], K, model_sigma(paths, k), tau)
        elif rule == "hmc":
            _, chi = _features(S[k], model_sigma(paths, k), tau, K)
            phi = chi @ fit["coefs"][k][1]
        else:
            raise ValueError(rule)
        gains += phi * (S[k2] - S[k])
    return premium + gains - np.maximum(S[n] - K, 0.0)
