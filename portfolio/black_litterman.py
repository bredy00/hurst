"""
Black-Litterman (Black & Litterman 1992; He & Litterman 1999), the only portfolio rule the
trial uses.

  equilibrium   Pi = delta_eq Sigma w_mkt: the returns that make the market portfolio optimal
                for an investor of risk aversion delta_eq. It is "pre-determined by the market":
                caps and a long-run covariance fix it, and delta_eq only scales it.
  posterior     mu_BL = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P' Omega^-1 Q]
                computed in the equivalent update form Pi + tau Sigma P' (P tau Sigma P' +
                Omega)^-1 (Q - P Pi), which inverts only the k x k view matrix, and
                M = [(tau Sigma)^-1 + P' Omega^-1 P]^-1, the posterior uncertainty of the mean.
  confidence    Omega = diag(P tau Sigma P') / c (He & Litterman's choice, scaled): c = 1 weighs
                each view like the prior, c -> 0 ignores the views, c -> inf imposes them.
  weights       w = (delta (Sigma_opt + M))^-1 mu_BL, unconstrained, the rest in cash; Sigma_opt
                may differ from the Sigma that set Pi (the trial's short-run risk forecast).
"""

import numpy as np


def equilibrium(Sigma, w_mkt, delta_eq):
    return delta_eq * Sigma @ w_mkt


def omega(P, Sigma, tau, confidence):
    return np.diag(np.diag(P @ (tau * Sigma) @ P.T)) / confidence


def posterior(Pi, Sigma, P, Q, Omega, tau):
    """(mu_BL, M) in the update form."""
    tS = tau * Sigma
    if P is None or len(P) == 0:
        return Pi.copy(), tS
    A = P @ tS @ P.T + Omega
    K = np.linalg.solve(A, P @ tS).T                  # tS P' A^-1 (A, tS symmetric)
    mu = Pi + K @ (Q - P @ Pi)
    M = tS - K @ P @ tS
    return mu, 0.5 * (M + M.T)


def posterior_information_form(Pi, Sigma, P, Q, Omega, tau):
    """The textbook form, with the N x N inverses; kept to test the update form against."""
    iT = np.linalg.inv(tau * Sigma)
    iO = np.linalg.inv(Omega)
    M = np.linalg.inv(iT + P.T @ iO @ P)
    return M @ (iT @ Pi + P.T @ iO @ Q), M


def weights(mu, Sigma_opt, M, delta):
    return np.linalg.solve(delta * (Sigma_opt + M), mu)
