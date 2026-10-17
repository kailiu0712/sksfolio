"""Support-restricted linear algebra for a constrained-prox dual Newton step.

Solve (regularization I + tau K J K.T) direction = rhs, where
J = diag(d) - h h.T / den is the generalized perspective-prox derivative.
The positive-derivative support can be much smaller than the active row count.
No active rows or components of the right-hand side are discarded.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def _factor(ka, d, h, den, tau):
    """Return U with U U.T = tau K J K.T, without an n-by-n array."""
    support = np.flatnonzero(d > 0.0)
    if not len(support):
        return np.empty((ka.shape[0], 0)), support
    sqrt_d = np.sqrt(d[support])
    u = np.asarray(ka[:, support] * sqrt_d[None, :], order='F')
    u *= np.sqrt(tau)
    if den > 0.0:
        # J = sqrt(D)(I-bb.T)sqrt(D).  Its symmetric square root uses
        # I - bb.T/(1 + sqrt(1-||b||^2)), including the singular ||b||=1 case.
        b = h[support] / sqrt_d / np.sqrt(den)
        norm2 = float(b @ b)
        if norm2 > 1.0 + 1e-9:
            raise ValueError('The supplied proximal Jacobian is not positive semidefinite')
        coefficient = 1.0 / (1.0 + np.sqrt(max(0.0, 1.0 - norm2)))
        u -= coefficient * np.outer(u @ b, b)
    return u, support


def inner_diagonal(ka, d, h, den, tau):
    """Compute diag(tau K J K.T) for selecting the existing regularizer."""
    support = np.flatnonzero(d > 0.0)
    if not len(support):
        return np.zeros(ka.shape[0])
    kp = ka[:, support]
    diagonal = tau * ((kp * kp) @ d[support])
    if den > 0.0:
        kh = kp @ h[support]
        diagonal -= (tau / den) * kh * kh
    return np.maximum(diagonal, 0.0)


def _ridge_solve(gram, regularization, rhs):
    matrix = gram.copy()
    matrix.flat[::len(matrix)+1] += regularization
    try:
        result = linalg.cho_solve(linalg.cho_factor(matrix, lower=True,
                                                  check_finite=False),
                                 rhs, check_finite=False)
        return result, False
    except linalg.LinAlgError:
        # A ridge smaller than floating-point resolution can disappear on
        # the diagonal.  Retain it explicitly in this spectral fallback.
        eigenvalues, vectors = linalg.eigh(gram, check_finite=False)
        coefficients = vectors.T @ rhs
        denominator = np.maximum(eigenvalues, 0.0) + regularization
        coefficients = (coefficients / denominator if rhs.ndim == 1
                        else coefficients / denominator[:, None])
        return vectors @ coefficients, True


def solve_inner(ka, d, h, den, tau, regularization, rhs):
    """Return a direction and linear-algebra diagnostics.

    If positive primal derivatives are fewer than active exposure rows, a
    thin QR factorization reduces the ridge solve to that support dimension.
    The orthogonal right-hand-side component is divided by the ridge rather
    than dropped, preserving the original regularized Newton equation.
    Otherwise form and factor the smaller active-row Gram matrix.
    """
    if regularization <= 0.0 or tau <= 0.0:
        raise ValueError('tau and regularization must be positive')
    m = ka.shape[0]
    rhs = np.asarray(rhs)
    if m == 0:
        return rhs.copy(), dict(method='empty', support_size=0,
                                system_dimension=0, spectral_fallback=False)
    u, support = _factor(ka, np.asarray(d), np.asarray(h), den, tau)
    r = len(support)
    if r == 0:
        return rhs / regularization, dict(method='zero_support', support_size=0,
                                          system_dimension=0, spectral_fallback=False)
    if r < m:
        q, triangular = linalg.qr(u, mode='economic', check_finite=False)
        coordinates = q.T @ rhs
        parallel, fallback = _ridge_solve(triangular @ triangular.T,
                                          regularization, coordinates)
        orthogonal = rhs - q @ coordinates
        direction = q @ parallel + orthogonal / regularization
        method = 'support_qr'
        dimension = r
    else:
        direction, fallback = _ridge_solve(u @ u.T, regularization, rhs)
        method = 'active_cholesky'
        dimension = m
    return direction, dict(method=method, support_size=r,
                           system_dimension=dimension, spectral_fallback=fallback)


def run_checks():
    """Verify both branches, zero support, and a singular prox derivative."""
    rng = np.random.default_rng(9071)
    max_direction_error = 0.0
    max_relative_residual = 0.0
    cases = 0
    for n, m, support_size in ((30, 20, 5), (30, 7, 20),
                                (30, 20, 0), (100, 40, 12)):
        for coupling in (0.0, 0.4, 0.999999, 1.0):
            for _ in range(5):
                ka = rng.normal(size=(m, n))
                d, h = np.zeros(n), np.zeros(n)
                p = rng.choice(n, support_size, replace=False)
                d[p] = rng.uniform(0.1, 1.0, support_size)
                den = 0.0
                if coupling and support_size:
                    h[p] = rng.normal(size=support_size) * np.sqrt(d[p])
                    den = float(np.sum(h[p]**2 / d[p]) / coupling)
                tau = float(np.exp(rng.uniform(-3, 1)))
                regularization = float(np.exp(rng.uniform(-8, -2)))
                rhs = rng.normal(size=m)
                gram = tau * ((ka*d) @ ka.T)
                if den > 0.0:
                    kh = ka @ h
                    gram -= tau * np.outer(kh, kh) / den
                matrix = gram + regularization*np.eye(m)
                expected = linalg.solve(matrix, rhs, assume_a='pos')
                actual, _ = solve_inner(ka, d, h, den, tau, regularization, rhs)
                direction_error = np.linalg.norm(actual-expected) / max(1.0, np.linalg.norm(expected))
                residual = np.linalg.norm(matrix@actual-rhs) / max(1.0, np.linalg.norm(rhs))
                max_direction_error = max(max_direction_error, float(direction_error))
                max_relative_residual = max(max_relative_residual, float(residual))
                assert direction_error < 2e-8, direction_error
                assert residual < 2e-8, residual
                cases += 1
    return dict(cases=cases, max_relative_direction_error=max_direction_error,
                max_relative_linear_residual=max_relative_residual)


if __name__ == '__main__':
    print(run_checks())
