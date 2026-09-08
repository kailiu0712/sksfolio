"""Experimental regularized active-set Newton steps for the split prox dual.

The Hessian uses the diagonal-minus-rank-one derivative of the PAVA pool.
Regularization is applied to the Newton system, never to the objective or
constraints. Dependent interval rows are retained.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve


def pava_jacobian_parts(values, point, gamma, k):
    """Return d, pool and beta such that J=diag(d)-beta*pool*pool.T.

This is a generalized derivative away from pool transitions; at a transition
the selected adjacent piece supplies a valid limiting derivative.
"""
    n = len(values)
    top = np.zeros(n, dtype=bool)
    top[np.argpartition(values, n-k)[n-k:]] = True
    interior = (point > 0.0) & (point < 1.0)
    singleton = np.minimum(np.maximum(values, 0.0)/(1.0+gamma), 1.0)
    pool = interior & (~top | (np.abs(point-singleton) > 1e-12))
    diagonal = np.zeros(n)
    diagonal[interior & top & ~pool] = 1.0/(1.0+gamma)
    diagonal[pool] = 1.0
    count = int(pool.sum())
    if not count:
        return diagonal, pool, 0.0
    boundary = float(np.mean(values[pool]-point[pool]))
    beta = (gamma/(gamma*count+np.count_nonzero(pool & top))
            if boundary < gamma else 1.0/count)
    return diagonal, pool, beta


def minimize_split_dual(fun, initial, *, bounds, hessian, pgtol, maxiter,
                        maxls=40, **unused):
    """L-BFGS-compatible experimental projected Newton driver."""
    x = initial.copy()
    bounded = np.array([lo is not None for lo, hi in bounds])
    value, gradient = fun(x)
    evaluations = 1
    status = 'iteration limit'
    for iteration in range(maxiter):
        projected = gradient.copy()
        projected[bounded & (x <= 0) & (gradient > 0)] = 0
        norm = float(np.max(np.abs(projected)))
        if norm <= pgtol:
            status = 'CONVERGENCE: projected gradient'
            break
        free = ~bounded | (x > 0) | (gradient < 0)
        indices = np.flatnonzero(free)
        matrix = hessian(indices)
        ridge = max(1e-12, min(1e-3, norm)*1e-3)*max(float(np.max(np.diag(matrix))), 1e-6)
        matrix.flat[::len(indices)+1] += ridge
        direction = np.zeros_like(x)
        try:
            direction[indices] = solve(matrix, -gradient[indices], assume_a='pos', check_finite=False)
        except np.linalg.LinAlgError:
            direction = -projected
        accepted = False
        # Safeguard with a projected gradient step when Newton is unsuitable.
        for attempt in range(2):
            if attempt:
                direction = -projected/max(float(np.max(np.diag(matrix))), 1e-6)
            step = 1.0
            for search in range(maxls):
                candidate = x + step*direction
                candidate[bounded] = np.maximum(candidate[bounded], 0.0)
                delta = candidate-x
                slope = float(gradient @ delta)
                if slope < 0:
                    next_value, next_gradient = fun(candidate)
                    evaluations += 1
                    if next_value <= value + 1e-4*slope + 1e-15*max(1.0, abs(value)):
                        accepted = True
                        break
                step *= 0.5
            if accepted:
                break
        if not accepted:
            status = 'line search stalled'
            break
        x, value, gradient = candidate, next_value, next_gradient
    return x, value, {'task': status, 'nit': iteration+1, 'funcalls': evaluations}
