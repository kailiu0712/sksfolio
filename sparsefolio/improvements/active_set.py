"""Reduced-support semismooth primal-dual KKT polishing.

A rough custom ALM solve identifies a small primal support.  This module then
solves the joint primal-dual fixed-point equations, with dense linear systems
restricted to that support and the currently active exposure rows.  The
independent core.evaluate check remains the only successful stopping test.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from scipy import linalg

from ..core import evaluate
try:
    from .fast_core import prox
except ImportError:
    from ..core import prox
from .alm_fast import solve as warm_solve


def _project_domain(y, lo, hi):
    y = y.copy()
    y[np.isneginf(lo)] = np.maximum(y[np.isneginf(lo)], 0)
    y[np.isposinf(hi)] = np.minimum(y[np.isposinf(hi)], 0)
    return y


def polish(p, x, y, time_limit=2.0, tol=1e-6, feas_tol=1e-7,
           max_iter=100, config=None):
    """Try to solve the common certificate from a custom approximate iterate."""
    config = config or {}
    start = time.perf_counter()
    deadline = start + time_limit
    x, y = x.copy(), y.copy()
    B, K = p.B, p.K
    n, m = p.n, len(p.lo)
    lf = max(float(linalg.eigvalsh(B @ B.T, check_finite=False)[-1]), 1e-12)
    tau = config.get('tau_factor', 1.0) / lf
    eq = p.lo == p.hi
    counts = dict(kkt_solves=0, backtracks=0, gradient_fallbacks=0,
                  max_support=0, max_active=0, prox_calls=0)
    history = []
    best = None
    best_merit = np.inf

    def point(xx, yy):
        q = xx - tau * (B.T @ (B @ xx) + p.c + K.T @ yy)
        px, z, theta, d, h, den = prox(q, tau, p.gamma, p.u, p.k)
        kx = K @ xx
        primal = xx - px
        dual = kx - np.clip(kx + yy, p.lo, p.hi)
        merit = float(primal @ primal / tau**2 + dual @ dual)
        active = eq | (kx + yy < p.lo) | (kx + yy > p.hi)
        counts['prox_calls'] += 1
        return px, primal, dual, d, h, den, active, merit

    current = point(x, y)
    status = 'iteration_limit'
    regularization_scale = config.get('regularization', 1e-7)
    for iteration in range(max_iter + 1):
        px, rp, rd, d, h, den, active, merit = current
        metrics = evaluate(p, px, y)
        score = max(metrics['feasibility'] / feas_tol, metrics['rel_gap'] / tol)
        history.append(dict(iteration=iteration, seconds=time.perf_counter()-start,
                            residual=merit, **metrics))
        if best is None or score < best_merit:
            best = (px.copy(), y.copy(), metrics)
            best_merit = score
        if score <= 1:
            status = 'converged'
            break
        if time.perf_counter() >= deadline:
            status = 'time_limit'
            break
        if iteration == max_iter:
            break
        support = d > 1e-12
        pp = np.flatnonzero(support)
        aa = np.flatnonzero(active)
        counts['max_support'] = max(counts['max_support'], len(pp))
        counts['max_active'] = max(counts['max_active'], len(aa))
        if not len(pp):
            status = 'empty_support'
            break
        # A zero proximal derivative can mean either zero weight or an asset
        # fixed at its upper cap, so use the residual rather than assuming zero.
        dx = -rp.copy()
        dx[pp] = 0.0
        dy = rd.copy()
        dy[aa] = 0.0
        ds, hs = d[pp], h[pp]
        jp = np.diag(ds)
        if den > 0:
            jp -= np.outer(hs, hs) / den
        bp = B[:, pp]
        kap = K[np.ix_(aa, pp)]
        hp = bp.T @ bp
        matrix = np.empty((len(pp)+len(aa), len(pp)+len(aa)))
        matrix[:len(pp), :len(pp)] = np.eye(len(pp)) - jp + tau * (jp @ hp)
        matrix[:len(pp), len(pp):] = tau * (jp @ kap.T)
        matrix[len(pp):, :len(pp)] = kap
        matrix[len(pp):, len(pp):] = 0.0
        rhs = np.concatenate((-rp[pp] - tau * (jp @ (bp.T @ (B @ dx) + K[:, pp].T @ dy)),
                              -rd[aa] - K[aa] @ dx))
        # This regularization is a numerical safeguard for temporary rank loss
        # when guessed active rows outnumber the current primal support.
        reg = regularization_scale * max(1.0, np.sqrt(merit))
        matrix[np.arange(len(pp)), np.arange(len(pp))] += reg
        ii = np.arange(len(pp), len(pp)+len(aa))
        matrix[ii, ii] -= reg
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', linalg.LinAlgWarning)
            try:
                solution = linalg.solve(matrix, rhs, check_finite=False)
            except linalg.LinAlgError:
                solution = linalg.lstsq(matrix, rhs, cond=1e-12, check_finite=False)[0]
        counts['kkt_solves'] += 1
        dx[pp], dy[aa] = solution[:len(pp)], solution[len(pp):]
        norm = max(np.linalg.norm(dx), np.linalg.norm(dy))
        radius = 10.0 * (1.0 + np.linalg.norm(x) + np.linalg.norm(y))
        if norm > radius:
            dx *= radius / norm
            dy *= radius / norm
        alpha = 1.0
        accepted = False
        for _ in range(25):
            xt = x + alpha * dx
            yt = _project_domain(y + alpha * dy, p.lo, p.hi)
            trial = point(xt, yt)
            if trial[-1] <= (1 - 1e-4*alpha)*merit:
                x, y, current = xt, yt, trial
                accepted = True
                break
            alpha *= 0.5
            counts['backtracks'] += 1
        if not accepted:
            # A short explicit Condat--Vu fallback changes the guessed face.
            # Its conservative norm bound avoids a second spectral setup.
            k2 = float(np.sum(K*K))
            sigma = 0.45 / max(tau*k2, 1e-12)
            for _ in range(25):
                oldx = x
                x = prox(x-tau*(B.T@(B@x)+p.c+K.T@y), tau, p.gamma, p.u, p.k)[0]
                q = y + sigma * (K @ (2*x-oldx))
                y = q-sigma*np.clip(q/sigma,p.lo,p.hi)
            counts['gradient_fallbacks'] += 1
            current = point(x, y)
    bx, by, metrics = best
    counts.update(seconds=time.perf_counter()-start, selected_merit=best_merit)
    return dict(x=bx, y=by, status=status, iterations=iteration,
                metrics=metrics, diagnostics=counts, history=history,
                seconds=counts['seconds'])


def solve(problem, mode='active_set', time_limit=15.0, tol=1e-6,
          feas_tol=1e-7, max_iter=100, config=None):
    """Custom ALM warmup followed by a reduced-support joint Newton solve."""
    if mode != 'active_set':
        raise ValueError('mode must be active_set')
    config = config or {}
    start = time.perf_counter()
    warm = warm_solve(problem, time_limit=time_limit,
                      tol=config.get('warm_tol', 0.05),
                      feas_tol=config.get('warm_feas_tol', 1e-3),
                      config=config.get('warm_config', {'certificate_stride': 5}))
    elapsed = time.perf_counter()-start
    if elapsed >= time_limit:
        warm['diagnostics']['warm_seconds'] = elapsed
        warm['diagnostics']['mode'] = mode
        warm['status'] = 'time_limit'
        return warm
    result = polish(problem, warm['x'], warm['y'], time_limit-elapsed, tol,
                    feas_tol, max_iter, config)
    result['diagnostics'].update(warm_seconds=elapsed,
                                 warm_diagnostics=warm['diagnostics'], mode=mode)
    result['seconds'] = time.perf_counter()-start
    result['diagnostics']['seconds'] = result['seconds']
    return result
