"""Independent first-order solvers for the perspective root relaxation.

The PDHG modes use the Condat--Vu forward primal-dual splitting, since the
quadratic objective is treated by a gradient step.  The adaptive mode uses
residual balancing and optional ergodic restarts for a *finite* burn-in only.
After burn-in it is the ordinary fixed-step method.  ALM uses an accelerated
proximal-gradient inner solve with the standard gradient-based restart.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import eigh

try:
    from .core import evaluate, prox
except ImportError:
    from .core import evaluate, prox


def _norm_squared(a):
    """Compute the spectral norm squared once, including its setup cost."""
    if a.size == 0:
        return 0.0
    gram = a @ a.T if a.shape[0] <= a.shape[1] else a.T @ a
    largest = eigh(gram, subset_by_index=[len(gram) - 1, len(gram) - 1],
                   eigvals_only=True, check_finite=False)[0]
    return max(0.0, float(largest)) * (1.0 + 1e-12)


def _initial_x(problem):
    x = np.minimum(problem.u, np.full(problem.n, 1.0 / problem.n))
    budget = np.sum(x / problem.u)
    if budget > problem.k:
        x *= problem.k / budget
    return x


def _metric(metrics, tol, feas_tol):
    feas = float(metrics.get("feasibility", math.inf))
    gap = float(metrics.get("rel_gap", math.inf))
    if not (np.isfinite(feas) and np.isfinite(gap)):
        return math.inf
    return max(feas / feas_tol, abs(gap) / tol)


class _Monitor:
    def __init__(self, problem, start, time_limit, tol, feas_tol):
        self.problem = problem
        self.start = start
        self.deadline = start + time_limit
        self.tol = tol
        self.feas_tol = feas_tol
        self.best_merit = math.inf
        self.best = None
        self.history = []
        self.certificates = 0

    def expired(self):
        return time.perf_counter() >= self.deadline

    def check(self, x, y, iteration):
        metrics = evaluate(self.problem, x, y)
        merit = _metric(metrics, self.tol, self.feas_tol)
        self.certificates += 1
        self.history.append({"iteration": iteration,
                             "seconds": time.perf_counter() - self.start,
                             "feasibility": float(metrics.get("feasibility", math.inf)),
                             "rel_gap": float(metrics.get("rel_gap", math.inf))})
        if self.best is None or merit < self.best_merit:
            self.best_merit = merit
            self.best = (x.copy(), y.copy(), metrics.copy(), iteration)
        return merit <= 1.0, merit

    def finish(self, x, y, iteration, status, diagnostics):
        # A final certificate makes the returned iterate reproducibly auditable.
        self.check(x, y, iteration)
        bx, by, metrics, selected_iteration = self.best
        if self.best_merit <= 1.0:
            status = "converged"
        diagnostics.update({"seconds": time.perf_counter() - self.start,
                            "certificates": self.certificates,
                            "selected_iteration": selected_iteration,
                            "history": self.history})
        return {"x": bx, "y": by, "iterations": iteration, "status": status,
                "metrics": metrics, "diagnostics": diagnostics,
                "seconds": diagnostics["seconds"]}


def solve(problem, mode="pdhg", time_limit=20.0, tol=1e-6,
          feas_tol=1e-7, max_iter=100000, rho=None):
    """Solve using a certificate rather than a reference solver's objective.

    ``rho`` is an optional primal/dual step-balance multiplier for the two
    PDHG modes.  For ALM it is the initial penalty.  The adaptive PDHG rule is
    explicitly a practical burn-in heuristic, not a claim of a new adaptive
    convergence theorem.  All continuing fixed steps satisfy the Condat--Vu
    sufficient condition tau*L/2 + tau*sigma*||K||^2 < 1.
    """
    if mode not in {"pdhg", "adaptive_pdhg", "alm_apg"}:
        raise ValueError(f"Unknown first-order mode {mode!r}")
    if rho is not None and rho <= 0:
        raise ValueError("rho must be positive")
    start = time.perf_counter()
    monitor = _Monitor(problem, start, time_limit, tol, feas_tol)
    smooth_lipschitz = _norm_squared(problem.B)
    constraint_norm_sq = _norm_squared(problem.K)
    x = _initial_x(problem)
    y = np.zeros(problem.K.shape[0])
    setup_seconds = time.perf_counter() - start
    common = {"mode": mode, "smooth_lipschitz": smooth_lipschitz,
              "constraint_norm_squared": constraint_norm_sq,
              "setup_seconds": setup_seconds}
    if mode == "alm_apg":
        return _alm_apg(problem, x, y, smooth_lipschitz, constraint_norm_sq,
                        monitor, max_iter, rho, common)
    return _pdhg(problem, x, y, smooth_lipschitz, constraint_norm_sq,
                 monitor, max_iter, rho, mode == "adaptive_pdhg", common)


def _pdhg(p, x, y, lf, k2, monitor, max_iter, rho, adaptive, diagnostics):
    # This leaves at least 0.055 slack in the sufficient convergence condition.
    tau_max = 0.99 / max(lf, 1e-14)
    tau = min(tau_max, (1.0 if rho is None else rho) *
              0.99 / max(lf + math.sqrt(k2), 1e-14))
    sigma = 0.45 / (tau * k2) if k2 > 0 else 1.0
    initial_tau, initial_sigma = tau, sigma
    # Finite adaptation makes all infinite tails conventional fixed-step CV.
    burnin_limit = min(2000, max_iter // 5)
    balance_updates = 0
    restarts = 0
    ave_x = np.zeros_like(x)
    ave_y = np.zeros_like(y)
    ave_count = 0
    status = "iteration_limit"
    iteration = 0
    gradient = p.B.T @ (p.B @ x) + p.c
    if monitor.check(x, y, 0)[0]:
        return monitor.finish(x, y, 0, "converged", diagnostics)
    for iteration in range(1, max_iter + 1):
        if iteration % 10 == 1 and monitor.expired():
            status = "time_limit"
            break
        old_x, old_y, old_gradient = x, y, gradient
        x = prox(old_x - tau * (old_gradient + p.K.T @ old_y), tau,
                 p.gamma, p.u, p.k)[0]
        q = old_y + sigma * (p.K @ (2.0 * x - old_x))
        y = q - sigma * np.clip(q / sigma, p.lo, p.hi)
        gradient = p.B.T @ (p.B @ x) + p.c
        if adaptive and iteration <= burnin_limit:
            ave_x += x
            ave_y += y
            ave_count += 1
        if iteration % 25 == 0:
            done, current_merit = monitor.check(x, y, iteration)
            if done:
                status = "converged"
                break
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                status = "numerical_failure"
                break
        if adaptive and iteration <= burnin_limit and iteration % 100 == 0:
            dx, dy = old_x - x, old_y - y
            rp = dx / tau + gradient - old_gradient - p.K.T @ dy
            rd = dy / sigma - p.K @ dx
            # Euclidean residual balance in the row-scaled problem.
            nrp, nrd = np.linalg.norm(rp), np.linalg.norm(rd)
            factor = np.clip(math.sqrt((nrp + 1e-15) / (nrd + 1e-15)),
                             0.5, 2.0)
            new_tau = np.clip(tau * factor, initial_tau * 1e-4, tau_max)
            if abs(math.log(new_tau / tau)) > 1e-3:
                sigma *= tau / new_tau
                tau = new_tau
                balance_updates += 1
            # Both points get the same independent primal-dual certificate.
            # Restarts are restricted to burn-in, just like step adaptation.
            if ave_count and iteration % 500 == 0:
                averaged_x, averaged_y = ave_x / ave_count, ave_y / ave_count
                done, averaged_merit = monitor.check(averaged_x, averaged_y, iteration)
                if done:
                    x, y = averaged_x, averaged_y
                    status = "converged"
                    break
                if averaged_merit < 0.8 * current_merit:
                    x, y = averaged_x.copy(), averaged_y.copy()
                    gradient = p.B.T @ (p.B @ x) + p.c
                    restarts += 1
                ave_x.fill(0.0)
                ave_y.fill(0.0)
                ave_count = 0
    diagnostics.update({"tau_initial": initial_tau, "sigma_initial": initial_sigma,
                        "tau_final": tau, "sigma_final": sigma,
                        "cv_condition": tau * lf / 2 + tau * sigma * k2,
                        "adaptation_iterations": burnin_limit if adaptive else 0,
                        "balance_updates": balance_updates, "restarts": restarts})
    return monitor.finish(x, y, iteration, status, diagnostics)


def _alm_apg(p, x, y, lf, k2, monitor, max_iter, rho, diagnostics):
    penalty = (max(1e-3, lf / max(k2, 1e-12)) if rho is None else float(rho))
    initial_penalty = penalty
    penalty_ceiling = penalty * 1e7
    tolerance_scale = max(lf, np.linalg.norm(p.c, ord=np.inf), 1e-3)
    inner_start_tol = 0.01 * tolerance_scale
    previous_feasibility = math.inf
    status = "iteration_limit"
    total = 0
    outer = 0
    restarts = 0
    last_y_candidate = y.copy()
    raw_k = p.raw_K if p.raw_K is not None else p.K
    raw_lo = p.raw_lo if p.raw_lo is not None else p.lo
    raw_hi = p.raw_hi if p.raw_hi is not None else p.hi
    if monitor.check(x, y, 0)[0]:
        return monitor.finish(x, y, 0, "converged", diagnostics)
    while total < max_iter:
        if monitor.expired():
            status = "time_limit"
            break
        outer += 1
        lipschitz = max(lf + penalty * k2, 1e-14)
        step = 0.99 / lipschitz
        # A geometric forcing schedule is summable and reaches certificate
        # accuracy without requiring hundreds of multiplier updates.
        inner_tol = max(1e-11, inner_start_tol * 0.5**min(outer, 60))
        extrapolated = x.copy()
        acceleration = 1.0
        # The limit prevents spending the whole time on an early inner solve.
        inner_limit = min(max_iter - total, max(1000, 500 * outer))
        for inner in range(inner_limit):
            if total % 10 == 0 and monitor.expired():
                status = "time_limit"
                break
            k_extrapolated = p.K @ extrapolated
            shifted = k_extrapolated + y / penalty
            penalty_gradient = penalty * (shifted - np.clip(shifted, p.lo, p.hi))
            grad = p.B.T @ (p.B @ extrapolated) + p.c + p.K.T @ penalty_gradient
            new_x = prox(extrapolated - step * grad, step, p.gamma, p.u, p.k)[0]
            mapping_norm = np.linalg.norm((extrapolated - new_x) / step, ord=np.inf)
            next_acceleration = (1 + math.sqrt(1 + 4 * acceleration**2)) / 2
            if np.dot(extrapolated - new_x, new_x - x) > 0:
                next_acceleration = 1.0
                restarts += 1
            next_extrapolated = new_x + ((acceleration - 1) / next_acceleration) * (new_x - x)
            if next_acceleration == 1.0:
                next_extrapolated = new_x.copy()
            x, extrapolated, acceleration = new_x, next_extrapolated, next_acceleration
            total += 1
            if total % 25 == 0:
                shifted_x = p.K @ x + y / penalty
                last_y_candidate = penalty * (shifted_x - np.clip(shifted_x, p.lo, p.hi))
                if monitor.check(x, last_y_candidate, total)[0]:
                    status = "converged"
                    break
            if mapping_norm <= inner_tol:
                break
        shifted_x = p.K @ x + y / penalty
        new_y = penalty * (shifted_x - np.clip(shifted_x, p.lo, p.hi))
        last_y_candidate = new_y
        if status in {"converged", "time_limit"}:
            break
        done, _ = monitor.check(x, new_y, total)
        if done:
            status = "converged"
            break
        raw_values = raw_k @ x
        feasibility = np.linalg.norm(raw_values - np.clip(raw_values, raw_lo, raw_hi), ord=np.inf)
        y = new_y
        # Once feasibility is adequate, further penalty growth only worsens
        # conditioning.  In particular, never react to roundoff stagnation.
        if (outer >= 2 and mapping_norm <= inner_tol
                and feasibility > max(0.25 * monitor.feas_tol, 2.0 * inner_tol)
                and feasibility > 0.7 * previous_feasibility):
            penalty = min(2.0 * penalty, penalty_ceiling)
        previous_feasibility = feasibility
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            status = "numerical_failure"
            break
    diagnostics.update({"outer_iterations": outer, "inner_iterations": total,
                        "penalty_initial": initial_penalty, "penalty_final": penalty,
                        "acceleration_restarts": restarts})
    return monitor.finish(x, last_y_candidate, total, status, diagnostics)
