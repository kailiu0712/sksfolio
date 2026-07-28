"""Backtracking FISTA for linearly constrained perspective relaxations.

The backend solves

    minimize 0.5 * ||B.T @ x||^2 - rho * mu.T @ x + omega * G_k(x)
    subject to lower <= C @ x <= upper.

The interval rows are kept inside the proximal step.  The exact budget-only
case uses the specialized scalar Brent/PAVA oracle.  All other row structures
use a warm-started dual FISTA oracle whose primitive is the same PAVA prox.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from ..pdhg.safe_dual import SafeDualEvaluator
from ..problem import evaluate_solution, perspective_value
from .linear_prox import LinearConstraintProx


@dataclass(frozen=True)
class _Settings:
    tolerance: float
    feasibility_tolerance: float
    time_limit: Optional[float]
    max_iterations: int
    history_interval: int
    initial_lipschitz: Optional[float]
    backtracking_factor: float
    step_growth: float
    line_search_tolerance: float
    max_backtracks: int
    restart_strategy: str
    restart_period: int
    restart_eta: float
    restart_check_interval: int
    restart_hinder_lubin_beta: float
    restart_function_tolerance: float
    norm_iterations: int
    prox_tolerance: float
    prox_max_iterations: int
    pava_backend: str
    prox_oracle: str
    prox_adaptive_restart: bool
    majorization_max_iterations: int
    majorization_polish: bool
    majorization_max_lift_variables: int
    dual_bound_cutoff: Optional[float]


def _as_options(options: Optional[Any]) -> Dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    if hasattr(options, "__dict__"):
        return {
            key: value
            for key, value in vars(options).items()
            if not key.startswith("_")
        }
    raise TypeError("options must be a mapping, object, or None")


def _settings(options: Optional[Any]) -> Tuple[_Settings, int]:
    supplied = _as_options(options)
    tolerance = float(supplied.get("tolerance", 1e-6))
    feasibility_tolerance = float(
        supplied.get(
            "feasibility_tolerance",
            max(tolerance, 1e-8),
        )
    )
    raw_limit = supplied.get("time_limit")
    time_limit = None if raw_limit is None else float(raw_limit)
    max_iterations = int(supplied.get("max_iterations", 10_000))
    history_interval = int(
        supplied.get(
            "history_interval",
            supplied.get("check_interval", 25),
        )
    )
    raw_lipschitz = supplied.get("initial_lipschitz")
    initial_lipschitz = (
        None if raw_lipschitz is None else float(raw_lipschitz)
    )
    backtracking_factor = float(
        supplied.get("backtracking_factor", 2.0)
    )
    step_growth = float(supplied.get("step_growth", 1.1))
    line_search_tolerance = float(
        supplied.get("line_search_tolerance", 1e-12)
    )
    max_backtracks = int(supplied.get("max_backtracks", 60))
    raw_restart_strategy = supplied.get("restart_strategy")
    if raw_restart_strategy is None:
        restart_strategy = (
            "gradient"
            if bool(supplied.get("adaptive_restart", True))
            else "none"
        )
    else:
        restart_strategy = str(raw_restart_strategy).lower().replace(
            "-",
            "_",
        )
    restart_strategy = {
        "off": "none",
        "disabled": "none",
        "adaptive": "gradient",
        "odonoghue": "gradient",
        "odonoghue_gradient": "gradient",
        "objective": "function",
        "function_value": "function",
        "fixed": "periodic",
        "fixed_period": "periodic",
        "hinder": "hinder_lubin",
        "distance": "hinder_lubin",
        "distance_potential": "hinder_lubin",
        "gap": "primal_dual_gap",
        "primal_dual": "primal_dual_gap",
        "duality_gap": "primal_dual_gap",
    }.get(restart_strategy, restart_strategy)
    restart_period = int(supplied.get("restart_period", 25))
    restart_eta = float(
        supplied.get("restart_eta", math.exp(2.0))
    )
    restart_check_interval = int(
        supplied.get("restart_check_interval", 1)
    )
    restart_hinder_lubin_beta = float(
        supplied.get("restart_hinder_lubin_beta", 0.25)
    )
    restart_function_tolerance = float(
        supplied.get("restart_function_tolerance", 0.0)
    )
    norm_iterations = int(supplied.get("norm_iterations", 20))
    prox_tolerance = float(supplied.get("prox_tolerance", 1e-8))
    prox_max_iterations = int(
        supplied.get("prox_max_iterations", 1000)
    )
    pava_backend = str(
        supplied.get("pava_backend", "partial_sort")
    ).lower().replace("-", "_")
    pava_backend = {
        "full": "full_sort",
        "partial": "partial_sort",
        "topk": "partial_sort",
    }.get(pava_backend, pava_backend)
    prox_oracle = str(
        supplied.get("prox_oracle", "auto")
    ).lower().replace("-", "_")
    prox_oracle = {
        "general": "dual_fista",
        "dual": "dual_fista",
        "apg": "dual_fista",
        "budget_scalar": "budget",
        "majorization": "majorization_qp",
        "qp": "majorization_qp",
    }.get(prox_oracle, prox_oracle)
    prox_adaptive_restart = bool(
        supplied.get("prox_adaptive_restart", True)
    )
    majorization_max_iterations = int(
        supplied.get("majorization_max_iterations", 20_000)
    )
    majorization_polish = bool(
        supplied.get("majorization_polish", True)
    )
    majorization_max_lift_variables = int(
        supplied.get("majorization_max_lift_variables", 5_000_000)
    )
    raw_cutoff = supplied.get("dual_bound_cutoff")
    dual_bound_cutoff = (
        None if raw_cutoff is None else float(raw_cutoff)
    )
    threads = int(supplied.get("threads", 0))

    positive = {
        "tolerance": tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "backtracking_factor": backtracking_factor,
        "step_growth": step_growth,
        "prox_tolerance": prox_tolerance,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if time_limit is not None and (
        not math.isfinite(time_limit) or time_limit <= 0.0
    ):
        raise ValueError("time_limit must be positive and finite")
    if initial_lipschitz is not None and (
        not math.isfinite(initial_lipschitz)
        or initial_lipschitz <= 0.0
    ):
        raise ValueError(
            "initial_lipschitz must be positive and finite"
        )
    if backtracking_factor <= 1.0:
        raise ValueError("backtracking_factor must exceed one")
    if step_growth < 1.0:
        raise ValueError("step_growth must be at least one")
    if (
        not math.isfinite(line_search_tolerance)
        or line_search_tolerance < 0.0
    ):
        raise ValueError(
            "line_search_tolerance must be finite and nonnegative"
        )
    for name, value in (
        ("max_iterations", max_iterations),
        ("history_interval", history_interval),
        ("max_backtracks", max_backtracks),
        ("norm_iterations", norm_iterations),
        ("prox_max_iterations", prox_max_iterations),
        ("majorization_max_iterations", majorization_max_iterations),
        (
            "majorization_max_lift_variables",
            majorization_max_lift_variables,
        ),
        ("restart_period", restart_period),
        ("restart_check_interval", restart_check_interval),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if pava_backend not in {"full_sort", "partial_sort"}:
        raise ValueError(
            "pava_backend must be 'full_sort' or 'partial_sort'"
        )
    if prox_oracle not in {
        "auto",
        "pava",
        "budget",
        "dual_fista",
        "majorization_qp",
    }:
        raise ValueError(
            "prox_oracle must be 'auto', 'pava', 'budget', "
            "'dual_fista', or 'majorization_qp'"
        )
    if restart_strategy not in {
        "none",
        "gradient",
        "function",
        "periodic",
        "hinder_lubin",
        "primal_dual_gap",
    }:
        raise ValueError(
            "restart_strategy must be 'none', 'gradient', 'function', "
            "'periodic', 'hinder_lubin', or 'primal_dual_gap'"
        )
    if not math.isfinite(restart_eta) or restart_eta <= 1.0:
        raise ValueError("restart_eta must be finite and exceed one")
    if (
        not math.isfinite(restart_hinder_lubin_beta)
        or not 0.0 < restart_hinder_lubin_beta < 1.0
    ):
        raise ValueError(
            "restart_hinder_lubin_beta must lie in (0, 1)"
        )
    if (
        not math.isfinite(restart_function_tolerance)
        or restart_function_tolerance < 0.0
    ):
        raise ValueError(
            "restart_function_tolerance must be finite and nonnegative"
        )
    if dual_bound_cutoff is not None and not math.isfinite(
        dual_bound_cutoff
    ):
        raise ValueError("dual_bound_cutoff must be finite")
    if threads < 0:
        raise ValueError("threads must be nonnegative")

    return (
        _Settings(
            tolerance=tolerance,
            feasibility_tolerance=feasibility_tolerance,
            time_limit=time_limit,
            max_iterations=max_iterations,
            history_interval=history_interval,
            initial_lipschitz=initial_lipschitz,
            backtracking_factor=backtracking_factor,
            step_growth=step_growth,
            line_search_tolerance=line_search_tolerance,
            max_backtracks=max_backtracks,
            restart_strategy=restart_strategy,
            restart_period=restart_period,
            restart_eta=restart_eta,
            restart_check_interval=restart_check_interval,
            restart_hinder_lubin_beta=restart_hinder_lubin_beta,
            restart_function_tolerance=restart_function_tolerance,
            norm_iterations=norm_iterations,
            prox_tolerance=prox_tolerance,
            prox_max_iterations=prox_max_iterations,
            pava_backend=pava_backend,
            prox_oracle=prox_oracle,
            prox_adaptive_restart=prox_adaptive_restart,
            majorization_max_iterations=majorization_max_iterations,
            majorization_polish=majorization_polish,
            majorization_max_lift_variables=(
                majorization_max_lift_variables
            ),
            dual_bound_cutoff=dual_bound_cutoff,
        ),
        threads,
    )


def _matvec(matrix: Any, value: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ value, dtype=float).reshape(-1)


def _smooth_value(
    B: Any,
    mu: np.ndarray,
    return_reward: float,
    x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    factor = _matvec(B.T, x)
    value = (
        0.5 * float(factor @ factor)
        - return_reward * float(mu @ x)
    )
    return value, factor


def _gradient(
    B: Any,
    mu: np.ndarray,
    return_reward: float,
    factor: np.ndarray,
) -> np.ndarray:
    return _matvec(B, factor) - return_reward * mu


def _estimate_lipschitz(
    B: Any,
    dimension: int,
    rank: int,
    iterations: int,
    perspective_weight: float,
) -> Tuple[float, str]:
    gram_work = dimension * rank * rank
    if rank <= 256 and gram_work <= 50_000_000:
        gram = B.T @ B
        if sparse.issparse(gram):
            gram = gram.toarray()
        gram = np.asarray(gram, dtype=float)
        estimate = float(np.linalg.eigvalsh(gram)[-1])
        kind = "exact_factor_gram"
    else:
        rng = np.random.default_rng(2849)
        vector = rng.standard_normal(rank)
        vector /= max(float(np.linalg.norm(vector)), 1e-300)
        estimate = 0.0
        for _ in range(iterations):
            image = _matvec(B.T, _matvec(B, vector))
            norm = float(np.linalg.norm(image))
            if norm == 0.0:
                estimate = 0.0
                break
            vector = image / norm
            estimate = float(vector @ _matvec(B.T, _matvec(B, vector)))
        estimate *= 1.05
        kind = "factor_power_estimate"

    # A positive trial curvature is also required when the smooth term is
    # affine.  This floor keeps the PAVA scale in a conservative range.
    floor = max(1e-12, perspective_weight / 1e6)
    return max(estimate, floor), kind


def _maximum_violation(instance: Any, x: np.ndarray) -> float:
    values = _matvec(instance.C, x)
    lower = np.asarray(instance.lower, dtype=float).reshape(-1)
    upper = np.asarray(instance.upper, dtype=float).reshape(-1)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    linear = 0.0
    if np.any(finite_lower):
        linear = max(
            linear,
            float(np.max(lower[finite_lower] - values[finite_lower])),
        )
    if np.any(finite_upper):
        linear = max(
            linear,
            float(np.max(values[finite_upper] - upper[finite_upper])),
        )
    return max(
        linear,
        0.0,
        -float(np.min(x)),
        float(np.max(x)) - 1.0,
        float(np.sum(x)) - float(instance.k),
    )


def _prox(
    argument: np.ndarray,
    gamma: float,
    oracle: Any,
    tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    result = oracle.solve(
        argument,
        gamma,
        tolerance=tolerance,
    )
    if hasattr(result, "constraint_multiplier"):
        point = result.x
        multiplier = result.constraint_multiplier
        pava_calls = result.pava_calls
        constraint_violation = result.constraint_violation
        scaled_violation = result.scaled_constraint_violation
        fixed_point_residual = result.fixed_point_residual
        restarts = result.restarts
        inner_backtracks = result.line_search_backtracks
        warm_started = result.warm_started
        method = result.method
        prox_converged = bool(result.converged)
    else:
        point = result.x
        multiplier = result.multiplier
        if (
            point is None
            or multiplier is None
            or not result.has_solution
        ):
            raise RuntimeError(
                "majorization QP prox returned no finite solution "
                f"(status={result.status})"
            )
        pava_calls = 0
        constraint_violation = (
            math.inf
            if result.linear_constraint_violation is None
            else result.linear_constraint_violation
        )
        scaled_violation = constraint_violation
        residuals = [
            value
            for value in (
                result.primal_residual,
                result.dual_residual,
                result.maximum_constraint_violation,
            )
            if value is not None
        ]
        fixed_point_residual = (
            max(residuals) if residuals else math.inf
        )
        restarts = 0
        inner_backtracks = 0
        warm_started = result.warm_start_used
        method = oracle.method
        residual_tolerance = max(10.0 * tolerance, 1e-8)
        feasibility_tolerance = max(tolerance, 1e-9)
        prox_converged = bool(
            result.converged
            and fixed_point_residual <= residual_tolerance
            and result.maximum_constraint_violation is not None
            and result.maximum_constraint_violation
            <= feasibility_tolerance
        )
    return (
        np.asarray(point, dtype=float).reshape(-1),
        {
            "constraint_multiplier": np.asarray(
                multiplier,
                dtype=float,
            ).reshape(-1),
            "iterations": int(result.iterations),
            "pava_calls": int(pava_calls),
            "constraint_violation": float(constraint_violation),
            "scaled_constraint_violation": float(scaled_violation),
            "fixed_point_residual": float(fixed_point_residual),
            "converged": prox_converged,
            "oracle": method,
            "inner_lipschitz": float(
                getattr(result, "lipschitz", 0.0)
            ),
            "restarts": int(restarts),
            "line_search_backtracks": int(inner_backtracks),
            "warm_started": bool(warm_started),
        },
    )


def _dual_candidate(
    evaluator: SafeDualEvaluator,
    factor: np.ndarray,
    constraint_multiplier: np.ndarray,
) -> Dict[str, Any]:
    return evaluator.evaluate(
        factor,
        constraint_multiplier,
    )


def _solve(
    instance: Any,
    settings: _Settings,
    prox_oracle: Any,
    dual_evaluator: SafeDualEvaluator,
    setup_seconds: float,
    initial_lipschitz: float,
    lipschitz_kind: str,
) -> Dict[str, Any]:
    B = instance.B
    mu = np.asarray(instance.mu, dtype=float).reshape(-1)
    omega = float(instance.perspective_weight)
    return_reward = float(instance.return_reward)
    k = int(instance.k)
    x = np.asarray(instance.anchor, dtype=float).reshape(-1).copy()
    y = x.copy()
    momentum = 1.0
    lipschitz = initial_lipschitz
    minimum_lipschitz = lipschitz
    maximum_lipschitz = lipschitz

    initial_factor = _matvec(B.T, x)
    initial_certificate = _dual_candidate(
        dual_evaluator,
        initial_factor,
        np.zeros(int(instance.rows), dtype=float),
    )
    initial_dual = initial_certificate.get("dual_bound")
    best_dual = (
        -math.inf if initial_dual is None else float(initial_dual)
    )
    best_certificate = initial_certificate
    last_dual = initial_dual
    maximum_dual_correction = float(
        initial_certificate.get("dual_domain_correction", 0.0)
    )
    dual_evaluations = 1
    initial_diagnostics = evaluate_solution(instance, x)
    initial_objective = initial_diagnostics.get("objective")
    objective_evaluations = 1
    previous_objective = (
        None
        if initial_objective is None
        else float(initial_objective)
    )
    restart_epoch_gap = (
        float(initial_objective) - float(initial_dual)
        if initial_objective is not None and initial_dual is not None
        else None
    )
    restart_initial_gap = restart_epoch_gap
    restart_iterations: list[int] = []
    restart_reasons: list[str] = []
    restart_metrics: list[Optional[float]] = []
    restart_thresholds: list[Optional[float]] = []
    restart_epochs = 0
    restart_checks = 0
    restart_dual_evaluations = 0
    restart_objective_evaluations = 0
    restart_bookkeeping_seconds = 0.0
    hinder_epoch_start = x.copy()
    hinder_previous_displacement: Optional[float] = None
    hinder_previous_length: Optional[int] = None
    restart_epoch_age = 0

    history: list[Dict[str, Any]] = []
    status = "iteration_limit"
    completed = 0
    backtracks = 0
    restarts = 0
    prox_calls = 0
    pava_calls = 0
    prox_inner_iterations = 0
    prox_inner_restarts = 0
    prox_inner_backtracks = 0
    prox_warm_starts = 0
    prox_converged_calls = 0
    maximum_prox_constraint_violation = 0.0
    maximum_prox_scaled_constraint_violation = 0.0
    maximum_prox_fixed_point_residual = 0.0
    last_prox_constraint_violation = math.inf
    last_prox_scaled_constraint_violation = math.inf
    last_prox_fixed_point_residual = math.inf
    last_prox_converged = False
    gradient_evaluations = 0
    initial_residual: Optional[float] = None
    residual = math.inf
    relative_residual = math.inf
    violation = _maximum_violation(instance, x)
    accepted_multiplier = np.zeros(int(instance.rows), dtype=float)
    accepted_step = 1.0 / lipschitz
    solve_start = time.perf_counter()
    last_recorded = 0

    def register_certificate(
        factor: np.ndarray,
        multiplier: np.ndarray,
    ) -> Dict[str, Any]:
        nonlocal best_certificate
        nonlocal best_dual
        nonlocal dual_evaluations
        nonlocal last_dual
        nonlocal maximum_dual_correction

        certificate = _dual_candidate(
            dual_evaluator,
            factor,
            multiplier,
        )
        dual_evaluations += 1
        last_dual = certificate.get("dual_bound")
        maximum_dual_correction = max(
            maximum_dual_correction,
            float(
                certificate.get(
                    "dual_domain_correction",
                    0.0,
                )
            ),
        )
        if last_dual is not None and float(last_dual) > best_dual:
            best_dual = float(last_dual)
            best_certificate = certificate
        return certificate

    def composite_objective(
        point: np.ndarray,
        smooth: float,
    ) -> Optional[float]:
        perspective = perspective_value(
            point,
            k,
            tolerance=max(1e-8, settings.prox_tolerance),
        )
        value = smooth + omega * perspective
        return float(value) if math.isfinite(value) else None

    for iteration in range(1, settings.max_iterations + 1):
        if (
            settings.time_limit is not None
            and time.perf_counter() - solve_start >= settings.time_limit
        ):
            status = "time_limit"
            break

        smooth_y, factor_y = _smooth_value(
            B,
            mu,
            return_reward,
            y,
        )
        gradient = _gradient(
            B,
            mu,
            return_reward,
            factor_y,
        )
        gradient_evaluations += 1
        old_lipschitz = lipschitz
        trial_lipschitz = max(
            lipschitz / settings.step_growth,
            omega / 1e6,
            1e-12,
        )
        accepted = False
        trial_details: Dict[str, Any] = {}
        committed_prox_state = prox_oracle.snapshot()
        if prox_oracle.exact_budget:
            prox_target = settings.prox_tolerance
        elif initial_residual is None:
            prox_target = max(settings.prox_tolerance, 1e-3)
        elif relative_residual <= 10.0 * settings.tolerance:
            prox_target = settings.prox_tolerance
        else:
            prox_target = max(
                settings.prox_tolerance,
                min(1e-3, 0.1 * relative_residual),
            )
        for _ in range(settings.max_backtracks):
            prox_oracle.restore(committed_prox_state)
            step = 1.0 / trial_lipschitz
            trial_x, trial_details = _prox(
                y - step * gradient,
                step * omega,
                prox_oracle,
                prox_target,
            )
            prox_calls += 1
            prox_inner_iterations += int(trial_details["iterations"])
            prox_inner_restarts += int(trial_details["restarts"])
            prox_inner_backtracks += int(
                trial_details["line_search_backtracks"]
            )
            prox_warm_starts += int(trial_details["warm_started"])
            prox_converged_calls += int(trial_details["converged"])
            last_prox_converged = bool(trial_details["converged"])
            last_prox_constraint_violation = float(
                trial_details["constraint_violation"]
            )
            last_prox_scaled_constraint_violation = float(
                trial_details["scaled_constraint_violation"]
            )
            last_prox_fixed_point_residual = float(
                trial_details["fixed_point_residual"]
            )
            maximum_prox_constraint_violation = max(
                maximum_prox_constraint_violation,
                last_prox_constraint_violation,
            )
            maximum_prox_scaled_constraint_violation = max(
                maximum_prox_scaled_constraint_violation,
                last_prox_scaled_constraint_violation,
            )
            maximum_prox_fixed_point_residual = max(
                maximum_prox_fixed_point_residual,
                last_prox_fixed_point_residual,
            )
            pava_calls += int(trial_details["pava_calls"])
            difference = trial_x - y
            smooth_trial, trial_factor = _smooth_value(
                B,
                mu,
                return_reward,
                trial_x,
            )
            model = (
                smooth_y
                + float(gradient @ difference)
                + 0.5
                * trial_lipschitz
                * float(difference @ difference)
            )
            scale = max(1.0, abs(smooth_y), abs(smooth_trial))
            if (
                math.isfinite(smooth_trial)
                and smooth_trial
                <= model + settings.line_search_tolerance * scale
            ):
                accepted = True
                break
            trial_lipschitz *= settings.backtracking_factor
            backtracks += 1
        if not accepted:
            prox_oracle.restore(committed_prox_state)
            raise RuntimeError(
                "FISTA line search failed after "
                f"{settings.max_backtracks} trials"
            )

        lipschitz = trial_lipschitz
        minimum_lipschitz = min(minimum_lipschitz, lipschitz)
        maximum_lipschitz = max(maximum_lipschitz, lipschitz)
        accepted_step = 1.0 / lipschitz
        accepted_multiplier = np.asarray(
            trial_details["constraint_multiplier"],
            dtype=float,
        ).reshape(-1)

        residual = lipschitz * float(np.linalg.norm(trial_x - y))
        if initial_residual is None:
            initial_residual = max(1.0, residual)
        relative_residual = residual / initial_residual
        violation = _maximum_violation(instance, trial_x)
        completed = iteration

        restart_clock = time.perf_counter()
        restart_epoch_age += 1
        restart_now = False
        restart_reason: Optional[str] = None
        restart_metric: Optional[float] = None
        restart_threshold: Optional[float] = None
        paired_gap: Optional[float] = None
        candidate_objective: Optional[float] = None
        iteration_certificate: Optional[Dict[str, Any]] = None
        strategy = settings.restart_strategy

        if strategy == "gradient":
            restart_checks += 1
            restart_metric = float(
                (y - trial_x) @ (trial_x - x)
            )
            restart_threshold = 0.0
            restart_now = restart_metric > restart_threshold
            restart_reason = (
                "odonoghue_candes_gradient"
                if restart_now
                else None
            )
        elif strategy == "function":
            restart_checks += 1
            candidate_objective = composite_objective(
                trial_x,
                smooth_trial,
            )
            objective_evaluations += 1
            restart_objective_evaluations += 1
            if (
                candidate_objective is not None
                and previous_objective is not None
                and violation <= settings.feasibility_tolerance
            ):
                objective_scale = max(
                    1.0,
                    abs(candidate_objective),
                    abs(previous_objective),
                )
                restart_metric = (
                    candidate_objective - previous_objective
                )
                restart_threshold = (
                    settings.restart_function_tolerance
                    * objective_scale
                )
                # For an inexact constrained prox, the literal indicator
                # objective is finite only once the accepted point is
                # feasible to the requested outer tolerance.
                restart_now = bool(
                    restart_metric > restart_threshold
                )
            if (
                candidate_objective is not None
                and violation <= settings.feasibility_tolerance
            ):
                previous_objective = candidate_objective
            restart_reason = (
                "odonoghue_candes_function"
                if restart_now
                else None
            )
        elif strategy == "periodic":
            restart_checks += 1
            restart_metric = float(restart_epoch_age)
            restart_threshold = float(settings.restart_period)
            restart_now = restart_epoch_age >= settings.restart_period
            restart_reason = "fixed_period" if restart_now else None
        elif strategy == "hinder_lubin":
            restart_checks += 1
            displacement = float(
                np.linalg.norm(trial_x - hinder_epoch_start)
            )
            restart_metric = displacement / float(
                (restart_epoch_age + 1) ** 2
            )
            if (
                hinder_previous_displacement is None
                or hinder_previous_length is None
            ):
                # Hinder--Lubin suggest a one-step first epoch when no
                # previous displacement is available.
                restart_threshold = restart_metric
                restart_now = restart_epoch_age == 1
                restart_reason = (
                    "hinder_lubin_initial_epoch"
                    if restart_now
                    else None
                )
            else:
                restart_threshold = (
                    settings.restart_hinder_lubin_beta
                    * hinder_previous_displacement
                    / float((hinder_previous_length + 1) ** 2)
                )
                restart_now = restart_metric <= restart_threshold
                restart_reason = (
                    "hinder_lubin_distance_potential"
                    if restart_now
                    else None
                )
        elif (
            strategy == "primal_dual_gap"
            and iteration % settings.restart_check_interval == 0
        ):
            restart_checks += 1
            candidate_objective = composite_objective(
                trial_x,
                smooth_trial,
            )
            objective_evaluations += 1
            restart_objective_evaluations += 1
            constraint_multiplier = (
                accepted_multiplier / accepted_step
            )
            iteration_certificate = register_certificate(
                trial_factor,
                constraint_multiplier,
            )
            restart_dual_evaluations += 1
            current_dual = iteration_certificate.get("dual_bound")
            if (
                candidate_objective is not None
                and current_dual is not None
            ):
                paired_gap = (
                    candidate_objective - float(current_dual)
                )
                restart_metric = paired_gap
                if restart_epoch_gap is not None:
                    restart_threshold = (
                        restart_epoch_gap / settings.restart_eta
                    )
                    numerical_scale = max(
                        1.0,
                        abs(candidate_objective),
                        abs(float(current_dual)),
                    )
                    restart_now = bool(
                        violation <= settings.feasibility_tolerance
                        and last_prox_converged
                        and paired_gap
                        >= -1e-12 * numerical_scale
                        and paired_gap <= restart_threshold
                    )
            restart_reason = (
                "current_primal_dual_gap_contraction"
                if restart_now
                else None
            )

        if restart_now:
            if strategy == "hinder_lubin":
                hinder_previous_displacement = float(
                    np.linalg.norm(trial_x - hinder_epoch_start)
                )
                hinder_previous_length = restart_epoch_age
                hinder_epoch_start = trial_x.copy()
            if strategy == "primal_dual_gap" and paired_gap is not None:
                restart_epoch_gap = max(0.0, paired_gap)
            next_momentum = 1.0
            next_y = trial_x.copy()
            restarts += 1
            restart_epochs += 1
            restart_iterations.append(int(iteration))
            restart_reasons.append(str(restart_reason))
            restart_metrics.append(restart_metric)
            restart_thresholds.append(restart_threshold)
            restart_epoch_age = 0
        else:
            next_momentum = 0.5 * (
                1.0
                + math.sqrt(
                    1.0
                    + 4.0
                    * (lipschitz / old_lipschitz)
                    * momentum
                    * momentum
                )
            )
            next_y = trial_x + (
                (momentum - 1.0) / next_momentum
            ) * (trial_x - x)

        restart_bookkeeping_seconds += (
            time.perf_counter() - restart_clock
        )
        checkpoint = (
            iteration == 1
            or iteration % settings.history_interval == 0
            or (
                relative_residual <= settings.tolerance
                and violation <= settings.feasibility_tolerance
            )
            or (
                strategy == "primal_dual_gap"
                and iteration % settings.restart_check_interval == 0
            )
            or restart_now
            or iteration == settings.max_iterations
        )
        if checkpoint:
            # Proximal multipliers are in the step-scaled problem.
            # Dividing by the accepted step gives multipliers for the
            # original objective and original, unscaled constraint rows.
            constraint_multiplier = (
                accepted_multiplier / accepted_step
            )
            if iteration_certificate is None:
                iteration_certificate = register_certificate(
                    trial_factor,
                    constraint_multiplier,
                )
            current_dual = iteration_certificate.get("dual_bound")
            if candidate_objective is None:
                candidate_objective = composite_objective(
                    trial_x,
                    smooth_trial,
                )
                objective_evaluations += 1
            objective = candidate_objective
            gap = (
                float(objective) - best_dual
                if objective is not None and math.isfinite(best_dual)
                else None
            )
            history.append(
                {
                    "iteration": int(iteration),
                    "elapsed_seconds": float(
                        time.perf_counter() - solve_start
                    ),
                    "objective": objective,
                    "dual_objective": current_dual,
                    "best_dual_bound": (
                        best_dual if math.isfinite(best_dual) else None
                    ),
                    "best_dual_lower_bound": (
                        best_dual if math.isfinite(best_dual) else None
                    ),
                    "primal_dual_gap": gap,
                    "paired_primal_dual_gap": (
                        float(objective) - float(current_dual)
                        if objective is not None
                        and current_dual is not None
                        else None
                    ),
                    "residual": float(residual),
                    "relative_residual": float(relative_residual),
                    "violation": float(violation),
                    "lipschitz": float(lipschitz),
                    "primal_step": float(accepted_step),
                    "prox_oracle": trial_details["oracle"],
                    "prox_converged": bool(
                        trial_details["converged"]
                    ),
                    "prox_inner_iterations": int(
                        trial_details["iterations"]
                    ),
                    "prox_fixed_point_residual": float(
                        trial_details["fixed_point_residual"]
                    ),
                    "prox_target": float(prox_target),
                    "prox_constraint_violation": float(
                        trial_details["constraint_violation"]
                    ),
                    "restart": restart_now,
                    "restart_strategy": strategy,
                    "restart_reason": restart_reason,
                    "restart_metric": restart_metric,
                    "restart_threshold": restart_threshold,
                    "restart_epoch": int(restart_epochs),
                    "restart_epoch_age": int(restart_epoch_age),
                    "cumulative_restarts": int(restarts),
                    "cumulative_backtracks": int(backtracks),
                }
            )
            last_recorded = iteration
            if (
                settings.dual_bound_cutoff is not None
                and math.isfinite(best_dual)
                and best_dual >= settings.dual_bound_cutoff
            ):
                status = "dual_bound_cutoff"

        x = trial_x
        y = next_y
        momentum = next_momentum
        if status == "dual_bound_cutoff":
            break
        if (
            relative_residual <= settings.tolerance
            and violation <= settings.feasibility_tolerance
            and last_prox_converged
        ):
            status = "converged"
            break

    if (
        status == "iteration_limit"
        and completed > 0
        and not last_prox_converged
    ):
        status = "prox_iteration_limit"
    if completed > 0 and last_recorded != completed:
        factor = _matvec(B.T, x)
        constraint_multiplier = accepted_multiplier / accepted_step
        certificate = register_certificate(
            factor,
            constraint_multiplier,
        )
        current_dual = certificate.get("dual_bound")
        smooth = (
            0.5 * float(factor @ factor)
            - return_reward * float(mu @ x)
        )
        objective = composite_objective(x, smooth)
        objective_evaluations += 1
        history.append(
            {
                "iteration": int(completed),
                "elapsed_seconds": float(
                    time.perf_counter() - solve_start
                ),
                "objective": objective,
                "dual_objective": current_dual,
                "best_dual_bound": (
                    best_dual if math.isfinite(best_dual) else None
                ),
                "best_dual_lower_bound": (
                    best_dual if math.isfinite(best_dual) else None
                ),
                "primal_dual_gap": (
                    float(objective) - best_dual
                    if objective is not None and math.isfinite(best_dual)
                    else None
                ),
                "paired_primal_dual_gap": (
                    float(objective) - float(current_dual)
                    if objective is not None and current_dual is not None
                    else None
                ),
                "residual": float(residual),
                "relative_residual": float(relative_residual),
                "violation": float(violation),
                "lipschitz": float(lipschitz),
                "primal_step": float(accepted_step),
                "prox_oracle": prox_oracle.method,
                "prox_converged": bool(last_prox_converged),
                "prox_inner_iterations": int(
                    trial_details.get("iterations", 0)
                ),
                "prox_fixed_point_residual": float(
                    last_prox_fixed_point_residual
                ),
                "prox_target": float(prox_target),
                "prox_constraint_violation": float(
                    last_prox_constraint_violation
                ),
                "restart": False,
                "restart_strategy": settings.restart_strategy,
                "restart_reason": None,
                "restart_metric": None,
                "restart_threshold": None,
                "restart_epoch": int(restart_epochs),
                "restart_epoch_age": int(restart_epoch_age),
                "cumulative_restarts": int(restarts),
                "cumulative_backtracks": int(backtracks),
            }
        )

    final_diagnostics = evaluate_solution(instance, x)
    final_objective = final_diagnostics.get("objective")
    best_factor = np.asarray(
        best_certificate["factor_dual"],
        dtype=float,
    ).reshape(-1)
    best_scaled_constraint = np.asarray(
        best_certificate["constraint_dual_scaled"],
        dtype=float,
    ).reshape(-1)
    best_original_constraint = np.asarray(
        best_certificate["constraint_dual_original"],
        dtype=float,
    ).reshape(-1)
    anchor_objective = initial_objective
    anchor_gap = (
        float(anchor_objective) - best_dual
        if anchor_objective is not None and math.isfinite(best_dual)
        else None
    )
    solve_seconds = time.perf_counter() - solve_start

    variant = {
        "none": "standard",
        "gradient": "adaptive-restart",
        "function": "function-restart",
        "periodic": "periodic-restart",
        "hinder_lubin": "hinder-lubin-restart",
        "primal_dual_gap": "primal-dual-gap-restart",
    }[settings.restart_strategy]
    restart_certificate = {
        "none": "disabled",
        "gradient": (
            "odonoghue_candes_generalized_gradient_inner_product"
        ),
        "function": (
            "odonoghue_candes_composite_objective_increase"
        ),
        "periodic": "accepted_iteration_period",
        "hinder_lubin": "hinder_lubin_distance_potential",
        "primal_dual_gap": "current_primal_minus_current_safe_dual",
    }[settings.restart_strategy]

    return {
        "x": x,
        "status": status,
        "success": status == "converged",
        "has_solution": True,
        "language": "python",
        "solver": "fista",
        "method": "backtracking FISTA",
        "variant": variant,
        "certified_optimal": False,
        "objective_gap_certified": False,
        "objective": final_objective,
        "external_objective": final_objective,
        "effective_tolerance": settings.tolerance,
        "effective_residual_tolerance": settings.tolerance,
        "feasibility_tolerance": settings.feasibility_tolerance,
        "stopping_certificate": (
            "initial_normalized_composite_gradient_mapping"
            "_inner_dual_fixed_point_and_original_scale_feasibility"
        ),
        "iterations": int(completed),
        "iteration_kind": "accepted_fista_iterations",
        "gradient_evaluations": int(gradient_evaluations),
        "prox_calls": int(prox_calls),
        "prox_converged_calls": int(prox_converged_calls),
        "prox_all_calls_converged": (
            prox_converged_calls == prox_calls
        ),
        "pava_calls": int(pava_calls),
        "prox_scalar_evaluations": int(prox_inner_iterations),
        "prox_inner_iterations": int(prox_inner_iterations),
        "prox_inner_restarts": int(prox_inner_restarts),
        "prox_inner_line_search_backtracks": int(
            prox_inner_backtracks
        ),
        "prox_warm_starts": int(prox_warm_starts),
        "pava_backend": settings.pava_backend,
        "prox_oracle_requested": settings.prox_oracle,
        "prox_oracle_used": prox_oracle.method,
        "prox_exact_budget_fast_path": bool(prox_oracle.exact_budget),
        "prox_operator_norm_squared": float(prox_oracle.lipschitz),
        "prox_operator_norm_kind": prox_oracle.lipschitz_kind,
        "prox_adaptive_restart": settings.prox_adaptive_restart,
        "majorization_max_iterations": (
            settings.majorization_max_iterations
        ),
        "majorization_polish": settings.majorization_polish,
        "majorization_max_lift_variables": (
            settings.majorization_max_lift_variables
        ),
        "prox_tolerance": settings.prox_tolerance,
        "prox_max_iterations": settings.prox_max_iterations,
        "prox_multiplier_available": True,
        "maximum_prox_equality_residual": float(
            maximum_prox_constraint_violation
        ),
        "maximum_prox_constraint_violation": float(
            maximum_prox_constraint_violation
        ),
        "maximum_prox_scaled_constraint_violation": float(
            maximum_prox_scaled_constraint_violation
        ),
        "maximum_prox_fixed_point_residual": float(
            maximum_prox_fixed_point_residual
        ),
        "last_prox_constraint_violation": float(
            last_prox_constraint_violation
        ),
        "last_prox_scaled_constraint_violation": float(
            last_prox_scaled_constraint_violation
        ),
        "last_prox_fixed_point_residual": float(
            last_prox_fixed_point_residual
        ),
        "last_prox_converged": bool(last_prox_converged),
        "restarts": int(restarts),
        "restart_strategy": settings.restart_strategy,
        "restart_certificate": restart_certificate,
        "restart_action": (
            "momentum_reset"
            if settings.restart_strategy != "none"
            else "disabled"
        ),
        "restart_period": int(settings.restart_period),
        "restart_eta": float(settings.restart_eta),
        "restart_contraction": float(1.0 / settings.restart_eta),
        "restart_check_interval": int(
            settings.restart_check_interval
        ),
        "restart_hinder_lubin_beta": float(
            settings.restart_hinder_lubin_beta
        ),
        "restart_function_tolerance": float(
            settings.restart_function_tolerance
        ),
        "restart_checks": int(restart_checks),
        "restart_epochs": int(restart_epochs),
        "restart_iterations": restart_iterations,
        "restart_reasons": restart_reasons,
        "restart_metrics": restart_metrics,
        "restart_thresholds": restart_thresholds,
        "restart_initial_paired_gap": restart_initial_gap,
        "restart_final_epoch_paired_gap": restart_epoch_gap,
        "restart_dual_evaluations": int(restart_dual_evaluations),
        "restart_objective_evaluations": int(
            restart_objective_evaluations
        ),
        "restart_bookkeeping_seconds": float(
            restart_bookkeeping_seconds
        ),
        "restart_exact_prox_assumption_satisfied": bool(
            prox_oracle.exact_budget
        ),
        "restart_inexact_prox_scope": (
            "exact_budget_brent_pava"
            if prox_oracle.exact_budget
            else "empirical_with_inexact_general_interval_prox"
        ),
        "restart_gap_certificate_pairing": (
            "current_primal_current_dual_for_trigger"
        ),
        "restart_safe_bound_aggregation": (
            "maximum_over_all_evaluated_dual_certificates"
        ),
        "restart_observed_gap_rate_qualification": (
            "Q_linear_only_under_the_paper_singleton_dual_"
            "subdifferential_condition_otherwise_empirical_trigger"
        ),
        "restart_strategy_theory_note": {
            "none": "baseline_without_restart",
            "gradient": (
                "odonoghue_candes_practical_heuristic"
            ),
            "function": (
                "odonoghue_candes_practical_heuristic"
            ),
            "periodic": (
                "linear_under_composite_local_quadratic_growth_"
                "with_exact_prox"
            ),
            "hinder_lubin": (
                "published_AGD_bound_assumes_smooth_strong_"
                "convexity_not_met_by_low_rank_markowitz_loss"
            ),
            "primal_dual_gap": (
                "paper_observed_gap_rule_with_corrected_"
                "singleton_condition_for_Q_linear_gap"
            ),
        }[settings.restart_strategy],
        "objective_evaluations": int(objective_evaluations),
        "line_search_backtracks": int(backtracks),
        "line_search_mode": "smooth_quadratic_majorization",
        "backtracking_factor": settings.backtracking_factor,
        "step_growth": settings.step_growth,
        "line_search_tolerance": settings.line_search_tolerance,
        "residual": float(residual),
        "relative_residual": float(relative_residual),
        "initial_residual": (
            float(initial_residual)
            if initial_residual is not None
            else None
        ),
        "residual_coordinates": "original_objective_problem",
        "violation": float(violation),
        "initial_lipschitz": float(initial_lipschitz),
        "final_lipschitz": float(lipschitz),
        "minimum_lipschitz": float(minimum_lipschitz),
        "maximum_lipschitz": float(maximum_lipschitz),
        "lipschitz_estimate_kind": lipschitz_kind,
        "final_primal_step": float(1.0 / lipschitz),
        "history": history,
        "dual_bound": (
            best_dual if math.isfinite(best_dual) else None
        ),
        "best_dual_lower_bound": (
            best_dual if math.isfinite(best_dual) else None
        ),
        "initial_dual_objective": initial_dual,
        "last_evaluated_dual_objective": last_dual,
        "dual_bound_evaluations": int(dual_evaluations),
        "dual_bound_available": math.isfinite(best_dual),
        "dual_bound_safe_in_exact_arithmetic": True,
        "dual_bound_floating_point_certified": False,
        "dual_bound_factor": best_factor,
        "dual_bound_constraint_scaled": best_scaled_constraint,
        "dual_bound_constraint_original": best_original_constraint,
        "maximum_dual_domain_correction": float(
            maximum_dual_correction
        ),
        "dual_bound_kind": "fenchel_weak_duality_lower_bound",
        "dual_bound_units": "original_objective",
        "dual_bound_formula": (
            "-0.5*||p||^2 - support_[lower,upper](q) "
            "- (perspective_weight*G_k)^*("
            "return_reward*mu - B*p - C.T*q)"
        ),
        "dual_bound_cutoff": settings.dual_bound_cutoff,
        "anchor_primal_upper_bound": anchor_objective,
        "anchor_primal_dual_gap": anchor_gap,
        "bound_consistency_violation": (
            max(best_dual - float(anchor_objective), 0.0)
            if anchor_objective is not None and math.isfinite(best_dual)
            else None
        ),
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "total_seconds": float(setup_seconds + solve_seconds),
        "complexity_per_iteration": (
            "O(d*r + warm_started_sparse_QP(d*k))"
            if prox_oracle.method == "majorization_qp_osqp"
            else (
                "O(d*r + prox_inner_iterations*"
                "(nnz(C)+PAVA(d,k)))"
            )
        ),
        "constraint_scope": (
            "single_exact_budget_equality"
            if prox_oracle.exact_budget
            else "general_interval_rows"
        ),
    }


def solve_fista(
    instance: Any,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve an interval-constrained instance with backtracking FISTA."""
    settings, threads = _settings(options)
    if threads:
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError as error:
            raise RuntimeError(
                "threadpoolctl is required to enforce FISTA thread limits"
            ) from error
        thread_context = threadpool_limits(
            limits=threads,
            user_api="blas",
        )
    else:
        thread_context = nullcontext()

    with thread_context:
        total_start = time.perf_counter()
        B = instance.B
        if len(B.shape) != 2:
            raise ValueError("B must be a matrix")
        dimension, rank = map(int, B.shape)
        if settings.prox_oracle == "majorization_qp":
            from .majorization_qp import MajorizationQPOracle

            prox_oracle = MajorizationQPOracle(
                instance.C,
                instance.lower,
                instance.upper,
                int(instance.k),
                tolerance=settings.prox_tolerance,
                max_iterations=settings.majorization_max_iterations,
                polish=settings.majorization_polish,
                max_lift_variables=(
                    settings.majorization_max_lift_variables
                ),
            )
        else:
            prox_oracle = LinearConstraintProx(
                instance.C,
                instance.lower,
                instance.upper,
                int(instance.k),
                pava_method=settings.pava_backend,
                tolerance=settings.prox_tolerance,
                max_iterations=settings.prox_max_iterations,
                adaptive_restart=settings.prox_adaptive_restart,
                use_budget_fast_path=(
                    settings.prox_oracle not in {"dual_fista", "pava"}
                ),
            )
            if (
                settings.prox_oracle == "budget"
                and not prox_oracle.exact_budget
            ):
                raise ValueError(
                    "prox_oracle='budget' requires the single exact row "
                    "1.T @ x = 1"
                )
            if (
                settings.prox_oracle == "pava"
                and prox_oracle.rows != 0
            ):
                raise ValueError(
                    "prox_oracle='pava' requires no linear rows"
                )
        if settings.initial_lipschitz is None:
            initial_lipschitz, lipschitz_kind = _estimate_lipschitz(
                B,
                dimension,
                rank,
                settings.norm_iterations,
                float(instance.perspective_weight),
            )
        else:
            initial_lipschitz = settings.initial_lipschitz
            lipschitz_kind = "user_supplied"
        dual_evaluator = SafeDualEvaluator(instance)
        setup_seconds = time.perf_counter() - total_start
        result = _solve(
            instance,
            settings,
            prox_oracle,
            dual_evaluator,
            setup_seconds,
            initial_lipschitz,
            lipschitz_kind,
        )
        try:
            from threadpoolctl import threadpool_info

            threadpools = threadpool_info()
        except ImportError:
            threadpools = []

    result["threads"] = threads
    result["thread_limit_kind"] = (
        "blas" if threads else "runtime_default"
    )
    result["threadpools"] = threadpools
    return result


__all__ = ["solve_fista"]
