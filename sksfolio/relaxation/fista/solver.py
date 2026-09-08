"""Backtracking FISTA for linearly constrained perspective relaxations.

The backend solves

    minimize 0.5 * ||B.T @ x||^2 - rho * mu.T @ x + omega * G_k(x)
    subject to lower <= C @ x <= upper.

The interval rows are kept inside the proximal step. Exactly two corrected
algorithms are supported: a curvature-synchronous dual-FISTA oracle and a
split-dual L-BFGS-B oracle with corrected dual-FISTA fallback.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from .._threading import limit_blas_threads, threadpool_info
from ..safe_dual import SafeDualEvaluator
from ..problem import evaluate_solution, perspective_value
from ..state import RelaxationState
from .linear_prox import LinearConstraintProx


_IMPLEMENTATION = "python"
_LANGUAGE = "python"


def _fista_momentum(momentum: float) -> float:
    """Return the classical FISTA momentum parameter."""
    return 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))


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
    prox_target_ceiling: float
    prox_max_iterations: int
    pava_backend: str
    prox_oracle: str
    prox_adaptive_restart: bool
    prox_lbfgs_memory: int
    prox_lbfgs_max_line_search: int
    prox_lbfgs_fallback: bool
    dual_bound_cutoff: Optional[float]
    warm_start: Optional[RelaxationState]
    resume_acceleration: bool


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
    # A 1.5 growth factor on a rejected outer trial wastes less of the step
    # than doubling does, and measured better on the tuned 0808 benchmark
    # settings. This changes only the accepted outer-trial policy, not the
    # corrected 0821 momentum bookkeeping.
    backtracking_factor = float(supplied.get("backtracking_factor", 1.5))
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
    # Early outer iterations do not benefit from driving the prox much below
    # 1e-2; a tighter ceiling mostly adds inner work without changing the
    # accepted outer step. Keep the corrected algorithm and relax only the
    # inexact-prox target used during the outer solve.
    prox_target_ceiling = float(supplied.get("prox_target_ceiling", 1e-2))
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
        supplied.get("prox_oracle", "dual_lbfgs")
    ).lower().replace("-", "_")
    prox_oracle = {
        "general": "dual_fista",
        "dual": "dual_fista",
        "apg": "dual_fista",
        "lbfgs": "dual_lbfgs",
        "lbfgsb": "dual_lbfgs",
        "l_bfgs": "dual_lbfgs",
        "l_bfgs_b": "dual_lbfgs",
        "dual_lbfgsb": "dual_lbfgs",
    }.get(prox_oracle, prox_oracle)
    prox_adaptive_restart = bool(
        supplied.get("prox_adaptive_restart", True)
    )
    prox_lbfgs_memory = int(
        supplied.get("prox_lbfgs_memory", 10)
    )
    prox_lbfgs_max_line_search = int(
        supplied.get("prox_lbfgs_max_line_search", 40)
    )
    prox_lbfgs_fallback = bool(
        supplied.get("prox_lbfgs_fallback", True)
    )
    raw_cutoff = supplied.get("dual_bound_cutoff")
    dual_bound_cutoff = (
        None if raw_cutoff is None else float(raw_cutoff)
    )
    threads = int(supplied.get("threads", 0))
    warm_start = RelaxationState.coerce(
        supplied.get("warm_start", supplied.get("initial_state"))
    )
    resume_acceleration = bool(
        supplied.get("resume_acceleration", False)
    )

    positive = {
        "tolerance": tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "backtracking_factor": backtracking_factor,
        "step_growth": step_growth,
        "prox_tolerance": prox_tolerance,
        "prox_target_ceiling": prox_target_ceiling,
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
        ("prox_lbfgs_memory", prox_lbfgs_memory),
        (
            "prox_lbfgs_max_line_search",
            prox_lbfgs_max_line_search,
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
    if prox_oracle not in {"dual_fista", "dual_lbfgs"}:
        raise ValueError(
            "prox_oracle must be 'dual_fista' or 'dual_lbfgs'"
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
            prox_target_ceiling=prox_target_ceiling,
            prox_max_iterations=prox_max_iterations,
            pava_backend=pava_backend,
            prox_oracle=prox_oracle,
            prox_adaptive_restart=prox_adaptive_restart,
            prox_lbfgs_memory=prox_lbfgs_memory,
            prox_lbfgs_max_line_search=(
                prox_lbfgs_max_line_search
            ),
            prox_lbfgs_fallback=prox_lbfgs_fallback,
            dual_bound_cutoff=dual_bound_cutoff,
            warm_start=warm_start,
            resume_acceleration=resume_acceleration,
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
    function_evaluations = int(
        getattr(result, "function_evaluations", pava_calls)
    )
    fallback_used = bool(getattr(result, "fallback_used", False))
    optimizer_status = str(getattr(result, "optimizer_status", ""))
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
            "function_evaluations": int(function_evaluations),
            "fallback_used": bool(fallback_used),
            "optimizer_status": optimizer_status,
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
    warm_state = settings.warm_start
    if warm_state is None:
        x = np.asarray(instance.anchor, dtype=float).reshape(-1).copy()
    else:
        x = warm_state.compatible_primal(instance)
    y = x.copy()
    momentum = 1.0
    warm_start_used = warm_state is not None
    warm_start_exact_structure = bool(
        warm_state is not None
        and tuple(str(name) for name in instance.constraint_names)
        == warm_state.constraint_ids
    )
    acceleration_resumed = False
    if (
        warm_state is not None
        and settings.resume_acceleration
        and warm_start_exact_structure
    ):
        candidate_y = warm_state.arrays.get("momentum_x")
        candidate_momentum = warm_state.scalars.get("momentum", 1.0)
        if candidate_y is not None and candidate_y.shape == x.shape:
            y = candidate_y.copy()
            acceleration_resumed = True
        try:
            candidate_momentum = float(candidate_momentum)
        except (TypeError, ValueError):
            candidate_momentum = 1.0
        if math.isfinite(candidate_momentum) and candidate_momentum >= 1.0:
            momentum = candidate_momentum
    lipschitz = initial_lipschitz
    minimum_lipschitz = lipschitz
    maximum_lipschitz = lipschitz

    initial_factor = _matvec(B.T, x)
    if (
        warm_state is not None
        and warm_state.factor_dual is not None
        and warm_state.factor_dual.shape == initial_factor.shape
    ):
        # A parent-node dual certificate remains a valid candidate after
        # child rows are added: mapped new-row multipliers are zero.  Reusing
        # that certificate preserves the parent's safe lower bound before the
        # first child iteration instead of rebuilding p from an infeasible
        # child warm start.
        initial_factor = warm_state.factor_dual.copy()
    initial_constraint_multiplier = np.zeros(
        int(instance.rows),
        dtype=float,
    )
    if warm_state is not None:
        mapped = warm_state.mapped_constraint_dual(instance)
        if mapped is not None:
            initial_constraint_multiplier = mapped
    initial_certificate = _dual_candidate(
        dual_evaluator,
        initial_factor,
        initial_constraint_multiplier,
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
    initial_violation = _maximum_violation(instance, x)
    initial_primal_upper_bound = (
        float(initial_objective)
        if initial_objective is not None
        and initial_violation <= settings.feasibility_tolerance
        else None
    )
    objective_evaluations = 1
    previous_objective = (
        None
        if initial_primal_upper_bound is None
        else float(initial_primal_upper_bound)
    )
    restart_epoch_gap = (
        float(initial_primal_upper_bound) - float(initial_dual)
        if initial_primal_upper_bound is not None and initial_dual is not None
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
    prox_function_evaluations = 0
    prox_lbfgs_fallback_calls = 0
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
    violation = initial_violation
    accepted_multiplier = initial_constraint_multiplier / lipschitz
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
        previous_lipschitz = lipschitz
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
            prox_target = max(
                settings.prox_tolerance,
                settings.prox_target_ceiling,
            )
        elif relative_residual <= 10.0 * settings.tolerance:
            prox_target = settings.prox_tolerance
        else:
            prox_target = max(
                settings.prox_tolerance,
                min(
                    settings.prox_target_ceiling,
                    0.1 * relative_residual,
                ),
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
            prox_function_evaluations += int(
                trial_details["function_evaluations"]
            )
            prox_lbfgs_fallback_calls += int(
                trial_details["fallback_used"]
            )
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
                and violation <= settings.feasibility_tolerance
                and last_prox_converged
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
            # Classical corrected FISTA momentum. Its Lyapunov proof applies
            # when the accepted curvature sequence is nondecreasing.
            next_momentum = _fista_momentum(momentum)
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
            current_is_feasible = bool(
                violation <= settings.feasibility_tolerance
            )
            gap = (
                float(objective) - best_dual
                if current_is_feasible
                and objective is not None
                and math.isfinite(best_dual)
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
                        if current_is_feasible
                        and objective is not None
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
        current_is_feasible = bool(
            violation <= settings.feasibility_tolerance
        )
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
                    if current_is_feasible
                    and objective is not None
                    and math.isfinite(best_dual)
                    else None
                ),
                "paired_primal_dual_gap": (
                    float(objective) - float(current_dual)
                    if current_is_feasible
                    and objective is not None
                    and current_dual is not None
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
    anchor_objective = initial_primal_upper_bound
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

    prox_state = (
        prox_oracle.export_warm_start()
        if hasattr(prox_oracle, "export_warm_start")
        else {}
    )
    final_factor = _matvec(B.T, x)
    final_constraint_multiplier = (
        accepted_multiplier / accepted_step
        if accepted_step > 0.0
        else initial_constraint_multiplier.copy()
    )
    state_arrays: Dict[str, np.ndarray] = {
        "momentum_x": y.copy(),
        "current_factor_dual": final_factor.copy(),
        "current_constraint_dual": final_constraint_multiplier.copy(),
    }
    internal_dual = prox_state.get("internal_dual")
    if internal_dual is not None:
        state_arrays["prox_dual_internal"] = np.asarray(
            internal_dual,
            dtype=float,
        ).reshape(-1)
    restart_state = RelaxationState(
        backend="fista",
        implementation=_IMPLEMENTATION,
        dimension=int(instance.dimension),
        k=k,
        constraint_ids=tuple(
            str(name) for name in instance.constraint_names
        ),
        x=x,
        # Public dual fields always retain the strongest recomputable safe
        # certificate.  Algorithm-specific current duals stay private above.
        factor_dual=best_factor,
        constraint_dual=best_original_constraint,
        arrays=state_arrays,
        scalars={
            "lipschitz": float(lipschitz),
            "momentum": float(momentum),
            "budget_eta": prox_state.get("budget_eta"),
            "fresh_restart_epoch_recommended": True,
        },
    )

    return {
        "x": x,
        "status": status,
        "success": status == "converged",
        "has_solution": True,
        "language": _LANGUAGE,
        "implementation": _IMPLEMENTATION,
        "native_solver_core": _IMPLEMENTATION == "native",
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
        "prox_function_evaluations": int(
            prox_function_evaluations
        ),
        "prox_lbfgs_fallback_calls": int(
            prox_lbfgs_fallback_calls
        ),
        "prox_newton_enabled": bool(prox_oracle.semismooth_newton),
        "pava_backend": settings.pava_backend,
        "prox_oracle_requested": settings.prox_oracle,
        "prox_oracle_used": prox_oracle.method,
        "prox_exact_budget_fast_path": bool(prox_oracle.exact_budget),
        "prox_operator_norm_squared": float(prox_oracle.lipschitz),
        "prox_operator_norm_kind": prox_oracle.lipschitz_kind,
        "prox_inner_momentum_rule": "curvature_synchronous_fista",
        "prox_inner_curvature_policy": (
            "decrease_then_backtrack_with_synchronized_extrapolation"
        ),
        "prox_adaptive_restart": settings.prox_adaptive_restart,
        "prox_lbfgs_memory": int(settings.prox_lbfgs_memory),
        "prox_lbfgs_max_line_search": int(
            settings.prox_lbfgs_max_line_search
        ),
        "prox_lbfgs_fallback_enabled": bool(
            settings.prox_lbfgs_fallback
        ),
        "prox_tolerance": settings.prox_tolerance,
        "prox_target_ceiling": settings.prox_target_ceiling,
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
        "momentum_rule": "classical_fista",
        "algorithm_variant": "corrected",
        "accelerated_rate_theory_scope": (
            "nondecreasing_accepted_curvature_exact_or_"
            "summably_inexact_prox_without_heuristic_restart"
        ),
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
        "anchor_violation": float(initial_violation),
        "anchor_is_numerically_feasible": bool(
            initial_primal_upper_bound is not None
        ),
        "bound_consistency_violation": (
            max(best_dual - float(anchor_objective), 0.0)
            if anchor_objective is not None and math.isfinite(best_dual)
            else None
        ),
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "total_seconds": float(setup_seconds + solve_seconds),
        "complexity_per_iteration": (
            "O(d*r + prox_lbfgs_evaluations*"
            "(nnz(C)+PAVA(d,k)+memory*rows(C)))"
            if prox_oracle.method == "row_scaled_dual_lbfgsb"
            else "O(d*r + prox_inner_iterations*(nnz(C)+PAVA(d,k)))"
        ),
        "constraint_scope": (
            "single_exact_budget_equality"
            if prox_oracle.exact_budget
            else "general_interval_rows"
        ),
        "warm_start_used": bool(warm_start_used),
        "warm_start_exact_structure": bool(
            warm_start_exact_structure
        ),
        "warm_start_structure_match_kind": "constraint_ids_only",
        "warm_start_acceleration_resumed": bool(
            acceleration_resumed
        ),
        "restart_state": restart_state.to_dict(copy=False),
    }


def solve_fista(
    instance: Any,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve an interval-constrained instance with backtracking FISTA."""
    settings, threads = _settings(options)
    if threads:
        try:
            thread_context = limit_blas_threads(threads)
        except RuntimeError as error:
            raise RuntimeError(
                "threadpoolctl is required to enforce FISTA thread limits"
            ) from error
    else:
        thread_context = nullcontext()

    with thread_context:
        total_start = time.perf_counter()
        B = instance.B
        if len(B.shape) != 2:
            raise ValueError("B must be a matrix")
        dimension, rank = map(int, B.shape)
        prox_oracle = LinearConstraintProx(
            instance.C,
            instance.lower,
            instance.upper,
            int(instance.k),
            pava_method=settings.pava_backend,
            tolerance=settings.prox_tolerance,
            max_iterations=settings.prox_max_iterations,
            adaptive_restart=settings.prox_adaptive_restart,
            use_budget_fast_path=True,
            dual_solver=(
                "lbfgs"
                if settings.prox_oracle == "dual_lbfgs"
                else "fista"
            ),
            lbfgs_memory=settings.prox_lbfgs_memory,
            lbfgs_max_line_search=settings.prox_lbfgs_max_line_search,
            lbfgs_fallback=settings.prox_lbfgs_fallback,
            semismooth_newton=bool(_as_options(options).get("prox_newton", False)),
        )
        if settings.initial_lipschitz is None:
            state_lipschitz = (
                settings.warm_start.scalars.get("lipschitz")
                if settings.warm_start is not None
                else None
            )
            try:
                state_lipschitz = float(state_lipschitz)
            except (TypeError, ValueError):
                state_lipschitz = None
        else:
            state_lipschitz = None
        if (
            settings.initial_lipschitz is None
            and state_lipschitz is not None
            and math.isfinite(state_lipschitz)
            and state_lipschitz > 0.0
        ):
            initial_lipschitz = state_lipschitz
            lipschitz_kind = "warm_start_state"
        elif settings.initial_lipschitz is None:
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
        if settings.warm_start is not None and hasattr(
            prox_oracle,
            "initialize_warm_start",
        ):
            mapped_multiplier = settings.warm_start.mapped_constraint_dual(
                instance
            )
            same_rows = bool(
                tuple(str(name) for name in instance.constraint_names)
                == settings.warm_start.constraint_ids
            )
            internal_dual = (
                settings.warm_start.arrays.get("prox_dual_internal")
                if same_rows
                else None
            )
            prox_oracle.initialize_warm_start(
                original_multiplier=mapped_multiplier,
                prox_step=1.0 / initial_lipschitz,
                internal_dual=internal_dual,
                budget_eta=settings.warm_start.scalars.get(
                    "budget_eta"
                ),
            )
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
        threadpools = threadpool_info()

    result["threads"] = threads
    result["thread_limit_kind"] = (
        "blas" if threads else "runtime_default"
    )
    result["threadpool_controller"] = (
        "cached_threadpool_controller" if threads else "not_used"
    )
    result["threadpools"] = threadpools
    return result


__all__ = ["solve_fista"]
