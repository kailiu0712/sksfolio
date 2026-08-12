"""Restarted accelerated proximal gradient on the SC-SDG.

This implements Algorithms 2--3 of Fercoq (2025) on the fully split
perspective Markowitz saddle problem

    min_x max_{p,q} omega G_k(x) - rho mu.T x
        + <R x, p> + <C x, q>
        - 0.5 ||p||^2 - sigma_[lower,upper](q).

The implementation uses the same normalized factor operator, row-scaled
constraints, PAVA oracle, and safe Fenchel certificate as the PDHG backend.
An optional positive scaling of the constraint dual block is implemented as
an exact change of coordinates, not as an unproved diagonal-metric variant.
The optional backtracking extension tests the smooth SC-SDG component against
its local quadratic upper model.  It is an implementation extension: the
paper explicitly leaves line search as future work.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..problem import perspective_value
from ..state import RelaxationState
from ..pdhg.pava import prox as pava_prox
from ..pdhg.safe_dual import _dual_bound
from ..pdhg.solver import (
    _constraint_adjoint,
    _constraint_forward,
    _factor_adjoint,
    _factor_forward,
    _frobenius_operator_bound,
    _gram_data,
    _gram_norm,
    _interval_dual_prox,
    _maximum_violation,
    _objective,
    _operator_norm,
    _prepare_problem,
)


PAPER_URL = "https://arxiv.org/pdf/2511.03442"
_IMPLEMENTATION = (
    "native" if __name__.endswith("._native_solver") else "python"
)
_LANGUAGE = "cython" if _IMPLEMENTATION == "native" else "python"


@dataclass(frozen=True)
class _Settings:
    tolerance: float
    feasibility_tolerance: float
    time_limit: Optional[float]
    max_iterations: int
    check_interval: int
    restart_check_interval: int
    min_restart_iterations: int
    restart_factor: float
    theta_parameter: float
    continuation_offset: float
    target_cbar: float
    step_ratio: float
    constraint_weight: Optional[float]
    operator_norm_mode: str
    norm_iterations: int
    norm_safety: float
    pava_backend: str
    dual_bound_cutoff: Optional[float]
    normalize_objective: bool
    center_return_objective: bool
    restart: bool
    line_search: bool
    line_search_mode: str
    line_search_auto_row_threshold: int
    line_search_initial_scale: float
    line_search_growth: float
    line_search_shrink: float
    line_search_max_scale: float
    line_search_safety: float
    line_search_tolerance: float
    line_search_max_backtracks: int
    line_search_reset_on_restart: bool
    lbfgs_variant: str
    lbfgs_delta_ratio: float
    lbfgs_memory: int
    lbfgs_max_line_search: int
    lbfgs_min_function_evaluations: int
    lbfgs_max_function_evaluations: int
    lbfgs_tolerance: float
    warm_start: Optional[RelaxationState]


@dataclass
class _Point:
    x: np.ndarray
    p: np.ndarray
    q: np.ndarray

    def copy(self) -> "_Point":
        return _Point(self.x.copy(), self.p.copy(), self.q.copy())


@dataclass(frozen=True)
class _LinearImage:
    factor: np.ndarray
    constraint: np.ndarray
    adjoint: np.ndarray


@dataclass(frozen=True)
class _OracleEvaluation:
    proximal_point: _Point
    gradient: _Point
    gap: Optional[float]
    raw_gap: Optional[float]
    smooth_value: Optional[float]
    residual: float
    point_image: _LinearImage
    proximal_image: _LinearImage


@dataclass(frozen=True)
class _LbfgsPolish:
    candidate: _Point
    candidate_image: _LinearImage
    center_oracle: _OracleEvaluation
    fixed_oracle: _OracleEvaluation
    current_oracle: _OracleEvaluation
    iterations: int
    function_evaluations: int
    pava_calls: int
    product_prox_calls: int
    gap_evaluations: int
    gradient_evaluations: int
    objective_initial: float
    objective_final: float
    gradient_norm: float
    optimizer_status: str


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
        supplied.get("feasibility_tolerance", tolerance)
    )
    raw_limit = supplied.get("time_limit")
    time_limit = None if raw_limit is None else float(raw_limit)
    max_iterations = int(supplied.get("max_iterations", 100_000))
    check_interval = int(supplied.get("check_interval", 25))
    restart_check_interval = int(
        supplied.get("restart_check_interval", 1)
    )
    min_restart_iterations = int(
        supplied.get("min_restart_iterations", 1)
    )
    restart_factor = float(supplied.get("restart_factor", 0.5))
    theta_parameter = float(supplied.get("theta_parameter", 2.0))
    continuation_offset = float(
        supplied.get("continuation_offset", 3.0)
    )
    target_cbar = float(supplied.get("target_cbar", 0.1))
    step_ratio = float(supplied.get("step_ratio", 1.0))
    raw_constraint_weight = supplied.get(
        "constraint_dual_weight",
        "auto",
    )
    constraint_weight = (
        None
        if raw_constraint_weight is None
        or str(raw_constraint_weight).lower() == "auto"
        else float(raw_constraint_weight)
    )
    operator_norm_mode = str(
        supplied.get("operator_norm_mode", "auto")
    ).lower().replace("-", "_")
    norm_iterations = int(supplied.get("norm_iterations", 30))
    norm_safety = float(supplied.get("norm_safety", 1.05))
    pava_backend = str(
        supplied.get("pava_backend", "partial_sort")
    ).lower().replace("-", "_")
    pava_backend = {
        "full": "full_sort",
        "partial": "partial_sort",
        "topk": "partial_sort",
    }.get(pava_backend, pava_backend)
    raw_cutoff = supplied.get("dual_bound_cutoff")
    dual_bound_cutoff = (
        None if raw_cutoff is None else float(raw_cutoff)
    )
    normalize_objective = bool(
        supplied.get("normalize_objective", True)
    )
    center_return_objective = bool(
        supplied.get("center_return_objective", False)
    )
    restart = bool(supplied.get("restart", True))
    line_search = bool(supplied.get("line_search", True))
    line_search_mode = str(
        supplied.get("line_search_mode", "auto")
    ).lower().replace("-", "_")
    line_search_mode = {
        "direct": "operator",
        "directional": "operator",
        "smooth": "majorization",
        "exact": "majorization",
    }.get(line_search_mode, line_search_mode)
    line_search_auto_row_threshold = int(
        supplied.get("line_search_auto_row_threshold", 16)
    )
    line_search_initial_scale = float(
        supplied.get("line_search_initial_scale", 1.0)
    )
    line_search_growth = float(
        supplied.get("line_search_growth", 1.1)
    )
    line_search_shrink = float(
        supplied.get("line_search_shrink", 0.5)
    )
    line_search_max_scale = float(
        supplied.get("line_search_max_scale", 1024.0)
    )
    line_search_safety = float(
        supplied.get("line_search_safety", 0.99)
    )
    line_search_tolerance = float(
        supplied.get("line_search_tolerance", 1e-12)
    )
    line_search_max_backtracks = int(
        supplied.get("line_search_max_backtracks", 40)
    )
    line_search_reset_on_restart = bool(
        supplied.get("line_search_reset_on_restart", False)
    )
    raw_lbfgs_variant = supplied.get("lbfgs_variant")
    if raw_lbfgs_variant is None:
        lbfgs_variant = (
            "restart_safe"
            if bool(supplied.get("lbfgs", False))
            else "off"
        )
    else:
        lbfgs_variant = str(raw_lbfgs_variant).lower().replace(
            "-",
            "_",
        )
    lbfgs_variant = {
        "none": "off",
        "disabled": "off",
        "safe": "restart_safe",
        "safeguarded": "restart_safe",
        "restart": "restart_safe",
        "reference": "paper",
        "paper_reference": "paper",
        "restart_paper": "restart_reference",
    }.get(lbfgs_variant, lbfgs_variant)
    lbfgs_delta_ratio = float(
        supplied.get("lbfgs_delta_ratio", 0.01)
    )
    lbfgs_memory = int(supplied.get("lbfgs_memory", 10))
    lbfgs_max_line_search = int(
        supplied.get("lbfgs_max_line_search", 40)
    )
    lbfgs_min_function_evaluations = int(
        supplied.get("lbfgs_min_function_evaluations", 10)
    )
    lbfgs_max_function_evaluations = int(
        supplied.get("lbfgs_max_function_evaluations", 200)
    )
    lbfgs_tolerance = float(
        supplied.get("lbfgs_tolerance", 1e-10)
    )
    threads = int(supplied.get("threads", 0))
    warm_start = RelaxationState.coerce(
        supplied.get("warm_start", supplied.get("initial_state"))
    )

    positive = {
        "tolerance": tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "restart_factor": restart_factor,
        "theta_parameter": theta_parameter,
        "continuation_offset": continuation_offset,
        "target_cbar": target_cbar,
        "step_ratio": step_ratio,
        "norm_safety": norm_safety,
        "line_search_initial_scale": line_search_initial_scale,
        "line_search_growth": line_search_growth,
        "line_search_max_scale": line_search_max_scale,
        "lbfgs_delta_ratio": lbfgs_delta_ratio,
        "lbfgs_tolerance": lbfgs_tolerance,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if constraint_weight is not None and (
        not math.isfinite(constraint_weight)
        or constraint_weight <= 0.0
    ):
        raise ValueError(
            "constraint_dual_weight must be positive, finite, or 'auto'"
        )
    if time_limit is not None and (
        not math.isfinite(time_limit) or time_limit <= 0.0
    ):
        raise ValueError("time_limit must be positive and finite")
    for name, value in (
        ("max_iterations", max_iterations),
        ("check_interval", check_interval),
        ("restart_check_interval", restart_check_interval),
        ("min_restart_iterations", min_restart_iterations),
        ("norm_iterations", norm_iterations),
        ("line_search_max_backtracks", line_search_max_backtracks),
        ("lbfgs_memory", lbfgs_memory),
        ("lbfgs_max_line_search", lbfgs_max_line_search),
        (
            "lbfgs_min_function_evaluations",
            lbfgs_min_function_evaluations,
        ),
        (
            "lbfgs_max_function_evaluations",
            lbfgs_max_function_evaluations,
        ),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < restart_factor < 1.0:
        raise ValueError("restart_factor must lie in (0, 1)")
    if theta_parameter < 2.0:
        raise ValueError("theta_parameter must be at least 2")
    if continuation_offset < theta_parameter:
        raise ValueError(
            "continuation_offset must be at least theta_parameter"
        )
    if not 0.0 < target_cbar < 1.0:
        raise ValueError("target_cbar must lie in (0, 1)")
    if norm_safety < 1.0:
        raise ValueError("norm_safety must be at least one")
    if operator_norm_mode not in {
        "auto",
        "gram",
        "power",
        "frobenius",
    }:
        raise ValueError(
            "operator_norm_mode must be auto, gram, power, or frobenius"
        )
    if pava_backend not in {"full_sort", "partial_sort"}:
        raise ValueError(
            "pava_backend must be 'full_sort' or 'partial_sort'"
        )
    if dual_bound_cutoff is not None and not math.isfinite(
        dual_bound_cutoff
    ):
        raise ValueError("dual_bound_cutoff must be finite")
    if line_search_growth < 1.0:
        raise ValueError("line_search_growth must be at least one")
    if not 0.0 < line_search_shrink < 1.0:
        raise ValueError("line_search_shrink must lie in (0, 1)")
    if line_search_auto_row_threshold < 0:
        raise ValueError(
            "line_search_auto_row_threshold must be nonnegative"
        )
    if line_search_mode not in {
        "auto",
        "operator",
        "majorization",
    }:
        raise ValueError(
            "line_search_mode must be 'auto', 'operator', or "
            "'majorization'"
        )
    if not 0.0 < line_search_safety <= 1.0:
        raise ValueError("line_search_safety must lie in (0, 1]")
    if line_search_initial_scale > line_search_max_scale:
        raise ValueError(
            "line_search_initial_scale cannot exceed "
            "line_search_max_scale"
        )
    if (
        not math.isfinite(line_search_tolerance)
        or line_search_tolerance < 0.0
    ):
        raise ValueError(
            "line_search_tolerance must be finite and nonnegative"
        )
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    if lbfgs_variant not in {
        "off",
        "paper",
        "restart_reference",
        "restart_safe",
    }:
        raise ValueError(
            "lbfgs_variant must be 'off', 'paper', "
            "'restart_reference', or 'restart_safe'"
        )
    if (
        lbfgs_min_function_evaluations
        > lbfgs_max_function_evaluations
    ):
        raise ValueError(
            "lbfgs_min_function_evaluations cannot exceed "
            "lbfgs_max_function_evaluations"
        )

    return (
        _Settings(
            tolerance=tolerance,
            feasibility_tolerance=feasibility_tolerance,
            time_limit=time_limit,
            max_iterations=max_iterations,
            check_interval=check_interval,
            restart_check_interval=restart_check_interval,
            min_restart_iterations=min_restart_iterations,
            restart_factor=restart_factor,
            theta_parameter=theta_parameter,
            continuation_offset=continuation_offset,
            target_cbar=target_cbar,
            step_ratio=step_ratio,
            constraint_weight=constraint_weight,
            operator_norm_mode=operator_norm_mode,
            norm_iterations=norm_iterations,
            norm_safety=norm_safety,
            pava_backend=pava_backend,
            dual_bound_cutoff=dual_bound_cutoff,
            normalize_objective=normalize_objective,
            center_return_objective=center_return_objective,
            restart=restart,
            line_search=line_search,
            line_search_mode=line_search_mode,
            line_search_auto_row_threshold=(
                line_search_auto_row_threshold
            ),
            line_search_initial_scale=line_search_initial_scale,
            line_search_growth=line_search_growth,
            line_search_shrink=line_search_shrink,
            line_search_max_scale=line_search_max_scale,
            line_search_safety=line_search_safety,
            line_search_tolerance=line_search_tolerance,
            line_search_max_backtracks=line_search_max_backtracks,
            line_search_reset_on_restart=line_search_reset_on_restart,
            lbfgs_variant=lbfgs_variant,
            lbfgs_delta_ratio=lbfgs_delta_ratio,
            lbfgs_memory=lbfgs_memory,
            lbfgs_max_line_search=lbfgs_max_line_search,
            lbfgs_min_function_evaluations=(
                lbfgs_min_function_evaluations
            ),
            lbfgs_max_function_evaluations=(
                lbfgs_max_function_evaluations
            ),
            lbfgs_tolerance=lbfgs_tolerance,
            warm_start=warm_start,
        ),
        threads,
    )


def _combine(left: _Point, right: _Point, right_weight: float) -> _Point:
    left_weight = 1.0 - right_weight
    return _Point(
        left_weight * left.x + right_weight * right.x,
        left_weight * left.p + right_weight * right.p,
        left_weight * left.q + right_weight * right.q,
    )


def _combine_image(
    left: _LinearImage,
    right: _LinearImage,
    right_weight: float,
) -> _LinearImage:
    left_weight = 1.0 - right_weight
    return _LinearImage(
        left_weight * left.factor + right_weight * right.factor,
        left_weight * left.constraint + right_weight * right.constraint,
        left_weight * left.adjoint + right_weight * right.adjoint,
    )


def _interval_support(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if value.size == 0:
        return 0.0
    positive = value > 0.0
    negative = value < 0.0
    if np.any(positive & ~np.isfinite(upper)):
        return math.inf
    if np.any(negative & ~np.isfinite(lower)):
        return math.inf
    result = 0.0
    if np.any(positive):
        result += float(value[positive] @ upper[positive])
    if np.any(negative):
        result += float(value[negative] @ lower[negative])
    return result


def _f_value(problem: Any, x: np.ndarray) -> float:
    value = perspective_value(x, problem.k, tolerance=1e-7)
    if not math.isfinite(value):
        return math.inf
    return (
        problem.perspective_weight * value
        - problem.return_reward * float(problem.mu @ x)
    )


def _product_value(
    problem: Any,
    point: _Point,
    constraint_scale: float,
) -> float:
    support = _interval_support(
        constraint_scale * point.q,
        problem.lower,
        problem.upper,
    )
    return (
        _f_value(problem, point.x)
        + 0.5 * float(point.p @ point.p)
        + support
    )


def _forward(
    problem: Any,
    x: np.ndarray,
    constraint_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        _factor_forward(problem, x),
        constraint_scale * _constraint_forward(problem, x),
    )


def _adjoint(
    problem: Any,
    p: np.ndarray,
    q: np.ndarray,
    constraint_scale: float,
) -> np.ndarray:
    result = _factor_adjoint(problem, p)
    if problem.constraints:
        result = result + constraint_scale * _constraint_adjoint(
            problem,
            q,
        )
    return np.asarray(result, dtype=float).reshape(-1)


def _linear_image(
    problem: Any,
    point: _Point,
    constraint_scale: float,
) -> _LinearImage:
    factor, constraint = _forward(
        problem,
        point.x,
        constraint_scale,
    )
    adjoint = _adjoint(
        problem,
        point.p,
        point.q,
        constraint_scale,
    )
    return _LinearImage(factor, constraint, adjoint)


def _prox_product(
    problem: Any,
    point: _Point,
    step_x: float,
    step_y: float,
    constraint_scale: float,
    pava_backend: str,
) -> _Point:
    x = pava_prox(
        point.x + step_x * problem.return_reward * problem.mu,
        step_x * problem.perspective_weight,
        problem.k,
        pava_backend,
    )
    p = point.p / (1.0 + step_y)
    if problem.constraints:
        q = _interval_dual_prox(
            point.q,
            step_y * constraint_scale,
            problem.lower,
            problem.upper,
        )
    else:
        q = point.q.copy()
    return _Point(x, p, q)


def _smoothed_oracle(
    problem: Any,
    point: _Point,
    beta_x: float,
    beta_y: float,
    constraint_scale: float,
    pava_backend: str,
    *,
    evaluate_gap: bool,
    evaluate_smooth: bool = False,
    point_image: Optional[_LinearImage] = None,
) -> _OracleEvaluation:
    image = (
        _linear_image(problem, point, constraint_scale)
        if point_image is None
        else point_image
    )
    shifted = _Point(
        point.x - image.adjoint / beta_x,
        point.p + image.factor / beta_y,
        point.q + image.constraint / beta_y,
    )
    proximal = _prox_product(
        problem,
        shifted,
        1.0 / beta_x,
        1.0 / beta_y,
        constraint_scale,
        pava_backend,
    )
    proximal_image = _linear_image(
        problem,
        proximal,
        constraint_scale,
    )
    gradient = _Point(
        proximal_image.adjoint
        + beta_x * (proximal.x - point.x),
        -proximal_image.factor
        + beta_y * (proximal.p - point.p),
        -proximal_image.constraint
        + beta_y * (proximal.q - point.q),
    )

    dx = point.x - proximal.x
    dp = point.p - proximal.p
    dq = point.q - proximal.q
    residual_squared = (
        beta_x * float(dx @ dx)
        + beta_y * float(dp @ dp)
        + beta_y * float(dq @ dq)
    )
    residual = math.sqrt(max(0.0, residual_squared))

    if not evaluate_gap and not evaluate_smooth:
        return _OracleEvaluation(
            proximal,
            gradient,
            None,
            None,
            None,
            residual,
            image,
            proximal_image,
        )

    value_proximal = _product_value(
        problem,
        proximal,
        constraint_scale,
    )
    coupling = (
        -float(point.p @ proximal_image.factor)
        - float(point.q @ proximal_image.constraint)
        + float(image.factor @ proximal.p)
        + float(image.constraint @ proximal.q)
    )
    quadratic = 0.5 * residual_squared
    smooth_value = -value_proximal + coupling - quadratic
    if not evaluate_gap:
        return _OracleEvaluation(
            proximal,
            gradient,
            None,
            None,
            float(smooth_value),
            residual,
            image,
            proximal_image,
        )

    value_point = _product_value(
        problem,
        point,
        constraint_scale,
    )
    raw_gap = value_point + smooth_value
    scale = (
        1.0
        + abs(value_point)
        + abs(value_proximal)
        + abs(coupling)
        + abs(quadratic)
    )
    if raw_gap < -1e-9 * scale:
        raise ArithmeticError(
            "the computed self-centered smoothed gap is materially "
            f"negative ({raw_gap}); check the saddle scaling"
        )
    gap = max(0.0, float(raw_gap))
    return _OracleEvaluation(
        proximal,
        gradient,
        gap,
        float(raw_gap),
        float(smooth_value),
        residual,
        image,
        proximal_image,
    )


def _pack_point(point: _Point) -> np.ndarray:
    return np.concatenate((point.x, point.p, point.q))


def _unpack_point(problem: Any, values: np.ndarray) -> _Point:
    dimension = int(problem.dimension)
    factors = int(problem.factors)
    first = dimension
    second = first + factors
    return _Point(
        np.asarray(values[:first], dtype=float).copy(),
        np.asarray(values[first:second], dtype=float).copy(),
        np.asarray(values[second:], dtype=float).copy(),
    )


def _lbfgs_polish(
    problem: Any,
    point: _Point,
    point_image: _LinearImage,
    beta_x: float,
    beta_y: float,
    beta_x0: float,
    beta_y0: float,
    gamma_x: float,
    gamma_y: float,
    constraint_scale: float,
    pava_backend: str,
    *,
    delta_ratio: float,
    memory: int,
    max_line_search: int,
    function_budget: int,
    tolerance: float,
) -> _LbfgsPolish:
    """Apply the paper's fixed-center fully smoothed L-BFGS add-on."""
    current_oracle = _smoothed_oracle(
        problem,
        point,
        beta_x,
        beta_y,
        constraint_scale,
        pava_backend,
        evaluate_gap=True,
        point_image=point_image,
    )
    if (
        current_oracle.gap is None
        or current_oracle.smooth_value is None
    ):
        raise RuntimeError("L-BFGS center gap evaluation failed")
    center = _Point(
        -current_oracle.gradient.x,
        -current_oracle.gradient.p,
        -current_oracle.gradient.q,
    )
    delta_x = delta_ratio * beta_x
    delta_y = delta_ratio * beta_y
    initial_vector = _pack_point(point)
    pava_calls = 1
    product_prox_calls = 1
    gap_evaluations = 1
    gradient_evaluations = 1
    function_evaluations = 0
    cached_vector: Optional[np.ndarray] = None
    cached_value = math.inf
    cached_gradient: Optional[np.ndarray] = None
    cached_phi: Optional[_OracleEvaluation] = None

    def objective_gradient(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        nonlocal pava_calls, product_prox_calls
        nonlocal gradient_evaluations, function_evaluations
        nonlocal cached_vector, cached_value, cached_gradient
        nonlocal cached_phi
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if (
            cached_vector is not None
            and np.array_equal(vector, cached_vector)
            and cached_gradient is not None
        ):
            return cached_value, cached_gradient
        candidate = _unpack_point(problem, vector)
        if np.array_equal(vector, initial_vector):
            phi = current_oracle
        else:
            phi = _smoothed_oracle(
                problem,
                candidate,
                beta_x,
                beta_y,
                constraint_scale,
                pava_backend,
                evaluate_gap=False,
                evaluate_smooth=True,
            )
            pava_calls += 1
            product_prox_calls += 1
            gradient_evaluations += 1
        if phi.smooth_value is None:
            raise RuntimeError("smooth SC-SDG evaluation failed")
        shifted = _Point(
            candidate.x + delta_x * center.x,
            candidate.p + delta_y * center.p,
            candidate.q + delta_y * center.q,
        )
        proximal = _prox_product(
            problem,
            shifted,
            delta_x,
            delta_y,
            constraint_scale,
            pava_backend,
        )
        pava_calls += 1
        product_prox_calls += 1
        dx = candidate.x - proximal.x
        dp = candidate.p - proximal.p
        dq = candidate.q - proximal.q
        fully_smoothed_value = (
            _product_value(problem, proximal, constraint_scale)
            + float(dx @ center.x)
            + float(dp @ center.p)
            + float(dq @ center.q)
            + 0.5 * float(dx @ dx) / delta_x
            + 0.5
            * (float(dp @ dp) + float(dq @ dq))
            / delta_y
        )
        gradient = _Point(
            center.x + dx / delta_x + phi.gradient.x,
            center.p + dp / delta_y + phi.gradient.p,
            center.q + dq / delta_y + phi.gradient.q,
        )
        value = fully_smoothed_value + float(phi.smooth_value)
        packed_gradient = _pack_point(gradient)
        function_evaluations += 1
        cached_vector = vector.copy()
        cached_value = float(value)
        cached_gradient = packed_gradient
        cached_phi = phi
        return cached_value, cached_gradient

    objective_initial, _ = objective_gradient(initial_vector)
    active_tolerance = min(
        tolerance,
        max(float(current_oracle.gap) / 10.0, 1e-15),
    )
    optimization = minimize(
        objective_gradient,
        initial_vector,
        method="L-BFGS-B",
        jac=True,
        tol=active_tolerance,
        options={
            "maxfun": function_budget,
            "maxiter": function_budget,
            "maxcor": memory,
            "maxls": max_line_search,
            "ftol": active_tolerance,
            "gtol": active_tolerance,
        },
    )
    final_vector = np.asarray(
        optimization.x,
        dtype=float,
    ).reshape(-1)
    objective_final, final_gradient = objective_gradient(final_vector)
    if cached_phi is None:
        raise RuntimeError("L-BFGS returned no smooth oracle evaluation")
    raw_point = _unpack_point(problem, final_vector)
    correction_trial = _Point(
        raw_point.x - gamma_x * cached_phi.gradient.x,
        raw_point.p - gamma_y * cached_phi.gradient.p,
        raw_point.q - gamma_y * cached_phi.gradient.q,
    )
    candidate = _prox_product(
        problem,
        correction_trial,
        gamma_x,
        gamma_y,
        constraint_scale,
        pava_backend,
    )
    pava_calls += 1
    product_prox_calls += 1
    candidate_image = _linear_image(
        problem,
        candidate,
        constraint_scale,
    )
    candidate_current = _smoothed_oracle(
        problem,
        candidate,
        beta_x,
        beta_y,
        constraint_scale,
        pava_backend,
        evaluate_gap=True,
        point_image=candidate_image,
    )
    pava_calls += 1
    product_prox_calls += 1
    gap_evaluations += 1
    gradient_evaluations += 1
    if beta_x == beta_x0 and beta_y == beta_y0:
        candidate_fixed = candidate_current
    else:
        candidate_fixed = _smoothed_oracle(
            problem,
            candidate,
            beta_x0,
            beta_y0,
            constraint_scale,
            pava_backend,
            evaluate_gap=True,
            point_image=candidate_image,
        )
        pava_calls += 1
        product_prox_calls += 1
        gap_evaluations += 1
        gradient_evaluations += 1
    if candidate_fixed.gap is None or candidate_current.gap is None:
        raise RuntimeError("L-BFGS candidate gap evaluation failed")
    return _LbfgsPolish(
        candidate=candidate,
        candidate_image=candidate_image,
        center_oracle=current_oracle,
        fixed_oracle=candidate_fixed,
        current_oracle=candidate_current,
        iterations=int(optimization.nit),
        function_evaluations=int(function_evaluations),
        pava_calls=int(pava_calls),
        product_prox_calls=int(product_prox_calls),
        gap_evaluations=int(gap_evaluations),
        gradient_evaluations=int(gradient_evaluations),
        objective_initial=float(objective_initial),
        objective_final=float(objective_final),
        gradient_norm=float(
            np.linalg.norm(final_gradient, ord=np.inf)
        ),
        optimizer_status=str(optimization.message),
    )


def _majorization_test(
    center: _Point,
    candidate: _Point,
    gradient: _Point,
    center_smooth_value: float,
    candidate_smooth_value: float,
    gamma_x: float,
    gamma_y: float,
    tolerance: float,
) -> Tuple[bool, float, float]:
    """Test the local quadratic upper model of the smooth SC-SDG term."""
    dx = candidate.x - center.x
    dp = candidate.p - center.p
    dq = candidate.q - center.q
    linear = (
        float(gradient.x @ dx)
        + float(gradient.p @ dp)
        + float(gradient.q @ dq)
    )
    quadratic = 0.5 * (
        float(dx @ dx) / gamma_x
        + (float(dp @ dp) + float(dq @ dq)) / gamma_y
    )
    model = center_smooth_value + linear + quadratic
    scale = (
        1.0
        + abs(candidate_smooth_value)
        + abs(center_smooth_value)
        + abs(linear)
        + abs(quadratic)
    )
    slack = model - candidate_smooth_value
    accepted = candidate_smooth_value <= model + tolerance * scale
    return bool(accepted), float(slack), float(scale)


def _operator_majorization_test(
    momentum: _Point,
    candidate_momentum: _Point,
    momentum_image: _LinearImage,
    candidate_image: _LinearImage,
    beta_x: float,
    beta_y: float,
    gamma_x: float,
    gamma_y: float,
    safety: float,
    tolerance: float,
) -> Tuple[bool, float, float, float]:
    """Check a sufficient directional upper-model inequality.

    The smooth SC-SDG component satisfies an upper Taylor model whose
    curvature along ``delta`` is

        ||K delta_x||^2 / beta_y
        + ||K.T delta_y||^2 / beta_x.

    This test therefore needs matrix products but no additional PAVA call.
    """
    dx = candidate_momentum.x - momentum.x
    dp = candidate_momentum.p - momentum.p
    dq = candidate_momentum.q - momentum.q
    factor_dx = candidate_image.factor - momentum_image.factor
    constraint_dx = (
        candidate_image.constraint - momentum_image.constraint
    )
    adjoint_dy = candidate_image.adjoint - momentum_image.adjoint
    curvature = (
        (
            float(factor_dx @ factor_dx)
            + float(constraint_dx @ constraint_dx)
        )
        / beta_y
        + float(adjoint_dy @ adjoint_dy) / beta_x
    )
    metric = (
        float(dx @ dx) / gamma_x
        + (float(dp @ dp) + float(dq @ dq)) / gamma_y
    )
    model_limit = safety * metric
    scale = 1.0 + abs(curvature) + abs(model_limit)
    slack = model_limit - curvature
    accepted = curvature <= model_limit + tolerance * scale
    ratio = curvature / metric if metric > 0.0 else 0.0
    return bool(accepted), float(slack), float(scale), float(ratio)


def _operator_setup(
    problem: Any,
    settings: _Settings,
) -> Dict[str, Any]:
    base_weight = max(1.0, 0.0064 * float(problem.dimension))
    row_multiplier = min(
        16.0,
        max(1.0, float(problem.constraints) / 16.0),
    )
    constraint_weight = (
        base_weight * row_multiplier
        if settings.constraint_weight is None
        else settings.constraint_weight
    )
    dual_dimension = problem.factors + problem.constraints
    gram_work = (
        problem.dimension * problem.factors * problem.factors
    )
    gram_affordable = (
        dual_dimension <= 512 and gram_work <= 3.0e8
    )
    use_gram = (
        settings.operator_norm_mode == "gram"
        or (
            settings.operator_norm_mode == "auto"
            and gram_affordable
        )
    )
    if use_gram:
        norm = _gram_norm(
            *_gram_data(problem),
            constraint_weight,
        )
        return {
            "operator_norm": norm,
            "beta_norm": norm,
            "operator_norm_kind": "exact_dual_gram",
            "theorem_step_certified": True,
            "constraint_weight": constraint_weight,
        }

    power_norm = _operator_norm(
        problem,
        constraint_weight,
        settings.norm_iterations,
    )
    if settings.operator_norm_mode == "frobenius":
        step_norm = _frobenius_operator_bound(
            problem,
            constraint_weight,
        )
        kind = "frobenius_upper_bound"
        certified = True
    else:
        step_norm = settings.norm_safety * power_norm
        kind = "power_estimate_with_safety"
        certified = False
    return {
        "operator_norm": step_norm,
        "beta_norm": power_norm,
        "operator_norm_kind": kind,
        "theorem_step_certified": certified,
        "constraint_weight": constraint_weight,
    }


def _history_point(
    problem: Any,
    iteration: int,
    elapsed: float,
    point: _Point,
    gap: float,
    initial_gap: float,
    residual: float,
    initial_residual: float,
    violation: float,
    dual_objective: float,
    best_dual_bound: float,
    restart: bool,
    epoch: int,
    epoch_iteration: int,
    beta_x: float,
    beta_y: float,
    theta: float,
    step_scale: float,
    cumulative_backtracks: int,
) -> Dict[str, Any]:
    return {
        "iteration": int(iteration),
        "elapsed_seconds": float(elapsed),
        "objective": float(_objective(problem, point.x)),
        "smoothed_gap": float(gap),
        "relative_smoothed_gap": float(
            gap / max(initial_gap, 1e-300)
        ),
        "residual": float(residual),
        "relative_residual": float(
            residual / max(initial_residual, 1e-300)
        ),
        "violation": float(violation),
        "dual_objective": (
            float(dual_objective)
            if math.isfinite(dual_objective)
            else None
        ),
        "best_dual_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "restart": bool(restart),
        "epoch": int(epoch),
        "epoch_iteration": int(epoch_iteration),
        "beta_x": float(beta_x),
        "beta_y": float(beta_y),
        "theta": float(theta),
        "step_scale": float(step_scale),
        "cumulative_backtracks": int(cumulative_backtracks),
    }


def _solve(
    instance: Any,
    settings: _Settings,
    threadpools: list[Dict[str, Any]],
    threads: int,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    problem = _prepare_problem(
        instance,
        normalize_objective=settings.normalize_objective,
        center_return_objective=settings.center_return_objective,
    )
    line_search_mode = settings.line_search_mode
    if line_search_mode == "auto":
        line_search_mode = (
            "majorization"
            if (
                problem.constraints
                <= settings.line_search_auto_row_threshold
            )
            else "operator"
        )
    operator = _operator_setup(problem, settings)
    operator_norm = float(operator["operator_norm"])
    beta_norm = float(operator["beta_norm"])
    if (
        not math.isfinite(operator_norm)
        or not math.isfinite(beta_norm)
        or operator_norm <= 0.0
        or beta_norm <= 0.0
    ):
        raise RuntimeError("the stacked saddle operator is zero or nonfinite")

    root_product = (
        beta_norm
        * math.sqrt(
            settings.target_cbar * settings.theta_parameter
        )
        / settings.continuation_offset
    )
    beta_x0 = root_product * settings.step_ratio
    beta_y0 = root_product / settings.step_ratio
    achieved_cbar = (
        beta_x0
        * beta_y0
        * settings.continuation_offset**2
        / (settings.theta_parameter * beta_norm**2)
    )
    constraint_weight = float(operator["constraint_weight"])
    constraint_scale = math.sqrt(constraint_weight)
    setup_seconds = time.perf_counter() - total_start

    if settings.warm_start is None:
        point = _Point(
            problem.anchor.copy(),
            np.zeros(problem.factors),
            np.zeros(problem.constraints),
        )
        warm_start_used = False
        warm_start_exact_structure = False
    else:
        state = settings.warm_start
        point_x = state.compatible_primal(problem)
        point_p = np.zeros(problem.factors)
        if (
            state.factor_dual is not None
            and state.factor_dual.shape == (problem.factors,)
        ):
            point_p = (
                state.factor_dual.copy()
                / math.sqrt(problem.objective_scale)
            )
        point_q = np.zeros(problem.constraints)
        mapped_constraint = state.mapped_constraint_dual(problem)
        if mapped_constraint is not None:
            internal_constraint = (
                mapped_constraint * problem.row_norms
                - problem.original_return_reward
                * problem.return_centering_coefficients
            ) / problem.objective_scale
            point_q = internal_constraint / constraint_scale
        point = _Point(point_x, point_p, point_q)
        warm_start_used = True
        warm_start_exact_structure = bool(
            state.constraint_ids == problem.constraint_names
        )
    momentum_point = point.copy()
    initial_oracle = _smoothed_oracle(
        problem,
        point,
        beta_x0,
        beta_y0,
        constraint_scale,
        settings.pava_backend,
        evaluate_gap=True,
    )
    if initial_oracle.gap is None or initial_oracle.raw_gap is None:
        raise RuntimeError("initial smoothed-gap evaluation failed")
    point_image = initial_oracle.point_image
    momentum_image = point_image
    initial_gap = float(initial_oracle.gap)
    initial_residual = max(float(initial_oracle.residual), 1e-16)
    current_gap = initial_gap
    current_raw_gap = float(initial_oracle.raw_gap)
    residual = float(initial_oracle.residual)
    relative_residual = residual / initial_residual
    violation = _maximum_violation(problem, point.x)

    (
        best_dual_bound,
        best_constraint,
        maximum_dual_correction,
    ) = _dual_bound(
        problem,
        point.p,
        constraint_scale * point.q,
    )
    best_factor = point.p.copy()
    best_dual_source = "primary_iterate"
    (
        initial_proximal_dual,
        initial_proximal_constraint,
        initial_proximal_correction,
    ) = _dual_bound(
        problem,
        initial_oracle.proximal_point.p,
        constraint_scale * initial_oracle.proximal_point.q,
    )
    if initial_proximal_dual > best_dual_bound:
        best_dual_bound = initial_proximal_dual
        best_factor = initial_oracle.proximal_point.p.copy()
        best_constraint = initial_proximal_constraint.copy()
        best_dual_source = "fixed_gap_proximal_point"
    maximum_dual_correction = max(
        maximum_dual_correction,
        initial_proximal_correction,
    )
    best_residual = residual
    best_violation = violation
    dual_evaluations = 2
    pava_calls = 1
    gap_evaluations = 1
    product_prox_calls = 1
    gradient_evaluations = 0
    line_search_evaluations = 0
    line_search_backtracks = 0
    line_search_failures = 0
    lbfgs_calls = 0
    lbfgs_accepted_calls = 0
    lbfgs_rejected_calls = 0
    lbfgs_restart_calls = 0
    lbfgs_anticipated_calls = 0
    lbfgs_iterations = 0
    lbfgs_function_evaluations = 0
    lbfgs_pava_calls = 0
    lbfgs_previous_epoch_length = (
        settings.lbfgs_min_function_evaluations
    )
    lbfgs_anticipated_done = False
    lbfgs_since_checkpoint = False
    lbfgs_accepted_since_checkpoint = False
    lbfgs_trigger_since_checkpoint: Optional[str] = None
    lbfgs_last_optimizer_status: Optional[str] = None
    lbfgs_last_gradient_norm: Optional[float] = None
    lbfgs_last_objective_reduction: Optional[float] = None
    accepted_step_scale = (
        settings.line_search_initial_scale
        if settings.line_search
        else 1.0
    )
    minimum_step_scale = (
        math.inf if settings.line_search else 1.0
    )
    maximum_step_scale = (
        0.0 if settings.line_search else 1.0
    )
    sum_step_scales = 0.0
    minimum_majorization_slack = math.inf
    maximum_accepted_curvature_ratio = 0.0
    restarts = 0
    epoch = 0
    epoch_iteration = 0
    restart_target = (
        initial_gap * settings.restart_factor
        if settings.restart
        else -math.inf
    )
    minimum_raw_gap = current_raw_gap
    beta_x = beta_x0
    beta_y = beta_y0
    theta = 1.0
    history = [
        _history_point(
            problem,
            0,
            0.0,
            point,
            current_gap,
            initial_gap,
            residual,
            initial_residual,
            violation,
            best_dual_bound,
            best_dual_bound,
            False,
            epoch,
            epoch_iteration,
            beta_x,
            beta_y,
            theta,
            accepted_step_scale,
            line_search_backtracks,
        )
    ]
    history[0].update(
        {
            "lbfgs_triggered": False,
            "lbfgs_accepted": False,
            "lbfgs_trigger": None,
            "cumulative_lbfgs_calls": 0,
        }
    )
    status = "iteration_limit"
    completed = 0
    restart_since_checkpoint = False
    solve_start = time.perf_counter()

    for iteration in range(1, settings.max_iterations + 1):
        elapsed_before = time.perf_counter() - solve_start
        if (
            settings.time_limit is not None
            and elapsed_before >= settings.time_limit
        ):
            status = "time_limit"
            break

        local_iteration = epoch_iteration
        theta = settings.theta_parameter / (
            local_iteration + settings.theta_parameter
        )
        beta_multiplier = settings.continuation_offset / (
            local_iteration + settings.continuation_offset
        )
        beta_x = beta_x0 * beta_multiplier
        beta_y = beta_y0 * beta_multiplier
        denominator = (
            operator_norm * operator_norm
            + 2.0 * beta_x * beta_y
        )
        base_gamma_x = beta_y / denominator
        base_gamma_y = beta_x / denominator

        extrapolated = _combine(point, momentum_point, theta)
        extrapolated_image = _combine_image(
            point_image,
            momentum_image,
            theta,
        )
        oracle = _smoothed_oracle(
            problem,
            extrapolated,
            beta_x,
            beta_y,
            constraint_scale,
            settings.pava_backend,
            evaluate_gap=(
                settings.line_search
                and line_search_mode == "majorization"
            ),
            point_image=extrapolated_image,
        )
        gradient_evaluations += 1
        pava_calls += 1
        product_prox_calls += 1
        if (
            settings.line_search
            and line_search_mode == "majorization"
        ):
            gap_evaluations += 1
            if oracle.smooth_value is None:
                raise RuntimeError(
                    "line-search center evaluation did not return "
                    "the smooth SC-SDG value"
                )
            trial_scale = min(
                settings.line_search_max_scale,
                accepted_step_scale * settings.line_search_growth,
            )
        elif settings.line_search:
            trial_scale = min(
                settings.line_search_max_scale,
                accepted_step_scale * settings.line_search_growth,
            )
        else:
            trial_scale = 1.0

        next_point: Optional[_Point] = None
        next_momentum: Optional[_Point] = None
        next_point_image: Optional[_LinearImage] = None
        next_momentum_image: Optional[_LinearImage] = None
        accepted_trial_oracle: Optional[_OracleEvaluation] = None
        accepted_slack = math.inf
        accepted_curvature_ratio = 0.0
        attempts = (
            settings.line_search_max_backtracks + 1
            if settings.line_search
            else 1
        )
        for attempt in range(attempts):
            gamma_x = trial_scale * base_gamma_x
            gamma_y = trial_scale * base_gamma_y
            step_x = gamma_x / theta
            step_y = gamma_y / theta
            trial = _Point(
                momentum_point.x - step_x * oracle.gradient.x,
                momentum_point.p - step_y * oracle.gradient.p,
                momentum_point.q - step_y * oracle.gradient.q,
            )
            candidate_momentum = _prox_product(
                problem,
                trial,
                step_x,
                step_y,
                constraint_scale,
                settings.pava_backend,
            )
            pava_calls += 1
            product_prox_calls += 1
            candidate_point = _combine(
                point,
                candidate_momentum,
                theta,
            )
            candidate_momentum_image = _linear_image(
                problem,
                candidate_momentum,
                constraint_scale,
            )
            candidate_point_image = _combine_image(
                point_image,
                candidate_momentum_image,
                theta,
            )
            if not settings.line_search:
                next_point = candidate_point
                next_momentum = candidate_momentum
                next_point_image = candidate_point_image
                next_momentum_image = candidate_momentum_image
                break

            line_search_evaluations += 1
            if line_search_mode == "operator":
                (
                    accepted,
                    slack,
                    _,
                    curvature_ratio,
                ) = _operator_majorization_test(
                    momentum_point,
                    candidate_momentum,
                    momentum_image,
                    candidate_momentum_image,
                    beta_x,
                    beta_y,
                    gamma_x,
                    gamma_y,
                    settings.line_search_safety,
                    settings.line_search_tolerance,
                )
                candidate_oracle = None
            else:
                candidate_oracle = _smoothed_oracle(
                    problem,
                    candidate_point,
                    beta_x,
                    beta_y,
                    constraint_scale,
                    settings.pava_backend,
                    evaluate_gap=True,
                    point_image=candidate_point_image,
                )
                pava_calls += 1
                product_prox_calls += 1
                gradient_evaluations += 1
                gap_evaluations += 1
                if candidate_oracle.smooth_value is None:
                    raise RuntimeError(
                        "line-search trial evaluation did not return "
                        "the smooth SC-SDG value"
                    )
                accepted, slack, _ = _majorization_test(
                    extrapolated,
                    candidate_point,
                    oracle.gradient,
                    float(oracle.smooth_value),
                    float(candidate_oracle.smooth_value),
                    gamma_x,
                    gamma_y,
                    settings.line_search_tolerance,
                )
                curvature_ratio = math.nan
            if accepted:
                next_point = candidate_point
                next_momentum = candidate_momentum
                next_point_image = candidate_point_image
                next_momentum_image = candidate_momentum_image
                accepted_trial_oracle = candidate_oracle
                accepted_slack = slack
                accepted_curvature_ratio = curvature_ratio
                break
            if attempt >= settings.line_search_max_backtracks:
                break
            line_search_backtracks += 1
            trial_scale *= settings.line_search_shrink
            if (
                not math.isfinite(trial_scale)
                or trial_scale <= np.finfo(float).tiny
            ):
                break

        if (
            next_point is None
            or next_momentum is None
            or next_point_image is None
            or next_momentum_image is None
        ):
            line_search_failures += 1
            raise RuntimeError(
                "SC-SDG line search failed after "
                f"{settings.line_search_max_backtracks} backtracks"
            )
        if settings.line_search:
            accepted_step_scale = trial_scale
            minimum_step_scale = min(
                minimum_step_scale,
                accepted_step_scale,
            )
            maximum_step_scale = max(
                maximum_step_scale,
                accepted_step_scale,
            )
            minimum_majorization_slack = min(
                minimum_majorization_slack,
                accepted_slack,
            )
            if math.isfinite(accepted_curvature_ratio):
                maximum_accepted_curvature_ratio = max(
                    maximum_accepted_curvature_ratio,
                    accepted_curvature_ratio,
                )
        else:
            accepted_step_scale = 1.0
        sum_step_scales += accepted_step_scale
        point = next_point
        momentum_point = next_momentum
        point_image = next_point_image
        momentum_image = next_momentum_image
        completed = iteration
        epoch_iteration += 1

        elapsed = time.perf_counter() - solve_start
        restart_due = bool(
            settings.restart
            and epoch_iteration >= settings.min_restart_iterations
            and epoch_iteration % settings.restart_check_interval == 0
        )
        checkpoint_due = bool(
            iteration % settings.check_interval == 0
            or iteration == settings.max_iterations
            or (
                settings.time_limit is not None
                and elapsed >= settings.time_limit
            )
        )
        lbfgs_anticipated_due = bool(
            settings.lbfgs_variant == "paper"
            and not lbfgs_anticipated_done
            and epoch_iteration
            >= 2 * lbfgs_previous_epoch_length
        )
        if (
            not restart_due
            and not checkpoint_due
            and not lbfgs_anticipated_due
        ):
            continue

        fixed_oracle = _smoothed_oracle(
            problem,
            point,
            beta_x0,
            beta_y0,
            constraint_scale,
            settings.pava_backend,
            evaluate_gap=True,
            point_image=point_image,
        )
        pava_calls += 1
        product_prox_calls += 1
        gap_evaluations += 1
        if fixed_oracle.gap is None or fixed_oracle.raw_gap is None:
            raise RuntimeError("fixed smoothed-gap evaluation failed")
        current_gap = float(fixed_oracle.gap)
        current_raw_gap = float(fixed_oracle.raw_gap)
        minimum_raw_gap = min(minimum_raw_gap, current_raw_gap)
        residual = float(fixed_oracle.residual)
        relative_residual = residual / initial_residual

        restart_now = bool(
            restart_due
            and current_gap <= restart_target
        )
        lbfgs_trigger: Optional[str] = None
        if settings.lbfgs_variant != "off":
            if restart_now:
                lbfgs_trigger = "restart"
            elif lbfgs_anticipated_due:
                lbfgs_trigger = "anticipated"

        if lbfgs_trigger is not None:
            epoch_length = max(epoch_iteration, 1)
            requested_budget = max(
                settings.lbfgs_min_function_evaluations,
                (
                    epoch_length
                    if lbfgs_trigger == "restart"
                    else lbfgs_previous_epoch_length
                ),
            )
            function_budget = min(
                requested_budget,
                settings.lbfgs_max_function_evaluations,
            )
            polish = _lbfgs_polish(
                problem,
                point,
                point_image,
                beta_x,
                beta_y,
                beta_x0,
                beta_y0,
                gamma_x,
                gamma_y,
                constraint_scale,
                settings.pava_backend,
                delta_ratio=settings.lbfgs_delta_ratio,
                memory=settings.lbfgs_memory,
                max_line_search=settings.lbfgs_max_line_search,
                function_budget=function_budget,
                tolerance=settings.lbfgs_tolerance,
            )
            lbfgs_calls += 1
            lbfgs_iterations += polish.iterations
            lbfgs_function_evaluations += (
                polish.function_evaluations
            )
            lbfgs_pava_calls += polish.pava_calls
            pava_calls += polish.pava_calls
            product_prox_calls += polish.product_prox_calls
            gap_evaluations += polish.gap_evaluations
            gradient_evaluations += polish.gradient_evaluations
            if lbfgs_trigger == "restart":
                lbfgs_restart_calls += 1
            else:
                lbfgs_anticipated_calls += 1
                lbfgs_anticipated_done = True
            lbfgs_since_checkpoint = True
            lbfgs_trigger_since_checkpoint = lbfgs_trigger
            lbfgs_last_optimizer_status = polish.optimizer_status
            lbfgs_last_gradient_norm = polish.gradient_norm
            lbfgs_last_objective_reduction = (
                polish.objective_initial - polish.objective_final
            )

            (
                polish_dual,
                polish_constraint,
                polish_correction,
            ) = _dual_bound(
                problem,
                polish.candidate.p,
                constraint_scale * polish.candidate.q,
            )
            dual_evaluations += 1
            maximum_dual_correction = max(
                maximum_dual_correction,
                polish_correction,
            )
            if polish_dual > best_dual_bound:
                best_dual_bound = polish_dual
                best_factor = polish.candidate.p.copy()
                best_constraint = polish_constraint.copy()
                best_dual_source = "lbfgs_polished_candidate"

            fixed_improved = bool(
                float(polish.fixed_oracle.gap)
                <= current_gap
                + 1e-14 * max(1.0, abs(current_gap))
            )
            current_improved = bool(
                float(polish.current_oracle.gap)
                < float(polish.center_oracle.gap)
            )
            if settings.lbfgs_variant == "restart_safe":
                accept_polish = fixed_improved
            else:
                accept_polish = bool(
                    (lbfgs_trigger == "restart" and fixed_improved)
                    or current_improved
                )
            if accept_polish:
                point = polish.candidate
                point_image = polish.candidate_image
                fixed_oracle = polish.fixed_oracle
                current_gap = float(fixed_oracle.gap)
                current_raw_gap = float(fixed_oracle.raw_gap)
                minimum_raw_gap = min(
                    minimum_raw_gap,
                    current_raw_gap,
                )
                residual = float(fixed_oracle.residual)
                relative_residual = residual / initial_residual
                lbfgs_accepted_calls += 1
                lbfgs_accepted_since_checkpoint = True
            else:
                lbfgs_rejected_calls += 1

        if restart_now:
            lbfgs_previous_epoch_length = max(
                settings.lbfgs_min_function_evaluations,
                epoch_iteration,
            )
            lbfgs_anticipated_done = False
        if restart_now:
            restarts += 1
            epoch += 1
            momentum_point = point.copy()
            momentum_image = point_image
            epoch_iteration = 0
            restart_target *= settings.restart_factor
            if settings.line_search_reset_on_restart:
                accepted_step_scale = (
                    settings.line_search_initial_scale
                )
            restart_since_checkpoint = True

        if not checkpoint_due:
            continue

        elapsed = time.perf_counter() - solve_start
        violation = _maximum_violation(problem, point.x)
        dual_candidates = [
            ("primary_iterate", point),
            ("momentum_point", momentum_point),
            ("fixed_gap_proximal_point", fixed_oracle.proximal_point),
            ("continuation_proximal_point", oracle.proximal_point),
        ]
        if accepted_trial_oracle is not None:
            dual_candidates.append(
                (
                    "line_search_proximal_point",
                    accepted_trial_oracle.proximal_point,
                )
            )
        dual_objective = -math.inf
        for source, candidate in dual_candidates:
            (
                candidate_dual,
                safe_constraint,
                dual_correction,
            ) = _dual_bound(
                problem,
                candidate.p,
                constraint_scale * candidate.q,
            )
            dual_evaluations += 1
            maximum_dual_correction = max(
                maximum_dual_correction,
                dual_correction,
            )
            if source == "primary_iterate":
                dual_objective = candidate_dual
            if candidate_dual > best_dual_bound:
                best_dual_bound = candidate_dual
                best_factor = candidate.p.copy()
                best_constraint = safe_constraint.copy()
                best_dual_source = source
        if (
            residual < best_residual
            or (
                residual <= best_residual * (1.0 + 1e-12)
                and violation < best_violation
            )
        ):
            best_residual = residual
            best_violation = violation

        history_row = _history_point(
            problem,
            iteration,
            elapsed,
            point,
            current_gap,
            initial_gap,
            residual,
            initial_residual,
            violation,
            dual_objective,
            best_dual_bound,
            restart_since_checkpoint,
            epoch,
            epoch_iteration,
            beta_x,
            beta_y,
            theta,
            accepted_step_scale,
            line_search_backtracks,
        )
        history_row.update(
            {
                "lbfgs_triggered": lbfgs_since_checkpoint,
                "lbfgs_accepted": (
                    lbfgs_accepted_since_checkpoint
                ),
                "lbfgs_trigger": lbfgs_trigger_since_checkpoint,
                "cumulative_lbfgs_calls": int(lbfgs_calls),
            }
        )
        history.append(history_row)
        restart_since_checkpoint = False
        lbfgs_since_checkpoint = False
        lbfgs_accepted_since_checkpoint = False
        lbfgs_trigger_since_checkpoint = None
        if (
            settings.dual_bound_cutoff is not None
            and best_dual_bound >= settings.dual_bound_cutoff
        ):
            status = "dual_bound_cutoff"
            break
        if (
            relative_residual <= settings.tolerance
            and violation <= settings.feasibility_tolerance
        ):
            best_residual = residual
            best_violation = violation
            status = "converged"
            break
        if (
            settings.time_limit is not None
            and elapsed >= settings.time_limit
        ):
            status = "time_limit"
            break

    if completed > 0:
        final_oracle = _smoothed_oracle(
            problem,
            point,
            beta_x0,
            beta_y0,
            constraint_scale,
            settings.pava_backend,
            evaluate_gap=True,
            point_image=point_image,
        )
        pava_calls += 1
        product_prox_calls += 1
        gap_evaluations += 1
        if final_oracle.gap is None or final_oracle.raw_gap is None:
            raise RuntimeError("final smoothed-gap evaluation failed")
        current_gap = float(final_oracle.gap)
        current_raw_gap = float(final_oracle.raw_gap)
        minimum_raw_gap = min(minimum_raw_gap, current_raw_gap)
        residual = float(final_oracle.residual)
        relative_residual = residual / initial_residual
        violation = _maximum_violation(problem, point.x)
        final_dual_objective = -math.inf
        for source, candidate in (
            ("primary_iterate", point),
            ("final_fixed_gap_proximal_point", final_oracle.proximal_point),
        ):
            (
                candidate_dual,
                safe_constraint,
                dual_correction,
            ) = _dual_bound(
                problem,
                candidate.p,
                constraint_scale * candidate.q,
            )
            dual_evaluations += 1
            maximum_dual_correction = max(
                maximum_dual_correction,
                dual_correction,
            )
            if source == "primary_iterate":
                final_dual_objective = candidate_dual
            if candidate_dual > best_dual_bound:
                best_dual_bound = candidate_dual
                best_factor = candidate.p.copy()
                best_constraint = safe_constraint.copy()
                best_dual_source = source
        best_residual = min(best_residual, residual)
        best_violation = min(best_violation, violation)
        final_elapsed = time.perf_counter() - solve_start
        final_history = _history_point(
            problem,
            completed,
            final_elapsed,
            point,
            current_gap,
            initial_gap,
            residual,
            initial_residual,
            violation,
            final_dual_objective,
            best_dual_bound,
            restart_since_checkpoint,
            epoch,
            epoch_iteration,
            beta_x,
            beta_y,
            theta,
            accepted_step_scale,
            line_search_backtracks,
        )
        final_history.update(
            {
                "lbfgs_triggered": lbfgs_since_checkpoint,
                "lbfgs_accepted": (
                    lbfgs_accepted_since_checkpoint
                ),
                "lbfgs_trigger": lbfgs_trigger_since_checkpoint,
                "cumulative_lbfgs_calls": int(lbfgs_calls),
            }
        )
        if history and history[-1]["iteration"] == completed:
            history[-1] = final_history
        else:
            history.append(final_history)
        if (
            status == "iteration_limit"
            and settings.dual_bound_cutoff is not None
            and best_dual_bound >= settings.dual_bound_cutoff
        ):
            status = "dual_bound_cutoff"

    solve_seconds = time.perf_counter() - solve_start
    original_factor_dual = (
        math.sqrt(problem.objective_scale) * best_factor
    )
    original_scaled_constraint_dual = (
        problem.objective_scale * best_constraint
        + problem.original_return_reward
        * problem.return_centering_coefficients
    )
    original_constraint_dual = (
        original_scaled_constraint_dual / problem.row_norms
        if problem.row_norms.size
        else original_scaled_constraint_dual.copy()
    )
    initial_dual_bound, _, _ = _dual_bound(
        problem,
        np.zeros(problem.factors),
        np.zeros(problem.constraints),
    )
    anchor_violation = _maximum_violation(problem, problem.anchor)
    anchor_objective = _objective(problem, problem.anchor)
    anchor_feasible = (
        math.isfinite(anchor_objective)
        and anchor_violation <= 1e-8
    )
    anchor_gap = (
        anchor_objective - best_dual_bound
        if anchor_feasible and math.isfinite(best_dual_bound)
        else None
    )

    current_original_factor = (
        math.sqrt(problem.objective_scale) * point.p
    )
    current_internal_constraint = constraint_scale * point.q
    current_original_scaled_constraint = (
        problem.objective_scale * current_internal_constraint
        + problem.original_return_reward
        * problem.return_centering_coefficients
    )
    current_original_constraint = (
        current_original_scaled_constraint / problem.row_norms
        if problem.row_norms.size
        else current_original_scaled_constraint.copy()
    )
    restart_state = RelaxationState(
        backend="scsdg",
        implementation=_IMPLEMENTATION,
        dimension=problem.dimension,
        k=problem.k,
        constraint_ids=problem.constraint_names,
        x=point.x,
        # Keep the strongest safe certificate in the portable fields so a
        # child node inherits the parent bound immediately.  The current
        # algorithm coordinates remain available privately for a future
        # exact same-problem continuation mode.
        factor_dual=original_factor_dual,
        constraint_dual=original_constraint_dual,
        arrays={
            "momentum_x": point.x.copy(),
            "momentum_factor_dual": current_original_factor.copy(),
            "momentum_constraint_dual": (
                current_original_constraint.copy()
            ),
            "current_factor_dual": current_original_factor.copy(),
            "current_constraint_dual": (
                current_original_constraint.copy()
            ),
        },
        scalars={
            "beta_x": float(beta_x),
            "beta_y": float(beta_y),
            "step_scale": float(accepted_step_scale),
            "fresh_restart_epoch_recommended": True,
        },
    )

    return {
        "x": point.x,
        "status": status,
        "language": _LANGUAGE,
        "implementation": _IMPLEMENTATION,
        "native_solver_core": _IMPLEMENTATION == "native",
        "solver": "scsdg",
        "method": (
            "restarted accelerated proximal gradient on the "
            "self-centered smoothed duality gap"
            + (
                " with local majorization line search"
                if settings.line_search
                else ""
            )
            + (
                " and fixed-center L-BFGS restart polishing"
                if settings.lbfgs_variant != "off"
                else ""
            )
        ),
        "variant": (
            (
                "accelerated-halving-restart"
                if settings.restart
                else "accelerated-no-restart"
            )
            + ("-linesearch" if settings.line_search else "-fixed-step")
            + (
                f"-lbfgs-{settings.lbfgs_variant}"
                if settings.lbfgs_variant != "off"
                else ""
            )
        ),
        "paper": PAPER_URL,
        "certified_optimal": False,
        "effective_tolerance": settings.tolerance,
        "effective_residual_tolerance": settings.tolerance,
        "feasibility_tolerance": settings.feasibility_tolerance,
        "stopping_certificate": (
            "fixed_beta_self_centered_proximal_residual"
            "_and_original_scale_feasibility"
        ),
        "objective_gap_certified": False,
        "dual_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "best_dual_lower_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "initial_dual_objective": (
            float(initial_dual_bound)
            if math.isfinite(initial_dual_bound)
            else None
        ),
        "last_evaluated_dual_objective": (
            history[-1].get("dual_objective") if history else None
        ),
        "dual_bound_evaluations": int(dual_evaluations),
        "best_dual_candidate_source": best_dual_source,
        "dual_candidates_per_checkpoint": (
            5
            if (
                settings.line_search
                and line_search_mode == "majorization"
            )
            else 4
        ),
        "dual_bound_available": math.isfinite(best_dual_bound),
        "dual_bound_safe_in_exact_arithmetic": True,
        "dual_bound_floating_point_certified": False,
        "dual_bound_factor": original_factor_dual,
        "dual_bound_constraint_scaled": original_scaled_constraint_dual,
        "dual_bound_constraint_original": original_constraint_dual,
        "dual_bound_factor_internal": best_factor.copy(),
        "dual_bound_constraint_internal": best_constraint.copy(),
        "maximum_dual_domain_correction": float(
            maximum_dual_correction
        ),
        "dual_bound_kind": "fenchel_weak_duality_lower_bound",
        "dual_bound_units": "original_objective",
        "dual_bound_cutoff": settings.dual_bound_cutoff,
        "anchor_primal_upper_bound": (
            float(anchor_objective) if anchor_feasible else None
        ),
        "anchor_violation": float(anchor_violation),
        "anchor_is_numerically_feasible": bool(anchor_feasible),
        "anchor_primal_dual_gap": (
            float(anchor_gap) if anchor_gap is not None else None
        ),
        "relative_anchor_primal_dual_gap": (
            anchor_gap / max(1.0, abs(anchor_objective))
            if anchor_gap is not None
            else None
        ),
        "dual_bound_formula": (
            "-0.5*||p||^2 - support_[lower,upper](q) "
            "- (perspective_weight*G_k)^*("
            "return_reward*mu - B*p - C_scaled.T*q)"
        ),
        "iterations": int(completed),
        "iteration_kind": "accepted_scsdg_accelerated_iterations",
        "gradient_evaluations": int(gradient_evaluations),
        "product_prox_calls": int(product_prox_calls),
        "pava_calls": int(pava_calls),
        "pava_backend": settings.pava_backend,
        "smoothed_gap_evaluations": int(gap_evaluations),
        "line_search_enabled": bool(settings.line_search),
        "line_search_mode": (
            (
                "directional_operator_upper_model"
                if line_search_mode == "operator"
                else "smooth_scsdg_weighted_quadratic_majorization"
            )
            if settings.line_search
            else "disabled"
        ),
        "line_search_mode_requested": (
            settings.line_search_mode
            if settings.line_search
            else "disabled"
        ),
        "line_search_mode_resolved": (
            line_search_mode if settings.line_search else "disabled"
        ),
        "line_search_auto_row_threshold": int(
            settings.line_search_auto_row_threshold
        ),
        "line_search_is_paper_extension": bool(
            settings.line_search
        ),
        "line_search_trials": int(line_search_evaluations),
        "line_search_model_evaluations": int(
            line_search_evaluations
        ),
        "line_search_backtracks": int(line_search_backtracks),
        "line_search_failures": int(line_search_failures),
        "line_search_initial_scale": float(
            settings.line_search_initial_scale
        ),
        "line_search_final_scale": float(accepted_step_scale),
        "line_search_minimum_accepted_scale": float(
            minimum_step_scale
            if math.isfinite(minimum_step_scale)
            else accepted_step_scale
        ),
        "line_search_maximum_accepted_scale": float(
            maximum_step_scale
            if maximum_step_scale > 0.0
            else accepted_step_scale
        ),
        "line_search_average_accepted_scale": float(
            sum_step_scales / completed
            if completed
            else accepted_step_scale
        ),
        "line_search_minimum_majorization_slack": (
            float(minimum_majorization_slack)
            if math.isfinite(minimum_majorization_slack)
            else None
        ),
        "line_search_maximum_accepted_curvature_ratio": (
            float(maximum_accepted_curvature_ratio)
            if (
                settings.line_search
                and line_search_mode == "operator"
            )
            else None
        ),
        "line_search_growth": float(settings.line_search_growth),
        "line_search_shrink": float(settings.line_search_shrink),
        "line_search_max_scale": float(
            settings.line_search_max_scale
        ),
        "line_search_safety": float(settings.line_search_safety),
        "line_search_tolerance": float(
            settings.line_search_tolerance
        ),
        "line_search_max_backtracks": int(
            settings.line_search_max_backtracks
        ),
        "line_search_reset_on_restart": bool(
            settings.line_search_reset_on_restart
        ),
        "lbfgs_enabled": settings.lbfgs_variant != "off",
        "lbfgs_variant": settings.lbfgs_variant,
        "lbfgs_fixed_center": settings.lbfgs_variant != "off",
        "lbfgs_calls": int(lbfgs_calls),
        "lbfgs_accepted_calls": int(lbfgs_accepted_calls),
        "lbfgs_rejected_calls": int(lbfgs_rejected_calls),
        "lbfgs_restart_calls": int(lbfgs_restart_calls),
        "lbfgs_anticipated_calls": int(lbfgs_anticipated_calls),
        "lbfgs_iterations": int(lbfgs_iterations),
        "lbfgs_function_evaluations": int(
            lbfgs_function_evaluations
        ),
        "lbfgs_pava_calls": int(lbfgs_pava_calls),
        "lbfgs_delta_ratio": float(settings.lbfgs_delta_ratio),
        "lbfgs_memory": int(settings.lbfgs_memory),
        "lbfgs_max_line_search": int(
            settings.lbfgs_max_line_search
        ),
        "lbfgs_min_function_evaluations": int(
            settings.lbfgs_min_function_evaluations
        ),
        "lbfgs_max_function_evaluations": int(
            settings.lbfgs_max_function_evaluations
        ),
        "lbfgs_tolerance": float(settings.lbfgs_tolerance),
        "lbfgs_last_optimizer_status": lbfgs_last_optimizer_status,
        "lbfgs_last_gradient_norm": lbfgs_last_gradient_norm,
        "lbfgs_last_objective_reduction": (
            lbfgs_last_objective_reduction
        ),
        "lbfgs_acceptance_rule": {
            "off": "disabled",
            "paper": "reference_fixed_or_current_gap_improvement",
            "restart_reference": (
                "restart_fixed_or_current_gap_improvement"
            ),
            "restart_safe": "restart_fixed_beta_gap_nonincrease",
        }[settings.lbfgs_variant],
        "lbfgs_trigger_rule": {
            "off": "disabled",
            "paper": "restart_or_twice_previous_epoch_length",
            "restart_reference": "restart_only",
            "restart_safe": "restart_only",
        }[settings.lbfgs_variant],
        "lbfgs_is_paper_extension_without_rate_proof": bool(
            settings.lbfgs_variant != "off"
        ),
        "restarts": int(restarts),
        "restart_certificate": (
            "fixed_beta_self_centered_smoothed_gap_halving"
            if settings.restart
            else "disabled"
        ),
        "restart_action": (
            "momentum_and_continuation_reset"
            if settings.restart
            else "disabled"
        ),
        "restart_factor": float(settings.restart_factor),
        "restart_check_interval": int(
            settings.restart_check_interval
        ),
        "minimum_restart_iterations": int(
            settings.min_restart_iterations
        ),
        "residual": float(residual),
        "relative_residual": float(
            residual / initial_residual
        ),
        "initial_residual": float(initial_residual),
        "best_checkpoint_residual": float(best_residual),
        "smoothed_gap": float(current_gap),
        "relative_smoothed_gap": float(
            current_gap / max(initial_gap, 1e-300)
        ),
        "initial_smoothed_gap": float(initial_gap),
        "minimum_raw_smoothed_gap": float(minimum_raw_gap),
        "violation": float(violation),
        "best_checkpoint_violation": float(best_violation),
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "total_seconds": float(setup_seconds + solve_seconds),
        "history": history,
        "theta_parameter": float(settings.theta_parameter),
        "continuation_offset": float(
            settings.continuation_offset
        ),
        "target_cbar": float(settings.target_cbar),
        "achieved_cbar_using_beta_norm": float(achieved_cbar),
        "beta_x0": float(beta_x0),
        "beta_y0": float(beta_y0),
        "beta_ratio": float(beta_x0 / beta_y0),
        "step_ratio": float(settings.step_ratio),
        "constraint_dual_weight": float(
            constraint_weight
        ),
        "constraint_dual_weight_requested": (
            "auto"
            if settings.constraint_weight is None
            else float(settings.constraint_weight)
        ),
        "constraint_dual_coordinate_scale": float(
            constraint_scale
        ),
        "operator_norm": float(operator_norm),
        "beta_calibration_norm": float(beta_norm),
        "operator_norm_kind": operator["operator_norm_kind"],
        "operator_norm_certified": bool(
            operator["theorem_step_certified"]
        ),
        "theorem_step_certified": bool(
            operator["theorem_step_certified"]
            and not settings.line_search
            and settings.lbfgs_variant == "off"
        ),
        "paper_rate_applies_without_extension": bool(
            not settings.line_search
            and settings.lbfgs_variant == "off"
        ),
        "warm_start_used": bool(warm_start_used),
        "warm_start_exact_structure": bool(
            warm_start_exact_structure
        ),
        "warm_start_acceleration_resumed": False,
        "restart_state": restart_state.to_dict(copy=False),
        "objective_normalization_requested": bool(
            settings.normalize_objective
        ),
        "objective_normalization_applied": bool(
            settings.normalize_objective
            and problem.objective_scale != 1.0
        ),
        "objective_normalization_scale": float(
            problem.objective_scale
        ),
        "internal_perspective_weight": float(
            problem.perspective_weight
        ),
        "original_perspective_weight": float(
            problem.original_perspective_weight
        ),
        "internal_return_reward": float(problem.return_reward),
        "original_return_reward": float(
            problem.original_return_reward
        ),
        "constraint_operator_storage": (
            problem.constraint_operator_storage
        ),
        "constraint_operator_density": float(
            problem.constraint_operator_density
        ),
        "constraint_operator_entries": int(
            problem.constraint_operator_entries
        ),
        "constraint_operator_nnz": int(
            problem.constraint_operator_nnz
        ),
        "constraint_row_normalization_applied": bool(
            problem.row_norms.size
        ),
        "complexity_per_iteration": (
            (
                (
                    "one continuation oracle and one product proximal "
                    "map per operator line-search trial"
                )
                if line_search_mode == "operator"
                else (
                    "one continuation oracle plus two product proximal "
                    "maps per smooth-majorization line-search trial"
                )
            )
            + (
                ", with scheduled fixed-gap evaluations"
            )
            if settings.line_search
            else (
                "two product proximal maps and two applications each "
                "of the stacked operator and its adjoint, plus "
                "scheduled fixed-gap evaluations"
            )
            + (
                "; each L-BFGS value-gradient evaluation uses two "
                "product proximal maps"
                if settings.lbfgs_variant != "off"
                else ""
            )
        ),
        "threads": int(threads),
        "thread_limit_kind": (
            "blas" if threads else "runtime_default"
        ),
        "threadpools": threadpools,
    }


def solve_scsdg(
    instance: Any,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve one perspective relaxation with restarted SC-SDG APG."""
    settings, threads = _settings(options)
    if threads:
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError as error:
            raise RuntimeError(
                "threadpoolctl is required to enforce SC-SDG thread limits"
            ) from error
        thread_context = threadpool_limits(
            limits=threads,
            user_api="blas",
        )
    else:
        thread_context = nullcontext()

    with thread_context:
        try:
            from threadpoolctl import threadpool_info

            threadpools = threadpool_info()
        except ImportError:
            threadpools = []
        return _solve(instance, settings, threadpools, threads)


__all__ = ["PAPER_URL", "solve_scsdg"]
