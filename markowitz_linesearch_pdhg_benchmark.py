#!/usr/bin/env python3
"""Line-search PDHG with adaptive restart for the Markowitz benchmark.

This script solves exactly the same perspective-relaxed Markowitz model
as ``markowitz_pdhg_benchmark.py``.  The only algorithmic change is the
Malitsky--Pock line search.  Its ordering is useful here: every main
iteration evaluates the PAVA proximal oracle exactly once (diagnostic
checkpoints make extra residual calls).  The default small dual Gram
cache also makes rejected trials independent of the portfolio dimension.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import markowitz_pdhg_benchmark as common


common.np = np
OUTPUT_DIRECTORY = Path(__file__).resolve().parent


@dataclass
class LineSearchHistoryPoint:
    iteration: int
    elapsed_seconds: float
    residual: float
    relative_residual: float
    violation: float
    objective: float
    budget_error: float
    return_shortfall: float
    restart: bool
    candidate: str
    primal_step: float
    cumulative_backtracks: int


@dataclass
class LineSearchPdhgResult:
    x: Any
    factor_dual: Any
    constraint_dual: Any
    objective: float
    residual: float
    relative_residual: float
    violation: float
    iterations: int
    pava_calls: int
    restarts: int
    setup_seconds: float
    seconds: float
    total_seconds: float
    status: str
    history: List[LineSearchHistoryPoint]
    tau: float
    sigma_factor: float
    sigma_constraint: float
    operator_norm: float
    line_search_backtracks: int
    minimum_tau: float
    maximum_tau: float
    line_search_delta: float
    line_search_shrink: float
    norm_iterations: int
    gram_line_search: bool


def _pava_step(
    instance: common.MarkowitzInstance,
    argument: Any,
    tau: float,
    pava_backend: str,
    bisection_steps: int,
    boundary: Optional[float],
) -> Tuple[Any, Optional[float]]:
    gamma = tau * instance.perspective_weight
    if pava_backend == "exact":
        return (
            common.exact_long_only_pava_prox(
                argument,
                gamma,
                instance.k,
            ),
            None,
        )
    x, next_boundary = common.long_only_pava_prox(
        argument,
        gamma,
        instance.k,
        bisection_steps=bisection_steps,
        initial_boundary=boundary,
        return_boundary=True,
    )
    return x, next_boundary


def adaptive_restarted_linesearch_pdhg(
    instance: common.MarkowitzInstance,
    tolerance: float,
    feasibility_tolerance: float,
    max_iterations: int,
    check_interval: int,
    min_epoch: int,
    max_epoch: int,
    restart_factor: float,
    step_ratio: float,
    step_safety: float,
    constraint_dual_weight: float,
    pava_backend: str,
    bisection_steps: int,
    line_search_delta: float,
    line_search_shrink: float,
    norm_iterations: int,
    gram_line_search: bool,
) -> LineSearchPdhgResult:
    """Run block-preconditioned line-search PDHG with residual restarts."""
    if not 0.0 < tolerance:
        raise ValueError("tolerance must be positive")
    if not 0.0 < feasibility_tolerance:
        raise ValueError("feasibility_tolerance must be positive")
    if not 0.0 < restart_factor < 1.0:
        raise ValueError("restart_factor must lie in (0, 1)")
    if not 0.0 < step_safety:
        raise ValueError("step_safety must be positive")
    if not 0.0 < step_ratio:
        raise ValueError("step_ratio must be positive")
    if not 0.0 < constraint_dual_weight:
        raise ValueError("constraint_dual_weight must be positive")
    if not 0.0 < line_search_delta < 1.0:
        raise ValueError("line_search_delta must lie in (0, 1)")
    if not 0.0 < line_search_shrink < 1.0:
        raise ValueError("line_search_shrink must lie in (0, 1)")
    if norm_iterations < 1:
        raise ValueError("norm_iterations must be positive")
    if check_interval < 1:
        raise ValueError("check_interval must be positive")
    if min_epoch < check_interval:
        raise ValueError("min_epoch must be at least check_interval")
    if max_epoch < min_epoch:
        raise ValueError("max_epoch must be at least min_epoch")

    setup_start = time.perf_counter()
    scaled_c, scaled_lower, scaled_upper, _ = (
        common.normalized_constraints(
            instance.constraint_matrix,
            instance.lower_bounds,
            instance.upper_bounds,
        )
    )
    factor_gram = None
    factor_constraint_gram = None
    constraint_gram = None
    if gram_line_search:
        factor_gram = (
            instance.factor_loadings.T @ instance.factor_loadings
        )
        factor_constraint_gram = (
            instance.factor_loadings.T @ scaled_c.T
        )
        constraint_gram = scaled_c @ scaled_c.T
        square_root_weight = math.sqrt(constraint_dual_weight)
        transformed_gram = np.block(
            [
                [
                    factor_gram,
                    square_root_weight * factor_constraint_gram,
                ],
                [
                    square_root_weight
                    * factor_constraint_gram.T,
                    constraint_dual_weight * constraint_gram,
                ],
            ]
        )
        norm_estimate = math.sqrt(
            max(
                0.0,
                float(np.linalg.eigvalsh(transformed_gram)[-1]),
            )
        )
    else:
        norm_estimate = common.operator_norm(
            instance.factor_loadings,
            scaled_c,
            constraint_weight=constraint_dual_weight,
            iterations=norm_iterations,
        )
    if norm_estimate <= 0.0:
        raise RuntimeError("the stacked primal-dual operator is zero")

    alpha_factor = step_ratio * step_ratio
    alpha_constraint = alpha_factor * constraint_dual_weight
    tau = step_safety / (step_ratio * norm_estimate)
    reference_tau = tau
    reference_sigma_factor = reference_tau * alpha_factor
    reference_sigma_constraint = reference_tau * alpha_constraint

    x = instance.feasible_anchor.copy()
    factor_dual = np.zeros(instance.factors)
    constraint_dual = np.zeros(scaled_c.shape[0])
    factor_image = instance.factor_loadings.T @ x
    constraint_image = scaled_c @ x
    adjoint_dual = np.zeros_like(x)
    setup_seconds = time.perf_counter() - setup_start

    start = time.perf_counter()
    initial_residual, pava_calls = common.pdhg_residual(
        instance,
        scaled_c,
        scaled_lower,
        scaled_upper,
        x,
        factor_dual,
        constraint_dual,
        reference_tau,
        reference_sigma_factor,
        reference_sigma_constraint,
        pava_backend,
        bisection_steps,
    )
    residual_scale = max(initial_residual, 1e-16)
    anchor_residual = initial_residual

    theta = 1.0
    minimum_tau = tau
    maximum_tau = tau
    line_search_backtracks = 0
    main_pava_boundary: Optional[float] = None

    epoch_start = 0
    epoch_weight = 0.0
    epoch_sum_x = np.zeros_like(x)
    epoch_sum_factor = np.zeros_like(factor_dual)
    epoch_sum_constraint = np.zeros_like(constraint_dual)
    restarts = 0
    history: List[LineSearchHistoryPoint] = []

    best_x = x.copy()
    best_factor = factor_dual.copy()
    best_constraint = constraint_dual.copy()
    best_residual = initial_residual
    best_violation = common.constraint_violation(
        instance.constraint_matrix,
        instance.lower_bounds,
        instance.upper_bounds,
        x,
    )
    epoch_best = (
        initial_residual,
        x.copy(),
        factor_dual.copy(),
        constraint_dual.copy(),
        "anchor",
    )
    status = "iteration limit"
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        primal_argument = (
            x
            - tau * adjoint_dual
            + tau
            * instance.return_reward
            * instance.expected_returns
        )
        x_next, main_pava_boundary = _pava_step(
            instance,
            primal_argument,
            tau,
            pava_backend,
            bisection_steps,
            main_pava_boundary,
        )
        pava_calls += 1
        factor_image_next = instance.factor_loadings.T @ x_next
        constraint_image_next = scaled_c @ x_next

        tau_trial = tau * math.sqrt(1.0 + theta)
        accepted = False
        for _ in range(80):
            theta_trial = tau_trial / tau
            extrapolated_factor = (
                factor_image_next
                + theta_trial * (factor_image_next - factor_image)
            )
            extrapolated_constraint = (
                constraint_image_next
                + theta_trial
                * (constraint_image_next - constraint_image)
            )
            sigma_factor_trial = tau_trial * alpha_factor
            sigma_constraint_trial = tau_trial * alpha_constraint

            factor_trial = (
                factor_dual
                + sigma_factor_trial * extrapolated_factor
            ) / (1.0 + sigma_factor_trial)
            constraint_value = (
                constraint_dual
                + sigma_constraint_trial * extrapolated_constraint
            )
            constraint_trial = common.interval_dual_prox(
                constraint_value,
                sigma_constraint_trial,
                scaled_lower,
                scaled_upper,
            )

            factor_difference = factor_trial - factor_dual
            constraint_difference = (
                constraint_trial - constraint_dual
            )
            if gram_line_search:
                adjoint_norm_squared = float(
                    factor_difference
                    @ factor_gram
                    @ factor_difference
                    + 2.0
                    * factor_difference
                    @ factor_constraint_gram
                    @ constraint_difference
                    + constraint_difference
                    @ constraint_gram
                    @ constraint_difference
                )
                adjoint_norm_squared = max(
                    0.0,
                    adjoint_norm_squared,
                )
            else:
                adjoint_difference = (
                    instance.factor_loadings @ factor_difference
                    + scaled_c.T @ constraint_difference
                )
                adjoint_norm_squared = float(
                    adjoint_difference @ adjoint_difference
                )
            left_squared = (
                tau_trial * tau_trial * adjoint_norm_squared
            )
            right_squared = line_search_delta * line_search_delta * (
                float(factor_difference @ factor_difference)
                / alpha_factor
                + float(
                    constraint_difference @ constraint_difference
                )
                / alpha_constraint
            )
            if (
                math.isfinite(left_squared)
                and left_squared
                <= right_squared * (1.0 + 1e-12) + 1e-30
            ):
                accepted = True
                break
            tau_trial *= line_search_shrink
            line_search_backtracks += 1

        if not accepted:
            raise RuntimeError(
                "PDHG line search failed after 80 backtracks"
            )
        if gram_line_search:
            adjoint_difference = (
                instance.factor_loadings @ factor_difference
                + scaled_c.T @ constraint_difference
            )

        x = x_next
        factor_image = factor_image_next
        constraint_image = constraint_image_next
        factor_dual = factor_trial
        constraint_dual = constraint_trial
        adjoint_dual += adjoint_difference
        tau = tau_trial
        theta = theta_trial
        minimum_tau = min(minimum_tau, tau)
        maximum_tau = max(maximum_tau, tau)
        completed_iterations = iteration

        epoch_weight += tau
        epoch_sum_x += tau * x
        epoch_sum_factor += tau * factor_dual
        epoch_sum_constraint += tau * constraint_dual

        if iteration % check_interval != 0 and iteration < max_iterations:
            continue

        # This weighted point is only a practical restart candidate; it
        # is not advertised as the formal Malitsky--Pock ergodic output.
        average_x = epoch_sum_x / epoch_weight
        average_factor = epoch_sum_factor / epoch_weight
        average_constraint = epoch_sum_constraint / epoch_weight
        average_residual, calls = common.pdhg_residual(
            instance,
            scaled_c,
            scaled_lower,
            scaled_upper,
            average_x,
            average_factor,
            average_constraint,
            reference_tau,
            reference_sigma_factor,
            reference_sigma_constraint,
            pava_backend,
            bisection_steps,
        )
        pava_calls += calls
        last_residual, calls = common.pdhg_residual(
            instance,
            scaled_c,
            scaled_lower,
            scaled_upper,
            x,
            factor_dual,
            constraint_dual,
            reference_tau,
            reference_sigma_factor,
            reference_sigma_constraint,
            pava_backend,
            bisection_steps,
        )
        pava_calls += calls

        if average_residual <= last_residual:
            candidate_x = average_x
            candidate_factor = average_factor
            candidate_constraint = average_constraint
            candidate_residual = average_residual
            candidate_name = "restart_average"
        else:
            candidate_x = x.copy()
            candidate_factor = factor_dual.copy()
            candidate_constraint = constraint_dual.copy()
            candidate_residual = last_residual
            candidate_name = "last"

        candidate_violation = common.constraint_violation(
            instance.constraint_matrix,
            instance.lower_bounds,
            instance.upper_bounds,
            candidate_x,
        )
        if candidate_residual < epoch_best[0]:
            epoch_best = (
                candidate_residual,
                candidate_x.copy(),
                candidate_factor.copy(),
                candidate_constraint.copy(),
                candidate_name,
            )
        if (
            candidate_residual < best_residual
            or (
                candidate_residual
                <= best_residual * (1.0 + 1e-12)
                and candidate_violation < best_violation
            )
        ):
            best_x = candidate_x.copy()
            best_factor = candidate_factor.copy()
            best_constraint = candidate_constraint.copy()
            best_residual = candidate_residual
            best_violation = candidate_violation

        epoch_length = iteration - epoch_start
        restart_now = (
            epoch_length >= min_epoch
            and candidate_residual
            <= restart_factor * anchor_residual
        )
        forced_restart = epoch_length >= max_epoch
        if forced_restart and not restart_now:
            (
                candidate_residual,
                candidate_x,
                candidate_factor,
                candidate_constraint,
                candidate_name,
            ) = epoch_best
            candidate_violation = common.constraint_violation(
                instance.constraint_matrix,
                instance.lower_bounds,
                instance.upper_bounds,
                candidate_x,
            )
            restart_now = True

        values = instance.constraint_matrix @ candidate_x
        budget_error = abs(values[0] - 1.0)
        return_shortfall = max(
            0.0,
            instance.lower_bounds[1] - values[1],
        )
        history.append(
            LineSearchHistoryPoint(
                iteration=iteration,
                elapsed_seconds=time.perf_counter() - start,
                residual=candidate_residual,
                relative_residual=candidate_residual / residual_scale,
                violation=candidate_violation,
                objective=common.markowitz_objective(
                    instance,
                    candidate_x,
                ),
                budget_error=float(budget_error),
                return_shortfall=float(return_shortfall),
                restart=restart_now,
                candidate=candidate_name,
                primal_step=tau,
                cumulative_backtracks=line_search_backtracks,
            )
        )

        if (
            candidate_residual / residual_scale <= tolerance
            and candidate_violation <= feasibility_tolerance
        ):
            best_x = candidate_x.copy()
            best_factor = candidate_factor.copy()
            best_constraint = candidate_constraint.copy()
            best_residual = candidate_residual
            best_violation = candidate_violation
            status = "converged"
            break

        if restart_now:
            x = candidate_x.copy()
            factor_dual = candidate_factor.copy()
            constraint_dual = candidate_constraint.copy()
            factor_image = instance.factor_loadings.T @ x
            constraint_image = scaled_c @ x
            adjoint_dual = (
                instance.factor_loadings @ factor_dual
                + scaled_c.T @ constraint_dual
            )
            theta = 1.0
            main_pava_boundary = None
            anchor_residual = candidate_residual
            epoch_start = iteration
            epoch_weight = 0.0
            epoch_sum_x.fill(0.0)
            epoch_sum_factor.fill(0.0)
            epoch_sum_constraint.fill(0.0)
            restarts += 1
            epoch_best = (
                candidate_residual,
                candidate_x.copy(),
                candidate_factor.copy(),
                candidate_constraint.copy(),
                "anchor",
            )

    seconds = time.perf_counter() - start
    return LineSearchPdhgResult(
        x=best_x,
        factor_dual=best_factor,
        constraint_dual=best_constraint,
        objective=common.markowitz_objective(instance, best_x),
        residual=best_residual,
        relative_residual=best_residual / residual_scale,
        violation=best_violation,
        iterations=completed_iterations,
        pava_calls=pava_calls,
        restarts=restarts,
        setup_seconds=setup_seconds,
        seconds=seconds,
        total_seconds=setup_seconds + seconds,
        status=status,
        history=history,
        tau=tau,
        sigma_factor=tau * alpha_factor,
        sigma_constraint=tau * alpha_constraint,
        operator_norm=norm_estimate,
        line_search_backtracks=line_search_backtracks,
        minimum_tau=minimum_tau,
        maximum_tau=maximum_tau,
        line_search_delta=line_search_delta,
        line_search_shrink=line_search_shrink,
        norm_iterations=norm_iterations,
        gram_line_search=gram_line_search,
    )


def write_history_csv(
    path: Path,
    result: LineSearchPdhgResult,
    reference_objective: Optional[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "iteration",
        "elapsed_seconds",
        "residual",
        "relative_residual",
        "violation",
        "objective",
        "relative_objective_error",
        "budget_error",
        "return_shortfall",
        "restart",
        "candidate",
        "primal_step",
        "cumulative_backtracks",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in result.history:
            row = asdict(point)
            if reference_objective is None:
                row["relative_objective_error"] = math.nan
            else:
                row["relative_objective_error"] = abs(
                    point.objective - reference_objective
                ) / max(1.0, abs(reference_objective))
            writer.writerow(row)


def make_fallback_convergence_plot(
    path: Path,
    result: LineSearchPdhgResult,
    reference_objective: Optional[float],
    gurobi_result: Optional[Dict[str, Any]],
) -> None:
    """Draw the convergence chart with Pillow when matplotlib is absent."""
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 610
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    try:
        regular = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            18,
        )
        small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            15,
        )
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            21,
        )
    except OSError:
        regular = ImageFont.load_default()
        small = regular
        title_font = regular

    iterations = np.asarray(
        [point.iteration for point in result.history],
        dtype=float,
    )
    times = result.setup_seconds + np.asarray(
        [point.elapsed_seconds for point in result.history],
        dtype=float,
    )
    residuals = np.asarray(
        [max(point.relative_residual, 1e-18) for point in result.history]
    )
    violations = np.asarray(
        [max(point.violation, 1e-18) for point in result.history]
    )
    series = [
        ("relative KKT residual", residuals, "#1769aa"),
        ("constraint violation", violations, "#d1495b"),
    ]
    if reference_objective is not None:
        objective_errors = np.asarray(
            [
                max(
                    abs(point.objective - reference_objective)
                    / max(1.0, abs(reference_objective)),
                    1e-18,
                )
                for point in result.history
            ]
        )
        series.append(
            ("relative objective error", objective_errors, "#16856b")
        )

    def draw_panel(
        box: Tuple[int, int, int, int],
        x_values: Any,
        plotted_series: List[Tuple[str, Any, str]],
        title: str,
        x_label: str,
        vertical_end: Optional[float] = None,
        logarithmic_x: bool = False,
    ) -> None:
        left, top, right, bottom = box
        plot_left = left + 92
        plot_top = top + 58
        plot_right = right - 24
        plot_bottom = bottom - 68
        all_values = np.concatenate(
            [values for _, values, _ in plotted_series]
        )
        positive = all_values[
            np.isfinite(all_values) & (all_values > 0.0)
        ]
        y_max_log = math.ceil(math.log10(float(np.max(positive))))
        y_min_log = math.floor(math.log10(float(np.min(positive))))
        y_min_log = max(y_min_log, y_max_log - 14)
        if y_min_log == y_max_log:
            y_min_log -= 1
        positive_x = x_values[
            np.isfinite(x_values) & (x_values > 0.0)
        ]
        x_min = (
            max(float(np.min(positive_x)) * 0.8, 1e-12)
            if logarithmic_x
            else 0.0
        )
        x_max = max(
            float(np.max(x_values)) if x_values.size else 1.0,
            vertical_end or 0.0,
            1e-12,
        )

        def px(value: float) -> float:
            if logarithmic_x:
                clipped = max(value, x_min)
                return plot_left + (plot_right - plot_left) * (
                    (math.log(clipped) - math.log(x_min))
                    / (math.log(x_max) - math.log(x_min))
                )
            return plot_left + (plot_right - plot_left) * (
                (value - x_min) / (x_max - x_min)
            )

        def py(value: float) -> float:
            log_value = math.log10(max(value, 10.0 ** y_min_log))
            return plot_bottom - (plot_bottom - plot_top) * (
                (log_value - y_min_log) / (y_max_log - y_min_log)
            )

        draw.text(
            ((left + right) // 2, top + 8),
            title,
            fill="#202124",
            font=title_font,
            anchor="ma",
        )
        for exponent in range(y_min_log, y_max_log + 1):
            y_pixel = py(10.0 ** exponent)
            draw.line(
                (plot_left, y_pixel, plot_right, y_pixel),
                fill="#e3e6e8",
                width=1,
            )
            draw.text(
                (plot_left - 10, y_pixel),
                f"1e{exponent}",
                fill="#5f6368",
                font=small,
                anchor="rm",
            )
        for index in range(6):
            if logarithmic_x:
                value = math.exp(
                    math.log(x_min)
                    + (math.log(x_max) - math.log(x_min))
                    * index
                    / 5.0
                )
            else:
                value = x_max * index / 5.0
            x_pixel = px(value)
            draw.line(
                (x_pixel, plot_top, x_pixel, plot_bottom),
                fill="#eef0f2",
                width=1,
            )
            draw.text(
                (x_pixel, plot_bottom + 10),
                (
                    f"{value:,.0f}"
                    if not logarithmic_x and x_max >= 1000.0
                    else f"{value:.3g}"
                ),
                fill="#5f6368",
                font=small,
                anchor="ma",
            )
        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill="#5f6368",
            width=2,
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill="#5f6368",
            width=2,
        )

        for label, values, color in plotted_series:
            points = [
                (px(float(x_value)), py(float(y_value)))
                for x_value, y_value in zip(x_values, values)
                if math.isfinite(float(y_value))
            ]
            if len(points) >= 2:
                draw.line(points, fill=color, width=4, joint="curve")
            elif points:
                x_pixel, y_pixel = points[0]
                draw.ellipse(
                    (
                        x_pixel - 3,
                        y_pixel - 3,
                        x_pixel + 3,
                        y_pixel + 3,
                    ),
                    fill=color,
                )

        if vertical_end is not None:
            end_pixel = px(vertical_end)
            draw.line(
                (end_pixel, plot_top, end_pixel, plot_bottom),
                fill="#e59b3a",
                width=3,
            )

        legend_x = plot_left + 8
        legend_y = plot_top + 8
        for label, _, color in plotted_series:
            draw.line(
                (legend_x, legend_y + 8, legend_x + 26, legend_y + 8),
                fill=color,
                width=4,
            )
            draw.text(
                (legend_x + 34, legend_y),
                label,
                fill="#303134",
                font=small,
            )
            legend_y += 22
        if vertical_end is not None:
            draw.line(
                (legend_x, legend_y + 8, legend_x + 26, legend_y + 8),
                fill="#e59b3a",
                width=3,
            )
            draw.text(
                (legend_x + 34, legend_y),
                "Gurobi end-to-end",
                fill="#303134",
                font=small,
            )

        draw.text(
            ((plot_left + plot_right) // 2, bottom - 26),
            x_label,
            fill="#303134",
            font=regular,
            anchor="ma",
        )
    draw_panel(
        (20, 42, 740, 590),
        iterations,
        series,
        "Line-search PDHG convergence",
        "accepted PDHG iteration",
    )
    time_series = list(series)
    gurobi_end = None
    if gurobi_result is not None:
        gurobi_end = float(gurobi_result["driver_total_seconds"])
        trace = gurobi_result.get("barrier_trace", [])
        gurobi_times = []
        gurobi_gaps = []
        for point in trace:
            primal = float(point["primal_objective"])
            dual = float(point["dual_objective"])
            if math.isfinite(primal) and math.isfinite(dual):
                gurobi_times.append(
                    float(gurobi_result["build_seconds"])
                    + float(point["runtime"])
                )
                gurobi_gaps.append(
                    max(
                        abs(primal - dual)
                        / max(1.0, abs(primal), abs(dual)),
                        1e-18,
                    )
                )
        if gurobi_times:
            time_series.append(
                (
                    "Gurobi barrier gap",
                    np.asarray(gurobi_gaps),
                    "#e59b3a",
                )
            )
            combined_times = np.concatenate(
                [times, np.asarray(gurobi_times)]
            )
        else:
            combined_times = times
    else:
        combined_times = times
    if len(time_series) != len(series):
        padded_series = []
        for label, values, color in series:
            padding = np.full(
                combined_times.size - times.size,
                math.nan,
            )
            padded_series.append(
                (label, np.concatenate([values, padding]), color)
            )
        gurobi_padding = np.full(times.size, math.nan)
        padded_series.append(
            (
                "Gurobi barrier gap",
                np.concatenate(
                    [gurobi_padding, time_series[-1][1]]
                ),
                "#e59b3a",
            )
        )
        time_series = padded_series
    draw_panel(
        (760, 42, 1480, 590),
        combined_times,
        time_series,
        "Wall-clock convergence",
        "end-to-end time (seconds, log scale)",
        vertical_end=gurobi_end,
        logarithmic_x=True,
    )
    image.save(path)


def driver_parser() -> Any:
    parser = common.driver_parser()
    parser.description = (
        "Line-search PDHG with adaptive restart versus Gurobi for the "
        "long-only sparse Markowitz perspective relaxation"
    )
    parser.add_argument(
        "--line-search-delta",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--line-search-shrink",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--norm-iterations",
        type=int,
        default=8,
        help=(
            "short power iteration used to initialize the line search "
            "only when --no-gram-line-search is selected"
        ),
    )
    parser.add_argument(
        "--no-gram-line-search",
        action="store_true",
        help=(
            "avoid the small dual Gram matrix and use matrix products "
            "for rejected line-search trials"
        ),
    )
    parser.set_defaults(
        plot=OUTPUT_DIRECTORY / "markowitz_linesearch.png",
        history=OUTPUT_DIRECTORY / "markowitz_linesearch.csv",
        summary=OUTPUT_DIRECTORY / "markowitz_linesearch.json",
    )
    return parser


def driver_main(arguments: Sequence[str]) -> None:
    args = driver_parser().parse_args(arguments)
    pava_check = common.run_long_only_pava_checks()
    instance = common.generate_markowitz_instance(
        dimension=args.d,
        k=args.k,
        factors=args.factors,
        sectors=args.sectors,
        style_factors=args.style_factors,
        stress_constraints=args.stress_constraints,
        seed=args.seed,
        factor_scale=args.factor_scale,
        perspective_weight=args.perspective_weight,
        return_reward=args.return_reward,
        target_fraction=args.target_fraction,
        sector_band=args.sector_band,
        style_band=args.style_band,
        stress_band=args.stress_band,
    )
    constraint_dual_weight = (
        args.constraint_dual_weight
        if args.constraint_dual_weight is not None
        else max(1.0, 0.0064 * args.d)
    )
    result = adaptive_restarted_linesearch_pdhg(
        instance=instance,
        tolerance=args.tolerance,
        feasibility_tolerance=args.feasibility_tolerance,
        max_iterations=args.max_iterations,
        check_interval=args.check_interval,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        restart_factor=args.restart_factor,
        step_ratio=args.step_ratio,
        step_safety=args.step_safety,
        constraint_dual_weight=constraint_dual_weight,
        pava_backend=args.pava_backend,
        bisection_steps=args.pava_bisection_steps,
        line_search_delta=args.line_search_delta,
        line_search_shrink=args.line_search_shrink,
        norm_iterations=args.norm_iterations,
        gram_line_search=not args.no_gram_line_search,
    )
    print(
        "Line-search PDHG finished: "
        f"{result.total_seconds:.3f} s, "
        f"{result.iterations} accepted iterations, "
        f"{result.line_search_backtracks} backtracks, "
        f"relative residual = {result.relative_residual:.3e}, "
        f"violation = {result.violation:.3e}",
        flush=True,
    )

    gurobi: Optional[Dict[str, Any]] = None
    if not args.skip_gurobi:
        print("Starting the matching Gurobi QCP...", flush=True)
        gurobi = common.run_gurobi_subprocess(
            instance=instance,
            gurobi_python=args.gurobi_python,
            gurobi_threads=args.gurobi_threads,
            gurobi_tolerance=args.gurobi_tolerance,
            output_flag=int(args.gurobi_log),
        )

    reference_objective = None
    if gurobi is not None:
        reference_objective = common.markowitz_objective(
            instance,
            np.asarray(gurobi["x"]),
        )
    write_history_csv(args.history, result, reference_objective)
    try:
        common.make_convergence_plot(
            args.plot,
            result,
            reference_objective,
            gurobi,
        )
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        make_fallback_convergence_plot(
            args.plot,
            result,
            reference_objective,
            gurobi,
        )
    summary = common.write_summary(
        args.summary,
        instance,
        pava_check,
        result,
        gurobi,
    )
    summary["pdhg"].update(
        {
            "method": "Malitsky--Pock line search with adaptive restart",
            "line_search_backtracks": result.line_search_backtracks,
            "minimum_tau": result.minimum_tau,
            "maximum_tau": result.maximum_tau,
            "line_search_delta": result.line_search_delta,
            "line_search_shrink": result.line_search_shrink,
            "norm_iterations": result.norm_iterations,
            "gram_line_search": result.gram_line_search,
        }
    )
    with args.summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    common.print_summary(summary)
    print(
        "Line search: "
        f"{result.line_search_backtracks} backtracks, "
        f"tau in [{result.minimum_tau:.3e}, "
        f"{result.maximum_tau:.3e}]"
    )
    print(f"Convergence plot: {args.plot.resolve()}")
    print(f"Iteration history: {args.history.resolve()}")
    print(f"JSON summary: {args.summary.resolve()}")


if __name__ == "__main__":
    import sys

    driver_main(sys.argv[1:])
