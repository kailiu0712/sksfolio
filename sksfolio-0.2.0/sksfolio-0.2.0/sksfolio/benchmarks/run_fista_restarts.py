"""Compare every supported outer restart rule for line-search FISTA.

The benchmark uses the deterministic Bertsimas--Cory-Wright-shaped
Markowitz generator.  For each requested dimension it solves the same
factor instance with:

* the budget row only (``bcw``);
* the matched 261-row ``many`` profile (budget, sector, style, and stress
  rows).

Only the outer restart rule changes within a matched group.  The line
search, PAVA backend, proximal oracle, tolerance, and thread count are held
fixed.  The script writes one row per run, an aggregate table, the complete
checkpoint history, time-to-safe-bound targets, and a wall-clock convergence
plot containing every restart variant.  The plotted dual ordinate is the
distance to the best safe lower bound found by any matched run, so it never
relies on the possibly slightly infeasible final iterate of the general-row
prox.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bertsimas_cory_wright import BCWCase, generate_synthetic_case
from ..relaxation import solve_relaxation


PERIODS = (5, 10, 20, 50, 100)
GAP_ETA_POWERS = (1, 2, 3)
DEFAULT_DIMENSIONS = (499, 958, 3162)
DEFAULT_PROFILES = ("bcw", "many")
SAFE_BOUND_TARGETS = (1e-5, 1e-6, 5e-7, 1e-7)


@dataclass(frozen=True)
class RestartVariant:
    """One member of the matched outer-restart grid."""

    identifier: str
    label: str
    strategy: str
    period: int = 25
    eta: float = math.e**2


def restart_variants() -> tuple[RestartVariant, ...]:
    """Return the complete, deliberately non-selected comparison grid."""
    variants = [
        RestartVariant("none", "No restart", "none"),
        RestartVariant(
            "gradient",
            "Gradient restart",
            "gradient",
        ),
        RestartVariant(
            "function",
            "Function restart",
            "function",
        ),
        RestartVariant(
            "hinder_lubin",
            "Hinder--Lubin",
            "hinder_lubin",
        ),
    ]
    variants.extend(
        RestartVariant(
            f"periodic_k{period}",
            f"Periodic K={period}",
            "periodic",
            period=period,
        )
        for period in PERIODS
    )
    variants.extend(
        RestartVariant(
            f"primal_dual_gap_e{power}",
            f"Primal-dual gap eta=e^{power}",
            "primal_dual_gap",
            eta=math.exp(float(power)),
        )
        for power in GAP_ETA_POWERS
    )
    return tuple(variants)


def _case(
    dimension: int,
    rank: int,
    k: int,
    gamma_scale: float,
) -> BCWCase:
    return BCWCase(
        family="historical",
        universe=f"synthetic{dimension}",
        dimension=dimension,
        rank=rank,
        k=k,
        gamma_scale=gamma_scale,
        regime="unconstrained",
    )


def _rank_for_dimension(
    args: argparse.Namespace,
    dimension: int,
) -> int:
    """Return the paper-shaped rank unless one override is requested."""
    if args.rank is not None:
        return int(args.rank)
    return 50 if dimension <= 1_000 else 100


def _solver_options(
    args: argparse.Namespace,
    variant: RestartVariant,
) -> dict[str, Any]:
    """Build the common solver settings plus one restart specialization."""
    return {
        "tolerance": args.tolerance,
        "feasibility_tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "initial_lipschitz": args.initial_lipschitz,
        "backtracking_factor": args.backtracking_factor,
        "step_growth": args.step_growth,
        "line_search_tolerance": args.line_search_tolerance,
        "max_backtracks": args.max_backtracks,
        "history_interval": args.history_interval,
        "prox_tolerance": args.prox_tolerance,
        "prox_max_iterations": args.prox_max_iterations,
        "prox_oracle": args.prox_oracle,
        "prox_adaptive_restart": args.prox_adaptive_restart,
        "restart_strategy": variant.strategy,
        "restart_period": variant.period,
        "restart_eta": variant.eta,
        "restart_check_interval": args.restart_check_interval,
        "restart_hinder_lubin_beta": args.hinder_lubin_beta,
        "restart_function_tolerance": args.function_tolerance,
    }


def _constraint_signature(problem: Any) -> dict[str, Any]:
    names = tuple(problem.constraint_names)
    return {
        "constraint_count": len(names),
        "constraint_names": "|".join(names),
        "constraint_nnz": int(problem.C.nnz),
    }


def _base_record(
    *,
    args: argparse.Namespace,
    problem: Any,
    case: BCWCase,
    profile: str,
    repeat: int,
    order: int,
    variant_index: int,
    variant: RestartVariant,
) -> dict[str, Any]:
    return {
        "run_id": (
            f"d{case.dimension}-{profile}-rep{repeat:02d}-"
            f"{variant.identifier}"
        ),
        "instance_id": (
            f"{case.key}-seed{args.seed}-{profile}"
        ),
        "profile": profile,
        "dimension": case.dimension,
        "rank": case.rank,
        "k": case.k,
        "gamma_scale": case.gamma_scale,
        "gamma": case.gamma,
        "regime": case.regime,
        "seed": args.seed,
        "repeat": repeat,
        "execution_order": order,
        "variant_index": variant_index,
        "restart_id": variant.identifier,
        "restart_label": variant.label,
        "restart_strategy_requested": variant.strategy,
        "restart_period_requested": variant.period,
        "restart_eta_requested": variant.eta,
        "restart_contraction_requested": 1.0 / variant.eta,
        "pava": args.pava,
        "prox_oracle_requested": args.prox_oracle,
        "prox_tolerance_requested": args.prox_tolerance,
        "prox_max_iterations_requested": args.prox_max_iterations,
        "prox_adaptive_restart_requested": (
            args.prox_adaptive_restart
        ),
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "history_interval": args.history_interval,
        "restart_check_interval": args.restart_check_interval,
        "initial_lipschitz_requested": args.initial_lipschitz,
        "backtracking_factor": args.backtracking_factor,
        "step_growth": args.step_growth,
        "line_search_tolerance": args.line_search_tolerance,
        "hinder_lubin_beta": args.hinder_lubin_beta,
        "function_tolerance": args.function_tolerance,
        **_constraint_signature(problem),
    }


def _result_record(
    base: Mapping[str, Any],
    problem: Any,
    result: Any,
    observed_wall_seconds: float,
) -> dict[str, Any]:
    raw = result.raw
    diagnostics = raw.get("diagnostics", {})
    violations = diagnostics.get("violations", {})
    anchor_upper = raw.get("anchor_primal_upper_bound")
    safe_bound = result.safe_dual_bound
    anchor_safe_gap = (
        float(anchor_upper) - float(safe_bound)
        if anchor_upper is not None and safe_bound is not None
        else None
    )
    certificate = result.dual_certificate
    certificate_verified = (
        certificate.verify(problem) if certificate is not None else False
    )
    return {
        **base,
        "status": result.status,
        "error": None,
        "objective": result.objective,
        "safe_dual_bound": safe_bound,
        "anchor_primal_upper_bound": anchor_upper,
        "anchor_safe_gap": anchor_safe_gap,
        "certificate_verified": certificate_verified,
        "dual_bound_safe_in_exact_arithmetic": raw.get(
            "dual_bound_safe_in_exact_arithmetic"
        ),
        "dual_bound_floating_point_certified": raw.get(
            "dual_bound_floating_point_certified"
        ),
        "variant": raw.get("variant"),
        "restart_strategy": raw.get("restart_strategy"),
        "restart_certificate": raw.get("restart_certificate"),
        "restart_exact_prox_assumption_satisfied": raw.get(
            "restart_exact_prox_assumption_satisfied"
        ),
        "restart_inexact_prox_scope": raw.get(
            "restart_inexact_prox_scope"
        ),
        "restart_period": raw.get("restart_period"),
        "restart_eta": raw.get("restart_eta"),
        "restart_contraction": raw.get("restart_contraction"),
        "restart_checks": raw.get("restart_checks"),
        "restarts": raw.get("restarts"),
        "restart_epochs": raw.get("restart_epochs"),
        "restart_iterations": json.dumps(
            raw.get("restart_iterations", ())
        ),
        "restart_reasons": json.dumps(
            raw.get("restart_reasons", ())
        ),
        "restart_bookkeeping_seconds": raw.get(
            "restart_bookkeeping_seconds"
        ),
        "iterations": raw.get("iterations"),
        "residual": raw.get("residual"),
        "relative_residual": raw.get("relative_residual"),
        "maximum_violation": violations.get("maximum"),
        "linear_row_violation": violations.get("linear_rows"),
        "prox_oracle_used": raw.get("prox_oracle_used"),
        "prox_exact_budget_fast_path": raw.get(
            "prox_exact_budget_fast_path"
        ),
        "prox_all_calls_converged": raw.get(
            "prox_all_calls_converged"
        ),
        "last_prox_fixed_point_residual": raw.get(
            "last_prox_fixed_point_residual"
        ),
        "line_search_backtracks": raw.get(
            "line_search_backtracks"
        ),
        "gradient_evaluations": raw.get("gradient_evaluations"),
        "objective_evaluations": raw.get("objective_evaluations"),
        "prox_calls": raw.get("prox_calls"),
        "pava_calls": raw.get("pava_calls"),
        "prox_inner_iterations": raw.get("prox_inner_iterations"),
        "prox_inner_restarts": raw.get("prox_inner_restarts"),
        "dual_bound_evaluations": raw.get(
            "dual_bound_evaluations"
        ),
        "restart_dual_evaluations": raw.get(
            "restart_dual_evaluations"
        ),
        "restart_objective_evaluations": raw.get(
            "restart_objective_evaluations"
        ),
        "initial_lipschitz": raw.get("initial_lipschitz"),
        "final_lipschitz": raw.get("final_lipschitz"),
        "setup_seconds": raw.get("setup_seconds"),
        "solve_seconds": raw.get("solve_seconds"),
        "total_seconds": raw.get("total_seconds"),
        "wrapper_seconds": raw.get("wrapper_seconds"),
        "observed_wall_seconds": observed_wall_seconds,
    }


def _error_record(
    base: Mapping[str, Any],
    error: BaseException,
    observed_wall_seconds: float,
) -> dict[str, Any]:
    """Keep a failed grid member visible rather than silently selecting it."""
    return {
        **base,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "objective": None,
        "safe_dual_bound": None,
        "anchor_primal_upper_bound": None,
        "anchor_safe_gap": None,
        "certificate_verified": False,
        "iterations": None,
        "residual": None,
        "relative_residual": None,
        "maximum_violation": None,
        "restarts": None,
        "line_search_backtracks": None,
        "gradient_evaluations": None,
        "prox_calls": None,
        "pava_calls": None,
        "dual_bound_evaluations": None,
        "restart_bookkeeping_seconds": None,
        "setup_seconds": None,
        "solve_seconds": None,
        "total_seconds": None,
        "wrapper_seconds": None,
        "observed_wall_seconds": observed_wall_seconds,
    }


def _history_records(
    *,
    base: Mapping[str, Any],
    result: Any,
) -> list[dict[str, Any]]:
    raw = result.raw
    anchor_upper = raw.get("anchor_primal_upper_bound")
    setup_seconds = float(raw.get("setup_seconds", 0.0) or 0.0)
    tolerance = float(base["tolerance"])
    rows = []
    for checkpoint in raw.get("history", ()):
        best_dual = checkpoint.get("best_dual_bound")
        objective = checkpoint.get("objective")
        violation = checkpoint.get("violation")
        safe_anchor_gap = (
            float(anchor_upper) - float(best_dual)
            if anchor_upper is not None and best_dual is not None
            else None
        )
        current_feasible_gap = None
        if (
            objective is not None
            and best_dual is not None
            and violation is not None
            and float(violation) <= tolerance
        ):
            current_feasible_gap = (
                float(objective) - float(best_dual)
            )
        elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        rows.append(
            {
                "run_id": base["run_id"],
                "profile": base["profile"],
                "dimension": base["dimension"],
                "repeat": base["repeat"],
                "variant_index": base["variant_index"],
                "restart_id": base["restart_id"],
                "restart_label": base["restart_label"],
                "restart_strategy": base[
                    "restart_strategy_requested"
                ],
                "iteration": checkpoint.get("iteration"),
                "solve_wall_seconds": elapsed,
                "total_wall_seconds": setup_seconds + elapsed,
                "objective": objective,
                "dual_objective": checkpoint.get(
                    "dual_objective"
                ),
                "best_safe_dual_bound": best_dual,
                "anchor_primal_upper_bound": anchor_upper,
                "safe_anchor_gap": safe_anchor_gap,
                "current_feasible_safe_gap": current_feasible_gap,
                "paired_primal_dual_gap": checkpoint.get(
                    "paired_primal_dual_gap"
                ),
                "residual": checkpoint.get("residual"),
                "relative_residual": checkpoint.get(
                    "relative_residual"
                ),
                "violation": violation,
                "restart": checkpoint.get("restart"),
                "restart_reason": checkpoint.get("restart_reason"),
                "restart_metric": checkpoint.get("restart_metric"),
                "restart_threshold": checkpoint.get(
                    "restart_threshold"
                ),
                "cumulative_restarts": checkpoint.get(
                    "cumulative_restarts"
                ),
                "cumulative_backtracks": checkpoint.get(
                    "cumulative_backtracks"
                ),
            }
        )
    return rows


def _fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not names:
            return
        writer = csv.DictWriter(
            stream,
            fieldnames=names,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _finite(
    rows: Sequence[Mapping[str, Any]],
    name: str,
) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=float)


def _summary(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    *,
    full: bool = False,
) -> dict[str, Any]:
    values = _finite(rows, name)
    prefix = name
    if values.size == 0:
        result = {f"{prefix}_median": None}
        if full:
            result.update(
                {
                    f"{prefix}_q25": None,
                    f"{prefix}_q75": None,
                    f"{prefix}_min": None,
                    f"{prefix}_max": None,
                }
            )
        return result
    result = {f"{prefix}_median": float(np.median(values))}
    if full:
        result.update(
            {
                f"{prefix}_q25": float(np.quantile(values, 0.25)),
                f"{prefix}_q75": float(np.quantile(values, 0.75)),
                f"{prefix}_min": float(np.min(values)),
                f"{prefix}_max": float(np.max(values)),
            }
        )
    return result


def _aggregate(
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = (
        "profile",
        "dimension",
        "rank",
        "k",
        "gamma_scale",
        "seed",
        "variant_index",
        "restart_id",
        "restart_label",
        "restart_strategy_requested",
        "restart_period_requested",
        "restart_eta_requested",
        "pava",
        "prox_oracle_requested",
        "tolerance",
        "threads",
        "constraint_count",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        key = tuple(row.get(name) for name in keys)
        groups.setdefault(key, []).append(row)

    result = []
    for key, rows in groups.items():
        record = dict(zip(keys, key))
        statuses: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        record.update(
            {
                "runs": len(rows),
                "status_counts": json.dumps(
                    statuses,
                    sort_keys=True,
                ),
                "converged_runs": statuses.get("converged", 0),
                "certificate_verified_runs": sum(
                    bool(row.get("certificate_verified"))
                    for row in rows
                ),
            }
        )
        for metric in (
            "observed_wall_seconds",
            "wrapper_seconds",
            "total_seconds",
            "solve_seconds",
        ):
            record.update(_summary(rows, metric, full=True))
        for metric in (
            "iterations",
            "restarts",
            "line_search_backtracks",
            "gradient_evaluations",
            "prox_calls",
            "pava_calls",
            "prox_inner_iterations",
            "dual_bound_evaluations",
            "restart_dual_evaluations",
            "restart_bookkeeping_seconds",
            "residual",
            "relative_residual",
            "maximum_violation",
            "anchor_safe_gap",
            "safe_bound_error_to_reference",
        ):
            record.update(_summary(rows, metric))
        result.append(record)
    result.sort(
        key=lambda row: (
            int(row["dimension"]),
            str(row["profile"]),
            int(row["variant_index"]),
        )
    )
    return result


def _attach_reference_bounds(
    raw_rows: Sequence[dict[str, Any]],
    history_rows: Sequence[dict[str, Any]],
) -> None:
    """Attach a primal-feasibility-independent dual convergence target.

    For each matched (dimension, profile) group, the reference is the
    largest *safe* final lower bound produced by any strategy or repeat.
    This is not claimed to be the unknown optimum.  It is a reproducible
    best-available-bound target and remains meaningful when a many-row
    primal iterate has a small inexact-prox feasibility violation.
    """
    references: dict[tuple[int, str], float] = {}
    for row in raw_rows:
        bound = row.get("safe_dual_bound")
        if bound is None:
            continue
        numeric = float(bound)
        if not math.isfinite(numeric):
            continue
        key = (int(row["dimension"]), str(row["profile"]))
        references[key] = max(
            references.get(key, -math.inf),
            numeric,
        )

    for row in raw_rows:
        key = (int(row["dimension"]), str(row["profile"]))
        reference = references.get(key)
        bound = row.get("safe_dual_bound")
        row["best_reference_safe_bound"] = reference
        row["safe_bound_reference_kind"] = (
            "maximum_safe_final_bound_over_all_matched_variants_and_repeats"
            if reference is not None
            else None
        )
        row["safe_bound_error_to_reference"] = (
            max(float(reference) - float(bound), 0.0)
            if reference is not None and bound is not None
            else None
        )

    for row in history_rows:
        key = (int(row["dimension"]), str(row["profile"]))
        reference = references.get(key)
        bound = row.get("best_safe_dual_bound")
        row["best_reference_safe_bound"] = reference
        row["safe_bound_reference_kind"] = (
            "maximum_safe_final_bound_over_all_matched_variants_and_repeats"
            if reference is not None
            else None
        )
        row["safe_bound_error_to_reference"] = (
            max(float(reference) - float(bound), 0.0)
            if reference is not None and bound is not None
            else None
        )


def _time_to_bound(
    history_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate first wall time to each matched safe-bound target."""
    runs: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    run_keys = (
        "run_id",
        "dimension",
        "profile",
        "variant_index",
        "restart_id",
        "restart_label",
    )
    for row in history_rows:
        key = tuple(row.get(name) for name in run_keys)
        runs.setdefault(key, []).append(row)

    observations: list[dict[str, Any]] = []
    for key, rows in runs.items():
        identity = dict(zip(run_keys, key))
        for target in SAFE_BOUND_TARGETS:
            times = [
                float(row["total_wall_seconds"])
                for row in rows
                if row.get("safe_bound_error_to_reference") is not None
                and float(row["safe_bound_error_to_reference"])
                <= target
            ]
            observations.append(
                {
                    **identity,
                    "safe_bound_error_target": target,
                    "time_to_target_seconds": (
                        min(times) if times else None
                    ),
                }
            )

    group_keys = (
        "dimension",
        "profile",
        "variant_index",
        "restart_id",
        "restart_label",
        "safe_bound_error_target",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in observations:
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)

    result = []
    for key, rows in groups.items():
        record = dict(zip(group_keys, key))
        times = _finite(rows, "time_to_target_seconds")
        record.update(
            {
                "runs": len(rows),
                "target_reached_runs": int(times.size),
                "target_reached_all_runs": bool(
                    times.size == len(rows)
                ),
                "time_to_target_seconds_median": (
                    float(np.median(times)) if times.size else None
                ),
                "time_to_target_seconds_q25": (
                    float(np.quantile(times, 0.25))
                    if times.size
                    else None
                ),
                "time_to_target_seconds_q75": (
                    float(np.quantile(times, 0.75))
                    if times.size
                    else None
                ),
            }
        )
        result.append(record)
    result.sort(
        key=lambda row: (
            int(row["dimension"]),
            str(row["profile"]),
            float(row["safe_bound_error_target"]),
            int(row["variant_index"]),
        )
    )
    return result


def _plot_convergence(
    path: Path,
    history: Sequence[Mapping[str, Any]],
    variants: Sequence[RestartVariant],
    plot_dimension: int,
    profiles: Sequence[str],
) -> bool:
    selected = [
        row
        for row in history
        if int(row["dimension"]) == plot_dimension
        and int(row["repeat"]) == 0
    ]
    if not selected:
        return False

    cache = Path(tempfile.gettempdir()) / "sksfolio-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(
        len(profiles),
        2,
        figsize=(14.0, 4.2 * len(profiles)),
        squeeze=False,
        sharex=False,
    )
    colors = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, len(variants))
    )
    handles = []
    labels = []
    for profile_index, profile in enumerate(profiles):
        profile_rows = [
            row for row in selected if row["profile"] == profile
        ]
        gap_axis = axes[profile_index, 0]
        residual_axis = axes[profile_index, 1]
        for variant_index, variant in enumerate(variants):
            trace = sorted(
                (
                    row
                    for row in profile_rows
                    if row["restart_id"] == variant.identifier
                ),
                key=lambda row: int(row["iteration"]),
            )
            if not trace:
                continue
            x_values = np.asarray(
                [row["total_wall_seconds"] for row in trace],
                dtype=float,
            )
            gap_values = np.asarray(
                [
                    (
                        float(row["safe_bound_error_to_reference"])
                        if row["safe_bound_error_to_reference"]
                        is not None
                        else np.nan
                    )
                    for row in trace
                ],
                dtype=float,
            )
            residual_values = np.asarray(
                [
                    (
                        float(row["relative_residual"])
                        if row["relative_residual"] is not None
                        else np.nan
                    )
                    for row in trace
                ],
                dtype=float,
            )
            positive_gap = gap_values[
                np.isfinite(gap_values) & (gap_values > 0.0)
            ]
            positive_residual = residual_values[
                np.isfinite(residual_values)
                & (residual_values > 0.0)
            ]
            gap_floor = (
                max(float(np.min(positive_gap)) * 0.1, 1e-16)
                if positive_gap.size
                else 1e-16
            )
            residual_floor = (
                max(
                    float(np.min(positive_residual)) * 0.1,
                    1e-16,
                )
                if positive_residual.size
                else 1e-16
            )
            gap_line = gap_axis.plot(
                x_values,
                np.maximum(gap_values, gap_floor),
                color=colors[variant_index],
                linewidth=1.6,
                label=variant.label,
            )[0]
            residual_axis.plot(
                x_values,
                np.maximum(residual_values, residual_floor),
                color=colors[variant_index],
                linewidth=1.6,
                label=variant.label,
            )
            if profile_index == 0:
                handles.append(gap_line)
                labels.append(variant.label)
        row_count = (
            int(profile_rows[0]["constraint_count"])
            if profile_rows
            and "constraint_count" in profile_rows[0]
            else (1 if profile == "bcw" else 261)
        )
        gap_axis.set_title(
            f"{profile}: best-safe-bound error "
            f"({row_count} rows)"
        )
        residual_axis.set_title(
            f"{profile}: normalized residual ({row_count} rows)"
        )
        for axis in (gap_axis, residual_axis):
            axis.set_yscale("log")
            axis.set_xlabel("Wall time (seconds)")
            axis.grid(True, which="both", alpha=0.25)
        gap_axis.set_ylabel("Safe-bound error")
        residual_axis.set_ylabel("Relative residual")

    figure.suptitle(
        "Line-search FISTA outer-restart comparison "
        f"(d={plot_dimension}, repeat 0)",
        y=0.995,
    )
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Compare all line-search FISTA outer restart rules on "
            "matched budget-only and many-row Markowitz instances"
        )
    )
    argument_parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMENSIONS),
        help="asset dimensions; e.g. --dimensions 499 958 3162",
    )
    argument_parser.add_argument(
        "--profiles",
        nargs="+",
        choices=DEFAULT_PROFILES,
        default=list(DEFAULT_PROFILES),
    )
    argument_parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "one rank override; default uses rank 50 through d=1000 "
            "and rank 100 above d=1000"
        ),
    )
    argument_parser.add_argument("--k", type=int, default=10)
    argument_parser.add_argument(
        "--gamma-scale",
        type=float,
        default=100.0,
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument("--repeats", type=int, default=3)
    argument_parser.add_argument(
        "--target-iterations",
        type=int,
        default=200,
    )
    argument_parser.add_argument("--tolerance", type=float, default=1e-5)
    argument_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100_000,
    )
    argument_parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        help="per-variant time limit in seconds",
    )
    argument_parser.add_argument("--threads", type=int, default=1)
    argument_parser.add_argument(
        "--pava",
        choices=("full_sort", "partial_sort"),
        default="partial_sort",
    )
    argument_parser.add_argument(
        "--prox-oracle",
        choices=("auto", "budget", "dual_fista"),
        default="auto",
    )
    argument_parser.add_argument(
        "--prox-tolerance",
        type=float,
        default=1e-8,
    )
    argument_parser.add_argument(
        "--prox-max-iterations",
        type=int,
        default=1000,
    )
    argument_parser.add_argument(
        "--prox-adaptive-restart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "inner dual-FISTA restart; held fixed while outer restart "
            "rules are compared"
        ),
    )
    argument_parser.add_argument(
        "--initial-lipschitz",
        type=float,
        default=None,
    )
    argument_parser.add_argument(
        "--backtracking-factor",
        type=float,
        default=2.0,
    )
    argument_parser.add_argument(
        "--step-growth",
        type=float,
        default=1.1,
    )
    argument_parser.add_argument(
        "--line-search-tolerance",
        type=float,
        default=1e-12,
    )
    argument_parser.add_argument(
        "--max-backtracks",
        type=int,
        default=60,
    )
    argument_parser.add_argument(
        "--history-interval",
        type=int,
        default=1,
        help=(
            "common safe-bound/checkpoint interval; default 1 keeps "
            "dual-bound sampling identical across restart rules"
        ),
    )
    argument_parser.add_argument(
        "--restart-check-interval",
        type=int,
        default=1,
    )
    argument_parser.add_argument(
        "--hinder-lubin-beta",
        type=float,
        default=0.25,
    )
    argument_parser.add_argument(
        "--function-tolerance",
        type=float,
        default=0.0,
    )
    argument_parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle the matched strategy order reproducibly per repeat",
    )
    argument_parser.add_argument(
        "--order-seed",
        type=int,
        default=941,
    )
    argument_parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run an unrecorded two-iteration solve before each matched "
            "profile/dimension timing group"
        ),
    )
    argument_parser.add_argument(
        "--plot-dimension",
        type=int,
        default=None,
        help="dimension shown in the convergence plot; default is largest",
    )
    argument_parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "output/benchmark/fista_restart_comparison"
        ),
        help=(
            "base path; writes *_raw.csv, *_aggregate.csv, "
            "*_history.csv, *_time_to_bound.csv, and "
            "*_convergence.png"
        ),
    )
    argument_parser.add_argument(
        "--fail-fast",
        action="store_true",
    )
    return argument_parser


def _validate(
    args: argparse.Namespace,
    argument_parser: argparse.ArgumentParser,
) -> None:
    dimensions = tuple(dict.fromkeys(args.dimensions))
    args.dimensions = dimensions
    args.profiles = tuple(dict.fromkeys(args.profiles))
    if not dimensions:
        argument_parser.error("at least one dimension is required")
    if any(dimension < 10 for dimension in dimensions):
        argument_parser.error("every dimension must be at least 10")
    if "many" in args.profiles and any(
        dimension < 20 for dimension in dimensions
    ):
        argument_parser.error(
            "the many profile requires every dimension to be at least 20"
        )
    for dimension in dimensions:
        rank = _rank_for_dimension(args, dimension)
        if rank < 2 or rank >= dimension:
            argument_parser.error(
                "each effective rank must be at least 2 and smaller "
                "than its dimension"
            )
    if args.k < 1 or any(
        args.k > dimension for dimension in dimensions
    ):
        argument_parser.error(
            "k must lie between 1 and every requested dimension"
        )
    for name in (
        "repeats",
        "target_iterations",
        "max_iterations",
        "threads",
        "prox_max_iterations",
        "max_backtracks",
        "history_interval",
        "restart_check_interval",
    ):
        if getattr(args, name) < 1:
            argument_parser.error(f"{name} must be positive")
    for name in (
        "gamma_scale",
        "tolerance",
        "time_limit",
        "prox_tolerance",
        "backtracking_factor",
        "step_growth",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            argument_parser.error(f"{name} must be positive and finite")
    if args.backtracking_factor <= 1.0:
        argument_parser.error("backtracking_factor must exceed one")
    if args.step_growth < 1.0:
        argument_parser.error("step_growth must be at least one")
    if args.plot_dimension is None:
        args.plot_dimension = max(dimensions)
    if args.plot_dimension not in dimensions:
        argument_parser.error(
            "plot_dimension must be one of --dimensions"
        )


def run(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    variants = restart_variants()
    raw_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    variant_indices = {
        variant.identifier: index
        for index, variant in enumerate(variants)
    }

    for dimension in args.dimensions:
        case = _case(
            dimension,
            _rank_for_dimension(args, dimension),
            args.k,
            args.gamma_scale,
        )
        for profile_index, profile in enumerate(args.profiles):
            problem = generate_synthetic_case(
                case,
                seed=args.seed,
                target_iterations=args.target_iterations,
                constraint_profile=profile,
            )
            if args.warmup:
                print(
                    f"[{profile} d={dimension}] unrecorded warmup",
                    flush=True,
                )
                warmup_options = _solver_options(args, variants[0])
                warmup_options["max_iterations"] = min(
                    2,
                    args.max_iterations,
                )
                warmup_options["time_limit"] = min(
                    5.0,
                    args.time_limit,
                )
                solve_relaxation(
                    problem,
                    "fista",
                    pava=args.pava,
                    options=warmup_options,
                )
            for repeat in range(args.repeats):
                ordered = list(variants)
                if args.shuffle:
                    order_rng = np.random.default_rng(
                        args.order_seed
                        + 1009 * dimension
                        + 37 * profile_index
                        + repeat
                    )
                    order_rng.shuffle(ordered)
                for order, variant in enumerate(ordered):
                    variant_index = variant_indices[variant.identifier]
                    base = _base_record(
                        args=args,
                        problem=problem,
                        case=case,
                        profile=profile,
                        repeat=repeat,
                        order=order,
                        variant_index=variant_index,
                        variant=variant,
                    )
                    print(
                        f"[{profile} d={dimension} repeat={repeat + 1}/"
                        f"{args.repeats}] {variant.identifier}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    try:
                        result = solve_relaxation(
                            problem,
                            "fista",
                            pava=args.pava,
                            options=_solver_options(args, variant),
                        )
                        elapsed = time.perf_counter() - started
                        raw_rows.append(
                            _result_record(
                                base,
                                problem,
                                result,
                                elapsed,
                            )
                        )
                        history = _history_records(
                            base=base,
                            result=result,
                        )
                        for row in history:
                            row["constraint_count"] = int(
                                problem.rows
                            )
                        history_rows.extend(history)
                    except Exception as error:
                        elapsed = time.perf_counter() - started
                        raw_rows.append(
                            _error_record(base, error, elapsed)
                        )
                        if args.fail_fast:
                            raise
    raw_rows.sort(
        key=lambda row: (
            int(row["dimension"]),
            str(row["profile"]),
            int(row["repeat"]),
            int(row["variant_index"]),
        )
    )
    history_rows.sort(
        key=lambda row: (
            int(row["dimension"]),
            str(row["profile"]),
            int(row["repeat"]),
            int(row["variant_index"]),
            int(row["iteration"]),
        )
    )
    _attach_reference_bounds(raw_rows, history_rows)
    return raw_rows, history_rows


def main(arguments: Sequence[str] | None = None) -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(arguments)
    _validate(args, argument_parser)
    raw_rows, history_rows = run(args)
    aggregate_rows = _aggregate(raw_rows)
    target_rows = _time_to_bound(history_rows)

    prefix = args.output_prefix
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    raw_path = prefix.with_name(prefix.name + "_raw.csv")
    aggregate_path = prefix.with_name(
        prefix.name + "_aggregate.csv"
    )
    history_path = prefix.with_name(prefix.name + "_history.csv")
    targets_path = prefix.with_name(
        prefix.name + "_time_to_bound.csv"
    )
    plot_path = prefix.with_name(
        prefix.name + "_convergence.png"
    )
    _write_csv(raw_path, raw_rows)
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(history_path, history_rows)
    _write_csv(targets_path, target_rows)

    plot_written = False
    if args.plot:
        plot_written = _plot_convergence(
            plot_path,
            history_rows,
            restart_variants(),
            args.plot_dimension,
            args.profiles,
        )
    print(f"Raw runs: {raw_path.resolve()}")
    print(f"Aggregate: {aggregate_path.resolve()}")
    print(f"History: {history_path.resolve()}")
    print(f"Time to safe bound: {targets_path.resolve()}")
    if plot_written:
        print(f"Plot: {plot_path.resolve()}")
    elif args.plot:
        print("Plot not written: matplotlib or plottable history unavailable")


if __name__ == "__main__":
    main()
