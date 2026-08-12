"""Compare the L-BFGS extensions at one common safe-dual target.

The Gurobi solve supplies a tightly solved reference objective.  Every
first-order method receives exactly the same safe Fenchel-bound cutoff, so
the reported times are directly comparable.  Gurobi's own numerical bound
is reported separately and is never labelled as a saved safe certificate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .bertsimas_cory_wright import (
    BCWCase,
    CONSTRAINT_PROFILES,
    HISTORICAL_UNIVERSES,
    generate_synthetic_case,
)
from ..relaxation import solve_relaxation


METHODS = (
    "fista_auto",
    "fista_dual_fista",
    "fista_dual_lbfgs",
    "pdhg",
    "scsdg_fixed",
    "scsdg_linesearch",
    "scsdg_lbfgs_restart_safe",
    "scsdg_lbfgs_paper",
)
REFERENCE_METHOD = "gurobi_reference"


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _cases(args: argparse.Namespace) -> tuple[BCWCase, ...]:
    universes = tuple(args.universes or ())
    dimensions = tuple(args.dimensions or ())
    if not universes and not dimensions:
        universes = ("sp500",)

    cases = []
    for universe in universes:
        _, dimension, published_ranks = HISTORICAL_UNIVERSES[universe]
        rank = args.rank if args.rank is not None else published_ranks[0]
        cases.append(
            BCWCase(
                family="historical",
                universe=universe,
                dimension=dimension,
                rank=rank,
                k=args.k,
                gamma_scale=args.gamma_scale,
                regime=args.regime,
            )
        )
    for dimension in dimensions:
        rank = (
            args.rank
            if args.rank is not None
            else min(50, dimension - 1)
        )
        cases.append(
            BCWCase(
                family="historical",
                universe=f"synthetic_n{dimension}",
                dimension=dimension,
                rank=rank,
                k=args.k,
                gamma_scale=args.gamma_scale,
                regime=args.regime,
            )
        )

    for case in cases:
        if not 2 <= case.rank < case.dimension:
            raise ValueError(
                f"rank must lie in [2, {case.dimension - 1}] for "
                f"{case.universe}"
            )
        if not 1 <= case.k <= case.dimension:
            raise ValueError(
                f"k must lie in [1, {case.dimension}] for "
                f"{case.universe}"
            )
    return _unique(cases)


def _common_options(
    args: argparse.Namespace,
    cutoff: float,
) -> Dict[str, Any]:
    return {
        "threads": args.threads,
        "max_iterations": args.max_iterations,
        "time_limit": args.time_limit,
        "tolerance": args.solver_tolerance,
        "feasibility_tolerance": args.solver_tolerance,
        "dual_bound_cutoff": cutoff,
    }


def _fista_options(
    args: argparse.Namespace,
    cutoff: float,
    prox_oracle: str,
) -> Dict[str, Any]:
    return {
        **_common_options(args, cutoff),
        "history_interval": args.check_interval,
        "prox_tolerance": args.prox_tolerance,
        "prox_max_iterations": args.prox_max_iterations,
        "prox_oracle": prox_oracle,
        "prox_lbfgs_memory": args.prox_lbfgs_memory,
        "prox_lbfgs_max_line_search": (
            args.prox_lbfgs_max_line_search
        ),
        "prox_lbfgs_fallback": args.prox_lbfgs_fallback,
        "restart_strategy": args.fista_restart,
    }


def _scsdg_options(
    args: argparse.Namespace,
    cutoff: float,
    *,
    line_search: bool,
    lbfgs_variant: str,
) -> Dict[str, Any]:
    return {
        **_common_options(args, cutoff),
        "check_interval": args.check_interval,
        "restart_check_interval": args.scsdg_restart_check_interval,
        "theta_parameter": 2.0,
        "continuation_offset": args.scsdg_continuation_offset,
        "target_cbar": args.scsdg_cbar,
        "step_ratio": args.scsdg_step_ratio,
        "constraint_dual_weight": args.scsdg_constraint_weight,
        "restart": True,
        "line_search": line_search,
        "line_search_mode": args.scsdg_line_search_mode,
        "line_search_auto_row_threshold": (
            args.scsdg_line_search_auto_row_threshold
        ),
        "line_search_initial_scale": (
            args.scsdg_line_search_initial_scale
        ),
        "line_search_growth": args.scsdg_line_search_growth,
        "line_search_shrink": args.scsdg_line_search_shrink,
        "line_search_max_scale": args.scsdg_line_search_max_scale,
        "line_search_safety": args.scsdg_line_search_safety,
        "lbfgs_variant": lbfgs_variant,
        "lbfgs_delta_ratio": args.scsdg_lbfgs_delta_ratio,
        "lbfgs_memory": args.scsdg_lbfgs_memory,
        "lbfgs_max_line_search": args.scsdg_lbfgs_max_line_search,
        "lbfgs_min_function_evaluations": (
            args.scsdg_lbfgs_min_function_evaluations
        ),
        "lbfgs_max_function_evaluations": (
            args.scsdg_lbfgs_max_function_evaluations
        ),
        "lbfgs_tolerance": args.scsdg_lbfgs_tolerance,
    }


def _method_options(
    method: str,
    args: argparse.Namespace,
    cutoff: float,
) -> tuple[str, Dict[str, Any]]:
    if method == "fista_auto":
        return "fista", _fista_options(args, cutoff, "auto")
    if method == "fista_dual_fista":
        return "fista", _fista_options(args, cutoff, "dual_fista")
    if method == "fista_dual_lbfgs":
        return "fista", _fista_options(args, cutoff, "dual_lbfgs")
    if method == "pdhg":
        return "pdhg", {
            **_common_options(args, cutoff),
            "check_interval": args.check_interval,
            "min_epoch": max(2 * args.check_interval, 50),
            "max_epoch": args.pdhg_max_epoch,
        }
    if method == "scsdg_fixed":
        return "scsdg", _scsdg_options(
            args,
            cutoff,
            line_search=False,
            lbfgs_variant="off",
        )
    if method == "scsdg_linesearch":
        return "scsdg", _scsdg_options(
            args,
            cutoff,
            line_search=True,
            lbfgs_variant="off",
        )
    if method == "scsdg_lbfgs_restart_safe":
        return "scsdg", _scsdg_options(
            args,
            cutoff,
            line_search=True,
            lbfgs_variant="restart_safe",
        )
    if method == "scsdg_lbfgs_paper":
        return "scsdg", _scsdg_options(
            args,
            cutoff,
            line_search=False,
            lbfgs_variant="paper",
        )
    raise ValueError(f"unknown method: {method}")


def _relative_error(value: Any, optimum: float) -> float | None:
    if value is None:
        return None
    bound = float(value)
    if not math.isfinite(bound):
        return None
    return max(optimum - bound, 0.0) / max(1.0, abs(optimum))


def _case_fields(
    case: BCWCase,
    profile: str,
    seed: int,
    problem: Any,
) -> Dict[str, Any]:
    return {
        "case_id": f"{case.key}-{profile}-seed{seed}",
        "universe": case.universe,
        "constraint_profile": profile,
        "seed": seed,
        "dimension": problem.dimension,
        "rank": problem.rank,
        "rows": problem.rows,
        "k": problem.k,
        "gamma_scale": case.gamma_scale,
        "regime": case.regime,
    }


def _summary_row(
    method: str,
    backend: str,
    result: Any,
    optimum: float,
    cutoff: float,
    target: float,
    case_fields: Dict[str, Any],
    reference_seconds: float,
    problem: Any,
) -> Dict[str, Any]:
    raw = result.raw
    certificate = result.dual_certificate
    diagnostics = raw.get("diagnostics", {})
    violations = diagnostics.get("violations", {})
    safe_bound = result.safe_dual_bound
    return {
        **case_fields,
        "method": method,
        "backend": backend,
        "status": result.status,
        "safe_gap_target": target,
        "safe_dual_cutoff": cutoff,
        "reference_objective": optimum,
        "reference_solve_seconds": reference_seconds,
        "objective": result.objective,
        "objective_relative_error": (
            abs(float(result.objective) - optimum)
            / max(1.0, abs(optimum))
            if result.objective is not None
            else None
        ),
        "safe_dual_bound": safe_bound,
        "safe_dual_relative_error": _relative_error(
            safe_bound,
            optimum,
        ),
        "safe_target_reached": bool(
            safe_bound is not None
            and safe_bound
            >= cutoff - 1e-12 * max(1.0, abs(optimum))
        ),
        "solver_objective_bound": result.solver_objective_bound,
        "solver_bound_relative_error": _relative_error(
            result.solver_objective_bound,
            optimum,
        ),
        "certificate_verified": (
            certificate.verify(problem)
            if certificate is not None
            else False
        ),
        "maximum_violation": violations.get("maximum"),
        "iterations": raw.get("iterations"),
        "pava_calls": raw.get("pava_calls"),
        "restarts": raw.get("restarts"),
        "prox_oracle_requested": raw.get("prox_oracle_requested"),
        "prox_oracle_used": raw.get("prox_oracle_used"),
        "prox_inner_iterations": raw.get("prox_inner_iterations"),
        "prox_function_evaluations": raw.get(
            "prox_function_evaluations"
        ),
        "prox_lbfgs_fallback_calls": raw.get(
            "prox_lbfgs_fallback_calls"
        ),
        "line_search_enabled": raw.get("line_search_enabled"),
        "line_search_mode_resolved": raw.get(
            "line_search_mode_resolved"
        ),
        "line_search_trials": raw.get("line_search_trials"),
        "line_search_backtracks": raw.get("line_search_backtracks"),
        "line_search_final_scale": raw.get("line_search_final_scale"),
        "lbfgs_variant": raw.get("lbfgs_variant"),
        "lbfgs_calls": raw.get("lbfgs_calls"),
        "lbfgs_accepted_calls": raw.get("lbfgs_accepted_calls"),
        "lbfgs_rejected_calls": raw.get("lbfgs_rejected_calls"),
        "lbfgs_iterations": raw.get("lbfgs_iterations"),
        "lbfgs_function_evaluations": raw.get(
            "lbfgs_function_evaluations"
        ),
        "lbfgs_pava_calls": raw.get("lbfgs_pava_calls"),
        "solve_seconds": raw.get("solve_seconds"),
        "setup_seconds": raw.get("setup_seconds"),
        "end_to_end_seconds": raw.get("end_to_end_seconds"),
    }


def _reference_row(
    result: Any,
    optimum: float,
    target: float,
    case_fields: Dict[str, Any],
    problem: Any,
) -> Dict[str, Any]:
    return _summary_row(
        REFERENCE_METHOD,
        "gurobi.python",
        result,
        optimum,
        math.nan,
        target,
        case_fields,
        float(result.raw.get("end_to_end_seconds", math.nan)),
        problem,
    )


def _cached_reference_row(
    optimum: float,
    target: float,
    case_fields: Dict[str, Any],
    reference_seconds: float | None,
) -> Dict[str, Any]:
    """Record a previously computed Gurobi optimum without relabelling it."""
    return {
        **case_fields,
        "method": REFERENCE_METHOD,
        "backend": "gurobi.python",
        "status": "cached_optimal_reference",
        "safe_gap_target": target,
        "reference_objective": optimum,
        "reference_solve_seconds": reference_seconds,
        "objective": optimum,
        "end_to_end_seconds": reference_seconds,
        "certificate_verified": False,
        "reference_source": "user_supplied_cached_gurobi_objective",
    }


def _history_rows(
    method: str,
    result: Any,
    optimum: float,
    target: float,
    case_fields: Dict[str, Any],
) -> list[Dict[str, Any]]:
    rows = []
    for point in result.raw.get("history", []):
        bound = point.get(
            "best_dual_bound",
            point.get("best_dual_lower_bound"),
        )
        rows.append(
            {
                **case_fields,
                "method": method,
                "safe_gap_target": target,
                "iteration": point.get("iteration"),
                "elapsed_seconds": point.get("elapsed_seconds"),
                "best_dual_bound": bound,
                "safe_dual_relative_error": _relative_error(
                    bound,
                    optimum,
                ),
                "residual": point.get("residual"),
                "relative_residual": point.get("relative_residual"),
                "violation": point.get("violation"),
                "smoothed_gap": point.get("smoothed_gap"),
                "relative_smoothed_gap": point.get(
                    "relative_smoothed_gap"
                ),
                "restart": point.get("restart"),
                "lbfgs_triggered": point.get("lbfgs_triggered"),
                "lbfgs_accepted": point.get("lbfgs_accepted"),
                "lbfgs_trigger": point.get("lbfgs_trigger"),
                "cumulative_lbfgs_calls": point.get(
                    "cumulative_lbfgs_calls"
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    path: Path,
    histories: list[Dict[str, Any]],
    summaries: list[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    case_ids = _unique(row["case_id"] for row in summaries)
    columns = min(2, len(case_ids))
    row_count = int(math.ceil(len(case_ids) / columns))
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(7.0 * columns, 4.3 * row_count),
        squeeze=False,
    )
    for axis, case_id in zip(axes.flat, case_ids):
        selected_summaries = [
            row for row in summaries if row["case_id"] == case_id
        ]
        for method in METHODS:
            selected = [
                row
                for row in histories
                if row["case_id"] == case_id
                and row["method"] == method
                and row["elapsed_seconds"] is not None
                and row["safe_dual_relative_error"] is not None
            ]
            if not selected:
                continue
            times = np.asarray(
                [float(row["elapsed_seconds"]) for row in selected]
            )
            errors = np.maximum(
                np.asarray(
                    [
                        float(row["safe_dual_relative_error"])
                        for row in selected
                    ]
                ),
                1e-16,
            )
            axis.semilogy(times, errors, label=method, linewidth=1.8)
        reference = next(
            (
                row
                for row in selected_summaries
                if row["method"] == REFERENCE_METHOD
            ),
            None,
        )
        if reference is not None:
            reference_time = reference.get("end_to_end_seconds")
            if reference_time is not None and math.isfinite(
                float(reference_time)
            ):
                axis.axvline(
                    float(reference_time),
                    color="black",
                    linestyle=":",
                    linewidth=1.4,
                    label="Gurobi reference time",
                )
        target = float(selected_summaries[0]["safe_gap_target"])
        axis.axhline(
            target,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            label="safe target",
        )
        first = selected_summaries[0]
        axis.set_title(
            f"{first['universe']}, {first['constraint_profile']}, "
            f"n={first['dimension']}, m={first['rows']}"
        )
        axis.set_xlabel("solve time (seconds)")
        axis.set_ylabel("safe dual-bound relative error")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(case_ids) :]:
        axis.set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compare FISTA, inner dual L-BFGS-B, PDHG, and the SC-SDG "
            "L-BFGS variants at one safe-dual cutoff"
        )
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("lbfgs_comparison.csv"),
    )
    result.add_argument(
        "--history-output",
        type=Path,
        default=Path("lbfgs_comparison_history.csv"),
    )
    result.add_argument(
        "--plot",
        type=Path,
        default=Path("lbfgs_comparison.png"),
    )
    result.add_argument(
        "--universe",
        dest="universes",
        action="append",
        choices=tuple(HISTORICAL_UNIVERSES),
        help="repeat to benchmark several published universe sizes",
    )
    result.add_argument(
        "--dimension",
        dest="dimensions",
        action="append",
        type=int,
        help="repeat to add a custom synthetic dimension",
    )
    result.add_argument("--rank", type=int)
    result.add_argument("--k", type=int, default=10)
    result.add_argument(
        "--gamma-scale",
        type=float,
        choices=(1.0, 100.0),
        default=100.0,
    )
    result.add_argument(
        "--regime",
        choices=("unconstrained", "constrained"),
        default="unconstrained",
    )
    result.add_argument(
        "--constraint-profile",
        dest="constraint_profiles",
        action="append",
        choices=tuple(CONSTRAINT_PROFILES),
        help="repeat to compare budget-only and many-row profiles",
    )
    result.add_argument(
        "--method",
        dest="methods",
        action="append",
        choices=METHODS,
        help="repeat to run a subset; the default runs every variant",
    )
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--target-iterations", type=int, default=200)
    result.add_argument("--safe-gap-target", type=float, default=1e-6)
    result.add_argument("--solver-tolerance", type=float, default=1e-10)
    result.add_argument("--prox-tolerance", type=float, default=1e-9)
    result.add_argument("--prox-max-iterations", type=int, default=2_000)
    result.add_argument("--max-iterations", type=int, default=200_000)
    result.add_argument("--time-limit", type=float, default=60.0)
    result.add_argument(
        "--reference-time-limit",
        type=float,
        default=600.0,
    )
    result.add_argument(
        "--reference-objective",
        type=float,
        help=(
            "reuse a previously computed Gurobi optimum; valid only for "
            "one case/profile and useful when a license is unavailable"
        ),
    )
    result.add_argument(
        "--reference-seconds",
        type=float,
        help="optional recorded end-to-end time for the cached reference",
    )
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--check-interval", type=int, default=25)
    result.add_argument(
        "--fista-restart",
        choices=(
            "none",
            "gradient",
            "function",
            "periodic",
            "hinder_lubin",
            "primal_dual_gap",
        ),
        default="gradient",
    )
    result.add_argument("--prox-lbfgs-memory", type=int, default=10)
    result.add_argument(
        "--prox-lbfgs-max-line-search",
        type=int,
        default=40,
    )
    result.add_argument(
        "--prox-lbfgs-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument("--pdhg-max-epoch", type=int, default=2_000)
    result.add_argument("--scsdg-cbar", type=float, default=0.1)
    result.add_argument("--scsdg-step-ratio", type=float, default=1.0)
    result.add_argument(
        "--scsdg-constraint-weight",
        default="auto",
    )
    result.add_argument(
        "--scsdg-continuation-offset",
        type=float,
        default=3.0,
    )
    result.add_argument(
        "--scsdg-restart-check-interval",
        type=int,
        default=10,
    )
    result.add_argument(
        "--scsdg-line-search-mode",
        choices=("auto", "operator", "majorization"),
        default="auto",
    )
    result.add_argument(
        "--scsdg-line-search-auto-row-threshold",
        type=int,
        default=16,
    )
    result.add_argument(
        "--scsdg-line-search-initial-scale",
        type=float,
        default=1.0,
    )
    result.add_argument(
        "--scsdg-line-search-growth",
        type=float,
        default=1.1,
    )
    result.add_argument(
        "--scsdg-line-search-shrink",
        type=float,
        default=0.5,
    )
    result.add_argument(
        "--scsdg-line-search-max-scale",
        type=float,
        default=1024.0,
    )
    result.add_argument(
        "--scsdg-line-search-safety",
        type=float,
        default=0.99,
    )
    result.add_argument(
        "--scsdg-lbfgs-delta-ratio",
        type=float,
        default=0.01,
    )
    result.add_argument("--scsdg-lbfgs-memory", type=int, default=10)
    result.add_argument(
        "--scsdg-lbfgs-max-line-search",
        type=int,
        default=40,
    )
    result.add_argument(
        "--scsdg-lbfgs-min-function-evaluations",
        type=int,
        default=10,
    )
    result.add_argument(
        "--scsdg-lbfgs-max-function-evaluations",
        type=int,
        default=200,
    )
    result.add_argument(
        "--scsdg-lbfgs-tolerance",
        type=float,
        default=1e-10,
    )
    return result


def _validate(args: argparse.Namespace) -> None:
    positive = {
        "safe-gap-target": args.safe_gap_target,
        "solver-tolerance": args.solver_tolerance,
        "prox-tolerance": args.prox_tolerance,
        "time-limit": args.time_limit,
        "reference-time-limit": args.reference_time_limit,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if args.reference_objective is not None and not math.isfinite(
        args.reference_objective
    ):
        raise ValueError("reference-objective must be finite")
    if args.reference_seconds is not None and (
        not math.isfinite(args.reference_seconds)
        or args.reference_seconds <= 0.0
    ):
        raise ValueError("reference-seconds must be positive and finite")


def main(arguments: Sequence[str] | None = None) -> None:
    args = parser().parse_args(arguments)
    _validate(args)
    cases = _cases(args)
    profiles = _unique(args.constraint_profiles or ("bcw",))
    methods = _unique(args.methods or METHODS)
    if (
        args.reference_objective is not None
        and len(cases) * len(profiles) != 1
    ):
        raise ValueError(
            "--reference-objective requires exactly one case and one "
            "constraint profile"
        )

    summaries: list[Dict[str, Any]] = []
    histories: list[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for profile in profiles:
            instance_seed = args.seed + 10_000 * case_index
            problem = generate_synthetic_case(
                case,
                seed=instance_seed,
                target_iterations=args.target_iterations,
                constraint_profile=profile,
            )
            fields = _case_fields(
                case,
                profile,
                instance_seed,
                problem,
            )
            if args.reference_objective is None:
                reference = solve_relaxation(
                    problem,
                    "gurobi.python",
                    options={
                        "threads": args.threads,
                        "tolerance": min(args.solver_tolerance, 1e-10),
                        "time_limit": args.reference_time_limit,
                        "log": False,
                    },
                )
                if (
                    reference.status != "optimal"
                    or reference.objective is None
                ):
                    raise RuntimeError(
                        "an optimal Gurobi reference is required for "
                        f"{fields['case_id']}; got {reference.status}"
                    )
                optimum = float(reference.objective)
                reference_seconds = float(
                    reference.raw.get("end_to_end_seconds", math.nan)
                )
                reference_row = _reference_row(
                    reference,
                    optimum,
                    args.safe_gap_target,
                    fields,
                    problem,
                )
            else:
                optimum = float(args.reference_objective)
                reference_seconds = (
                    float(args.reference_seconds)
                    if args.reference_seconds is not None
                    else math.nan
                )
                reference_row = _cached_reference_row(
                    optimum,
                    args.safe_gap_target,
                    fields,
                    args.reference_seconds,
                )
            scale = max(1.0, abs(optimum))
            cutoff = optimum - args.safe_gap_target * scale
            summaries.append(reference_row)

            for method in methods:
                backend, options = _method_options(
                    method,
                    args,
                    cutoff,
                )
                result = solve_relaxation(
                    problem,
                    backend,
                    variant="metric-linesearch-restart",
                    pava="partial_sort",
                    options=options,
                )
                summaries.append(
                    _summary_row(
                        method,
                        backend,
                        result,
                        optimum,
                        cutoff,
                        args.safe_gap_target,
                        fields,
                        reference_seconds,
                        problem,
                    )
                )
                histories.extend(
                    _history_rows(
                        method,
                        result,
                        optimum,
                        args.safe_gap_target,
                        fields,
                    )
                )

    _write_csv(args.output, summaries)
    _write_csv(args.history_output, histories)
    if args.plot is not None:
        _plot(args.plot, histories, summaries)
    for row in summaries:
        elapsed = row.get("end_to_end_seconds")
        error = row.get("safe_dual_relative_error")
        print(
            row["case_id"],
            row["method"],
            row["status"],
            f"{float(elapsed):.6g}s" if elapsed is not None else "n/a",
            (
                f"safe_error={float(error):.3e}"
                if error is not None
                else "safe_error=n/a"
            ),
        )


if __name__ == "__main__":
    main()
