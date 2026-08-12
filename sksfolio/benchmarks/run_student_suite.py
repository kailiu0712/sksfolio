"""One-instance benchmark for the public relaxation and exact-solve APIs.

The driver deliberately reports JSON rather than a presentation table.  It
is intended to be a reproducible starting point for student experiments and
keeps optional commercial solvers as structured ``unavailable`` records.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from ..bnb import solve_bnb
from ..incumbent import (
    solve_gurobi_incumbent,
    solve_incumbent,
    solve_mosek_incumbent,
)
from ..relaxation import load_instance_bundle, solve_relaxation
from .instance_generator import generate_instance


SCHEMA_VERSION = 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the student comparison suite on one constrained sparse "
            "Markowitz instance and emit machine-readable JSON"
        )
    )
    result.add_argument("--bundle", type=Path, default=None)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--dimension", type=int, default=500)
    result.add_argument("--rank", type=int, default=20)
    result.add_argument("--k", type=int, default=25)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--gamma-scale", type=float, default=100.0)
    result.add_argument("--sectors", type=int, default=10)
    result.add_argument("--styles", type=int, default=4)
    result.add_argument("--stresses", type=int, default=4)
    result.add_argument("--sector-band", type=float, default=0.08)
    result.add_argument("--style-band", type=float, default=0.20)
    result.add_argument("--stress-band", type=float, default=0.01)
    result.add_argument(
        "--relaxation-time-limit",
        type=float,
        default=30.0,
        help="per-method continuous-relaxation time limit",
    )
    result.add_argument(
        "--relaxation-iterations",
        type=int,
        default=10_000,
    )
    result.add_argument("--relaxation-tolerance", type=float, default=1e-7)
    result.add_argument("--prox-tolerance", type=float, default=1e-8)
    result.add_argument("--prox-iterations", type=int, default=1_000)
    result.add_argument("--check-interval", type=int, default=25)
    result.add_argument(
        "--jump-optimizer",
        default=None,
        help=(
            "optional JuMP optimizer alias or Package.Constructor, for "
            "example clarabel, gurobi, or mosek"
        ),
    )
    result.add_argument(
        "--jump-mip-optimizer",
        action="append",
        choices=("gurobi", "mosek"),
        default=[],
        help=(
            "optional JuMP mixed-integer perspective backend; repeat the "
            "option to benchmark both gurobi and mosek"
        ),
    )
    result.add_argument("--incumbent-time-limit", type=float, default=15.0)
    result.add_argument("--mip-time-limit", type=float, default=60.0)
    result.add_argument("--node-limit", type=int, default=1_000_000)
    result.add_argument("--relative-gap", type=float, default=1e-4)
    result.add_argument("--absolute-gap", type=float, default=1e-8)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument(
        "--skip-commercial-relaxations",
        action="store_true",
    )
    result.add_argument("--skip-exact", action="store_true")
    result.add_argument("--skip-commercial-mip", action="store_true")
    return result


def _instance(args: argparse.Namespace) -> Any:
    if args.bundle is not None:
        return load_instance_bundle(args.bundle)
    return generate_instance(
        dimension=args.dimension,
        rank=args.rank,
        k=args.k,
        gamma_scale=args.gamma_scale,
        regime="hybrid",
        seed=args.seed,
        sectors=args.sectors,
        style_factors=args.styles,
        stress_constraints=args.stresses,
        target_fraction=0.3,
        target_iterations=200,
        sector_band=args.sector_band,
        style_band=args.style_band,
        stress_band=args.stress_band,
        annual_volatility=0.20,
        common_correlation=0.15,
    )


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _error_record(name: str, family: str, error: BaseException) -> dict[str, Any]:
    return {
        "method": name,
        "family": family,
        "status": "unavailable",
        "error": f"{type(error).__name__}: {error}",
    }


def _certificate_verified(result: Any, instance: Any) -> bool:
    certificate = result.dual_certificate
    if certificate is None:
        return False
    try:
        return bool(certificate.verify(instance))
    except (TypeError, ValueError):
        return False


def _relaxation_record(
    name: str,
    backend: str,
    result: Any,
    observed_seconds: float,
    instance: Any,
) -> dict[str, Any]:
    raw = result.raw
    diagnostics = raw.get("diagnostics", {})
    violations = diagnostics.get("violations", {})
    return {
        "method": name,
        "family": "continuous_relaxation",
        "backend": backend,
        "status": result.status,
        "objective": result.objective,
        "primal_feasible": result.primal_feasible,
        "primal_upper_bound": result.primal_upper_bound,
        "safe_dual_bound": result.safe_dual_bound,
        "safe_certificate_verified": _certificate_verified(result, instance),
        "solver_objective_bound": result.solver_objective_bound,
        "maximum_violation": _finite(violations.get("maximum")),
        "iterations": raw.get("iterations"),
        "restarts": raw.get("restarts"),
        "line_search_enabled": raw.get("line_search_enabled"),
        "prox_oracle_requested": raw.get("prox_oracle_requested"),
        "prox_oracle_used": raw.get("prox_oracle_used"),
        "solve_seconds": _finite(raw.get("solve_seconds")),
        "end_to_end_seconds": _finite(raw.get("end_to_end_seconds")),
        "observed_seconds": observed_seconds,
        "error": raw.get("error", raw.get("message")),
    }


def _continuous_configurations(
    args: argparse.Namespace,
) -> list[tuple[str, str, dict[str, Any]]]:
    common = {
        "threads": args.threads,
        "max_iterations": args.relaxation_iterations,
        "time_limit": args.relaxation_time_limit,
        "tolerance": args.relaxation_tolerance,
        "feasibility_tolerance": args.relaxation_tolerance,
    }
    fista = {
        **common,
        "history_interval": args.check_interval,
        "prox_tolerance": args.prox_tolerance,
        "prox_max_iterations": args.prox_iterations,
        "restart_strategy": "gradient",
    }
    configurations = [
        (
            "fista_dual_fista",
            "fista",
            {**fista, "prox_oracle": "dual_fista"},
        ),
        (
            "fista_dual_lbfgs",
            "fista",
            {**fista, "prox_oracle": "dual_lbfgs"},
        ),
    ]
    if not args.skip_commercial_relaxations:
        commercial = {
            "threads": args.threads,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "log": False,
            "warm_start": False,
        }
        configurations.extend(
            [
                ("gurobi_native", "gurobi.python", dict(commercial)),
                ("mosek_native", "mosek.python", dict(commercial)),
            ]
        )
    if args.jump_optimizer is not None:
        configurations.append(
            (
                f"jump_{args.jump_optimizer}",
                "jump.julia",
                {
                    "optimizer": args.jump_optimizer,
                    "threads": args.threads,
                    "time_limit": args.relaxation_time_limit,
                    "tolerance": args.relaxation_tolerance,
                    "log": False,
                    "warm_start": False,
                },
            )
        )
    return configurations


def _run_relaxations(
    instance: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name, backend, options in _continuous_configurations(args):
        start = time.perf_counter()
        try:
            result = solve_relaxation(
                instance,
                backend=backend,
                warm_start=False,
                options=options,
            )
        except Exception as error:
            records.append(
                _error_record(name, "continuous_relaxation", error)
            )
            continue
        elapsed = time.perf_counter() - start
        records.append(
            _relaxation_record(name, backend, result, elapsed, instance)
        )
        results[name] = result
    return records, results


def _exact_record(
    name: str,
    result: Any,
    observed_seconds: float,
    shared_preprocessing_seconds: float,
) -> dict[str, Any]:
    raw = result.raw
    upper = _finite(result.upper_bound)
    lower = _finite(
        getattr(result, "lower_bound", None)
        if name == "sksfolio_bnb"
        else raw.get("solver_objective_bound")
    )
    return {
        "method": name,
        "family": "sparse_exact",
        "status": result.status,
        "upper_bound": upper,
        "lower_bound": lower,
        "absolute_gap": (
            None if upper is None or lower is None else max(upper - lower, 0.0)
        ),
        "relative_gap": (
            None
            if upper is None or lower is None
            else max(upper - lower, 0.0) / max(1.0, abs(upper))
        ),
        "nodes": raw.get("nodes_processed", raw.get("node_count")),
        "root_screened": raw.get("root_screened_count"),
        "screening_fixings": raw.get("screening_fixings"),
        "restricted_qp_solves": raw.get("restricted_qp_solves"),
        "formulation": raw.get("formulation"),
        "model_build_seconds": _finite(raw.get("build_seconds")),
        "optimizer_seconds": _finite(raw.get("optimizer_seconds")),
        "commercial_total_seconds": _finite(raw.get("total_seconds")),
        "incumbent_polish_seconds": _finite(
            raw.get("incumbent_polish_seconds")
        ),
        "search_seconds": _finite(raw.get("solve_seconds")),
        "observed_search_seconds": observed_seconds,
        "shared_preprocessing_seconds": shared_preprocessing_seconds,
        "end_to_end_seconds": shared_preprocessing_seconds + observed_seconds,
        "error": raw.get("error"),
    }


def _run_exact(
    instance: Any,
    args: argparse.Namespace,
    relaxations: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relaxation_name = "fista_dual_lbfgs"
    relaxation = relaxations.get(relaxation_name)
    if relaxation is None:
        relaxation_name = "fista_dual_fista"
        relaxation = relaxations.get(relaxation_name)
    if relaxation is None:
        return (
            {
                "status": "unavailable",
                "error": "no FISTA root relaxation was available",
            },
            [],
        )

    root_seconds = _finite(relaxation.raw.get("end_to_end_seconds")) or 0.0
    incumbent_start = time.perf_counter()
    try:
        incumbent = solve_incumbent(
            instance,
            relaxation,
            method="auto",
            restricted_solver="osqp",
            time_limit=args.incumbent_time_limit,
            random_state=args.seed,
        )
    except Exception as error:
        return (
            {
                "status": "unavailable",
                "root_relaxation": relaxation_name,
                "error": f"{type(error).__name__}: {error}",
            },
            [],
        )
    incumbent_seconds = time.perf_counter() - incumbent_start
    preprocessing = root_seconds + incumbent_seconds
    warm_start = incumbent if incumbent.feasible else None
    matched_absolute_gap = float(args.absolute_gap)
    if incumbent.upper_bound is not None:
        matched_absolute_gap = max(
            matched_absolute_gap,
            float(args.relative_gap)
            * max(1.0, abs(float(incumbent.upper_bound))),
        )
    matched_relative_gap = 0.0
    shared = {
        "status": incumbent.status,
        "root_relaxation": relaxation_name,
        "restricted_solver": "osqp",
        "incumbent_feasible": incumbent.feasible,
        "incumbent_upper_bound": incumbent.upper_bound,
        "root_relaxation_seconds": root_seconds,
        "incumbent_seconds": incumbent_seconds,
        "shared_preprocessing_seconds": preprocessing,
        "requested_relative_gap": args.relative_gap,
        "requested_absolute_gap": args.absolute_gap,
        "effective_relative_gap": matched_relative_gap,
        "effective_absolute_gap": matched_absolute_gap,
    }
    records: list[dict[str, Any]] = []

    start = time.perf_counter()
    try:
        custom = solve_bnb(
            instance,
            relaxation=relaxation,
            incumbent=warm_start,
            restricted_solver="osqp",
            time_limit=args.mip_time_limit,
            node_limit=args.node_limit,
            relative_gap=matched_relative_gap,
            absolute_gap=matched_absolute_gap,
            options={
                "safe_screening": True,
                "node_dual_iterations": 15,
                "feasibility_tolerance": args.relaxation_tolerance,
                "relaxation_options": {"threads": args.threads},
            },
        )
        records.append(
            _exact_record(
                "sksfolio_bnb",
                custom,
                time.perf_counter() - start,
                preprocessing,
            )
        )
    except Exception as error:
        records.append(_error_record("sksfolio_bnb", "sparse_exact", error))

    if not args.skip_commercial_mip:
        start = time.perf_counter()
        try:
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                gurobi = solve_gurobi_incumbent(
                    instance,
                    warm_start=warm_start,
                    time_limit=args.mip_time_limit,
                    options={
                        "verbose": False,
                        "Threads": args.threads,
                        "MIPFocus": 0,
                        "MIPGap": matched_relative_gap,
                        "MIPGapAbs": matched_absolute_gap,
                        "FeasibilityTol": 1e-9,
                        "OptimalityTol": 1e-9,
                        "BarQCPConvTol": 1e-9,
                        "Seed": args.seed,
                    },
                )
            records.append(
                _exact_record(
                    "gurobi_perspective_miqcp",
                    gurobi,
                    time.perf_counter() - start,
                    preprocessing,
                )
            )
        except Exception as error:
            records.append(
                _error_record(
                    "gurobi_perspective_miqcp",
                    "sparse_exact",
                    error,
                )
            )

        start = time.perf_counter()
        try:
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                mosek = solve_mosek_incumbent(
                    instance,
                    warm_start=warm_start,
                    time_limit=args.mip_time_limit,
                    options={
                        "verbose": False,
                        "threads": args.threads,
                        "relative_gap": matched_relative_gap,
                        "absolute_gap": matched_absolute_gap,
                    },
                )
            records.append(
                _exact_record(
                    "mosek_perspective_misocp",
                    mosek,
                    time.perf_counter() - start,
                    preprocessing,
                )
            )
        except Exception as error:
            records.append(
                _error_record(
                    "mosek_perspective_misocp",
                    "sparse_exact",
                    error,
                )
            )

        for optimizer in dict.fromkeys(args.jump_mip_optimizer):
            name = f"jump_{optimizer}_perspective_mip"
            start = time.perf_counter()
            try:
                incumbent_module = importlib.import_module(
                    "sksfolio.incumbent"
                )
                solver = getattr(
                    incumbent_module,
                    "solve_jump_incumbent",
                    None,
                )
                if solver is None:
                    raise ImportError(
                        "solve_jump_incumbent is not available in this build"
                    )
                captured = io.StringIO()
                with redirect_stdout(captured), redirect_stderr(captured):
                    jump_result = solver(
                        instance,
                        optimizer=optimizer,
                        warm_start=warm_start,
                        time_limit=args.mip_time_limit,
                        options={
                            "log": False,
                            "julia_instantiate": True,
                            "threads": args.threads,
                            "relative_gap": matched_relative_gap,
                            "absolute_gap": matched_absolute_gap,
                        },
                    )
                records.append(
                    _exact_record(
                        name,
                        jump_result,
                        time.perf_counter() - start,
                        preprocessing,
                    )
                )
            except Exception as error:
                records.append(_error_record(name, "sparse_exact", error))
    return shared, records


def _exact_comparison(
    records: Sequence[Mapping[str, Any]],
    relative_gap: float,
    absolute_gap: float,
) -> dict[str, Any]:
    comparison_relative_tolerance = max(float(relative_gap), 1e-10)
    comparison_absolute_tolerance = max(float(absolute_gap), 1e-10)
    feasible = [
        record
        for record in records
        if _finite(record.get("upper_bound")) is not None
    ]
    if not feasible:
        return {"best_upper_bound": None, "fastest_comparable_method": None}
    best = min(float(record["upper_bound"]) for record in feasible)
    comparable = [
        record
        for record in feasible
        if (
            _finite(record.get("relative_gap")) is not None
            and float(record["relative_gap"])
            <= comparison_relative_tolerance
        )
        or (
            _finite(record.get("absolute_gap")) is not None
            and float(record["absolute_gap"])
            <= comparison_absolute_tolerance
        )
    ]
    fastest = min(
        comparable,
        key=lambda item: float(item.get("end_to_end_seconds", math.inf)),
        default=None,
    )
    return {
        "best_upper_bound": best,
        "comparison_relative_tolerance": comparison_relative_tolerance,
        "comparison_absolute_tolerance": comparison_absolute_tolerance,
        "fastest_comparable_method": (
            None if fastest is None else fastest.get("method")
        ),
        "methods": [
            {
                "method": record.get("method"),
                "objective_minus_best": (
                    None
                    if _finite(record.get("upper_bound")) is None
                    else float(record["upper_bound"]) - best
                ),
                "end_to_end_seconds": record.get("end_to_end_seconds"),
            }
            for record in records
        ],
    }


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    instance = _instance(args)
    constraint_names = list(instance.constraint_names)
    relaxation_records, relaxation_results = _run_relaxations(instance, args)
    if args.skip_exact:
        shared: dict[str, Any] = {"status": "skipped"}
        exact_records: list[dict[str, Any]] = []
    else:
        shared, exact_records = _run_exact(
            instance,
            args,
            relaxation_results,
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "instance": {
            "dimension": instance.dimension,
            "rank": instance.rank,
            "k": instance.k,
            "constraint_rows": instance.rows,
            "constraint_nnz": int(instance.C.nnz),
            "seed": args.seed,
            "gamma_scale": args.gamma_scale,
            "perspective_weight": instance.perspective_weight,
            "constraint_names": constraint_names,
            "constraint_families": {
                "budget": sum(name == "budget" for name in constraint_names),
                "minimum_return": sum(
                    name == "minimum_return" for name in constraint_names
                ),
                "sectors": sum(
                    name.startswith("sector_") for name in constraint_names
                ),
                "styles": sum(
                    name.startswith("style_") for name in constraint_names
                ),
                "stresses": sum(
                    name.startswith("stress_") for name in constraint_names
                ),
            },
        },
        "fairness": {
            "threads": args.threads,
            "relaxation_tolerance": args.relaxation_tolerance,
            "relaxation_time_limit_per_method": args.relaxation_time_limit,
            "mip_time_limit_per_method": args.mip_time_limit,
            "relative_mip_gap": args.relative_gap,
            "absolute_mip_gap": args.absolute_gap,
            "gap_policy": (
                "one absolute cutoff is derived from the shared incumbent "
                "and passed unchanged to all exact solvers"
            ),
            "continuous_warm_start": "disabled_for_every_method",
            "exact_warm_start": "one_shared_OSQP_incumbent",
            "bnb_safe_screening": True,
            "gurobi_mip_focus": 0,
            "jump_mip_optimizers": list(
                dict.fromkeys(args.jump_mip_optimizer)
            ),
        },
        "continuous_relaxations": relaxation_records,
        "shared_exact_preprocessing": shared,
        "sparse_exact_solvers": exact_records,
        "exact_comparison": _exact_comparison(
            exact_records,
            args.relative_gap,
            args.absolute_gap,
        ),
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
