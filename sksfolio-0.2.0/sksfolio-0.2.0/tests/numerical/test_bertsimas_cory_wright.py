"""Tests for the paper parameter grid and its continuous relaxation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import sksfolio.benchmarks.run_relaxations as benchmark_runner
from sksfolio.benchmarks.bertsimas_cory_wright import (
    BCWCase,
    generate_synthetic_case,
    historical_cases,
    iter_paper_cases,
    or_library_cases,
)
from sksfolio.benchmarks.run_pdhg_fista import (
    MATCHED_BACKENDS,
    MATCHED_OUTPUT,
    MATCHED_TOLERANCE,
)
from sksfolio.benchmarks.run_relaxations import (
    SYNTHETIC_PROVENANCE,
    _case,
    _optimal_gurobi_reference,
    _row,
    _run_configuration,
)
from sksfolio.relaxation import (
    evaluate_solution,
    registered_backends,
    solve_relaxation,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RESULTS = (
    ROOT
    / "benchmarks"
    / "bertsimas_cory_wright_2022"
    / "paper_reference_results.csv"
)
RELAXATION_RESULTS = (
    ROOT
    / "benchmarks"
    / "bertsimas_cory_wright_2022"
    / "sp500_rank50_k10_results.csv"
)
MATCHED_RELAXATION_RESULTS = (
    ROOT
    / "benchmarks"
    / "bertsimas_cory_wright_2022"
    / "sp500_rank50_k10_pdhg_fista_results.csv"
)
BEST_MATCHED_RELAXATION_RESULTS = (
    ROOT
    / "benchmarks"
    / "bertsimas_cory_wright_2022"
    / "sp500_rank50_k10_best_pdhg_fista_results.csv"
)


class BertsimasCoryWrightTests(unittest.TestCase):
    def test_exact_paper_grid_is_encoded(self) -> None:
        or_cases = or_library_cases()
        historical = historical_cases()

        def signature(case):
            return (
                case.family,
                case.universe,
                case.dimension,
                case.rank,
                case.k,
                case.gamma_scale,
                case.regime,
            )

        or_dimensions = {
            "port1": 31,
            "port2": 85,
            "port3": 89,
            "port4": 98,
            "port5": 225,
        }
        expected_or = {
            (
                "or_library",
                universe,
                dimension,
                dimension,
                k,
                100.0,
                regime,
            )
            for universe, dimension in or_dimensions.items()
            for k in (5, 10, 20)
            for regime in ("unconstrained", "constrained")
        }
        historical_profiles = {
            "sp500": (499, (50, 100, 150, 200)),
            "russell1000": (958, (50, 100, 200, 300)),
            "wilshire5000": (3162, (100, 200, 500, 1000)),
        }
        expected_historical = {
            (
                "historical",
                universe,
                dimension,
                rank,
                k,
                gamma_scale,
                regime,
            )
            for universe, (dimension, ranks) in historical_profiles.items()
            for rank in ranks
            for k in (10, 50, 100, 200)
            for gamma_scale in (1.0, 100.0)
            for regime in ("unconstrained", "constrained")
        }
        self.assertEqual({signature(case) for case in or_cases}, expected_or)
        self.assertEqual(
            {signature(case) for case in historical},
            expected_historical,
        )
        self.assertEqual(len(or_cases), len(expected_or))
        self.assertEqual(len(historical), len(expected_historical))
        self.assertEqual(
            {signature(case) for case in iter_paper_cases()},
            expected_or | expected_historical,
        )
        nominal = historical_cases(processed_dimensions=False)
        self.assertEqual(
            {(case.universe, case.dimension) for case in nominal},
            {
                ("sp500", 500),
                ("russell1000", 1000),
                ("wilshire5000", 3200),
            },
        )
        for case in historical:
            self.assertAlmostEqual(
                case.gamma,
                case.gamma_scale / math.sqrt(case.dimension),
            )

    def test_paper_reference_rows_are_context_not_golden_targets(self) -> None:
        with REFERENCE_RESULTS.open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            rows,
            [
                {
                    "table": "7",
                    "universe": "sp500",
                    "rank": "50",
                    "k": "10",
                    "gamma_scale": "1",
                    "regime": "unconstrained",
                    "algorithm3_seconds": "0.01",
                    "cplex_misoco_seconds": "0.54",
                    "source_note": "integer model on paper hardware",
                },
                {
                    "table": "8",
                    "universe": "russell1000",
                    "rank": "50",
                    "k": "10",
                    "gamma_scale": "1",
                    "regime": "unconstrained",
                    "algorithm3_seconds": "0.02",
                    "cplex_misoco_seconds": "7.77",
                    "source_note": "integer model on paper hardware",
                },
                {
                    "table": "9",
                    "universe": "wilshire5000",
                    "rank": "100",
                    "k": "10",
                    "gamma_scale": "1",
                    "regime": "unconstrained",
                    "algorithm3_seconds": "0.04",
                    "cplex_misoco_seconds": "15.07",
                    "source_note": "integer model on paper hardware",
                },
            ],
        )

    def test_saved_dual_bound_snapshot_obeys_weak_duality(self) -> None:
        with RELAXATION_RESULTS.open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        pdhg_rows = [row for row in rows if row["backend"] == "pdhg"]
        gurobi_rows = [
            row for row in rows if row["backend"] == "gurobi.python"
        ]
        self.assertEqual(len(pdhg_rows), 10)
        self.assertEqual(len(gurobi_rows), 1)
        self.assertEqual(
            {row["status"] for row in pdhg_rows},
            {"iteration_limit"},
        )
        self.assertEqual(gurobi_rows[0]["status"], "optimal")
        reference = float(gurobi_rows[0]["objective"])
        for row in pdhg_rows:
            self.assertEqual(row["certificate_verified"], "True")
            self.assertLessEqual(
                float(row["safe_dual_bound"]),
                reference + 1e-12,
            )
            self.assertGreaterEqual(
                float(row["safe_dual_gap_to_reference"]),
                -1e-12,
            )
        for variant in {
            row["variant"] for row in pdhg_rows
        }:
            pair = {
                row["pava"]: row
                for row in pdhg_rows
                if row["variant"] == variant
            }
            self.assertEqual(set(pair), {"full_sort", "partial_sort"})
            self.assertAlmostEqual(
                float(pair["full_sort"]["safe_dual_bound"]),
                float(pair["partial_sort"]["safe_dual_bound"]),
                places=12,
            )

    def test_saved_matched_snapshot_reports_limits_and_certificates(
        self,
    ) -> None:
        with MATCHED_RELAXATION_RESULTS.open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        pdhg_rows = [row for row in rows if row["backend"] == "pdhg"]
        fista_rows = [row for row in rows if row["backend"] == "fista"]
        gurobi_rows = [
            row for row in rows if row["backend"] == "gurobi.python"
        ]
        self.assertEqual(len(pdhg_rows), 10)
        self.assertEqual(len(fista_rows), 2)
        self.assertEqual(len(gurobi_rows), 1)
        self.assertEqual(len({row["instance_id"] for row in rows}), 1)
        self.assertEqual(
            {row["constraint_names"] for row in rows},
            {"budget"},
        )
        self.assertEqual({row["tolerance"] for row in rows}, {"1e-08"})
        self.assertEqual(
            {row["reference_tolerance"] for row in rows},
            {"1e-10"},
        )
        self.assertEqual(
            {row["status"] for row in pdhg_rows},
            {"iteration_limit"},
        )
        self.assertEqual(
            {row["status"] for row in fista_rows},
            {"converged"},
        )
        self.assertEqual(
            {row["variant"] for row in fista_rows},
            {"adaptive-restart"},
        )
        self.assertEqual(gurobi_rows[0]["status"], "optimal")
        self.assertEqual(gurobi_rows[0]["primal_feasible"], "True")
        reference = float(gurobi_rows[0]["objective"])
        for row in pdhg_rows + fista_rows:
            self.assertEqual(row["certificate_verified"], "True")
            self.assertLessEqual(
                float(row["safe_dual_bound"]),
                reference + 1e-12,
            )
            self.assertGreaterEqual(
                float(row["safe_dual_gap_to_reference"]),
                -1e-12,
            )
        for row in fista_rows:
            self.assertEqual(row["primal_feasible"], "True")
            self.assertLessEqual(
                abs(float(row["objective"]) - reference),
                1e-8,
            )
            self.assertLessEqual(
                float(row["safe_dual_gap_to_reference"]),
                1e-8,
            )
        by_pava = {row["pava"]: row for row in fista_rows}
        self.assertEqual(
            set(by_pava),
            {"full_sort", "partial_sort"},
        )
        self.assertAlmostEqual(
            float(by_pava["full_sort"]["objective"]),
            float(by_pava["partial_sort"]["objective"]),
            places=13,
        )
        self.assertAlmostEqual(
            float(by_pava["full_sort"]["safe_dual_bound"]),
            float(by_pava["partial_sort"]["safe_dual_bound"]),
            places=13,
        )

    def test_saved_best_head_to_head_is_feasible_and_weakly_dual(
        self,
    ) -> None:
        with BEST_MATCHED_RELAXATION_RESULTS.open(
            newline="",
            encoding="utf-8",
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["backend"] for row in rows},
            {"pdhg", "fista", "gurobi.python"},
        )
        self.assertEqual(len({row["instance_id"] for row in rows}), 1)
        self.assertEqual(
            {row["constraint_names"] for row in rows},
            {"budget"},
        )
        self.assertEqual({row["tolerance"] for row in rows}, {"1e-08"})
        self.assertTrue(
            all(row["primal_feasible"] == "True" for row in rows)
        )
        self.assertTrue(
            all(float(row["maximum_violation"]) <= 1e-8 for row in rows)
        )
        by_backend = {row["backend"]: row for row in rows}
        pdhg = by_backend["pdhg"]
        fista = by_backend["fista"]
        gurobi = by_backend["gurobi.python"]
        self.assertEqual(pdhg["status"], "converged")
        self.assertEqual(pdhg["variant"], "metric-linesearch-restart")
        self.assertEqual(pdhg["pava"], "partial_sort")
        self.assertEqual(pdhg["iterations"], "6050")
        self.assertEqual(fista["status"], "converged")
        self.assertEqual(fista["variant"], "adaptive-restart")
        self.assertEqual(fista["pava"], "partial_sort")
        self.assertEqual(fista["iterations"], "46")
        self.assertEqual(gurobi["status"], "optimal")
        reference = float(gurobi["objective"])
        self.assertEqual(float(gurobi["reference_objective"]), reference)
        for row in (pdhg, fista):
            self.assertEqual(row["certificate_verified"], "True")
            self.assertLessEqual(
                float(row["safe_dual_bound"]),
                reference + 1e-12,
            )
            self.assertGreaterEqual(
                float(row["safe_dual_gap_to_reference"]),
                -1e-12,
            )
            self.assertLessEqual(
                abs(float(row["objective"]) - reference),
                1e-8,
            )

    def test_runner_records_configuration_and_separate_timings(self) -> None:
        args = SimpleNamespace(
            universe="sp500",
            rank=50,
            k=10,
            gamma_scale=100.0,
            regime="unconstrained",
        )
        case = _case(args)
        self.assertEqual(case.dimension, 499)
        configuration = _run_configuration(
            seed=17,
            target_iterations=31,
            tolerance=2e-7,
            max_iterations=4321,
            time_limit=12.5,
            threads=3,
        )
        self.assertEqual(
            configuration,
            {
                "instance_provenance": SYNTHETIC_PROVENANCE,
                "paper_returns_used": False,
                "seed": 17,
                "target_iterations": 31,
                "tolerance": 2e-7,
                "feasibility_tolerance": 2e-7,
                "max_iterations": 4321,
                "time_limit": 12.5,
                "threads": 3,
                "commercial_log": False,
                "reference_tolerance": 1e-10,
                "effective_commercial_tolerance": 1e-10,
                "constraint_profile": "bcw",
                "sector_count": None,
                "style_factor_count": None,
                "stress_constraint_count": None,
                "sector_band": None,
                "style_band": None,
                "stress_band": None,
                "pdhg_check_interval": 50,
                "pdhg_min_epoch": 100,
                "pdhg_max_epoch": 2000,
                "fista_initial_lipschitz": None,
                "fista_backtracking_factor": 2.0,
                "fista_step_growth": 1.1,
                "fista_line_search_tolerance": 1e-12,
                "fista_adaptive_restart": True,
                "fista_history_interval": 25,
                "fista_prox_tolerance": 1e-8,
                "fista_prox_max_iterations": 1000,
                "fista_max_backtracks": 60,
                "fista_prox_oracles": "auto",
                "fista_major_max_iterations": 20_000,
            },
        )

        class Certificate:
            @staticmethod
            def verify(problem) -> bool:
                return problem.constraint_names == ["budget"]

        problem = SimpleNamespace(constraint_names=["budget"])
        result = SimpleNamespace(
            raw={
                "diagnostics": {"violations": {"maximum": 1e-8}},
                "iterations": 12,
                "solve_seconds": 2.0,
                "total_seconds": 5.0,
                "wrapper_seconds": 5.5,
            },
            dual_certificate=Certificate(),
            status="optimal",
            objective=1.0,
            safe_dual_bound=0.9,
        )
        row = _row(
            case,
            problem,
            "gurobi.python",
            "",
            "",
            result,
            configuration,
        )
        self.assertEqual(row["instance_provenance"], SYNTHETIC_PROVENANCE)
        self.assertFalse(row["paper_returns_used"])
        self.assertEqual(row["constraint_names"], "budget")
        self.assertEqual(row["constraint_count"], 1)
        self.assertTrue(row["primal_feasible"])
        self.assertEqual(row["solve_seconds"], 2.0)
        self.assertEqual(row["total_seconds"], 5.0)
        self.assertEqual(row["wrapper_seconds"], 5.5)

    def test_only_optimal_native_gurobi_is_a_reference(self) -> None:
        time_limited = [
            {
                "backend": "gurobi.python",
                "status": "time_limit",
                "objective": 1.2,
                "primal_feasible": True,
            },
        ]
        self.assertIsNone(_optimal_gurobi_reference(time_limited))
        infeasible = [
            {
                "backend": "gurobi.python",
                "status": "optimal",
                "objective": 0.9,
                "primal_feasible": False,
            },
        ]
        self.assertIsNone(_optimal_gurobi_reference(infeasible))
        optimal = time_limited + [
            {
                "backend": "gurobi.python",
                "status": "optimal",
                "objective": 1.0,
                "primal_feasible": True,
            },
        ]
        self.assertEqual(_optimal_gurobi_reference(optimal), 1.0)

    def test_matched_runner_uses_one_instance_and_all_method_pairs(
        self,
    ) -> None:
        self.assertIn("fista", registered_backends())
        self.assertEqual(
            MATCHED_BACKENDS,
            ("pdhg", "fista", "gurobi.python"),
        )
        self.assertEqual(
            MATCHED_OUTPUT.name,
            "bcw_matched_pdhg_fista_results.csv",
        )
        self.assertEqual(MATCHED_TOLERANCE, 1e-8)
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=100.0,
            regime="unconstrained",
        )
        problem = SimpleNamespace(constraint_names=["budget"])
        calls = []

        def fake_solve(received_problem, backend, **kwargs):
            calls.append((received_problem, backend, kwargs))
            status = "optimal" if backend == "gurobi.python" else (
                "iteration_limit"
            )
            return SimpleNamespace(
                raw={
                    "diagnostics": {
                        "violations": {"maximum": 1e-9},
                    },
                    "variant": (
                        "adaptive-restart"
                        if backend == "fista"
                        else ""
                    ),
                    "iterations": 7,
                    "residual": 2e-5,
                    "relative_residual": 3e-5,
                    "solve_seconds": 0.2,
                    "total_seconds": 0.3,
                },
                dual_certificate=None,
                status=status,
                objective=0.75 if status == "optimal" else 0.8,
                safe_dual_bound=(
                    None if status == "optimal" else 0.7
                ),
            )

        variants = (
            "fixed",
            "fixed-restart",
            "linesearch",
            "linesearch-restart",
            "metric-linesearch-restart",
        )
        pava_methods = ("full_sort", "partial_sort")
        with patch.object(
            benchmark_runner,
            "generate_synthetic_case",
            return_value=problem,
        ) as generate, patch.object(
            benchmark_runner,
            "solve_relaxation",
            side_effect=fake_solve,
        ):
            rows = benchmark_runner.run(
                case,
                backends=MATCHED_BACKENDS,
                variants=variants,
                pava_methods=pava_methods,
                seed=29,
                target_iterations=37,
                tolerance=1e-6,
                max_iterations=5000,
                time_limit=600.0,
                threads=1,
            )

        generate.assert_called_once_with(
            case,
            seed=29,
            target_iterations=37,
            constraint_profile="bcw",
            sectors=None,
            style_factors=None,
            stress_constraints=None,
            sector_band=None,
            style_band=None,
            stress_band=None,
        )
        self.assertEqual(len(calls), 13)
        self.assertTrue(
            all(received is problem for received, _, _ in calls)
        )
        self.assertEqual(len(rows), 13)
        self.assertEqual(
            {row["instance_id"] for row in rows},
            {
                (
                    f"{case.key}-{SYNTHETIC_PROVENANCE}"
                    "-seed29"
                ),
            },
        )
        self.assertEqual(
            {row["constraint_names"] for row in rows},
            {"budget"},
        )
        pdhg_rows = [row for row in rows if row["backend"] == "pdhg"]
        fista_rows = [row for row in rows if row["backend"] == "fista"]
        gurobi_rows = [
            row for row in rows if row["backend"] == "gurobi.python"
        ]
        self.assertEqual(len(pdhg_rows), 10)
        self.assertEqual(len(fista_rows), 2)
        self.assertEqual(len(gurobi_rows), 1)
        self.assertEqual(
            {
                (row["variant"], row["pava"])
                for row in pdhg_rows
            },
            {
                (variant, pava)
                for variant in variants
                for pava in pava_methods
            },
        )
        self.assertEqual(
            {(row["variant"], row["pava"]) for row in fista_rows},
            {
                ("adaptive-restart", "full_sort"),
                ("adaptive-restart", "partial_sort"),
            },
        )
        self.assertTrue(
            all(row["reference_objective"] == 0.75 for row in rows)
        )
        self.assertTrue(
            all(row["safe_dual_bound"] == 0.7 for row in fista_rows)
        )
        self.assertTrue(
            all(
                abs(row["safe_dual_gap_to_reference"] - 0.05)
                <= 1e-15
                for row in fista_rows
            )
        )
        fista_calls = [
            kwargs
            for _, backend, kwargs in calls
            if backend == "fista"
        ]
        self.assertEqual(
            {call["pava"] for call in fista_calls},
            {"full_sort", "partial_sort"},
        )
        self.assertTrue(
            all("variant" not in call for call in fista_calls)
        )
        self.assertTrue(
            all(
                call["options"]["adaptive_restart"]
                and call["options"]["step_growth"] == 1.1
                and call["options"]["prox_tolerance"] == 1e-8
                and call["options"]["prox_max_iterations"] == 1000
                and call["options"]["prox_oracle"] == "auto"
                for call in fista_calls
            )
        )
        gurobi_call = next(
            kwargs
            for _, backend, kwargs in calls
            if backend == "gurobi.python"
        )
        self.assertEqual(
            gurobi_call["options"]["tolerance"],
            1e-10,
        )

    def test_fista_benchmark_accepts_additional_linear_rows(self) -> None:
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=1.0,
            regime="constrained",
        )
        problem = SimpleNamespace(
            constraint_names=["budget", "minimum_return"],
        )
        result = SimpleNamespace(
            raw={
                "diagnostics": {"violations": {"maximum": 0.0}},
                "variant": "adaptive-restart",
                "prox_oracle_used": "row_scaled_dual_fista",
            },
            dual_certificate=None,
            status="converged",
            objective=1.0,
            safe_dual_bound=0.9,
        )
        with patch.object(
            benchmark_runner,
            "generate_synthetic_case",
            return_value=problem,
        ), patch.object(
            benchmark_runner,
            "solve_relaxation",
            return_value=result,
        ) as solve:
            rows = benchmark_runner.run(
                case,
                backends=("fista",),
                variants=(),
                pava_methods=("partial_sort",),
                seed=31,
                target_iterations=17,
                tolerance=1e-8,
                max_iterations=100,
                time_limit=5.0,
                threads=1,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["prox_oracle"],
            "row_scaled_dual_fista",
        )
        self.assertEqual(
            solve.call_args.kwargs["options"]["prox_oracle"],
            "auto",
        )

    def test_sp500_shaped_relaxation_has_safe_bound(self) -> None:
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=100.0,
            regime="unconstrained",
        )
        problem = generate_synthetic_case(
            case,
            seed=19,
            target_iterations=30,
        )
        self.assertEqual(problem.dimension, 499)
        self.assertEqual(problem.rank, 50)
        self.assertEqual(problem.constraint_names, ["budget"])
        self.assertEqual(problem.return_reward, 1.0)
        self.assertAlmostEqual(
            problem.perspective_weight,
            math.sqrt(499.0) / 100.0,
        )
        result = solve_relaxation(
            problem,
            "pdhg",
            variant="metric-linesearch-restart",
            pava="partial_sort",
            options={
                "threads": 1,
                "max_iterations": 100,
                "check_interval": 20,
                "min_epoch": 20,
                "max_epoch": 100,
            },
        )
        certificate = result.dual_certificate
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate.verify(problem))
        anchor_objective = evaluate_solution(
            problem,
            problem.anchor,
        )["objective"]
        self.assertLessEqual(
            result.safe_dual_bound,
            anchor_objective + 1e-10,
        )
        self.assertTrue(np.isfinite(result.safe_dual_bound))

    def test_constrained_regime_adds_the_paper_return_row(self) -> None:
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=1.0,
            regime="constrained",
        )
        problem = generate_synthetic_case(
            case,
            seed=23,
            target_iterations=30,
        )
        self.assertEqual(
            problem.constraint_names,
            ["budget", "minimum_return"],
        )
        self.assertEqual(problem.return_reward, 0.0)
        self.assertEqual(problem.lower[0], problem.upper[0])
        self.assertTrue(np.isposinf(problem.upper[1]))
        self.assertGreaterEqual(
            float(problem.mu @ problem.anchor),
            float(problem.lower[1]) - 1e-10,
        )

    def test_many_constraint_profile_builds_every_linear_block(
        self,
    ) -> None:
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=100.0,
            regime="constrained",
        )
        problem = generate_synthetic_case(
            case,
            seed=7,
            target_iterations=30,
            constraint_profile="many",
        )
        self.assertEqual(problem.rows, 262)
        self.assertEqual(problem.constraint_names[0], "budget")
        self.assertEqual(
            problem.constraint_names[1],
            "minimum_return",
        )
        self.assertEqual(
            sum(name.startswith("sector_") for name in problem.constraint_names),
            20,
        )
        self.assertEqual(
            sum(name.startswith("style_") for name in problem.constraint_names),
            40,
        )
        self.assertEqual(
            sum(
                name.startswith("stress_loss_")
                for name in problem.constraint_names
            ),
            200,
        )
        values = np.asarray(problem.C @ problem.anchor).reshape(-1)
        violation = max(
            float(np.max(np.maximum(problem.lower - values, 0.0))),
            float(np.max(np.maximum(values - problem.upper, 0.0))),
        )
        self.assertLessEqual(violation, 1e-12)
        self.assertEqual(
            problem.metadata["constraint_profile"],
            "many",
        )

    def test_constraint_profile_rejects_too_many_sectors(self) -> None:
        case = BCWCase(
            family="historical",
            universe="sp500",
            dimension=499,
            rank=50,
            k=10,
            gamma_scale=100.0,
            regime="constrained",
        )
        with self.assertRaisesRegex(
            ValueError,
            "sectors cannot exceed",
        ):
            generate_synthetic_case(
                case,
                seed=7,
                target_iterations=5,
                constraint_profile="many",
                sectors=500,
            )


if __name__ == "__main__":
    unittest.main()
