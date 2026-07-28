"""Restart-grid benchmark tests."""

from __future__ import annotations

import unittest

from sksfolio.benchmarks.run_fista_restarts import (
    _time_to_bound,
    restart_variants,
)


class FISTARestartBenchmarkTests(unittest.TestCase):
    def test_grid_contains_every_declared_variant(self) -> None:
        variants = restart_variants()
        identifiers = {variant.identifier for variant in variants}
        self.assertEqual(len(variants), 12)
        self.assertEqual(len(identifiers), len(variants))
        self.assertIn("none", identifiers)
        self.assertIn("gradient", identifiers)
        self.assertIn("function", identifiers)
        self.assertIn("hinder_lubin", identifiers)
        self.assertIn("periodic_k5", identifiers)
        self.assertIn("periodic_k100", identifiers)
        self.assertIn("primal_dual_gap_e1", identifiers)
        self.assertIn("primal_dual_gap_e3", identifiers)

    def test_time_to_bound_reports_unreached_runs(self) -> None:
        history = [
            {
                "run_id": "a",
                "dimension": 20,
                "profile": "bcw",
                "variant_index": 0,
                "restart_id": "none",
                "restart_label": "No restart",
                "total_wall_seconds": 0.2,
                "safe_bound_error_to_reference": 2e-6,
            },
            {
                "run_id": "a",
                "dimension": 20,
                "profile": "bcw",
                "variant_index": 0,
                "restart_id": "none",
                "restart_label": "No restart",
                "total_wall_seconds": 0.3,
                "safe_bound_error_to_reference": 5e-7,
            },
            {
                "run_id": "b",
                "dimension": 20,
                "profile": "bcw",
                "variant_index": 0,
                "restart_id": "none",
                "restart_label": "No restart",
                "total_wall_seconds": 0.25,
                "safe_bound_error_to_reference": 3e-6,
            },
        ]
        rows = _time_to_bound(history)
        millionth = next(
            row
            for row in rows
            if row["safe_bound_error_target"] == 1e-6
        )
        self.assertEqual(millionth["runs"], 2)
        self.assertEqual(millionth["target_reached_runs"], 1)
        self.assertFalse(millionth["target_reached_all_runs"])
        self.assertAlmostEqual(
            millionth["time_to_target_seconds_median"],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
