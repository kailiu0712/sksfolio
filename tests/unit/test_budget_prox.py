"""Tests for the full-investment perspective proximal oracle."""

from __future__ import annotations

import unittest

import numpy as np

from sksfolio.relaxation.fista import (
    prox_budget,
    prox_budget_details,
)
from sksfolio.relaxation.pava import prox as pava_prox


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = ordered - (
        cumulative - 1.0
    ) / np.arange(1, values.size + 1)
    count = int(np.flatnonzero(candidates > 0.0)[-1]) + 1
    threshold = (float(np.sum(ordered[:count])) - 1.0) / count
    return np.maximum(values - threshold, 0.0)


class BudgetProxTests(unittest.TestCase):
    def test_equality_and_long_only_domain(self) -> None:
        rng = np.random.default_rng(1701)
        for method in ("full_sort", "partial_sort"):
            for dimension, k in ((5, 2), (20, 7), (100, 13)):
                for _ in range(5):
                    values = rng.normal(size=dimension)
                    point = prox_budget(
                        values,
                        0.7,
                        k,
                        method,
                    )
                    self.assertLess(abs(float(np.sum(point)) - 1.0), 1e-9)
                    self.assertGreaterEqual(float(np.min(point)), 0.0)
                    self.assertLessEqual(float(np.max(point)), 1.0)

    def test_full_and_partial_sort_agree(self) -> None:
        rng = np.random.default_rng(812)
        for dimension in (6, 25, 200):
            for k in (2, max(2, dimension // 3), dimension - 1):
                for _ in range(8):
                    values = rng.normal(size=dimension)
                    gamma = float(rng.uniform(1e-3, 3.0))
                    full = prox_budget(
                        values,
                        gamma,
                        k,
                        "full_sort",
                    )
                    partial = prox_budget(
                        values,
                        gamma,
                        k,
                        "partial_sort",
                    )
                    np.testing.assert_allclose(
                        full,
                        partial,
                        atol=2e-10,
                        rtol=2e-10,
                    )

    def test_endpoint_k_values_match_independent_formulas(self) -> None:
        rng = np.random.default_rng(912)
        for dimension in (2, 7, 30):
            values = rng.normal(size=dimension)
            gamma = 1.3
            np.testing.assert_allclose(
                prox_budget(values, gamma, 1),
                _simplex_projection(values),
                atol=2e-13,
            )
            np.testing.assert_allclose(
                prox_budget(values, gamma, dimension),
                _simplex_projection(values / (1.0 + gamma)),
                atol=2e-13,
            )

    def test_brent_and_bisection_agree_and_report_details(self) -> None:
        values = np.array([1.7, -0.4, 0.2, 2.1, 0.8, -1.3])
        brent = prox_budget_details(
            values,
            0.9,
            3,
            root_method="brent",
        )
        bisection = prox_budget_details(
            values,
            0.9,
            3,
            root_method="bisection",
        )
        np.testing.assert_allclose(
            brent.x,
            bisection.x,
            atol=2e-10,
            rtol=2e-10,
        )
        self.assertLess(brent.equality_residual, 1e-10)
        self.assertLess(bisection.equality_residual, 1e-10)
        self.assertGreater(brent.evaluations, 0)
        self.assertGreater(bisection.evaluations, 0)
        self.assertLess(brent.evaluations, bisection.evaluations)

    def test_eta_has_the_lagrangian_shift_sign(self) -> None:
        values = np.array([0.1, 2.0, -0.7, 1.3, 0.4, -0.2, 0.8])
        gamma = 0.6
        k = 3
        for method in ("full_sort", "partial_sort"):
            details = prox_budget_details(
                values,
                gamma,
                k,
                method,
            )
            shifted = pava_prox(
                values - details.eta,
                gamma,
                k,
                method,
            )
            np.testing.assert_allclose(
                shifted,
                details.x,
                atol=2e-10,
                rtol=2e-10,
            )


if __name__ == "__main__":
    unittest.main()
