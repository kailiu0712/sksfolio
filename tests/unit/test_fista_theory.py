"""Numerical identity checks used by the FISTA convergence proof."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from sksfolio.relaxation.fista.linear_prox import (
    LinearConstraintProx,
    _fista_momentum as _inner_fista_momentum,
    _smooth_dual_value,
    _smooth_dual_value_scratch,
    _variable_fista_momentum,
)
from sksfolio.relaxation.fista.solver import (
    _fista_momentum as _outer_fista_momentum,
)
from sksfolio.relaxation.problem import perspective_value


class FISTATheoryChecks(unittest.TestCase):
    def test_allocation_light_dual_value_matches_reference(self) -> None:
        rng = np.random.default_rng(20260820)
        for dimension, k in ((8, 3), (31, 7), (64, 64)):
            shifted = rng.normal(size=dimension)
            point = rng.normal(size=dimension)
            gamma = 0.73
            expected = _smooth_dual_value(shifted, point, gamma, k)
            actual = _smooth_dual_value_scratch(
                shifted,
                point,
                gamma,
                k,
                np.empty(dimension),
                np.empty(dimension),
                np.empty(dimension),
            )
            self.assertAlmostEqual(actual, expected, places=12)

    def test_variable_momentum_is_synchronous_with_curvature(self) -> None:
        previous_momentum = 1.0
        previous_curvature = 0.8
        for curvature in (0.4, 1.2, 0.3, 0.9, 0.9):
            momentum = _variable_fista_momentum(
                previous_momentum,
                curvature,
                previous_curvature,
            )
            self.assertAlmostEqual(
                momentum * (momentum - 1.0) / curvature,
                previous_momentum
                * previous_momentum
                / previous_curvature,
                places=13,
            )
            previous_momentum = momentum
            previous_curvature = curvature

    def test_momentum_satisfies_nonincreasing_step_compatibility(
        self,
    ) -> None:
        curvatures = [0.7, 0.7, 1.4, 1.4, 2.8, 3.0]
        momentum = 1.0
        previous_momentum = None
        previous_curvature = None
        for curvature in curvatures:
            if previous_momentum is not None:
                left = momentum * (momentum - 1.0) / curvature
                right = (
                    previous_momentum * previous_momentum
                    / previous_curvature
                )
                self.assertLessEqual(left, right * (1.0 + 1e-14))
            outer_next = _outer_fista_momentum(momentum)
            inner_next = _inner_fista_momentum(momentum)
            self.assertAlmostEqual(outer_next, inner_next, places=15)
            self.assertAlmostEqual(
                outer_next * (outer_next - 1.0),
                momentum * momentum,
                places=14,
            )
            previous_momentum = momentum
            previous_curvature = curvature
            momentum = outer_next

    def test_exact_fista_lyapunov_telescopes_with_curvature_growth(
        self,
    ) -> None:
        hessian = np.diag([0.4, 1.7, 2.0])
        curvatures = [2.0, 2.0, 2.5, 3.0, 3.0, 4.0]
        point = np.array([0.8, -0.4, 0.3])
        extrapolated = point.copy()
        momentum = 1.0
        previous_energy = float(point @ point)

        for curvature in curvatures:
            next_point = extrapolated - hessian @ extrapolated / curvature
            affine_error = next_point + (momentum - 1.0) * (
                next_point - point
            )
            objective = 0.5 * float(next_point @ hessian @ next_point)
            energy = (
                2.0 * momentum * momentum * objective / curvature
                + float(affine_error @ affine_error)
            )
            self.assertLessEqual(energy, previous_energy + 1e-14)

            next_momentum = _outer_fista_momentum(momentum)
            extrapolated = next_point + (
                (momentum - 1.0) / next_momentum
            ) * (next_point - point)
            point = next_point
            momentum = next_momentum
            previous_energy = energy

    def test_perspective_is_not_strongly_convex(self) -> None:
        center = np.array([0.2, 0.2, 0.2])
        direction = np.array([1.0, -1.0, 0.0])
        displacement = 0.03
        values = [
            perspective_value(
                center + sign * displacement * direction,
                2,
                tolerance=0.0,
            )
            for sign in (-1.0, 0.0, 1.0)
        ]
        self.assertAlmostEqual(values[0], values[1], places=15)
        self.assertAlmostEqual(values[1], values[2], places=15)

    def test_smooth_dual_gradient_and_global_curvature_bound(self) -> None:
        rng = np.random.default_rng(20260819)
        dimension = 8
        matrix = rng.normal(size=(3, dimension))
        lower = np.array([-0.5, -0.4, -0.3])
        upper = np.array([0.6, 0.7, 0.8])
        oracle = LinearConstraintProx(
            sparse.csr_matrix(matrix),
            lower,
            upper,
            3,
            use_budget_fast_path=False,
            dual_solver="fista",
            adaptive_restart=False,
        )
        operator = np.asarray(oracle.scaled_operator)
        argument = rng.normal(size=dimension)
        gamma = 0.9
        dual = np.array([0.17, -0.11, 0.23])

        def smooth_value(value: np.ndarray) -> float:
            shifted = argument - operator.T @ value
            point = oracle._pava(shifted, gamma)
            return _smooth_dual_value(shifted, point, gamma, 3)

        shifted = argument - operator.T @ dual
        point = oracle._pava(shifted, gamma)
        conjugate_argument = -operator.T @ dual
        fenchel_value = (
            float(conjugate_argument @ point)
            - 0.5 * float((point - argument) @ (point - argument))
            - gamma * perspective_value(point, 3, tolerance=0.0)
        )
        self.assertAlmostEqual(
            smooth_value(dual),
            fenchel_value + 0.5 * float(argument @ argument),
            places=12,
        )
        analytic = -operator @ point
        finite_difference = np.empty(dual.size)
        delta = 1e-6
        for index in range(dual.size):
            direction = np.zeros_like(dual)
            direction[index] = delta
            finite_difference[index] = (
                smooth_value(dual + direction)
                - smooth_value(dual - direction)
            ) / (2.0 * delta)

        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2e-7,
            atol=2e-8,
        )
        spectral_norm_squared = float(np.linalg.norm(operator, 2) ** 2)
        self.assertGreaterEqual(
            oracle.lipschitz,
            spectral_norm_squared * (1.0 - 1e-12),
        )

    def test_lbfgs_certificate_uses_rigorous_dual_curvature(self) -> None:
        matrix = sparse.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 0.0, 0.0],
                ]
            )
        )
        oracle = LinearConstraintProx(
            matrix,
            np.array([0.5, -0.4]),
            np.array([0.5, 0.4]),
            2,
            tolerance=1e-9,
            max_iterations=2_000,
            use_budget_fast_path=False,
            dual_solver="lbfgs",
            lbfgs_fallback=False,
        )
        result = oracle.solve(
            np.array([0.9, 0.2, -0.1, 0.4]),
            gamma=7.0,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.lipschitz, oracle.lipschitz)

    def test_corrected_dual_fista_certifies_local_curvature(self) -> None:
        matrix = sparse.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 0.0, 0.0],
                ]
            )
        )
        oracle = LinearConstraintProx(
            matrix,
            np.array([0.5, -0.4]),
            np.array([0.5, 0.4]),
            2,
            tolerance=1e-9,
            max_iterations=2_000,
            use_budget_fast_path=False,
            dual_solver="fista",
            adaptive_restart=False,
        )
        result = oracle.solve(
            np.array([0.9, 0.2, -0.1, 0.4]),
            gamma=7.0,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.lipschitz, oracle.lipschitz)
        self.assertGreater(result.line_search_backtracks, 0)
        self.assertLessEqual(
            result.scaled_constraint_violation,
            1e-9,
        )

        committed = oracle.snapshot()
        shifted = np.array([0.901, 0.199, -0.099, 0.399])
        first_trial = oracle.solve(shifted, gamma=7.0)
        oracle.restore(committed)
        repeated_trial = oracle.solve(shifted, gamma=7.0)
        np.testing.assert_allclose(
            repeated_trial.x,
            first_trial.x,
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertEqual(
            repeated_trial.lipschitz,
            first_trial.lipschitz,
        )
        self.assertEqual(
            repeated_trial.pava_calls,
            first_trial.pava_calls,
        )


if __name__ == "__main__":
    unittest.main()
