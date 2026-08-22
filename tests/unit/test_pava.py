"""Tests for the two finite PAVA implementations."""

from __future__ import annotations

import unittest

import numpy as np

from sksfolio.relaxation.pava import (
    check_pava_oracles,
    native_partial_sort_available,
    prox,
    prox_full_sort,
    prox_native_partial_sort,
    prox_partial_sort,
    prox_python_partial_sort,
)


class PAVATests(unittest.TestCase):
    def test_random_full_and_partial_sort_agree(self) -> None:
        report = check_pava_oracles(seed=91, repetitions=15)
        self.assertEqual(report["cases"], 255.0)
        self.assertLess(report["maximum_absolute_error"], 2e-12)

    def test_ties_are_permutation_equivariant(self) -> None:
        values = np.array(
            [3.0, 3.0, 3.0, 1.5, 1.5, 0.0, -2.0, 3.0]
        )
        permutation = np.array([3, 0, 6, 2, 7, 5, 1, 4])
        expected = prox_partial_sort(values, 0.7, 3)
        permuted = prox_partial_sort(values[permutation], 0.7, 3)
        recovered = np.empty_like(permuted)
        recovered[permutation] = permuted
        np.testing.assert_allclose(recovered, expected, atol=2e-14)
        np.testing.assert_allclose(
            prox_full_sort(values, 0.7, 3),
            expected,
            atol=2e-14,
        )

    @unittest.skipUnless(
        native_partial_sort_available(),
        "native PAVA extension is not built",
    )
    def test_native_partial_sort_matches_python_reference(self) -> None:
        rng = np.random.default_rng(8201)
        for dimension in (2, 5, 31, 499, 1000):
            selected_values = {
                1,
                max(1, dimension // 7),
                max(1, dimension - 1),
                dimension,
            }
            for selected in selected_values:
                for gamma in (1e-4, 0.2, 1.0, 7.0):
                    values = rng.normal(size=dimension)
                    expected = prox_python_partial_sort(
                        values,
                        gamma,
                        selected,
                    )
                    result = prox_native_partial_sort(
                        values,
                        gamma,
                        selected,
                    )
                    np.testing.assert_allclose(
                        result,
                        expected,
                        rtol=2e-13,
                        atol=2e-13,
                    )

    def test_long_only_and_endpoint_cases(self) -> None:
        values = np.array([-4.0, -1.0, 0.0, 1.0, 5.0])
        result = prox(values, 2.0, values.size, "partial_sort")
        np.testing.assert_allclose(
            result,
            np.array([0.0, 0.0, 0.0, 1.0 / 3.0, 1.0]),
        )
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_invalid_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_sort"):
            prox(np.ones(4), 1.0, 2, "scalar_newton")

    def test_k_must_be_an_integer_and_not_boolean(self) -> None:
        for selected in (True, 1.5, np.nan):
            with self.subTest(k=selected):
                with self.assertRaisesRegex(ValueError, "integer"):
                    prox_partial_sort(np.ones(4), 1.0, selected)

    def test_implementations_are_split_across_public_modules(self) -> None:
        self.assertEqual(
            prox_full_sort.__module__,
            "sksfolio.relaxation.pava.full_sort",
        )
        self.assertEqual(
            prox_partial_sort.__module__,
            "sksfolio.relaxation.pava.partial_sort",
        )

    def test_extreme_float64_scales_require_rescaling(self) -> None:
        values = np.array([1e200, 2e200, 3e200])
        for oracle in (prox_full_sort, prox_partial_sort):
            with self.subTest(oracle=oracle.__module__):
                with self.assertRaisesRegex(ValueError, "rescale"):
                    oracle(values, 1e200, 1)


if __name__ == "__main__":
    unittest.main()
