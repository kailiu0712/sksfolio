"""Warm-started conditional-dual L-BFGS polishing for BnB nodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import fmin_l_bfgs_b

from ..relaxation.certificate import SafeDualCertificate
from ..screening.oracle import FenchelScreeningOracle
from .bounds import indices_from_mask
from .perspective import SelectorPerspectiveOracle


@dataclass(frozen=True)
class NodeDualResult:
    certificate: SafeDualCertificate
    oracle: FenchelScreeningOracle
    conditional_lower_bound: float
    iterations: int
    function_evaluations: int
    optimizer_status: str
    solve_seconds: float


class ConditionalDualPolisher:
    """Optimize a node's Fenchel certificate in factor-plus-row dimension.

    Every objective evaluation is a valid conditional lower bound. Therefore
    an iteration limit or nonsmooth top-k tie can weaken the result but cannot
    invalidate it.
    """

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.dimension = int(instance.dimension)
        self.rank = int(instance.rank)
        self.rows = int(instance.rows)
        self.B = instance.B
        self.C = sparse.csr_matrix(instance.C, dtype=float)
        self.lower = np.asarray(instance.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(instance.upper, dtype=float).reshape(-1)
        finite_lower = np.isfinite(self.lower)
        finite_upper = np.isfinite(self.upper)
        equality = (
            finite_lower
            & finite_upper
            & (self.lower == self.upper)
        )
        self.equal = np.flatnonzero(equality)
        self.upper_rows = np.flatnonzero(finite_upper & ~equality)
        self.lower_rows = np.flatnonzero(finite_lower & ~equality)
        self.coordinate_count = (
            self.rank
            + self.equal.size
            + self.upper_rows.size
            + self.lower_rows.size
        )
        self.bounds = (
            [(None, None)] * (self.rank + self.equal.size)
            + [(0.0, None)]
            * (self.upper_rows.size + self.lower_rows.size)
        )

    def _coordinates(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        pieces = [np.asarray(p, dtype=float).reshape(-1)]
        if self.equal.size:
            pieces.append(q[self.equal])
        if self.upper_rows.size:
            pieces.append(np.maximum(q[self.upper_rows], 0.0))
        if self.lower_rows.size:
            pieces.append(np.maximum(-q[self.lower_rows], 0.0))
        return np.concatenate(pieces)

    def _multipliers(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cursor = 0
        p = coordinates[: self.rank]
        cursor += self.rank
        q = np.zeros(self.rows)
        if self.equal.size:
            count = self.equal.size
            q[self.equal] = coordinates[cursor : cursor + count]
            cursor += count
        if self.upper_rows.size:
            count = self.upper_rows.size
            q[self.upper_rows] += coordinates[cursor : cursor + count]
            cursor += count
        if self.lower_rows.size:
            count = self.lower_rows.size
            q[self.lower_rows] -= coordinates[cursor : cursor + count]
        return p, q

    def solve(
        self,
        one_mask: int,
        zero_mask: int,
        initial_oracle: FenchelScreeningOracle,
        *,
        maximum_iterations: int = 20,
        memory: int = 10,
        maximum_line_search: int = 20,
    ) -> NodeDualResult:
        import time

        start = time.perf_counter()
        perspective = SelectorPerspectiveOracle(
            self.dimension,
            int(self.instance.k),
            forced_one=indices_from_mask(one_mask),
            forced_zero=indices_from_mask(zero_mask),
        )
        initial = self._coordinates(
            np.asarray(initial_oracle.factor_multiplier, dtype=float),
            np.asarray(initial_oracle.constraint_multiplier, dtype=float),
        )
        best_value = math.inf
        best_coordinates = initial.copy()
        evaluations = 0

        def objective_gradient(values: np.ndarray):
            nonlocal best_value, best_coordinates, evaluations
            p, q = self._multipliers(values)
            priced = (
                float(self.instance.return_reward)
                * np.asarray(self.instance.mu, dtype=float)
                - np.asarray(self.B @ p, dtype=float).reshape(-1)
            )
            if self.rows:
                priced -= np.asarray(self.C.T @ q, dtype=float).reshape(-1)
            selected = perspective.selected_free_indices(
                priced,
                float(self.instance.perspective_weight),
            )
            x = np.zeros(self.dimension)
            omega = float(self.instance.perspective_weight)
            if perspective.forced_one.size:
                x[perspective.forced_one] = np.clip(
                    priced[perspective.forced_one] / omega,
                    0.0,
                    1.0,
                )
            if selected.size:
                x[selected] = np.clip(
                    priced[selected] / omega,
                    0.0,
                    1.0,
                )
            row_values = (
                np.asarray(self.C @ x, dtype=float).reshape(-1)
                if self.rows
                else np.empty(0)
            )
            value = 0.5 * float(p @ p) + perspective.conjugate(
                priced,
                omega,
            )
            gradient_parts = [
                p - np.asarray(self.B.T @ x, dtype=float).reshape(-1)
            ]
            cursor = self.rank
            if self.equal.size:
                coordinates = values[cursor : cursor + self.equal.size]
                value += float(self.lower[self.equal] @ coordinates)
                gradient_parts.append(
                    self.lower[self.equal] - row_values[self.equal]
                )
                cursor += self.equal.size
            if self.upper_rows.size:
                coordinates = values[
                    cursor : cursor + self.upper_rows.size
                ]
                value += float(self.upper[self.upper_rows] @ coordinates)
                gradient_parts.append(
                    self.upper[self.upper_rows]
                    - row_values[self.upper_rows]
                )
                cursor += self.upper_rows.size
            if self.lower_rows.size:
                coordinates = values[
                    cursor : cursor + self.lower_rows.size
                ]
                value -= float(self.lower[self.lower_rows] @ coordinates)
                gradient_parts.append(
                    row_values[self.lower_rows]
                    - self.lower[self.lower_rows]
                )
            gradient = np.concatenate(gradient_parts)
            evaluations += 1
            if math.isfinite(value) and value < best_value:
                best_value = float(value)
                best_coordinates = np.asarray(values, dtype=float).copy()
            return float(value), gradient

        final, _, information = fmin_l_bfgs_b(
            objective_gradient,
            initial,
            bounds=self.bounds,
            m=max(1, int(memory)),
            factr=0.0,
            pgtol=1e-10,
            maxiter=max(1, int(maximum_iterations)),
            maxfun=max(20, 20 * int(maximum_iterations)),
            maxls=max(1, int(maximum_line_search)),
        )
        if not math.isfinite(best_value):
            objective_gradient(np.asarray(final, dtype=float))
        p, q = self._multipliers(best_coordinates)
        certificate = SafeDualCertificate(
            lower_bound=-float(best_value),
            factor_multiplier=np.asarray(p, dtype=float).copy(),
            constraint_multiplier=np.asarray(q, dtype=float).copy(),
        )
        oracle = FenchelScreeningOracle(self.instance, certificate)
        conditional = oracle.pattern_lower_bound(
            indices_from_mask(one_mask),
            indices_from_mask(zero_mask),
        )
        return NodeDualResult(
            certificate=certificate,
            oracle=oracle,
            conditional_lower_bound=float(conditional),
            iterations=int(information.get("nit", 0)),
            function_evaluations=evaluations,
            optimizer_status=str(information.get("task", "")),
            solve_seconds=time.perf_counter() - start,
        )


__all__ = ["ConditionalDualPolisher", "NodeDualResult"]
