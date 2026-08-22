"""Recomputable safe dual certificates for the perspective relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class SafeDualCertificate:
    """A Fenchel weak-duality lower bound in original problem units."""

    lower_bound: float
    factor_multiplier: np.ndarray
    constraint_multiplier: np.ndarray
    floating_point_certified: bool = False

    @classmethod
    def from_result(
        cls,
        result: Mapping[str, Any],
    ) -> "SafeDualCertificate":
        lower_bound = result.get(
            "best_dual_lower_bound",
            result.get("dual_bound"),
        )
        factor = result.get("dual_bound_factor")
        constraint = result.get("dual_bound_constraint_original")
        if lower_bound is None or factor is None or constraint is None:
            raise ValueError("result does not contain a dual certificate")
        return cls(
            lower_bound=float(lower_bound),
            factor_multiplier=np.asarray(
                factor,
                dtype=float,
            ).reshape(-1).copy(),
            constraint_multiplier=np.asarray(
                constraint,
                dtype=float,
            ).reshape(-1).copy(),
            floating_point_certified=bool(
                result.get(
                    "dual_bound_floating_point_certified",
                    False,
                )
            ),
        )

    def recompute(self, problem: Any) -> Dict[str, Any]:
        """Independently reevaluate the lower bound."""
        from .safe_dual import evaluate_dual_bound

        return evaluate_dual_bound(
            problem,
            self.factor_multiplier,
            self.constraint_multiplier,
        )

    def verify(
        self,
        problem: Any,
        tolerance: float = 1e-10,
    ) -> bool:
        """Check that an independent evaluation matches the stored value."""
        recomputed = self.recompute(problem)["dual_bound"]
        if recomputed is None:
            return False
        scale = max(1.0, abs(self.lower_bound), abs(recomputed))
        return abs(recomputed - self.lower_bound) <= tolerance * scale


__all__ = ["SafeDualCertificate"]
