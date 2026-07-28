"""Result objects returned by the public relaxation API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np

from .certificate import SafeDualCertificate


@dataclass
class RelaxationResult(Mapping[str, Any]):
    """Thin typed view over the complete backend result mapping."""

    raw: Dict[str, Any]

    @property
    def weights(self) -> Optional[np.ndarray]:
        value = self.raw.get("x")
        if value is None:
            return None
        return np.asarray(value, dtype=float).reshape(-1)

    @property
    def status(self) -> str:
        return str(self.raw.get("status", "unknown"))

    @property
    def objective(self) -> Optional[float]:
        diagnostics = self.raw.get("diagnostics", {})
        value = diagnostics.get(
            "objective",
            self.raw.get("external_objective"),
        )
        if value is None:
            return None
        bound = float(value)
        return bound if math.isfinite(bound) else None

    @property
    def safe_dual_bound(self) -> Optional[float]:
        value = self.raw.get(
            "best_dual_lower_bound",
            self.raw.get("dual_bound"),
        )
        if value is None:
            return None
        bound = float(value)
        return bound if math.isfinite(bound) else None

    @property
    def solver_objective_bound(self) -> Optional[float]:
        """Return a solver-reported numerical bound, if available.

        Unlike :attr:`safe_dual_bound`, this value has no saved,
        independently recomputable Fenchel certificate.
        """
        value = self.raw.get("solver_objective_bound")
        if value is None:
            details = self.raw.get("solver_details", {})
            if isinstance(details, Mapping):
                value = details.get("objective_bound")
        if value is None:
            return None
        bound = float(value)
        return bound if math.isfinite(bound) else None

    @property
    def dual_certificate(self) -> Optional[SafeDualCertificate]:
        try:
            return SafeDualCertificate.from_result(self.raw)
        except ValueError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


__all__ = ["RelaxationResult"]
