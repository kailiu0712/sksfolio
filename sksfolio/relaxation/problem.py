"""Problem data, portable storage, and common diagnostics.

The bundle format avoids JSON-encoding the large factor and constraint
matrices. Float arrays are little-endian Float64 binary files. The factor
matrix is stored in column-major order so Julia can reshape it without a
transpose. The linear constraint matrix is stored in zero-based CSC form,
which both SciPy and Julia's SparseArrays can reconstruct efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from scipy import sparse


SCHEMA_VERSION = 1
PathLike = Union[str, Path]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class MarkowitzInstance:
    """Factor-form continuous perspective Markowitz instance.

    Set ``anchor_must_be_feasible=False`` when ``feasible_anchor`` is only
    an initialization inherited from a parent branch-and-bound node.  Its
    long-only perspective domain is still validated, but newly introduced
    linear rows may cut it off.
    """

    factor_loadings: Any
    expected_returns: Any
    constraint_matrix: Any
    lower_bounds: Any
    upper_bounds: Any
    feasible_anchor: Any
    constraint_names: List[str]
    k: int
    perspective_weight: float
    return_reward: float
    anchor_must_be_feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    bundle_path: Optional[Path] = None

    @property
    def B(self) -> Any:
        return self.factor_loadings

    @property
    def mu(self) -> Any:
        return self.expected_returns

    @property
    def C(self) -> Any:
        return self.constraint_matrix

    @property
    def lower(self) -> Any:
        return self.lower_bounds

    @property
    def upper(self) -> Any:
        return self.upper_bounds

    @property
    def anchor(self) -> Any:
        return self.feasible_anchor

    @property
    def dimension(self) -> int:
        return int(self.factor_loadings.shape[0])

    @property
    def rank(self) -> int:
        return int(self.factor_loadings.shape[1])

    @property
    def rows(self) -> int:
        return int(self.constraint_matrix.shape[0])

    def validate(self, tolerance: float = 1e-8) -> None:
        if self.factor_loadings.ndim != 2:
            raise ValueError("factor_loadings must be a matrix")
        dimension, rank = self.factor_loadings.shape
        if dimension < 2 or rank < 1 or rank > dimension:
            raise ValueError("require 1 <= rank <= dimension")
        if np.shape(self.expected_returns) != (dimension,):
            raise ValueError("expected_returns has the wrong shape")
        if np.shape(self.feasible_anchor) != (dimension,):
            raise ValueError("feasible_anchor has the wrong shape")
        if self.constraint_matrix.shape[1] != dimension:
            raise ValueError("constraint_matrix has the wrong width")
        rows = int(self.constraint_matrix.shape[0])
        if np.shape(self.lower_bounds) != (rows,):
            raise ValueError("lower_bounds has the wrong shape")
        if np.shape(self.upper_bounds) != (rows,):
            raise ValueError("upper_bounds has the wrong shape")
        if len(self.constraint_names) != rows:
            raise ValueError("constraint_names has the wrong length")
        if len(set(str(name) for name in self.constraint_names)) != rows:
            raise ValueError("constraint_names must be unique")
        if not isinstance(self.anchor_must_be_feasible, (bool, np.bool_)):
            raise TypeError("anchor_must_be_feasible must be boolean")
        if not 1 <= int(self.k) <= dimension:
            raise ValueError("k must lie in {1, ..., dimension}")
        if (
            not math.isfinite(float(self.perspective_weight))
            or float(self.perspective_weight) <= 0.0
        ):
            raise ValueError("perspective_weight must be positive")
        if (
            not math.isfinite(float(self.return_reward))
            or float(self.return_reward) < 0.0
        ):
            raise ValueError("return_reward must be nonnegative")
        for name, values in (
            ("factor_loadings", self.factor_loadings),
            ("expected_returns", self.expected_returns),
            ("feasible_anchor", self.feasible_anchor),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains a nonfinite value")
        constraint_data = (
            self.constraint_matrix.data
            if sparse.issparse(self.constraint_matrix)
            else np.asarray(self.constraint_matrix)
        )
        if not np.all(np.isfinite(constraint_data)):
            raise ValueError("constraint_matrix contains a nonfinite value")
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("constraint bounds cannot contain NaN")
        if np.any(lower > upper):
            raise ValueError("a lower bound exceeds its upper bound")
        anchor = np.asarray(self.feasible_anchor)
        if np.min(anchor) < -tolerance or np.max(anchor) > 1.0 + tolerance:
            raise ValueError("feasible_anchor violates the long-only box")
        if np.sum(anchor) > float(self.k) + tolerance:
            raise ValueError("feasible_anchor violates the perspective budget")
        values = np.asarray(self.constraint_matrix @ anchor).reshape(-1)
        lower_error = np.where(
            np.isfinite(lower),
            np.maximum(lower - values, 0.0),
            0.0,
        )
        upper_error = np.where(
            np.isfinite(upper),
            np.maximum(values - upper, 0.0),
            0.0,
        )
        linear_error = (
            max(
                float(np.max(lower_error)),
                float(np.max(upper_error)),
            )
            if rows
            else 0.0
        )
        if self.anchor_must_be_feasible and linear_error > 10.0 * tolerance:
            raise ValueError("feasible_anchor violates a linear constraint")


def _write_f64(path: Path, values: Any, order: str = "C") -> None:
    array = np.asarray(values, dtype="<f8")
    np.asarray(array.ravel(order=order), dtype="<f8").tofile(path)


def _write_i64(path: Path, values: Any) -> None:
    np.asarray(values, dtype="<i8").tofile(path)


def save_instance_bundle(
    instance: MarkowitzInstance,
    path: PathLike,
    overwrite: bool = False,
) -> Path:
    """Write an instance without materializing a dense covariance matrix."""
    instance.validate()
    target = Path(path).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"bundle target is not a directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "metadata.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"bundle already exists: {target}; pass overwrite=True explicitly"
        )

    factor_path = target / "factor_loadings.f64"
    returns_path = target / "expected_returns.f64"
    lower_path = target / "lower_bounds.f64"
    upper_path = target / "upper_bounds.f64"
    anchor_path = target / "feasible_anchor.f64"
    _write_f64(factor_path, instance.factor_loadings, order="F")
    _write_f64(returns_path, instance.expected_returns)
    _write_f64(lower_path, instance.lower_bounds)
    _write_f64(upper_path, instance.upper_bounds)
    _write_f64(anchor_path, instance.feasible_anchor)

    matrix = sparse.csc_matrix(instance.constraint_matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    _write_i64(target / "constraint_colptr.i64", matrix.indptr)
    _write_i64(target / "constraint_rowval.i64", matrix.indices)
    _write_f64(target / "constraint_nzval.f64", matrix.data)

    metadata = {
        "schema": "markowitz-perspective-bundle",
        "schema_version": SCHEMA_VERSION,
        "dimension": instance.dimension,
        "rank": instance.rank,
        "constraint_rows": instance.rows,
        "constraint_nnz": int(matrix.nnz),
        "matrix_order": "column-major",
        "constraint_storage": "csc-zero-based",
        "float_dtype": "little-endian-float64",
        "index_dtype": "little-endian-int64",
        "k": int(instance.k),
        "perspective_weight": float(instance.perspective_weight),
        "return_reward": float(instance.return_reward),
        "anchor_must_be_feasible": bool(
            instance.anchor_must_be_feasible
        ),
        "constraint_names": list(instance.constraint_names),
        "files": {
            "factor_loadings": factor_path.name,
            "expected_returns": returns_path.name,
            "lower_bounds": lower_path.name,
            "upper_bounds": upper_path.name,
            "feasible_anchor": anchor_path.name,
            "constraint_colptr": "constraint_colptr.i64",
            "constraint_rowval": "constraint_rowval.i64",
            "constraint_nzval": "constraint_nzval.f64",
        },
        "generator": _json_safe(instance.metadata),
    }
    temporary_manifest = target / "metadata.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    temporary_manifest.replace(manifest_path)
    return target


def _memmap(
    path: Path,
    dtype: str,
    shape: Any,
    order: str = "C",
) -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"binary size mismatch for {path.name}: "
            f"expected {expected}, found {actual}"
        )
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        shape=shape,
        order=order,
    )


def load_instance_bundle(
    path: PathLike,
    validate: bool = True,
) -> MarkowitzInstance:
    """Memory-map a bundle and reconstruct its sparse constraint matrix."""
    root = Path(path).expanduser().resolve()
    with (root / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema") != "markowitz-perspective-bundle":
        raise ValueError("not a Markowitz perspective bundle")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema version")

    dimension = int(metadata["dimension"])
    rank = int(metadata["rank"])
    rows = int(metadata["constraint_rows"])
    nnz = int(metadata["constraint_nnz"])
    files = metadata["files"]
    factor_loadings = _memmap(
        root / files["factor_loadings"],
        "<f8",
        (dimension, rank),
        order="F",
    )
    expected_returns = _memmap(
        root / files["expected_returns"],
        "<f8",
        (dimension,),
    )
    lower = _memmap(root / files["lower_bounds"], "<f8", (rows,))
    upper = _memmap(root / files["upper_bounds"], "<f8", (rows,))
    anchor = _memmap(
        root / files["feasible_anchor"],
        "<f8",
        (dimension,),
    )
    colptr = _memmap(
        root / files["constraint_colptr"],
        "<i8",
        (dimension + 1,),
    )
    rowval = _memmap(
        root / files["constraint_rowval"],
        "<i8",
        (nnz,),
    )
    nzval = _memmap(
        root / files["constraint_nzval"],
        "<f8",
        (nnz,),
    )
    constraint_matrix = sparse.csc_matrix(
        (nzval, rowval, colptr),
        shape=(rows, dimension),
        copy=False,
    )
    instance = MarkowitzInstance(
        factor_loadings=factor_loadings,
        expected_returns=expected_returns,
        constraint_matrix=constraint_matrix,
        lower_bounds=lower,
        upper_bounds=upper,
        feasible_anchor=anchor,
        constraint_names=list(metadata["constraint_names"]),
        k=int(metadata["k"]),
        perspective_weight=float(metadata["perspective_weight"]),
        return_reward=float(metadata["return_reward"]),
        anchor_must_be_feasible=bool(
            metadata.get("anchor_must_be_feasible", True)
        ),
        metadata=dict(metadata.get("generator", {})),
        bundle_path=root,
    )
    if validate:
        instance.validate()
    return instance


def perspective_value(
    x: Any,
    k: int,
    tolerance: float = 1e-8,
) -> float:
    """Evaluate the long-only perspective envelope at x."""
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("x must be nonempty")
    if not 1 <= int(k) <= vector.size:
        raise ValueError("k must lie in {1, ..., dimension}")
    if np.any(~np.isfinite(vector)):
        return math.inf
    if (
        np.min(vector) < -tolerance
        or np.max(vector) > 1.0 + tolerance
        or np.sum(vector) > float(k) + tolerance
    ):
        return math.inf
    values = np.maximum(vector, 0.0)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    if positive.size <= k:
        return 0.5 * float(positive @ positive)
    # Only the largest k entries can be capped at the breakpoint.  Partial
    # selection therefore gives the same value without sorting all d entries.
    split = positive.size - k
    ordered = np.sort(np.partition(positive, split)[split:])[::-1]
    tail_sum = float(np.sum(positive))
    number_capped = 0
    while (
        number_capped < min(k, ordered.size)
        and (
            float(k - number_capped)
            * float(ordered[number_capped])
            >= tail_sum
        )
    ):
        tail_sum -= float(ordered[number_capped])
        number_capped += 1
    if number_capped >= k or tail_sum <= 0.0:
        raise ArithmeticError(
            "failed to locate the finite perspective breakpoint"
        )
    scale = float(k - number_capped) / tail_sum
    capped_value = float(
        ordered[:number_capped] @ ordered[:number_capped]
    )
    tail_value = tail_sum / scale
    return 0.5 * (capped_value + tail_value)


def evaluate_solution(
    instance: MarkowitzInstance,
    x: Any,
    domain_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Compute one solver-independent objective and feasibility report."""
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.shape != (instance.dimension,):
        raise ValueError("solution has the wrong dimension")
    factor_exposure = np.asarray(instance.B.T @ vector).reshape(-1)
    risk = 0.5 * float(factor_exposure @ factor_exposure)
    expected_return = float(instance.mu @ vector)
    perspective = perspective_value(
        vector,
        instance.k,
        tolerance=domain_tolerance,
    )
    objective = (
        risk
        - instance.return_reward * expected_return
        + instance.perspective_weight * perspective
    )
    values = np.asarray(instance.C @ vector).reshape(-1)
    lower = np.asarray(instance.lower)
    upper = np.asarray(instance.upper)
    lower_error = np.where(
        np.isfinite(lower),
        np.maximum(lower - values, 0.0),
        0.0,
    )
    upper_error = np.where(
        np.isfinite(upper),
        np.maximum(values - upper, 0.0),
        0.0,
    )
    linear_violation = (
        max(
            float(np.max(lower_error)),
            float(np.max(upper_error)),
        )
        if lower_error.size
        else 0.0
    )
    violations = {
        "linear_rows": linear_violation,
        "nonnegativity": max(0.0, -float(np.min(vector))),
        "upper_box": max(0.0, float(np.max(vector)) - 1.0),
        "perspective_budget": max(
            0.0,
            float(np.sum(vector)) - float(instance.k),
        ),
    }
    violations["maximum"] = max(violations.values())
    return {
        "objective": (
            objective if math.isfinite(objective) else None
        ),
        "risk": risk,
        "expected_return": expected_return,
        "perspective_value": (
            perspective if math.isfinite(perspective) else None
        ),
        "budget": float(np.sum(vector)),
        "maximum_weight": float(np.max(vector)),
        "positive_weights_above_1e-8": int(
            np.count_nonzero(vector > 1e-8)
        ),
        "violations": violations,
    }


__all__ = [
    "MarkowitzInstance",
    "evaluate_solution",
    "load_instance_bundle",
    "perspective_value",
    "save_instance_bundle",
]
