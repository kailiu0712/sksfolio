"""sksfolio: Sparse k-support portfolio optimization."""

from .relaxation import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_FISTA_PROX_ORACLE,
    DEFAULT_FISTA_RESTART,
    DEFAULT_PAVA,
    DEFAULT_PDHG_VARIANT,
    MarkowitzInstance,
    PerspectiveRelaxation,
    RelaxationResult,
    SafeDualCertificate,
    available_backends,
    registered_backends,
    solve_relaxation,
)
from ._version import __version__

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_FISTA_PROX_ORACLE",
    "DEFAULT_FISTA_RESTART",
    "DEFAULT_PAVA",
    "DEFAULT_PDHG_VARIANT",
    "MarkowitzInstance",
    "PerspectiveRelaxation",
    "RelaxationResult",
    "SafeDualCertificate",
    "available_backends",
    "registered_backends",
    "solve_relaxation",
    "__version__",
]
