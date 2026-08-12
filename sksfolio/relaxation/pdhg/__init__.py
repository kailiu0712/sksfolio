"""PDHG variants, PAVA proximal oracles, and safe dual evaluation."""

from .pava import (
    PAVA_METHODS,
    check_pava_oracles,
    prox,
    prox_full_sort,
    prox_partial_sort,
)
from .safe_dual import SafeDualCertificate, evaluate_dual_bound
from .solver import PDHG_VARIANTS, solve_pdhg

__all__ = [
    "PAVA_METHODS",
    "PDHG_VARIANTS",
    "SafeDualCertificate",
    "check_pava_oracles",
    "evaluate_dual_bound",
    "prox",
    "prox_full_sort",
    "prox_partial_sort",
    "solve_pdhg",
]
