"""Accelerated minimization of the self-centered smoothed duality gap."""

from .solver import solve_scsdg

SCSDG_LBFGS_VARIANTS = (
    "off",
    "restart_safe",
    "restart_reference",
    "paper",
)
SCSDG_LINE_SEARCH_MODES = (
    "auto",
    "operator",
    "majorization",
)

__all__ = [
    "SCSDG_LBFGS_VARIANTS",
    "SCSDG_LINE_SEARCH_MODES",
    "solve_scsdg",
]
