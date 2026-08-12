"""Reproducible benchmark profiles and instance generation."""

from .bertsimas_cory_wright import (
    BCWCase,
    GAMMA_SCALES,
    HISTORICAL_UNIVERSES,
    K_VALUES,
    OR_LIBRARY_UNIVERSES,
    generate_synthetic_case,
    historical_cases,
    or_library_cases,
)

__all__ = [
    "BCWCase",
    "GAMMA_SCALES",
    "HISTORICAL_UNIVERSES",
    "K_VALUES",
    "OR_LIBRARY_UNIVERSES",
    "generate_synthetic_case",
    "historical_cases",
    "or_library_cases",
]
