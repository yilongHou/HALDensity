"""Deprecated import path. Use haldensity.datasets; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.datasets")

from haldensity.datasets import (  # noqa: E402
    BivariateTruncatedNormal,
    Sinusoidal,
    StepFunction,
    TruncatedGMM,
    TruncatedNormal,
)

__all__ = [
    "BivariateTruncatedNormal",
    "Sinusoidal",
    "StepFunction",
    "TruncatedGMM",
    "TruncatedNormal",
]
