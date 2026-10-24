"""Deprecated import path. Use haldensity.full_uni; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.full_uni")

from haldensity.full_uni import (  # noqa: E402
    OptunaHyperparameterTuner,
)

__all__ = [
    "OptunaHyperparameterTuner",
]
