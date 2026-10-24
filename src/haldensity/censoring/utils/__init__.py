"""Deprecated import path. Use haldensity.core; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.core")

from haldensity.core import (  # noqa: E402
    kl_divergence,
)

__all__ = [
    "kl_divergence",
]
