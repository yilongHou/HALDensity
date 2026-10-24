"""Deprecated import path. Use haldensity.core.variance; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.core.variance")

from haldensity.core import (  # noqa: E402
    create_basis_functions,
    density_confidence_interval,
    estimate_covariance_beta,
)

__all__ = [
    "create_basis_functions",
    "density_confidence_interval",
    "estimate_covariance_beta",
]
