"""Deprecated import path. Use haldensity.core.basis and haldensity.datasets; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.core.basis and haldensity.datasets")

from haldensity.datasets import (  # noqa: E402
    BivariateTruncatedNormal,
    Sinusoidal,
    StepFunction,
    TruncatedGMM,
    TruncatedNormal,
)
from haldensity.core import (  # noqa: E402
    create_basis_functions,
    create_multivariate_basis,
)

__all__ = [
    "BivariateTruncatedNormal",
    "Sinusoidal",
    "StepFunction",
    "TruncatedGMM",
    "TruncatedNormal",
    "create_basis_functions",
    "create_multivariate_basis",
]
