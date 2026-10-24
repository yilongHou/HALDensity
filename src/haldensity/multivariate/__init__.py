"""Deprecated import path. Use haldensity.full_multi; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.full_multi")

from haldensity.datasets import (  # noqa: E402
    BivariateTruncatedNormal,
)
from haldensity.full_multi import (  # noqa: E402
    MultvarHAL,
    build_exact_k0_normalizer_bank,
    build_midpoint_quadrature,
    build_sobol_normalizer_bank,
    default_sobol_budget,
    exact_k0_cell_budget,
)
from haldensity.core.basis import (  # noqa: E402
    coerce_multivariate_data,
    expected_section_counts,
    get_basis_index,
    summarize_basis_catalog,
    summarize_design_matrix,
    theoretical_basis_count,
)
from haldensity.core import (  # noqa: E402
    create_multivariate_basis,
)

__all__ = [
    "BivariateTruncatedNormal",
    "MultvarHAL",
    "build_exact_k0_normalizer_bank",
    "build_midpoint_quadrature",
    "build_sobol_normalizer_bank",
    "coerce_multivariate_data",
    "create_multivariate_basis",
    "default_sobol_budget",
    "exact_k0_cell_budget",
    "expected_section_counts",
    "get_basis_index",
    "summarize_basis_catalog",
    "summarize_design_matrix",
    "theoretical_basis_count",
]
