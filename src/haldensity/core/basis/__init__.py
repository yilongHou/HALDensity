"""HAL basis construction: univariate (any order k) and multivariate tensor products (general d)."""

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.basis.multivariate import (
    coerce_multivariate_data,
    create_multivariate_basis,
    expected_section_counts,
    extract_term_type_metrics,
    get_basis_index,
    summarize_basis_catalog,
    summarize_design_matrix,
    theoretical_basis_count,
)

__all__ = [
    "create_basis_functions",
    "coerce_multivariate_data",
    "create_multivariate_basis",
    "expected_section_counts",
    "extract_term_type_metrics",
    "get_basis_index",
    "summarize_basis_catalog",
    "summarize_design_matrix",
    "theoretical_basis_count",
]
