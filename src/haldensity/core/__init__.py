"""Shared HAL machinery used by more than one data type."""

from haldensity.core.basis import create_basis_functions, create_multivariate_basis
from haldensity.core.univariate import BaseEstimator, HALFit, WeightedHALMLEEstimator
from haldensity.core.solvers import solve_cvxpy_problem
from haldensity.core.variance import density_confidence_interval, estimate_covariance_beta
from haldensity.core.metrics import kl_divergence

__all__ = [
    "create_basis_functions",
    "create_multivariate_basis",
    "BaseEstimator",
    "HALFit",
    "WeightedHALMLEEstimator",
    "solve_cvxpy_problem",
    "density_confidence_interval",
    "estimate_covariance_beta",
    "kl_divergence",
]
