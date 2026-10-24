"""Delta-method variance of the full-data HAL density and pointwise confidence intervals.

Basis cleaning (on by default): the SE is computed on the basis of an accurate, certified solution
(:mod:`haldensity.core.univariate.support`) instead of a loosely solved fit's pruned basis.
``support_refit="auto"`` (default): a ``CVXPYEstimator`` fit with ``solver_tolerance="tight"``
(its default) already is that solution, and its own basis is used, so the point estimate and the
SE share one solve; for other results that carry the fit's L1 bound (e.g. ``solver_tolerance=
"default"``) the problem is re-solved tightly once per fit, cached and reused in
:func:`density_confidence_interval`, and the point estimate stays the original fit.
``support_refit=None`` uses the original basis; a :class:`SupportRefit` can also be passed.
"""

from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm
import warnings

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.univariate.support import SupportRefit, refit_support

SupportRefitArg = Union[SupportRefit, None, str]
_REFIT_CACHE: "OrderedDict[tuple, SupportRefit]" = OrderedDict()
_REFIT_CACHE_SIZE = 16


class BasisCovariance(np.ndarray):
    """Cov(beta-hat) that remembers its basis: ``support_refit`` (None = the original basis).

    Behaves as a plain array; :func:`density_confidence_interval` reads ``support_refit`` so the SE
    uses the same basis as the covariance.
    """

    support_refit: Optional[SupportRefit] = None

    def __array_finalize__(self, obj):
        self.support_refit = getattr(obj, "support_refit", None)


def _refit_key(estimation_results: dict, basis_order: int) -> Optional[tuple]:
    if estimation_results.get('norm_constraint') is None:
        return None
    return (
        np.asarray(estimation_results['theta_hat'], dtype=float).tobytes(),
        np.asarray(estimation_results['data_points'], dtype=float).tobytes(),
        int(basis_order),
        float(estimation_results['norm_constraint']),
        int(estimation_results.get('n_grid_points', 200)),
        str(estimation_results.get('objective_normalizer', 'midpoint')),
        estimation_results.get('max_knots'),
    )


def _auto_refit(data: pd.DataFrame, estimation_results: dict, basis_order: int) -> Optional[SupportRefit]:
    """The cleaned basis of this fit, or None if the fit has no L1 bound.

    A fit made with ``solver_tolerance="tight"`` carries its certified solution (``support_refit``),
    which is used as it is; otherwise the problem is re-solved tightly once and cached.
    """
    carried = estimation_results.get('support_refit')
    if isinstance(carried, SupportRefit):
        return carried
    key = _refit_key(estimation_results, basis_order)
    if key is None:
        return None
    if key not in _REFIT_CACHE:
        refit = refit_support(data, int(basis_order), float(estimation_results['norm_constraint']),
                              n_grid_points=int(estimation_results.get('n_grid_points', 200)),
                              objective_normalizer=str(estimation_results.get('objective_normalizer', 'midpoint')),
                              max_knots=estimation_results.get('max_knots'))
        knots = np.unique(np.asarray(estimation_results['data_points'], dtype=float))
        if not np.array_equal(refit.knots, knots):
            raise ValueError("data do not match the fit's knots; pass the data the estimator was fitted on")
        _REFIT_CACHE[key] = refit
        while len(_REFIT_CACHE) > _REFIT_CACHE_SIZE:
            _REFIT_CACHE.popitem(last=False)
    _REFIT_CACHE.move_to_end(key)
    return _REFIT_CACHE[key]


NO_SE_WARNING = (
    "Basis cleaning failed: the final fit did not pass the optimality check, so no SE or interval is "
    "given (NaN); the point estimate is kept. For uncertainty use a bootstrap, or refit with another "
    "solver (e.g. MOSEK), with more solver time, or at another bound."
)


def _cleaning_failed(support_refit: Optional[SupportRefit]) -> bool:
    return support_refit is not None and not support_refit.certified


def _selection_size(theta_full: np.ndarray, basis_order: int) -> int:
    return int(basis_order + 1 + np.count_nonzero(np.asarray(theta_full)[basis_order + 1:]))


def _theta_for_se(estimation_results: dict, support_refit: Optional[SupportRefit]) -> np.ndarray:
    theta = np.array(estimation_results['theta_hat'])
    if support_refit is None:
        return theta
    knots = np.unique(np.array(estimation_results['data_points']))
    if support_refit.theta.shape != theta.shape or not np.array_equal(support_refit.knots, knots):
        raise ValueError("support_refit does not belong to this fit (different knots or basis)")
    return np.asarray(support_refit.theta, dtype=float)


def estimate_covariance_beta(
    data: pd.DataFrame,
    estimation_results: dict,
    basis_order: int,
    ridge_param: float = 1e-6,
    support_refit: SupportRefitArg = "auto",
) -> np.ndarray:
    """Estimate Cov(β̂) via observed Fisher information with ridge regularization.

    ``support_refit="auto"`` (default) uses the cleaned basis when the results carry the fit's
    L1 bound; ``None`` uses the original basis; a :class:`SupportRefit` is used as given. If basis
    cleaning failed, no SE is given: the covariance is NaN, with a warning.
    """
    if isinstance(support_refit, str):
        if support_refit != "auto":
            raise ValueError("support_refit must be 'auto', None or a SupportRefit")
        support_refit = _auto_refit(data, estimation_results, basis_order)
    theta_hat_full = _theta_for_se(estimation_results, support_refit)
    if _cleaning_failed(support_refit):
        # decision 5 (Yilong, 2026-10-24): point estimate yes, SE no
        warnings.warn(NO_SE_WARNING, UserWarning, stacklevel=2)
        size = _selection_size(theta_hat_full, basis_order)
        covariance = np.full((size, size), np.nan).view(BasisCovariance)
        covariance.support_refit = support_refit
        return covariance
    original_data_points = np.array(estimation_results['data_points'])
    grid_points_hal_original = np.unique(original_data_points)

    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        theta_hat_full[basis_order + 1:].nonzero()[0] + basis_order + 1,
    ])
    theta_hat = theta_hat_full[theta_selected_ind]

    n_samples = len(data)

    basis_matrix, _ = create_basis_functions(
        data,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    basis_matrix_selected = basis_matrix[:, theta_selected_ind]

    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    integration_grid = np.linspace(0, 1, n_grid_points)
    integration_df = pd.DataFrame({'W1': integration_grid})

    phi_integration, _ = create_basis_functions(
        integration_df,
        grid_points_hal_original,
        order=basis_order,
        include_intercept=True,
    )
    phi_integration_selected = phi_integration[:, theta_selected_ind]

    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)

    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = trapezoid(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant

    expected_phi = trapezoid(
        phi_integration_selected * density_integration[:, np.newaxis],
        dx=dx,
        axis=0,
    )

    scores = basis_matrix_selected - expected_phi[np.newaxis, :]
    observed_information = (scores.T @ scores) / n_samples

    ridge_matrix = ridge_param * np.eye(observed_information.shape[0])
    regularized_information = observed_information + ridge_matrix

    try:
        covariance_matrix = np.linalg.inv(regularized_information) / n_samples
    except np.linalg.LinAlgError:
        warnings.warn("Information matrix is singular. Using pseudo-inverse.")
        covariance_matrix = np.linalg.pinv(regularized_information) / n_samples

    covariance = covariance_matrix.view(BasisCovariance)
    covariance.support_refit = support_refit
    return covariance


def _evaluation_pieces(
    x_values: np.ndarray,
    theta_hat_full: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    n_grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(selected theta, basis at x on the selection, density at x, E[phi]) for one coefficient vector."""
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        theta_hat_full[basis_order + 1:].nonzero()[0] + basis_order + 1,
    ])
    theta_hat = theta_hat_full[theta_selected_ind]

    phi_eval, _ = create_basis_functions(
        pd.DataFrame({'W1': x_values}),
        knots,
        order=basis_order,
        include_intercept=True,
    )
    phi_eval_selected = phi_eval[:, theta_selected_ind]

    integration_grid = np.linspace(0, 1, n_grid_points)
    phi_integration, _ = create_basis_functions(
        pd.DataFrame({'W1': integration_grid}),
        knots,
        order=basis_order,
        include_intercept=True,
    )
    phi_integration_selected = phi_integration[:, theta_selected_ind]

    log_density_integration = phi_integration_selected @ theta_hat
    max_log_density = np.max(log_density_integration)
    density_unnorm = np.exp(log_density_integration - max_log_density)

    dx = integration_grid[1] - integration_grid[0]
    normalizing_constant = trapezoid(density_unnorm, dx=dx)
    density_integration = density_unnorm / normalizing_constant

    log_density_eval = phi_eval_selected @ theta_hat
    log_Z = np.log(normalizing_constant) + max_log_density
    density_eval = np.exp(log_density_eval - log_Z)

    expected_phi = trapezoid(
        phi_integration_selected * density_integration[:, np.newaxis],
        dx=dx,
        axis=0,
    )
    return theta_hat, phi_eval_selected, density_eval, expected_phi


def density_confidence_interval(
    x_values: np.ndarray,
    estimation_results: dict,
    cov_beta: np.ndarray,
    basis_order: int,
    alpha: float = 0.05,
    support_refit: SupportRefitArg = "auto",
) -> pd.DataFrame:
    """Compute delta-method CIs for density at x_values.

    The SE uses the same basis as ``cov_beta``: with ``support_refit="auto"`` (default) the
    cleaned basis computed by :func:`estimate_covariance_beta` for this fit if ``cov_beta`` has its
    size, else the original basis; ``None`` forces the original basis; a :class:`SupportRefit`
    is used as given. The ``density`` column is always the original fit. ``result.attrs`` records
    the basis used and whether it was certified. If basis cleaning failed (the final fit did not
    pass the optimality check), ``se``, ``lower`` and ``upper`` are NaN, with a warning (decision 5,
    Yilong 2026-10-24); ``support_refit=None`` still gives the original-basis SE.
    """
    original_theta = np.array(estimation_results['theta_hat'])
    if isinstance(support_refit, str):
        if support_refit != "auto":
            raise ValueError("support_refit must be 'auto', None or a SupportRefit")
        if isinstance(cov_beta, BasisCovariance):
            support_refit = cov_beta.support_refit       # the basis the covariance was computed on
        else:
            carried = estimation_results.get('support_refit')
            key = _refit_key(estimation_results, basis_order)
            cached = carried if isinstance(carried, SupportRefit) else (
                _REFIT_CACHE.get(key) if key is not None else None)
            size = cov_beta.shape[0]
            if cached is not None and _selection_size(cached.theta, basis_order) == size:
                support_refit = cached
            else:
                support_refit = None
    knots = np.unique(np.array(estimation_results['data_points']))
    n_grid_points = len(estimation_results.get('grid_points', np.linspace(0, 1, 200)))
    x_values = np.clip(x_values, 0, 1)
    if _cleaning_failed(support_refit):
        warnings.warn(NO_SE_WARNING, UserWarning, stacklevel=2)
        density_eval = _evaluation_pieces(x_values, original_theta, knots, basis_order, n_grid_points)[2]
        nan = np.full(len(x_values), np.nan)
        result_df = pd.DataFrame({'x': x_values, 'density': density_eval, 'se': nan, 'lower': nan, 'upper': nan})
        result_df.attrs.update(basis='cleaned', basis_certified=False, se='none: basis cleaning failed')
        return result_df

    theta_hat, phi_eval_selected, density_se_basis, expected_phi = _evaluation_pieces(
        x_values, _theta_for_se(estimation_results, support_refit), knots, basis_order, n_grid_points
    )
    if cov_beta.shape != (theta_hat.size, theta_hat.size):
        raise ValueError(
            f"cov_beta is {cov_beta.shape} but the basis has {theta_hat.size} functions; compute it with "
            "the same support_refit"
        )
    if support_refit is None:
        density_eval = density_se_basis
    else:
        # the point estimate stays the original fit; only the SE uses the cleaned basis
        density_eval = _evaluation_pieces(x_values, original_theta, knots, basis_order, n_grid_points)[2]

    n_eval = len(x_values)
    standard_errors = np.zeros(n_eval)
    for i in range(n_eval):
        grad_density = density_se_basis[i] * (phi_eval_selected[i] - expected_phi)
        variance_f = grad_density.T @ cov_beta @ grad_density
        if variance_f < 0:
            warnings.warn(f"Negative variance detected at x={x_values[i]:.3f}, setting to 0")
            variance_f = 0
        standard_errors[i] = np.sqrt(variance_f)

    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * standard_errors
    lower_bounds = np.maximum(density_eval - margin_of_error, 0)
    upper_bounds = density_eval + margin_of_error

    result_df = pd.DataFrame({
        'x': x_values,
        'density': density_eval,
        'se': standard_errors,
        'lower': lower_bounds,
        'upper': upper_bounds,
    })
    result_df.attrs['basis'] = 'original' if support_refit is None else 'cleaned'
    if support_refit is not None:
        result_df.attrs['basis_certified'] = bool(support_refit.certified)
        result_df.attrs['n_basis_original'] = int(original_theta[basis_order + 1:].nonzero()[0].size)
        result_df.attrs['n_basis_cleaned'] = int(support_refit.n_selected)

    return result_df
