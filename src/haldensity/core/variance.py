"""Delta-method variance of the full-data HAL density and pointwise confidence intervals.

Basis cleaning (on by default): the SE is computed on the cleaned basis of an accurate solution
(:mod:`haldensity.core.univariate.support`) instead of a loosely solved fit's pruned basis.
``support_refit="auto"`` (default): a ``CVXPYEstimator`` fit made with its tight final solve (the
default) already is that solution, and its own basis is used, so the point estimate and the SE share
one solve; for other results that carry the fit's L1 bound (e.g. ``_solver_tolerance="default"``) the
problem is re-solved tightly once per fit, cached and reused in :func:`density_confidence_interval`,
and the point estimate stays the original fit. ``support_refit=None`` uses the original basis; a
:class:`SupportRefit` can also be passed.

The band's centre (the density), its normalizing constant and E[phi] in the scores use the rule of the
accurate normalizer (Gauss-Legendre with breakpoints at the selected knots, exact for k = 0;
:func:`haldensity.core.univariate.normalizer.quadrature_rule`), so the centre is the density that
``get_density_at_points`` reports. Before 0.2.0 they used a 200-point trapezoid on [0, 1].

The information is inverted with the scaled floor (the default since 0.2.0; Yilong, 2026-10-24, after SE
round 2): a ridge of 1e-5 times each column's own score variance, which has no units, so the SE does not
change when the data's units do. The 0.1.0 rule, an absolute ridge 1e-6 that depends on the units, stays
available internally (``_inverse="ridge"``) as the reference the paper's experiments compute alongside.
"""

from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.options import REMOVED_IN, take_deprecated
from haldensity.core.univariate.normalizer import SUPPORT, log_unnormalized_density, quadrature_rule
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

    A fit made with the tight final solve (the default) carries its certified solution (``support_refit``),
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


INVERSES = ("scaled_floor", "ridge", "cutoff")
# the default rule's constant: a ridge of SCALED_FLOOR times each column's own score variance
SCALED_FLOOR = 1e-5
# the 0.1.0 default, an absolute ridge on the information; kept internally as a reference
REFERENCE_RIDGE = 1e-6
# a column whose centred score has an RMS below this fraction of the column's own RMS is constant (the
# intercept; relative to the column itself, so the test does not depend on the data's units)
CONSTANT_COLUMN_TOL = 1e-12
INFORMATIONS = ("empirical", "model")
INTERVAL_SCALES = ("density", "log")


def _covariance_from_information(information: np.ndarray, root: Optional[np.ndarray], n_samples: int,
                                 inverse: str, ridge_param: float, relative_cutoff: float,
                                 floor: float = SCALED_FLOOR, constant: Optional[np.ndarray] = None) -> np.ndarray:
    """Cov(beta-hat) = information^{-1} / n.

    "scaled_floor" (the package default): the information scaled to unit diagonal, R = D^-1 I D^-1 with d_j
    the RMS of the centred score of column j, and Cov = D^-1 (R + floor * I)^-1 D^-1 / n, i.e. a ridge of
    floor * d_j^2 on coordinate j: no units, so the SE does not change when the data's units do. The
    ``constant`` columns (the intercept, whose centred score is zero, and any other column constant on the
    data; :func:`_constant_columns`) are dropped: their rows and columns of Cov are zero.
    "ridge": the inverse of information + ridge_param * I (an absolute ridge, the 0.1.0 default).
    "cutoff": the pseudo-inverse from the singular values of ``root`` (information = root' root), dropping
    the directions whose eigenvalue is below ``relative_cutoff`` times the largest (the intercept's
    direction is exactly zero), so no constant depends on the scale of the basis functions.
    """
    if inverse == "scaled_floor":
        scale = np.sqrt(np.clip(np.diag(information), 0.0, None))
        keep = scale > 0 if constant is None else ~np.asarray(constant, dtype=bool)
        outer = np.outer(scale[keep], scale[keep])
        covariance = np.zeros_like(information)
        covariance[np.ix_(keep, keep)] = (np.linalg.inv(information[np.ix_(keep, keep)] / outer
                                                        + floor * np.eye(int(keep.sum()))) / outer / n_samples)
        return covariance
    if inverse == "ridge":
        regularized_information = information + ridge_param * np.eye(information.shape[0])
        try:
            return np.linalg.inv(regularized_information) / n_samples
        except np.linalg.LinAlgError:
            warnings.warn("Information matrix is singular. Using pseudo-inverse.")
            return np.linalg.pinv(regularized_information) / n_samples
    _, singular_values, vt = np.linalg.svd(root, full_matrices=False)
    eigenvalues = singular_values ** 2
    keep = eigenvalues > relative_cutoff * eigenvalues.max(initial=0.0)
    directions = vt[keep].T
    return (directions / eigenvalues[keep]) @ directions.T / n_samples


def _constant_columns(basis_at_data: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Columns to drop: the intercept (column 0, its centred score is zero up to rounding), any column with no
    spread on the data, and any column whose centred score is negligible against the column itself, a test that
    does not depend on the data's units (a threshold relative to the other columns would: squeezing the data
    shrinks them, not the rounding residue).

    A column constant on the data carries no information there, but its centred score need not be zero: at
    k = 0 the knot at min(x) is all ones on the data while its centred score is P(X < min(x)) under the fit
    (review C1, 2026-10-25: it was kept and added about p(x)^2 / n to the variance at every x >= min(x))."""
    rms_basis = np.sqrt(np.mean(basis_at_data ** 2, axis=0))
    rms_scores = np.sqrt(np.mean(scores ** 2, axis=0))
    constant = (rms_scores <= CONSTANT_COLUMN_TOL * rms_basis) | (np.ptp(basis_at_data, axis=0) == 0)
    constant[0] = True
    return constant


def _model_information_root(theta_hat_full: np.ndarray, knots: np.ndarray, basis_order: int,
                            support: tuple[float, float] = SUPPORT) -> np.ndarray:
    """R with R' R = Cov(phi) under the fitted density (the model-based information), on the selected basis."""
    points, weights, result = quadrature_rule(theta_hat_full, knots, basis_order, support)
    mass = weights * np.exp(log_unnormalized_density(points, theta_hat_full, knots, basis_order) - result.log_z)
    phi = _basis(points, _selected_knots(theta_hat_full, knots, basis_order), basis_order)
    return np.sqrt(mass)[:, np.newaxis] * (phi - mass @ phi)


def estimate_covariance_beta(
    data: pd.DataFrame,
    estimation_results: dict,
    basis_order: int,
    *,
    support_refit: SupportRefitArg = "auto",
    _inverse: str = "scaled_floor",
    _floor: float = SCALED_FLOOR,
    _ridge_param: Optional[float] = None,
    _relative_cutoff: float = 1e-10,
    _information: str = "empirical",
    _support: tuple[float, float] = SUPPORT,
    **deprecated,
) -> np.ndarray:
    """Estimate Cov(β̂) by the delta method: the empirical information P_n S S', inverted with the scaled floor.

    The scaled floor (default since 0.2.0; Yilong, 2026-10-24, after SE round 2) adds a ridge of 1e-5 times
    each column's own score variance (``_floor``), so the SE does not depend on the data's units; the
    intercept and any column constant on the data are dropped. ``support_refit="auto"`` (default) uses the
    cleaned basis when the results carry the fit's L1 bound; ``None`` uses the original basis; a
    :class:`SupportRefit` is used as given. If basis cleaning failed, no SE is given: the covariance is NaN,
    with a warning.

    Internal: ``_inverse="ridge"`` is the 0.1.0 rule, the information plus the absolute ridge
    ``_ridge_param`` (1e-6 when not given), kept as the reference the paper's experiments compute alongside; the 0.1.0
    option ``ridge_param`` still selects it (deprecated). ``_inverse="cutoff"`` inverts the information
    through its singular values with a relative cutoff ``_relative_cutoff`` (round 1; not adopted);
    ``_information="model"`` uses the covariance of the basis under the fitted density instead of the
    empirical information (a reference only; the paper recommends the empirical one). ``_support`` (default
    [0, 1]) is the interval the fitted density lives on, for tests that rescale the data's units.
    """
    if "ridge_param" in deprecated:
        # 0.1.0 option: it keeps the 0.1.0 rule, an absolute ridge; accepted through 0.2.x
        warnings.warn(
            f"estimate_covariance_beta(ridge_param=...) is deprecated and will be removed in {REMOVED_IN}. It selects "
            "the 0.1.0 rule, an absolute ridge on the information (internally _inverse='ridge', _ridge_param=...); "
            "the default since 0.2.0 is the scaled floor, which does not depend on the data's units.",
            DeprecationWarning, stacklevel=2)
        _inverse, _ridge_param = "ridge", float(deprecated.pop("ridge_param"))
    take_deprecated("estimate_covariance_beta", deprecated, {})  # anything else is an unknown argument
    if _ridge_param is not None and _inverse != "ridge":
        # review C3: on its own it would be ignored silently, since the default rule is the scaled floor
        raise ValueError("_ridge_param applies only with _inverse='ridge' (the absolute ridge); the default rule is "
                         "the scaled floor, set by _floor")
    ridge_param = REFERENCE_RIDGE if _ridge_param is None else float(_ridge_param)
    if _inverse not in INVERSES:
        raise ValueError(f"_inverse must be one of {INVERSES}, got {_inverse!r}")
    if _information not in INFORMATIONS:
        raise ValueError(f"_information must be one of {INFORMATIONS}, got {_information!r}")
    # rows with a missing value are not observations: the fit drops them, so the SE does too (review C5; they
    # counted as observations below every knot at k = 0 and made every SE NaN at k >= 1)
    data = data.loc[np.isfinite(np.asarray(data["W1"], dtype=float))]
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
    knots = np.unique(np.array(estimation_results['data_points']))
    n_samples = len(data)

    selected_knots = _selected_knots(theta_hat_full, knots, basis_order)
    _, expected_phi = _fitted_expectation(theta_hat_full, knots, basis_order, _support)
    basis_matrix_selected = _basis(data, selected_knots, basis_order)

    scores = basis_matrix_selected - expected_phi[np.newaxis, :]
    if _information == "model":
        root = _model_information_root(theta_hat_full, knots, basis_order, _support)
        information = root.T @ root
    else:
        root = scores / np.sqrt(n_samples)
        information = (scores.T @ scores) / n_samples
    covariance_matrix = _covariance_from_information(information, root, n_samples, _inverse, ridge_param,
                                                     _relative_cutoff, _floor,
                                                     _constant_columns(basis_matrix_selected, scores))

    covariance = covariance_matrix.view(BasisCovariance)
    covariance.support_refit = support_refit
    return covariance


def _selected_knots(theta_hat_full: np.ndarray, knots: np.ndarray, basis_order: int) -> np.ndarray:
    return knots[np.asarray(theta_hat_full)[basis_order + 1:] != 0]


def _basis(x: Union[pd.DataFrame, np.ndarray], knots: np.ndarray, basis_order: int) -> np.ndarray:
    """The basis at ``x`` restricted to the intercept, the powers and the given knots (float64)."""
    frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame({'W1': np.asarray(x, dtype=float)})
    return create_basis_functions(frame, knots, order=basis_order, include_intercept=True)[0]


def _expected_basis(points: np.ndarray, mass: np.ndarray, selected_knots: np.ndarray, basis_order: int,
                    chunk: int = 4096) -> np.ndarray:
    """sum_q mass_q phi(points_q) over the selected basis, ``chunk`` nodes at a time: the whole node-by-basis
    matrix was 64,064 x 1,002 float64 (514 MB) at 1,000 knots (review C4)."""
    expected = np.zeros(basis_order + 1 + selected_knots.size)
    for lo in range(0, points.size, chunk):
        expected += mass[lo:lo + chunk] @ _basis(points[lo:lo + chunk], selected_knots, basis_order)
    return expected


def _fitted_expectation(theta_hat_full: np.ndarray, knots: np.ndarray, basis_order: int,
                        support: tuple[float, float] = SUPPORT) -> tuple[float, np.ndarray]:
    """log Z and E[phi] over the selected basis under the fitted density, with the rule of the accurate
    normalizer (Gauss-Legendre with breakpoints at the selected knots; exact for k = 0)."""
    points, weights, result = quadrature_rule(theta_hat_full, knots, basis_order, support)
    mass = weights * np.exp(log_unnormalized_density(points, theta_hat_full, knots, basis_order) - result.log_z)
    selected = _selected_knots(theta_hat_full, knots, basis_order)
    return result.log_z, _expected_basis(points, mass, selected, basis_order)


def _density_at(x_values: np.ndarray, theta_hat_full: np.ndarray, knots: np.ndarray, basis_order: int,
                support: tuple[float, float] = SUPPORT) -> np.ndarray:
    """The fitted density at ``x_values`` with the accurate normalizer (log Z only, no E[phi])."""
    _, _, result = quadrature_rule(theta_hat_full, knots, basis_order, support)
    log_density = log_unnormalized_density(x_values, theta_hat_full, knots, basis_order) - result.log_z
    return np.exp(np.clip(log_density, -745.0, 709.0))


def _evaluation_pieces(
    x_values: np.ndarray,
    theta_hat_full: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    support: tuple[float, float] = SUPPORT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(selected theta, basis at x on the selection, density at x, E[phi]) for one coefficient vector; the
    density is normalized with the accurate normalizer, as ``get_density_at_points`` does."""
    theta_selected_ind = np.concatenate([
        np.arange(0, basis_order + 1),
        theta_hat_full[basis_order + 1:].nonzero()[0] + basis_order + 1,
    ])
    theta_hat = theta_hat_full[theta_selected_ind]
    log_z, expected_phi = _fitted_expectation(theta_hat_full, knots, basis_order, support)
    phi_eval_selected = _basis(x_values, _selected_knots(theta_hat_full, knots, basis_order), basis_order)
    log_density = log_unnormalized_density(x_values, theta_hat_full, knots, basis_order) - log_z
    density_eval = np.exp(np.clip(log_density, -745.0, 709.0))
    return theta_hat, phi_eval_selected, density_eval, expected_phi


def density_confidence_interval(
    x_values: np.ndarray,
    estimation_results: dict,
    cov_beta: np.ndarray,
    basis_order: int,
    alpha: float = 0.05,
    support_refit: SupportRefitArg = "auto",
    *,
    scale: str = "density",
    _support: tuple[float, float] = SUPPORT,
) -> pd.DataFrame:
    """Compute delta-method CIs for density at x_values.

    ``scale`` sets the interval's form from the same SE: ``"density"`` (default) is density +- z SE with
    the lower end cut at 0 (the paper's formula); ``"log"`` is the interval for the log density mapped back,
    density * exp(+- z SE / density), which is always positive (Yilong, 2026-10-24: an option while both are
    compared; the default is unchanged).

    The SE uses the same basis as ``cov_beta``: with ``support_refit="auto"`` (default) the
    cleaned basis computed by :func:`estimate_covariance_beta` for this fit if ``cov_beta`` has its
    size, else the original basis; ``None`` forces the original basis; a :class:`SupportRefit`
    is used as given. The ``density`` column is always the original fit. ``result.attrs`` records
    the basis used and whether it was certified. If basis cleaning failed (the final fit did not
    pass the optimality check), ``se``, ``lower`` and ``upper`` are NaN, with a warning (decision 5,
    Yilong 2026-10-24); ``support_refit=None`` still gives the original-basis SE.
    """
    if scale not in INTERVAL_SCALES:
        raise ValueError(f"scale must be one of {INTERVAL_SCALES}, got {scale!r}")
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
    x_values = np.clip(x_values, *_support)
    if _cleaning_failed(support_refit):
        warnings.warn(NO_SE_WARNING, UserWarning, stacklevel=2)
        density_eval = _density_at(x_values, original_theta, knots, basis_order, _support)
        nan = np.full(len(x_values), np.nan)
        result_df = pd.DataFrame({'x': x_values, 'density': density_eval, 'se': nan, 'lower': nan, 'upper': nan})
        result_df.attrs.update(basis='cleaned', basis_certified=False, se='none: basis cleaning failed', scale=scale)
        return result_df

    theta_hat, phi_eval_selected, density_se_basis, expected_phi = _evaluation_pieces(
        x_values, _theta_for_se(estimation_results, support_refit), knots, basis_order, _support
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
        density_eval = _density_at(x_values, original_theta, knots, basis_order, _support)

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
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.exp(margin_of_error / density_eval)
        lower_bounds, upper_bounds = density_eval / factor, density_eval * factor
    else:
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
    result_df.attrs['scale'] = scale
    if support_refit is not None:
        result_df.attrs['basis_certified'] = bool(support_refit.certified)
        result_df.attrs['n_basis_original'] = int(original_theta[basis_order + 1:].nonzero()[0].size)
        result_df.attrs['n_basis_cleaned'] = int(support_refit.n_selected)

    return result_df
