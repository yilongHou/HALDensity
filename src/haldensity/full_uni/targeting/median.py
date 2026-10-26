import numpy as np
import pandas as pd
from typing import Union, Any, Tuple
from haldensity.core.defaults import DEFAULT_BASIS_ORDER
from haldensity.full_uni.targeting.base import BaseTargetLearner
from haldensity.full_uni.targeting.exact import ExactDensity, cell_edges, converged_rule
from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.core.univariate.functionals import generic_compute_median_from_density



def _compute_median_from_density(targeted_fit: dict[str, Any]) -> Tuple[float, float]:
    """
    Compute median from density estimate using cumulative integration.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from M-step containing density estimates and grid information.
        
    Returns
    -------
    Tuple[float, float]
        Tuple of (median, density_at_median).
    """
    
    density = targeted_fit['estimated_density']
    grid_points = targeted_fit['grid_midpoints']
    
    if density.ndim != 1 or grid_points.ndim != 1:
        raise ValueError("Both density and grid_points must be 1-dimensional arrays.")
    if density.shape[0] != grid_points.shape[0]:
        raise ValueError("Density and grid_points must have the same length.")
    
    # Compute median from density using utility function
    median_value, density_at_median = generic_compute_median_from_density(grid_points, density)
    
    return median_value, density_at_median


class MedianTargetLearner(BaseTargetLearner):
    """
    Target learner that uses median-based targeting basis functions.
    
    This learner creates targeting basis functions that are indicators for 
    whether data points are less than the initial fit's median m. The M-step's
    integration rule has a cell edge at m, so the targeted density satisfies
    F(m) = P_n(X < m) at m itself.

    One targeting step at the initial median, as in the paper's median example (Yilong,
    2026-10-24: one step is good enough), so the targeted median need not be an empirical
    median; iterating the step would end at one.
    """

    def _initial_median(self, **kwargs) -> float:
        """The initial fit's median, exactly (cached per fit)."""
        old_theta = np.asarray(kwargs['_old_theta'], dtype=float).ravel()
        knots = np.asarray(kwargs['_grid_points_hal_selected'], dtype=float).ravel()
        key = (old_theta.tobytes(), knots.tobytes(), int(self.basis_order))
        cached = getattr(self, "_median_cache", None)
        if cached is None or cached[0] != key:
            def fixed(x: np.ndarray) -> np.ndarray:
                return log_unnormalized_density(x, old_theta, knots, self.basis_order)
            median = ExactDensity(fixed, converged_rule(fixed, cell_edges(knots))).quantile(0.5)
            self._median_cache = cached = (key, median)
        return cached[1]

    def _indicator(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """I(x < m), m the initial fit's median; shape (n, 1)."""
        return (np.asarray(x, dtype=float) < self._initial_median(**kwargs)).astype(float)[:, None]

    def targeting_breakpoints(self, **kwargs) -> np.ndarray:
        return np.array([self._initial_median(**kwargs)])

    def target_values(self, density: ExactDensity, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """The median of ``density`` and the density there, exactly."""
        median = density.quantile(0.5)
        return np.array([median]), density.density(np.array([median]))
    
    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at the data, (n_data, 1): I(x < m). kwargs carry '_old_theta' and
        '_grid_points_hal_selected' (the initial fit)."""
        return self._indicator(uncensored_data['W1'].values, **kwargs)
    
    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at integration points, (n_points, 1): I(x < m)."""
        return self._indicator(data_grid['W1'].values, **kwargs)
    
    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at the fine evaluation grid, (n_grid_full, 1): I(x < m)."""
        return self._indicator(data_grid_full['W1'].values, **kwargs)
    
    def influence_curve_at(self, density: ExactDensity, x: np.ndarray, **kwargs) -> np.ndarray:
        """(1/2 - I(x < m)) / f(m) under ``density``, m its median."""
        median = density.quantile(0.5)
        at = float(density.density(np.array([median]))[0])
        return ((0.5 - (np.asarray(x, dtype=float) < median)) / at)[:, None]

    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        uncensored_data: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Compute variance of median estimate using efficient influence curve (EIC).
        
        Based on the methodology where for median m:
        D*(x) = (0.5 - I(x < m)) / f(m)
        
        The variance is estimated as the sample variance of the EIC divided by n.
        This follows the asymptotic theory: Var(√n(Ψ_n - Ψ)) ≈ Var(D*) / n
        
        Parameters
        ----------
        targeted_fit : dict[str, Any]
            Results from the M-step containing density estimates and grid information.
        uncensored_data : pd.DataFrame
            The uncensored data points used to construct the EIC.
        **kwargs
            Additional parameters (not used for median variance estimation).
            
        Returns
        -------
        np.ndarray
            Estimated variance for the median (single element array).
        """
        cross_fitted = self._cross_fitted_variance(targeted_fit, uncensored_data)
        if cross_fitted is not None:                # each observation under its own fold's targeted fit
            return cross_fitted
        # Step 1: the targeted median and the density there, exact (the M-step's estimate)
        estimate = self._exact_estimate(targeted_fit)
        if estimate is None:
            median_value, density_at_median = _compute_median_from_density(targeted_fit)
        else:
            median_value = float(estimate[0])
            density_at_median = float(np.asarray(targeted_fit['density_at_estimate'], dtype=float).ravel()[0])
        
        # Step 2: Check for very small density values to avoid numerical issues
        if density_at_median < 1e-8:
            # Use a small positive value to avoid division by zero
            density_at_median = 1e-8
        
        # Step 3: Create indicator for values less than median
        data_values = uncensored_data['W1'].values
        data_values = np.asarray(data_values)  # Ensure numpy array
        indicator_less_than_median = (data_values < median_value).astype(float)
        
        # Step 4: Construct EIC using the formula: D*(x) = (0.5 - I(x < m)) / f(m)
        EIC = (0.5 - indicator_less_than_median) / density_at_median
        
        # Step 5: Compute variance as sample variance of EIC divided by n
        n_data = uncensored_data.shape[0]
        estimand_variance = np.var(EIC, ddof=1) / n_data
        
        # Return as single-element array for consistency
        return np.array([estimand_variance])


# Convenience function for backward compatibility
def median_targeting_M_step(
    uncensored_augmented: pd.DataFrame,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: int = 20,
    basis_order: int = DEFAULT_BASIS_ORDER,
    cross_fit: Union[str, bool] = "tuner",
    fit: Any = None,
    folds: Any = None
) -> dict:
    """
    Convenience function that wraps the MedianTargetLearner for backward compatibility.
    """
    learner = MedianTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.run_m_step(
        uncensored_augmented=uncensored_augmented,
        grid_points_hal_selected=grid_points_hal_selected,
        old_theta=old_theta,
        _old_theta=old_theta,  # serves as kwarg
        _grid_points_hal_selected=grid_points_hal_selected,  # serves as kwarg
        cross_fit=cross_fit,
        fit=fit,
        folds=folds
    )


def median_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    norm_constraint: int = 20,
    basis_order: int = DEFAULT_BASIS_ORDER
) -> np.ndarray:
    """
    Convenience function that wraps the MedianTargetLearner for estimating variance.
    
    Parameters
    ----------
    targeted_fit : dict[str, Any]
        Results from the M-step containing density estimates and grid information.
    uncensored_data : pd.DataFrame
        The uncensored data points used to construct the EIC.
    norm_constraint : int, optional
        L1-norm constraint (default is 20).
    basis_order : int, optional
        Order of the basis functions (default 2, DEFAULT_BASIS_ORDER).
        
    Returns
    -------
    np.ndarray
        Estimated variance for the median (single element array).
    """
    learner = MedianTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        uncensored_data=uncensored_data
    )
