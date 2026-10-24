"""Helpers for fitting interval-censored initial estimators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from haldensity.core.defaults import EM_DEFAULTS, TUNER_DEFAULTS


def fit_ic_init_estimator(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int,
    n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
    solver: str = TUNER_DEFAULTS.solver,
    use_secondary_solver: bool = TUNER_DEFAULTS.use_secondary_solver,
    tol: float = EM_DEFAULTS.tol,
    L_col: str = "L",
    R_col: str = "R",
) -> Any:
    """Fit an interval-censored init estimator on data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with interval columns.
    norm_constraint : float
        L1 norm constraint.
    basis_order : int
        Basis order.
    n_grid_points : int
        Number of grid points.
    solver : str
        CVXPY solver.
    use_secondary_solver : bool
        Whether to use fallback solvers.
    tol : float
        Tolerance for coefficient pruning.
    L_col : str
        Name of left interval column.
    R_col : str
        Name of right interval column.
        
    Returns
    -------
    Any
        Fitted estimator.
    """
    from haldensity.ic.estimators import IntervalCensoredInitEstimator
    
    estimator = IntervalCensoredInitEstimator(
        tol=tol,
        norm_constraint=norm_constraint,
        n_grid_points=n_grid_points,
        basis_order=basis_order,
        solver=solver,
        use_secondary_solver=use_secondary_solver,
        include_intercept_in_constraint=False,
    )
    estimator.fit(data, L_col=L_col, R_col=R_col)
    
    return estimator
