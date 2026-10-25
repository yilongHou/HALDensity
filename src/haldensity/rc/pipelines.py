"""Right-censored point estimators: one call each, with the same inputs and outputs.

All four estimate the event-time density from observed right-censored data
``(T, Delta)`` and report the plug-in survival and the density at the requested
time points. They share Stage 1: an IPCW-weighted HAL-MLE whose L1 bound M1 is
chosen by cross-validation (``RightCensoredInitTuner`` scored by the IPCW
log-likelihood).

========================  ======================================================
``method``                point estimator
========================  ======================================================
``"ipcw_hal"``            the Stage-1 IPCW-HAL fit itself (bound M1)
``"l1mle_m1"``            observed-data L1-MLE on the Stage-1 support at bound
                          M1, warm-started from Stage 1
``"cvl1mle"`` (default)   observed-data L1-MLE on the Stage-1 support with the
                          bound chosen by cross-validation over [M1, 2 M1]
                          (``RightCensoredObservedL1Tuner``)
``"undersmoothed"``       observed-data L1-MLE path over [M1, 2 M1]; for each
                          time point, the smallest bound whose final EIF mean
                          passes the TMLE threshold (if none passes, the bound
                          with the smallest absolute final EIF mean)
========================  ======================================================

``"cvl1mle"`` is the default because it has done best in most comparisons so
far; the default may change as the comparison continues. Survival is computed
the same way for all four (mass of the fitted density on its grid, interpolated
at each time point). Comparisons against a known truth belong in experiment
repositories, not in the package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from haldensity.rc.fit import RightCensoredHALFit
from haldensity.rc.km import KaplanMeier, last_time_as_event
from haldensity.rc.metrics import incomplete_loglik
from haldensity.rc.observed_mle import RightCensoredObservedL1MLE
from haldensity.rc.targeting.survival import right_censored_survival_targeting_M_step_v2
from haldensity.rc.tuners import RightCensoredInitTuner, RightCensoredObservedL1Tuner

METHODS: tuple[str, ...] = ("ipcw_hal", "l1mle_m1", "cvl1mle", "undersmoothed")
DEFAULT_METHOD = "cvl1mle"
DEFAULT_NORM_FACTORS: tuple[float, ...] = tuple(
    float(x) for x in np.round(np.linspace(1.0, 2.0, 11), 2)
)
DEFAULT_TARGETING_KWARGS: dict[str, Any] = {
    "mode": "iterative",
    "clip": 1e-6,
    "survival_clip": 1e-8,
    "max_iter": 25,
    "min_abs_eps": 1e-10,
    "min_score_tol": 1e-8,
    "store_pointwise_arrays": False,
}
_FIXED_L1_ARGS = {"working_grid_points", "norm_constraint", "basis_order", "warm_start_theta"}


@dataclass(frozen=True)
class RightCensoredPointEstimate:
    """Result of a right-censored point estimator.

    Attributes
    ----------
    method : str
        One of :data:`METHODS`.
    summary : pd.DataFrame
        One row per time point: ``t0``, ``survival`` (plug-in), ``density``,
        ``norm_factor`` (bound / M1) and ``norm_constraint`` (bound) of the
        fit used at that point.
    fits : dict[float, RightCensoredHALFit]
        Frozen fit for each norm factor used (one entry except for
        ``"undersmoothed"``, which may use a different bound per time point).
    stage1_norm_constraint : float
        M1, the cross-validated bound of the Stage-1 IPCW fit.
    details : dict
        Live estimators by norm factor (``"estimators"``, e.g. as initial fits
        for targeting), tuning results, path tables, selection diagnostics
        and runtimes in seconds.
    """

    method: str
    summary: pd.DataFrame
    fits: dict[float, RightCensoredHALFit]
    stage1_norm_constraint: float
    details: dict[str, Any]

    def fit_at(self, t0: float) -> RightCensoredHALFit:
        """Return the fit used at time point ``t0`` (one of the requested points)."""
        times = self.summary["t0"].to_numpy(dtype=float)
        match = np.flatnonzero(np.isclose(times, float(t0)))
        if match.size == 0:
            raise KeyError(f"t0={t0} is not one of the requested time points")
        return self.fits[float(self.summary["norm_factor"].iloc[int(match[0])])]


def _plugin_survival(estimator: Any, points: np.ndarray) -> np.ndarray:
    """Survival from the fitted density's mass on its grid, interpolated at ``points``."""
    _, density_grid = estimator.get_density()
    delta_j = np.asarray(estimator.delta_j, dtype=float)
    survival_grid = np.cumsum((np.asarray(density_grid, dtype=float) * delta_j)[::-1])[::-1]
    edge_survival = np.concatenate((survival_grid, [0.0]))
    grid_edges = np.asarray(estimator.grid_points, dtype=float)
    return np.interp(points, grid_edges, edge_survival, left=1.0, right=0.0)


def _compress_theta_to_selected_support(estimator: Any) -> np.ndarray:
    support = np.asarray(estimator.grid_points_hal_selected, dtype=float)
    all_knots = np.asarray(estimator._grid_points_hal, dtype=float)
    theta_full = np.asarray(estimator.theta_hat, dtype=float)
    basis_order = int(estimator.basis_order)
    knot_start = 1 + (basis_order if basis_order > 0 else 0)
    theta = np.zeros(knot_start + support.size, dtype=float)
    theta[:knot_start] = theta_full[:knot_start]
    for i, knot in enumerate(support):
        idx = np.where(np.isclose(all_knots, knot, atol=1e-10, rtol=0.0))[0]
        if idx.size == 0:
            raise RuntimeError("Selected support knot was not found in the fitted working grid.")
        theta[knot_start + i] = theta_full[knot_start + int(idx[0])]
    return theta


def _norm_factors(values: Optional[Sequence[float]]) -> list[float]:
    factors = list(DEFAULT_NORM_FACTORS) if values is None else [float(x) for x in values]
    if len(factors) == 0:
        raise ValueError("norm_constraint_factors must contain at least one value")
    if any(x <= 0.0 for x in factors):
        raise ValueError("norm_constraint_factors must contain positive values")
    return sorted(set(factors))


def _fit_stage1(
    observed_data: pd.DataFrame,
    *,
    cv_folds: int,
    n_trials: int,
    random_state: int,
    n_grid_points: int,
    param_overrides: Optional[dict[str, Any]],
    tuner_kwargs: Optional[dict[str, Any]],
) -> tuple[Any, Any, float]:
    kwargs = dict(tuner_kwargs or {})
    kwargs.setdefault("silent", True)
    kwargs.setdefault("use_conservative_adjustment", False)
    tic = time.perf_counter()
    tuner = RightCensoredInitTuner(
        data=observed_data,
        cv_folds=int(cv_folds),
        random_state=int(random_state),
        n_grid_points=int(n_grid_points),
        param_overrides=param_overrides,
        validation_metric="ipcw_loglik",
        **kwargs,
    )
    result = tuner.optimize(n_trials=int(n_trials))
    return result.estimator, result, float(time.perf_counter() - tic)


def _fit_l1mle(
    observed_data: pd.DataFrame,
    stage1_estimator: Any,
    norm_constraint: float,
    warm_start: np.ndarray,
    l1_kwargs: dict[str, Any],
) -> RightCensoredObservedL1MLE:
    return RightCensoredObservedL1MLE(
        working_grid_points=np.asarray(stage1_estimator.grid_points_hal_selected, dtype=float),
        norm_constraint=float(norm_constraint),
        basis_order=int(stage1_estimator.basis_order),
        warm_start_theta=warm_start,
        **l1_kwargs,
    ).fit(observed_data, time_col="T", delta_col="Delta")


def _fit_path(
    observed_data: pd.DataFrame,
    stage1_estimator: Any,
    factors: list[float],
    l1_kwargs: dict[str, Any],
) -> tuple[dict[float, Any], pd.DataFrame]:
    """Fit the observed-data L1-MLE along the factor path, anchored at the factor closest to 1."""
    base_norm = float(stage1_estimator.norm_constraint)
    grid = np.asarray(factors, dtype=float)
    anchor = int(np.argmin(np.abs(grid - 1.0)))
    order = [anchor] + list(range(anchor + 1, len(grid))) + list(range(anchor - 1, -1, -1))
    estimators: dict[float, Any] = {}
    rows = []
    warm_start = _compress_theta_to_selected_support(stage1_estimator)
    for idx in order:
        factor = float(grid[idx])
        tic = time.perf_counter()
        estimator = _fit_l1mle(observed_data, stage1_estimator, base_norm * factor, warm_start, l1_kwargs)
        results = estimator.get_results()
        estimators[factor] = estimator
        rows.append({
            "norm_factor": factor,
            "norm_constraint": base_norm * factor,
            "runtime_seconds": float(time.perf_counter() - tic),
            "observed_loglik": float(incomplete_loglik(estimator, observed_data)),
            "n_iterations_run": int(results.get("n_iterations_run", 0)),
            "converged": bool(results.get("converged", False)),
        })
        warm_start = np.asarray(estimator.theta_hat, dtype=float).copy()
    path_df = pd.DataFrame(rows).sort_values("norm_factor").reset_index(drop=True)
    return estimators, path_df


def _select_undersmoothed(
    observed_data: pd.DataFrame,
    points: np.ndarray,
    estimators: dict[float, Any],
    km: KaplanMeier,
    targeting_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per time point, the smallest factor whose final EIF mean passes; else the smallest |EIF mean|."""
    frames = []
    for factor, estimator in sorted(estimators.items()):
        targeted = right_censored_survival_targeting_M_step_v2(
            initial_estimator=estimator,
            observed_data=observed_data,
            targeting_points=points,
            km=km,
            **targeting_kwargs,
        )
        summary = targeted["summary"].copy()
        summary["norm_factor"] = float(factor)
        summary["passes_final"] = (
            np.abs(summary["eif_mean_final"].to_numpy(dtype=float))
            <= summary["threshold_final"].to_numpy(dtype=float) + 1e-12
        )
        frames.append(summary)
    long = pd.concat(frames, ignore_index=True).sort_values(["t0", "norm_factor"]).reset_index(drop=True)
    rows = []
    for t0, group in long.groupby("t0", sort=True):
        ordered = group.sort_values("norm_factor").reset_index(drop=True)
        passing = ordered[ordered["passes_final"]]
        if not passing.empty:
            selected, status = passing.iloc[0], "smallest_passing_factor"
        else:
            selected = ordered.iloc[int(np.argmin(np.abs(ordered["eif_mean_final"].to_numpy(dtype=float))))]
            status = "closest_failing_factor"
        rows.append({
            "t0": float(t0),
            "norm_factor": float(selected["norm_factor"]),
            "eif_mean_final": float(selected["eif_mean_final"]),
            "threshold_final": float(selected["threshold_final"]),
            "passes_final": bool(selected["passes_final"]),
            "psi_star_targeted_diagnostic": float(selected["psi_star"]),
            "n_iterations": int(selected["n_iterations"]),
            "selection_status": status,
        })
    return pd.DataFrame(rows).sort_values("t0").reset_index(drop=True), long


def fit_right_censored(
    observed_data: pd.DataFrame,
    targeting_points: Sequence[float],
    *,
    method: str = DEFAULT_METHOD,
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 0,
    n_grid_points: int = 200,
    stage1_param_overrides: Optional[dict[str, Any]] = None,
    stage1_tuner_kwargs: Optional[dict[str, Any]] = None,
    norm_constraint_factors: Optional[Sequence[float]] = None,
    l1_kwargs: Optional[dict[str, Any]] = None,
    targeting_kwargs: Optional[dict[str, Any]] = None,
) -> RightCensoredPointEstimate:
    """Fit a right-censored HAL point estimator (see the module docstring for ``method``).

    Parameters
    ----------
    observed_data : pd.DataFrame
        Columns ``T`` (observed time in ``[0, 1]``) and ``Delta`` (1 = event).
    targeting_points : sequence of float
        Time points at which survival and density are reported, sorted and
        de-duplicated in the summary (and, for
        ``"undersmoothed"``, at which the EIF rule picks the bound).
    method : str
        One of ``"ipcw_hal"``, ``"l1mle_m1"``, ``"cvl1mle"`` (default),
        ``"undersmoothed"``.
    cv_folds, random_state : int
        Folds and seed for the Stage-1 CV and, for ``"cvl1mle"``, the CV over
        the bound.
    n_trials : int
        Optuna trials for the Stage-1 CV.
    n_grid_points : int
        Fitting grid of the Stage-1 IPCW fit.
    stage1_param_overrides, stage1_tuner_kwargs : dict, optional
        Passed to ``RightCensoredInitTuner`` (search ranges; tuner options).
    norm_constraint_factors : sequence of float, optional
        Bound multipliers of M1 for ``"cvl1mle"`` and ``"undersmoothed"``
        (default 1.0, 1.1, ..., 2.0).
    l1_kwargs : dict, optional
        Options of ``RightCensoredObservedL1MLE`` (solver settings); the
        support, bound, basis order and warm start are fixed by the method.
    targeting_kwargs : dict, optional
        Options of the survival TMLE used by the ``"undersmoothed"`` rule
        (default :data:`DEFAULT_TARGETING_KWARGS`).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    points = np.unique(np.asarray(targeting_points, dtype=float).ravel())
    if points.size == 0:
        raise ValueError("targeting_points must contain at least one time point")
    l1 = dict(l1_kwargs or {})
    fixed = sorted(_FIXED_L1_ARGS.intersection(l1))
    if fixed:
        raise ValueError(f"l1_kwargs must not set arguments fixed by the method: {', '.join(fixed)}")

    observed_data = last_time_as_event(observed_data)
    stage1, stage1_result, stage1_seconds = _fit_stage1(
        observed_data,
        cv_folds=cv_folds,
        n_trials=n_trials,
        random_state=random_state,
        n_grid_points=n_grid_points,
        param_overrides=stage1_param_overrides,
        tuner_kwargs=stage1_tuner_kwargs,
    )
    m1 = float(stage1.norm_constraint)
    details: dict[str, Any] = {"stage1_tuning_result": stage1_result}
    runtime = {"stage1": stage1_seconds}
    tic = time.perf_counter()

    if method == "ipcw_hal":
        estimators = {1.0: stage1}
        chosen = np.ones(points.size)
    elif method == "l1mle_m1":
        warm_start = _compress_theta_to_selected_support(stage1)
        estimators = {1.0: _fit_l1mle(observed_data, stage1, m1, warm_start, l1)}
        chosen = np.ones(points.size)
    elif method == "cvl1mle":
        tuner = RightCensoredObservedL1Tuner(
            data=observed_data,
            stage1_estimator=stage1,
            cv_folds=int(cv_folds),
            random_state=int(random_state),
            norm_constraint_factors=_norm_factors(norm_constraint_factors),
            l1_kwargs=l1,
            silent=True,
        )
        result = tuner.optimize()
        factor = float(result.best_params["norm_constraint_factor"])
        estimators = {factor: result.estimator}
        chosen = np.full(points.size, factor)
        details["cv_tuning_result"] = result
    else:
        path_estimators, path_df = _fit_path(observed_data, stage1, _norm_factors(norm_constraint_factors), l1)
        km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")
        targeting = dict(DEFAULT_TARGETING_KWARGS)
        targeting.update(dict(targeting_kwargs or {}))
        selection, targeting_long = _select_undersmoothed(observed_data, points, path_estimators, km, targeting)
        if not np.allclose(selection["t0"].to_numpy(dtype=float), points):
            raise RuntimeError("Targeting returned time points that differ from the requested ones")
        chosen = selection["norm_factor"].to_numpy(dtype=float)
        estimators = {f: path_estimators[f] for f in sorted(set(chosen.tolist()))}
        details.update({
            "path": path_df,
            "path_estimators": path_estimators,
            "selection": selection,
            "targeting_summary_long": targeting_long,
            "targeting_kwargs": targeting,
        })

    runtime[method] = float(time.perf_counter() - tic)
    runtime["total"] = runtime["stage1"] + runtime[method]
    details.update({"estimators": estimators, "runtime_seconds": runtime})

    survival = np.empty(points.size)
    density = np.empty(points.size)
    for factor, estimator in estimators.items():
        mask = np.isclose(chosen, factor)
        survival[mask] = _plugin_survival(estimator, points[mask])
        density[mask] = np.asarray(estimator.get_density_at_points(points[mask]), dtype=float)
    summary = pd.DataFrame({
        "t0": points,
        "survival": survival,
        "density": density,
        "norm_factor": chosen,
        "norm_constraint": chosen * m1,
    })
    return RightCensoredPointEstimate(
        method=method,
        summary=summary,
        fits={float(f): est.get_fit() for f, est in estimators.items()},
        stage1_norm_constraint=m1,
        details=details,
    )


def _named(method: str, doc: str):
    def fit(observed_data: pd.DataFrame, targeting_points: Sequence[float], **kwargs: Any) -> RightCensoredPointEstimate:
        if "method" in kwargs:
            raise TypeError("method is fixed by this function; use fit_right_censored to choose it")
        return fit_right_censored(observed_data, targeting_points, method=method, **kwargs)

    fit.__name__ = f"fit_{method}"
    fit.__qualname__ = fit.__name__
    fit.__doc__ = f"{doc}\n\nSame inputs and output as :func:`fit_right_censored` with ``method={method!r}``."
    return fit


fit_ipcw_hal = _named("ipcw_hal", "Stage-1 IPCW-HAL fit (bound M1 chosen by CV).")
fit_l1mle_m1 = _named("l1mle_m1", "Observed-data L1-MLE on the Stage-1 support at bound M1.")
fit_cvl1mle = _named("cvl1mle", "Observed-data L1-MLE with the bound chosen by CV over [M1, 2 M1] (default method).")
fit_undersmoothed = _named(
    "undersmoothed", "Observed-data L1-MLE with the bound chosen per time point by the EIF-mean rule."
)
