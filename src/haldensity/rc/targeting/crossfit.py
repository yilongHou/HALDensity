"""Cross-fitted targeting for right-censored data (Yilong, 2026-10-26); ``cross_fit`` is explained in
:mod:`haldensity.core.crossfit`.

Fold v's initial fit and its censoring Kaplan-Meier come from the other folds (with the Kaplan-Meier convention, as
every fit applies it to its data); the validation fold is scored as observed. One epsilon for all folds solves the
cross-fitted score sum_v sum_{i in v} D*_v(O_i) = 0 with the full-sample step's solver (bracket, then Brent), and the
directions are recomputed from each fold's targeted fit after every update. ``mode`` works as in the full-sample
step: "iterative" until |P_n D*| <= max(min_score_tol, sigma / (sqrt(n) log n)), "one_step", or "auto" (no update when
the fold fits already pass). The estimate is the fold-size-weighted mean of the targeted fold fits' values; the SE is
the sd of the cross-fitted influence curve (each observation's under its own fold's targeted fit) over sqrt(n).

For entropy and density L2 targeting removes the first-order term, not the second-order remainder of the fold
fits (a Kullback-Leibler divergence and minus a squared L2 distance); that remainder was still visible at
n <= 1,600 in the CV-TMLE study (2026-10-26).
"""
from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from haldensity.core.crossfit import FoldFits, data_key, default_splits, resolve_cross_fit
from haldensity.rc.km import KaplanMeier, last_time_as_event
from haldensity.rc.targeting._common import (
    RCCensoringCache,
    RCInitialFit,
    RCTargetGrid,
    _build_censoring_cache,
    _build_initial_fit,
    _build_pointwise_target_grid,
    _build_wald_interval,
    _compute_dfg_t0_on_grid,
    _compute_targeting_gbar_floor,
    _find_score_bracket,
    _interp_survival_from_edges,
    _tilted_density_and_survival,
    _tmle_eif_values,
    _tmle_loglik_t0,
    _tmle_score_t0,
)
from haldensity.rc.targeting.eic import _build_unbounded_wald_interval_allow_nan, _gbar_floor_active


@dataclass(frozen=True)
class TargetSpec:
    """A target for the cross-fitted loop: its grid from an initial fit, its centred direction on a grid, its value
    on a grid, where its influence curve stops using G (survival's t0, RMST's tau) and its Wald interval."""

    target: str
    point_column: Optional[str]
    point: Optional[float]
    build: Callable[[RCInitialFit], RCTargetGrid]
    direction: Callable[[RCTargetGrid, RCCensoringCache, float], np.ndarray]
    value: Callable[[RCTargetGrid], float]
    horizon: Optional[float]
    interval: Callable[[float, float], tuple[float, float]]


def survival_spec(t0: float, survival_clip: float) -> TargetSpec:
    t0 = float(t0)
    return TargetSpec(
        "survival", "t0", t0, lambda fit: _build_pointwise_target_grid(fit, t0),
        lambda g, cache, floor: _compute_dfg_t0_on_grid(g, cache, t0, survival_clip, floor)[1],
        lambda g: float(_interp_survival_from_edges(g.grid_edges, g.edge_survival, np.array([t0]))[0]),
        t0, _build_wald_interval)


def rmst_spec(tau: float, survival_clip: float) -> TargetSpec:
    from haldensity.rc.targeting import rmst as R

    tau = float(tau)
    return TargetSpec(
        "rmst", "tau", tau, lambda fit: _build_pointwise_target_grid(fit, tau),
        lambda g, cache, floor: R._compute_rmst_direction_on_grid(g, cache, tau, survival_clip, floor)[1],
        lambda g: R._compute_rmst_from_target_grid(g, tau), tau,
        lambda center, se: R._build_rmst_wald_interval(center, se, tau))


def mean_spec(survival_clip: float) -> TargetSpec:
    from haldensity.rc.targeting import mean as M

    return TargetSpec(
        "mean", None, None, M._initial_fit_to_target_grid,
        lambda g, cache, floor: M._compute_mean_direction_on_grid(g, cache, survival_clip, floor)[1],
        lambda g: M._compute_mean_from_grid(g.density_grid, g.grid_midpoints, g.delta_j), None,
        _build_wald_interval)


def entropy_spec(survival_clip: float) -> TargetSpec:
    from haldensity.rc.targeting import entropy as E

    return TargetSpec(
        "entropy", None, None, E._initial_fit_to_target_grid,
        lambda g, cache, floor: E._compute_entropy_direction_on_grid(g, cache, survival_clip, floor)[1],
        lambda g: E._compute_entropy_from_grid(g.density_grid, g.delta_j), None,
        _build_unbounded_wald_interval_allow_nan)


def density_l2_spec(survival_clip: float) -> TargetSpec:
    from haldensity.rc.targeting import density_l2 as L

    return TargetSpec(
        "density_l2", None, None, L._initial_fit_to_target_grid,
        lambda g, cache, floor: L._compute_density_l2_direction_on_grid(g, cache, survival_clip, floor)[1],
        lambda g: L._compute_density_l2_from_grid(g.density_grid, g.delta_j), None,
        _build_unbounded_wald_interval_allow_nan)


# ---------------------------------------------------------------- the fold fits ---------------------------------------

def resolve_fold_fits(initial_estimator: Any, observed_data: pd.DataFrame, cross_fit: str) -> FoldFits:
    """The fold fits for ``cross_fit`` ("tuner", "refit" or "nested"); "tuner" falls back to "refit" when the estimator
    has no tuner fold fits for this data."""
    data = observed_data.reset_index(drop=True)
    key = data_key(data["T"].to_numpy(), data["Delta"].to_numpy())
    tuner_fits = getattr(initial_estimator, "fold_fits_", None)
    usable = tuner_fits is not None and tuner_fits.source == "tuner" and tuner_fits.matches(key)
    if cross_fit == "tuner" and usable:
        return tuner_fits
    if cross_fit == "tuner":
        cross_fit = "refit"
    splits = [(train.copy(), validation.copy()) for train, validation in tuner_fits.splits] if usable \
        else default_splits(len(data))
    make = _refit if cross_fit == "refit" else _nested
    estimators = [make(initial_estimator, data.iloc[train].reset_index(drop=True)) for train, _ in splits]
    return FoldFits(splits=splits, estimators=estimators, source=cross_fit, data_key=key)


def _refit(estimator: Any, train: pd.DataFrame) -> Any:
    """The estimator's own settings on a training fold. For a fit from fit_right_censored: Stage 1 at its chosen
    (k, M1) on the fold, then (except for "ipcw_hal") the observed-data L1-MLE at the chosen bound on that fold's own
    Stage-1 knots; otherwise the same class and settings."""
    from haldensity.rc.estimators import RightCensoredInitEstimator
    from haldensity.rc.observed_mle import RightCensoredObservedL1MLE

    pipeline = getattr(estimator, "rc_pipeline_", None)
    if pipeline is not None:
        from haldensity.rc.pipelines import _compress_theta_to_selected_support, _fit_l1mle
        from haldensity.rc.tuners import RightCensoredInitTuner

        recipe, stage1 = pipeline["recipe"], pipeline["stage1"]
        kwargs = dict(recipe.get("stage1_tuner_kwargs") or {})
        kwargs.setdefault("silent", True)
        kwargs.setdefault("use_conservative_adjustment", False)
        tuner = RightCensoredInitTuner(data=train, cv_folds=recipe["cv_folds"], random_state=recipe["random_state"],
                                       n_grid_points=recipe["n_grid_points"],
                                       param_overrides=recipe.get("stage1_param_overrides"),
                                       validation_metric="ipcw_loglik", **kwargs)
        fold_stage1 = tuner._fit_final_estimator({"basis_order": int(stage1.basis_order),
                                                  "norm_constraint": float(stage1.norm_constraint)})
        if recipe["method"] == "ipcw_hal":
            return fold_stage1
        return _fit_l1mle(train, fold_stage1, pipeline["norm_factor"] * float(stage1.norm_constraint),
                          _compress_theta_to_selected_support(fold_stage1), dict(recipe.get("l1_kwargs") or {}))
    if isinstance(estimator, RightCensoredObservedL1MLE):
        params = _constructor_params(estimator, skip=("warm_start_theta", "log_dir", "log_frequency"))
        return RightCensoredObservedL1MLE(**params).fit(train, time_col="T", delta_col="Delta")
    if isinstance(estimator, RightCensoredInitEstimator):
        from haldensity.rc.weights import compute_ipcw_weights

        converted = last_time_as_event(train)
        km = KaplanMeier().fit(converted, time_col="T", delta_col="Delta")
        t, delta = converted["T"].to_numpy(dtype=float), converted["Delta"].to_numpy(dtype=int)
        weights = compute_ipcw_weights(t, delta, lambda s: np.atleast_1d(km.predict(s)))
        params = _constructor_params(estimator, skip=("log_dir", "log_frequency"))
        fit = RightCensoredInitEstimator(**params)
        fit.fit(pd.DataFrame({"W1": t[delta == 1]}), sample_weights=weights[delta == 1])
        return fit
    raise ValueError(f"cross_fit cannot refit an estimator of type {type(estimator).__name__} on the folds; pass "
                     "an estimator from haldensity.rc.fit_right_censored, or cross_fit='none' for the full-sample "
                     "targeting step")


def _nested(estimator: Any, train: pd.DataFrame) -> Any:
    """fit_right_censored repeated on a training fold, with all its tuning (Stage 1's k, M1 and knots; the bound)."""
    from haldensity.rc.pipelines import fit_right_censored

    pipeline = getattr(estimator, "rc_pipeline_", None)
    if pipeline is None:
        raise ValueError("cross_fit='nested' repeats fit_right_censored inside each training fold, so it needs an "
                         "estimator from fit_right_censored")
    recipe = dict(pipeline["recipe"])
    method = recipe.pop("method")
    if method == "undersmoothed":
        raise ValueError("cross_fit='nested' is not available for method='undersmoothed' (its bound is chosen per "
                         "time point); use 'tuner' or 'refit'")
    fit = fit_right_censored(train, pipeline["targeting_points"], method=method, **recipe)
    (fold_estimator,) = fit.details["estimators"].values()
    return fold_estimator


def _constructor_params(estimator: Any, skip: Sequence[str] = ()) -> dict[str, Any]:
    params = {}
    for name, parameter in inspect.signature(type(estimator).__init__).parameters.items():
        if name == "self" or name in skip or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(estimator, name):
            params[name] = getattr(estimator, name)
    return params


# ---------------------------------------------------------------- the update loop -------------------------------------

@dataclass
class _Fold:
    fit: RCInitialFit
    cache: RCCensoringCache
    floor: float
    data: pd.DataFrame
    rows: np.ndarray


def _fold_parts(fold_fits: FoldFits, observed_data: pd.DataFrame, clip: float, floor_scale: Optional[float],
                use_floor: bool) -> list[_Fold]:
    data = observed_data.reset_index(drop=True)
    parts = []
    for (train, validation), estimator in zip(fold_fits.splits, fold_fits.estimators):
        km = KaplanMeier().fit(data.iloc[train].reset_index(drop=True), time_col="T", delta_col="Delta")
        floor = (_compute_targeting_gbar_floor(n_obs=len(train), scale=floor_scale, hard_min=clip) if use_floor
                 else 0.0)
        parts.append(_Fold(_build_initial_fit(estimator), _build_censoring_cache(km, clip=clip), floor,
                           data.iloc[validation].reset_index(drop=True), np.asarray(validation)))
    return parts


def _tilt(grid: RCTargetGrid, centred: np.ndarray, eps: float) -> RCTargetGrid:
    density, survival, edge_survival, _ = _tilted_density_and_survival(grid, centred, eps)
    return dataclasses.replace(grid, density_grid=density, log_density_grid=np.log(np.clip(density, 1e-300, None)),
                               survival_grid=survival, edge_survival=edge_survival)


def _solve_epsilon(grids, centred, folds, start, growth, max_width, fallback_bounds) -> tuple[float, bool]:
    """The full-sample step's _solve_epsilon_t0 with the score and log-likelihood summed over the folds."""
    from scipy.optimize import brentq, minimize_scalar

    def score(eps):
        return sum(_tmle_score_t0(eps, g, c, f.data) for g, c, f in zip(grids, centred, folds))

    def loglik(eps):
        return sum(_tmle_loglik_t0(eps, g, c, f.data) for g, c, f in zip(grids, centred, folds))

    score_at_zero = float(score(0.0))
    bracket, _ = _find_score_bracket(score, start=start, growth=growth, max_width=max_width)
    if bracket is not None:
        eps = float(bracket[0]) if bracket[0] == bracket[1] else float(brentq(score, bracket[0], bracket[1]))
        return eps, bool(np.isfinite(eps) and np.isfinite(score(eps)))
    optimum = minimize_scalar(lambda e: -loglik(e), bounds=fallback_bounds, method="bounded")
    eps = float(optimum.x)
    accepted = (bool(optimum.success) and bool(np.isfinite(eps))
                and abs(score(eps)) <= 0.9 * max(abs(score_at_zero), 1e-12) and loglik(eps) >= loglik(0.0) - 1e-10)
    return eps, accepted


def _path(spec: TargetSpec, folds: list[_Fold], n_obs: int, *, mode: str, max_iter: int, min_abs_eps: float,
          min_score_tol: float, eps_bracket_start: float, eps_bracket_growth: float, eps_bracket_max: float,
          eps_fallback_bounds: tuple[float, float]) -> tuple[list[dict], str, bool]:
    grids = [spec.build(f.fit) for f in folds]
    floor_active = any(_gbar_floor_active(g, f.cache, f.floor, spec.horizon) for g, f in zip(grids, folds))

    def evaluate(grids, eps):
        centred, eif = [], np.empty(n_obs)
        for g, f in zip(grids, folds):
            centred.append(spec.direction(g, f.cache, f.floor))
            eif[f.rows] = _tmle_eif_values(0.0, g, centred[-1], f.data)
        sigma = float(np.std(eif, ddof=1)) if n_obs > 1 else 0.0
        threshold = max(min_score_tol, sigma / (math.sqrt(n_obs) * math.log(n_obs))) if n_obs > 1 else min_score_tol
        psi = float(sum(f.rows.size * spec.value(g) for g, f in zip(grids, folds)) / n_obs)
        return centred, dict(psi=psi, eif=eif, eif_mean=float(np.mean(eif)), sigma=sigma,
                             se=sigma / math.sqrt(n_obs), threshold=threshold, epsilon=float(eps))

    centred, stage = evaluate(grids, 0.0)
    stages = [stage]
    if mode == "auto" and abs(stage["eif_mean"]) <= stage["threshold"]:
        return stages, "initial_score_tolerance", floor_active
    stop = "max_iter"
    for iteration in range(1, (1 if mode == "one_step" else int(max_iter)) + 1):
        eps, accepted = _solve_epsilon(grids, centred, folds, eps_bracket_start, eps_bracket_growth,
                                       eps_bracket_max, eps_fallback_bounds)
        if not accepted:
            stop = "solver_failure"
            break
        grids = [_tilt(g, c, eps) for g, c in zip(grids, centred)]
        centred, stage = evaluate(grids, eps)
        stages.append(stage)
        if mode == "one_step":
            stop = "one_step"
            break
        if abs(eps) < min_abs_eps:
            stop = "epsilon_tiny"
            break
        if abs(stage["eif_mean"]) <= stage["threshold"]:
            stop = "score_tolerance"
            break
    return stages, stop, floor_active


def run_cross_fitted(
    specs: Sequence[TargetSpec],
    initial_estimator: Any,
    observed_data: pd.DataFrame,
    *,
    cross_fit: str,
    mode: str = "iterative",
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    use_floor: bool = True,
    targeting_gbar_floor_scale: Optional[float] = 1.0,
    max_iter: int = 25,
    min_abs_eps: float = 1e-10,
    min_score_tol: float = 1e-8,
    eps_bracket_start: float = 0.25,
    eps_bracket_growth: float = 2.0,
    eps_bracket_max: float = 16.0,
    eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
    api_version: str = "cross_fitted",
) -> dict[str, Any]:
    """Cross-fitted targeting of each spec; the targeting functions' result dict ("summary", "metadata") plus
    "cross_fit" (the label, the folds' validation rows and each observation's final influence-curve value)."""
    cross_fit = resolve_cross_fit(cross_fit)
    if cross_fit == "none":
        raise ValueError("run_cross_fitted needs cross_fit 'tuner', 'refit' or 'nested'")
    if mode not in {"auto", "one_step", "iterative"}:
        raise ValueError("mode must be one of {'auto', 'one_step', 'iterative'}.")
    data = observed_data.reset_index(drop=True)
    fold_fits = resolve_fold_fits(initial_estimator, data, cross_fit)
    folds = _fold_parts(fold_fits, data, clip, targeting_gbar_floor_scale, use_floor)
    full_fit = _build_initial_fit(initial_estimator)
    n_obs = len(data)
    rows, eif_columns = [], []
    for spec in specs:
        stages, stop, floor_active = _path(
            spec, folds, n_obs, mode=mode, max_iter=max_iter, min_abs_eps=min_abs_eps, min_score_tol=min_score_tol,
            eps_bracket_start=eps_bracket_start, eps_bracket_growth=eps_bracket_growth,
            eps_bracket_max=eps_bracket_max, eps_fallback_bounds=eps_fallback_bounds)
        initial, one_step, final = stages[0], stages[1] if len(stages) > 1 else stages[0], stages[-1]
        ci_lower, ci_upper = spec.interval(final["psi"], final["se"])
        row = {"target": spec.target}
        if spec.point_column is not None:
            row[spec.point_column] = spec.point
        row.update({
            "psi_init": float(spec.value(spec.build(full_fit))),
            "psi_init_cross_fitted": initial["psi"],
            "psi_star": final["psi"],
            "epsilon": final["epsilon"],
            "estimand_variance": final["se"] ** 2,
            "standard_error": final["se"],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "eif_mean": final["eif_mean"],
            "sigma": final["sigma"],
            "threshold": final["threshold"],
            "n_iterations": len(stages) - 1,
            "stop_reason": stop,
            "gbar_floor_active": floor_active,
            "cross_fit": fold_fits.source,
            "n_folds": fold_fits.n_folds,
            "n_observations": n_obs,
        })
        for name, stage in (("initial_stage", initial), ("one_step", one_step), ("final", final)):
            row.update({f"psi_{name}": stage["psi"], f"eif_mean_{name}": stage["eif_mean"],
                        f"estimand_variance_{name}": stage["se"] ** 2, f"standard_error_{name}": stage["se"]})
        rows.append(row)
        eif_columns.append(final["eif"])
    return {
        "summary": pd.DataFrame(rows),
        "cross_fit": {"cross_fit": fold_fits.source, "n_folds": fold_fits.n_folds,
                      "validation_rows": [validation.copy() for _, validation in fold_fits.splits],
                      "eif_values": np.column_stack(eif_columns)},
        "metadata": {"api_version": api_version, "target": specs[0].target if specs else None, "mode": mode,
                     "cross_fit": fold_fits.source, "n_folds": fold_fits.n_folds, "n_observations": n_obs,
                     "clip": clip, "survival_clip": survival_clip, "use_floor": use_floor,
                     "targeting_gbar_floor_scale": targeting_gbar_floor_scale, "max_iter": max_iter,
                     "min_abs_eps": min_abs_eps, "min_score_tol": min_score_tol},
    }
