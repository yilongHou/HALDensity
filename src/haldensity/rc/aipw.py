"""Pointwise AIPW estimator of the right-censored survival probability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from haldensity.rc.estimators import RightCensoredInitEstimator
from haldensity.rc.km import KaplanMeier
from haldensity.rc.targeting._common import (
    RCCensoringCache,
    RCInitialFit,
    _build_censoring_cache,
    _build_initial_fit,
    _build_pointwise_target_grid,
    _build_wald_interval,
    _compute_dfg_t0_on_grid,
    _estimate_eic_variance,
    _evaluate_gbar,
    _evaluate_gbar_left,
    _evaluate_survival,
    _normalize_targeting_points,
    _serialize_censoring_cache,
    _serialize_initial_fit,
    _tmle_eif_values,
    _tmle_score_t0,
)


@dataclass
class PointwiseAIPWResult:
    t0: float
    psi_ipcw: float
    psi_aipw_raw: float
    psi_aipw_clipped: float
    if_mean_raw: float
    eic_based_variance: float
    eic_based_se: float
    eic_based_ci_lower_raw: float
    eic_based_ci_upper_raw: float
    eic_based_ci_lower: float
    eic_based_ci_upper: float
    score_at_zero: float
    score_at_zero_over_n: float
    diagnostics: dict[str, Any]


def _target_aipw_t0(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    data: pd.DataFrame,
    t0: float,
    survival_clip: float,
) -> PointwiseAIPWResult:
    target_grid = _build_pointwise_target_grid(initial_fit, t0)
    raw_direction, centered_direction, direction_details = _compute_dfg_t0_on_grid(
        target_grid, censoring_cache, t0, survival_clip
    )
    psi_ipcw = float(_evaluate_survival(target_grid, np.array([t0]), survival_clip)[0])
    eic_values = _tmle_eif_values(0.0, target_grid, centered_direction, data)
    if_mean_raw = float(np.mean(eic_values))
    psi_aipw_raw = float(psi_ipcw + if_mean_raw)
    psi_aipw_clipped = float(np.clip(psi_aipw_raw, 0.0, 1.0))
    eic_based_variance, eic_based_se = _estimate_eic_variance(eic_values)
    z_975 = 1.96
    eic_based_ci_lower_raw = float(psi_aipw_raw - z_975 * eic_based_se)
    eic_based_ci_upper_raw = float(psi_aipw_raw + z_975 * eic_based_se)
    eic_based_ci_lower, eic_based_ci_upper = _build_wald_interval(psi_aipw_raw, eic_based_se)
    score_at_zero = float(_tmle_score_t0(0.0, target_grid, centered_direction, data))
    score_at_zero_over_n = float(score_at_zero / data.shape[0]) if data.shape[0] > 0 else 0.0

    diagnostics = {
        "target_grid_edges": target_grid.grid_edges,
        "target_grid_midpoints": target_grid.grid_midpoints,
        "target_delta_j": target_grid.delta_j,
        "density_before_grid": target_grid.density_grid,
        "survival_before_grid": target_grid.survival_grid,
        "edge_survival_before_grid": target_grid.edge_survival,
        "raw_direction_grid": raw_direction,
        "centered_direction_grid": centered_direction,
        "direction_details": direction_details,
        "eic_values": eic_values,
        "eic_based_variance": eic_based_variance,
        "eic_based_se": eic_based_se,
        "eic_based_ci_lower_raw": eic_based_ci_lower_raw,
        "eic_based_ci_upper_raw": eic_based_ci_upper_raw,
        "eic_based_ci_lower": eic_based_ci_lower,
        "eic_based_ci_upper": eic_based_ci_upper,
        "gbar_right_on_target_grid": _evaluate_gbar(censoring_cache, target_grid.grid_midpoints),
        "gbar_left_on_target_grid": _evaluate_gbar_left(censoring_cache, target_grid.grid_midpoints),
    }
    return PointwiseAIPWResult(
        t0=float(t0),
        psi_ipcw=psi_ipcw,
        psi_aipw_raw=psi_aipw_raw,
        psi_aipw_clipped=psi_aipw_clipped,
        if_mean_raw=if_mean_raw,
        eic_based_variance=float(eic_based_variance),
        eic_based_se=float(eic_based_se),
        eic_based_ci_lower_raw=eic_based_ci_lower_raw,
        eic_based_ci_upper_raw=eic_based_ci_upper_raw,
        eic_based_ci_lower=eic_based_ci_lower,
        eic_based_ci_upper=eic_based_ci_upper,
        score_at_zero=score_at_zero,
        score_at_zero_over_n=score_at_zero_over_n,
        diagnostics=diagnostics,
    )


def _package_pointwise_aipw_fit(
    result: PointwiseAIPWResult,
    theta_selected: np.ndarray,
    store_pointwise_arrays: bool,
) -> dict[str, Any]:
    fit = {
        "t0": result.t0,
        "targeting_points": np.asarray([result.t0], dtype=float),
        "theta_selected": np.asarray(theta_selected, dtype=float).copy(),
        "psi_init": result.psi_ipcw,
        "psi_ipcw": result.psi_ipcw,
        "psi_aipw_raw": result.psi_aipw_raw,
        "psi_aipw_clipped": result.psi_aipw_clipped,
        "if_mean_raw": result.if_mean_raw,
        "estimand_variance": result.eic_based_variance,
        "standard_error": result.eic_based_se,
        "eic_based_variance": result.eic_based_variance,
        "eic_based_se": result.eic_based_se,
        "ci_lower": result.eic_based_ci_lower,
        "ci_upper": result.eic_based_ci_upper,
        "eic_based_ci_lower_raw": result.eic_based_ci_lower_raw,
        "eic_based_ci_upper_raw": result.eic_based_ci_upper_raw,
        "eic_based_ci_lower": result.eic_based_ci_lower,
        "eic_based_ci_upper": result.eic_based_ci_upper,
        "score_at_zero": result.score_at_zero,
        "score_at_zero_over_n": result.score_at_zero_over_n,
        "eif_mean": result.if_mean_raw,
        "abs_eif_mean": abs(result.if_mean_raw),
        "aipw_out_of_bounds": bool((result.psi_aipw_raw < 0.0) or (result.psi_aipw_raw > 1.0)),
        "target_grid_augmented_with_t0": result.diagnostics["direction_details"][
            "target_grid_augmented_with_t0"
        ],
        "raw_direction_mean_before_centering": result.diagnostics["direction_details"]["raw_mean"],
        "gbar_t0_right": result.diagnostics["direction_details"]["gbar_t0_right"],
    }
    if store_pointwise_arrays:
        fit.update(
            {
                "grid_eval": result.diagnostics["target_grid_edges"],
                "grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "delta_j": result.diagnostics["target_delta_j"],
                "estimated_density": result.diagnostics["density_before_grid"],
                "estimated_survival": result.diagnostics["survival_before_grid"],
                "target_grid_edges": result.diagnostics["target_grid_edges"],
                "target_grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "target_delta_j": result.diagnostics["target_delta_j"],
                "raw_direction": result.diagnostics["raw_direction_grid"],
                "centered_direction": result.diagnostics["centered_direction_grid"],
                "eic_values": result.diagnostics["eic_values"],
                "diagnostics": result.diagnostics,
            }
        )
    return fit


class RightCensoredSurvivalAIPWLearner:
    """Pointwise AIPW estimator for right-censored survival targets."""

    def __init__(
        self,
        *,
        clip: float = 1e-6,
        survival_clip: float = 1e-8,
        store_pointwise_arrays: bool = False,
    ) -> None:
        self.clip = float(clip)
        self.survival_clip = float(survival_clip)
        self.store_pointwise_arrays = bool(store_pointwise_arrays)

    def run(
        self,
        initial_estimator: RightCensoredInitEstimator,
        observed_data: pd.DataFrame,
        targeting_points: Union[float, Sequence[float], np.ndarray],
        *,
        km: Optional[KaplanMeier] = None,
        store_pointwise_arrays: Optional[bool] = None,
    ) -> dict[str, Any]:
        if not isinstance(observed_data, pd.DataFrame):
            raise TypeError("observed_data must be a pandas DataFrame.")
        required_cols = {"T", "Delta"}
        missing = required_cols.difference(observed_data.columns)
        if missing:
            raise ValueError(f"observed_data is missing required columns: {sorted(missing)}")

        targeting_points_arr = _normalize_targeting_points(targeting_points)
        if km is None:
            km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")

        initial_fit = _build_initial_fit(initial_estimator)
        censoring_cache = _build_censoring_cache(km, clip=self.clip)
        store_arrays = (
            self.store_pointwise_arrays
            if store_pointwise_arrays is None
            else bool(store_pointwise_arrays)
        )

        results = [
            _target_aipw_t0(
                initial_fit,
                censoring_cache,
                observed_data,
                float(t0),
                survival_clip=self.survival_clip,
            )
            for t0 in targeting_points_arr
        ]

        summary = pd.DataFrame(
            {
                "t0": [r.t0 for r in results],
                "psi_init": [r.psi_ipcw for r in results],
                "psi_ipcw": [r.psi_ipcw for r in results],
                "psi_aipw_raw": [r.psi_aipw_raw for r in results],
                "psi_aipw_clipped": [r.psi_aipw_clipped for r in results],
                "if_mean_raw": [r.if_mean_raw for r in results],
                "estimand_variance": [r.eic_based_variance for r in results],
                "standard_error": [r.eic_based_se for r in results],
                "eic_based_variance": [r.eic_based_variance for r in results],
                "eic_based_se": [r.eic_based_se for r in results],
                "ci_lower": [r.eic_based_ci_lower for r in results],
                "ci_upper": [r.eic_based_ci_upper for r in results],
                "eic_based_ci_lower_raw": [r.eic_based_ci_lower_raw for r in results],
                "eic_based_ci_upper_raw": [r.eic_based_ci_upper_raw for r in results],
                "eic_based_ci_lower": [r.eic_based_ci_lower for r in results],
                "eic_based_ci_upper": [r.eic_based_ci_upper for r in results],
                "score_at_zero": [r.score_at_zero for r in results],
                "score_at_zero_over_n": [r.score_at_zero_over_n for r in results],
                "eif_mean": [r.if_mean_raw for r in results],
                "abs_eif_mean": [abs(r.if_mean_raw) for r in results],
                "aipw_out_of_bounds": [
                    bool((r.psi_aipw_raw < 0.0) or (r.psi_aipw_raw > 1.0)) for r in results
                ],
                "gbar_t0_right": [
                    r.diagnostics["direction_details"]["gbar_t0_right"] for r in results
                ],
                "target_grid_augmented_with_t0": [
                    r.diagnostics["direction_details"]["target_grid_augmented_with_t0"]
                    for r in results
                ],
                "raw_direction_mean_before_centering": [
                    r.diagnostics["direction_details"]["raw_mean"] for r in results
                ],
            }
        ).sort_values("t0").reset_index(drop=True)

        return {
            "targeting_points": targeting_points_arr,
            "summary": summary,
            "pointwise_fits": [
                _package_pointwise_aipw_fit(
                    r,
                    theta_selected=initial_fit.theta_hat,
                    store_pointwise_arrays=store_arrays,
                )
                for r in results
            ],
            "initial_fit": _serialize_initial_fit(initial_fit),
            "censoring_cache": _serialize_censoring_cache(censoring_cache),
            "metadata": {
                "api_version": "aipw_v1",
                "n_targets": int(targeting_points_arr.size),
                "n_observations": int(observed_data.shape[0]),
                "clip": self.clip,
                "survival_clip": self.survival_clip,
                "store_pointwise_arrays": store_arrays,
                "pointwise_design": True,
                "estimator_family": "aipw",
            },
        }

    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        observed_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> np.ndarray:
        _ = observed_data
        _ = kwargs
        summary = targeted_fit["summary"]
        return summary["eic_based_variance"].to_numpy(dtype=float)


def right_censored_survival_aipw(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    targeting_points: Union[float, Sequence[float], np.ndarray],
    *,
    km: Optional[KaplanMeier] = None,
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    store_pointwise_arrays: bool = False,
) -> dict[str, Any]:
    learner = RightCensoredSurvivalAIPWLearner(
        clip=clip,
        survival_clip=survival_clip,
        store_pointwise_arrays=store_pointwise_arrays,
    )
    return learner.run(
        initial_estimator=initial_estimator,
        observed_data=observed_data,
        targeting_points=targeting_points,
        km=km,
    )


def right_censored_survival_aipw_estimand_variance(
    targeted_fit: dict[str, Any],
    observed_data: Optional[pd.DataFrame] = None,
    targeting_points: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    **kwargs,
) -> np.ndarray:
    _ = targeting_points
    learner = RightCensoredSurvivalAIPWLearner(**kwargs)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        observed_data=observed_data,
    )
