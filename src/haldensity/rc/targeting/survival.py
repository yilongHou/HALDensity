"""Right-censored survival-probability TMLE (pointwise, one-step and iterative)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from haldensity.rc.estimators import RightCensoredInitEstimator
from haldensity.rc.km import KaplanMeier, last_time_as_event
from haldensity.rc.targeting.eic import _compute_exact_survival_eif
from haldensity.rc.targeting._common import (
    PointwiseTMLEState,
    RCCensoringCache,
    RCInitialFit,
    _build_censoring_cache,
    _build_initial_fit,
    _build_pointwise_target_grid,
    _build_wald_interval,
    _compute_dfg_t0_on_grid,
    _compute_targeting_gbar_floor,
    _estimate_eic_variance,
    _estimate_eic_variance_from_sigma,
    _evaluate_gbar,
    _evaluate_gbar_left,
    _evaluate_survival,
    _interp_survival_from_edges,
    _iterative_tolerance_from_sigma,
    _normalize_targeting_points,
    _select_variance_column,
    _serialize_censoring_cache,
    _serialize_initial_fit,
    _solve_epsilon_t0,
    _state_to_target_grid,
    _tilted_density_and_survival,
    _tmle_eif_values,
    _tmle_loglik_t0,
    _tmle_score_t0,
)


@dataclass
class PointwiseTMLEResult:
    t0: float
    psi_init: float
    psi_star: float
    epsilon: float
    estimand_variance: float
    standard_error: float
    score_at_zero: float
    score_at_solution: float
    objective_at_solution: float
    eif_mean: float
    solve_method: str
    converged: bool
    diagnostics: dict[str, Any]


def _target_survival_t0(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    data: pd.DataFrame,
    t0: float,
    survival_clip: float,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
    compute_diagnostic_curves: bool,
) -> PointwiseTMLEResult:
    target_grid = _build_pointwise_target_grid(initial_fit, t0)
    raw_direction, centered_direction, direction_details = _compute_dfg_t0_on_grid(
        target_grid, censoring_cache, t0, survival_clip
    )
    psi_init = float(_evaluate_survival(target_grid, np.array([t0]), survival_clip)[0])
    initial_eic_values = _tmle_eif_values(0.0, target_grid, centered_direction, data)
    initial_eif_mean = float(np.mean(initial_eic_values))
    initial_estimand_variance, initial_standard_error = _estimate_eic_variance(initial_eic_values)
    epsilon_hat, converged, solver_details = _solve_epsilon_t0(
        target_grid,
        centered_direction,
        data,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
    )
    density_star, survival_star, edge_survival_star, normalizer = _tilted_density_and_survival(
        target_grid, centered_direction, epsilon_hat
    )
    psi_star = float(
        _interp_survival_from_edges(target_grid.grid_edges, edge_survival_star, np.array([t0]))[0]
    )
    score_at_solution = _tmle_score_t0(epsilon_hat, target_grid, centered_direction, data)
    objective_at_solution = _tmle_loglik_t0(epsilon_hat, target_grid, centered_direction, data)
    one_step_eic_values = _tmle_eif_values(epsilon_hat, target_grid, centered_direction, data)
    one_step_estimand_variance, one_step_standard_error = _estimate_eic_variance(
        one_step_eic_values
    )
    ci_lower, ci_upper = _build_wald_interval(psi_star, initial_standard_error)
    ci_lower_one_step, ci_upper_one_step = _build_wald_interval(psi_star, one_step_standard_error)
    if compute_diagnostic_curves:
        eps_window = max(1.0, abs(epsilon_hat) + 0.5)
        diagnostic_eps_grid = np.linspace(epsilon_hat - eps_window, epsilon_hat + eps_window, 161)
        diagnostic_loglik = np.array(
            [_tmle_loglik_t0(eps, target_grid, centered_direction, data) for eps in diagnostic_eps_grid],
            dtype=float,
        )
        diagnostic_score = np.array(
            [_tmle_score_t0(eps, target_grid, centered_direction, data) for eps in diagnostic_eps_grid],
            dtype=float,
        )
    else:
        diagnostic_eps_grid = np.empty(0, dtype=float)
        diagnostic_loglik = np.empty(0, dtype=float)
        diagnostic_score = np.empty(0, dtype=float)

    diagnostics = {
        "target_grid_edges": target_grid.grid_edges,
        "target_grid_midpoints": target_grid.grid_midpoints,
        "target_delta_j": target_grid.delta_j,
        "density_before_grid": target_grid.density_grid,
        "raw_direction_grid": raw_direction,
        "centered_direction_grid": centered_direction,
        "density_star_grid": density_star,
        "survival_star_grid": survival_star,
        "edge_survival_star_grid": edge_survival_star,
        "normalizer": normalizer,
        "direction_details": direction_details,
        "solver_details": solver_details,
        "eif_like_values": one_step_eic_values,
        "eif_values_initial_stage": initial_eic_values,
        "eif_values_one_step": one_step_eic_values,
        "eif_mean_initial_stage": initial_eif_mean,
        "estimand_variance": initial_estimand_variance,
        "standard_error": initial_standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "estimand_variance_initial_stage": initial_estimand_variance,
        "standard_error_initial_stage": initial_standard_error,
        "estimand_variance_one_step": one_step_estimand_variance,
        "standard_error_one_step": one_step_standard_error,
        "ci_lower_one_step": ci_lower_one_step,
        "ci_upper_one_step": ci_upper_one_step,
        "gbar_right_on_target_grid": _evaluate_gbar(censoring_cache, target_grid.grid_midpoints),
        "gbar_left_on_target_grid": _evaluate_gbar_left(censoring_cache, target_grid.grid_midpoints),
        "diagnostic_eps_grid": diagnostic_eps_grid,
        "diagnostic_loglik_values": diagnostic_loglik,
        "diagnostic_score_values": diagnostic_score,
    }
    return PointwiseTMLEResult(
        t0=float(t0),
        psi_init=psi_init,
        psi_star=psi_star,
        epsilon=epsilon_hat,
        estimand_variance=initial_estimand_variance,
        standard_error=initial_standard_error,
        score_at_zero=float(solver_details["score_at_zero"]),
        score_at_solution=float(score_at_solution),
        objective_at_solution=float(objective_at_solution),
        eif_mean=float(np.mean(one_step_eic_values)),
        solve_method=str(solver_details["method"]),
        converged=converged,
        diagnostics=diagnostics,
    )


def _package_pointwise_fit(
    result: PointwiseTMLEResult,
    theta_selected: np.ndarray,
    store_pointwise_arrays: bool,
) -> dict[str, Any]:
    fit = {
        "t0": result.t0,
        "targeting_points": np.asarray([result.t0], dtype=float),
        "theta_targeting": np.asarray([result.epsilon], dtype=float),
        "theta_selected": np.asarray(theta_selected, dtype=float).copy(),
        "epsilon": result.epsilon,
        "psi_init": result.psi_init,
        "psi_star": result.psi_star,
        "estimand_variance": result.estimand_variance,
        "standard_error": result.standard_error,
        "ci_lower": result.diagnostics["ci_lower"],
        "ci_upper": result.diagnostics["ci_upper"],
        "score_at_zero": result.score_at_zero,
        "score_at_solution": result.score_at_solution,
        "objective_at_solution": result.objective_at_solution,
        "eif_mean": result.eif_mean,
        "solve_method": result.solve_method,
        "converged": result.converged,
        "psi_initial_stage": result.psi_init,
        "eif_mean_initial_stage": result.diagnostics["eif_mean_initial_stage"],
        "estimand_variance_initial_stage": result.diagnostics["estimand_variance_initial_stage"],
        "standard_error_initial_stage": result.diagnostics["standard_error_initial_stage"],
        "psi_one_step": result.psi_star,
        "eif_mean_one_step": result.eif_mean,
        "estimand_variance_one_step": result.diagnostics["estimand_variance_one_step"],
        "standard_error_one_step": result.diagnostics["standard_error_one_step"],
        "score_at_solution_one_step": result.score_at_solution,
        "target_grid_augmented_with_t0": result.diagnostics["direction_details"][
            "target_grid_augmented_with_t0"
        ],
    }
    if store_pointwise_arrays:
        fit.update(
            {
                "grid_eval": result.diagnostics["target_grid_edges"],
                "grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "delta_j": result.diagnostics["target_delta_j"],
                "estimated_density": result.diagnostics["density_star_grid"],
                "estimated_survival": result.diagnostics["survival_star_grid"],
                "targeted_survival_grid": result.diagnostics["survival_star_grid"],
                "target_grid_edges": result.diagnostics["target_grid_edges"],
                "target_grid_midpoints": result.diagnostics["target_grid_midpoints"],
                "target_delta_j": result.diagnostics["target_delta_j"],
                "density_before_grid": result.diagnostics["density_before_grid"],
                "raw_direction": result.diagnostics["raw_direction_grid"],
                "centered_direction": result.diagnostics["centered_direction_grid"],
                "eic_values": result.diagnostics["eif_like_values"],
                "eic_values_initial_stage": result.diagnostics["eif_values_initial_stage"],
                "eic_values_one_step": result.diagnostics["eif_values_one_step"],
                "diagnostics": result.diagnostics,
            }
        )
    return fit


def _initialize_pointwise_state(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    t0: float,
) -> PointwiseTMLEState:
    target_grid = _build_pointwise_target_grid(initial_fit, float(t0))
    psi_init = float(
        _interp_survival_from_edges(target_grid.grid_edges, target_grid.edge_survival, np.array([t0]))[0]
    )
    return PointwiseTMLEState(
        t0=float(t0),
        grid_edges=target_grid.grid_edges.copy(),
        grid_midpoints=target_grid.grid_midpoints.copy(),
        delta_j=target_grid.delta_j.copy(),
        log_density_grid=target_grid.log_density_grid.copy(),
        density_grid=target_grid.density_grid.copy(),
        survival_grid=target_grid.survival_grid.copy(),
        edge_survival=target_grid.edge_survival.copy(),
        psi_current=psi_init,
        target_grid_augmented_with_t0=bool(target_grid.t0_inserted),
        censoring_cache=censoring_cache,
    )


def _evaluate_pointwise_state(
    state: PointwiseTMLEState,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> tuple[PointwiseTMLEState, dict[str, Any]]:
    target_grid = _state_to_target_grid(state)
    raw_direction, centered_direction, direction_details = _compute_dfg_t0_on_grid(
        target_grid,
        state.censoring_cache,
        state.t0,
        survival_clip,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    eif_values = _tmle_eif_values(0.0, target_grid, centered_direction, observed_data)
    eif_mean = float(np.mean(eif_values))
    eif_sigma = float(np.std(eif_values, ddof=1)) if eif_values.size > 1 else 0.0
    score_at_zero = float(_tmle_score_t0(0.0, target_grid, centered_direction, observed_data))
    psi_current = float(
        _interp_survival_from_edges(state.grid_edges, state.edge_survival, np.array([state.t0]))[0]
    )
    evaluated_state = PointwiseTMLEState(
        t0=state.t0,
        grid_edges=state.grid_edges.copy(),
        grid_midpoints=state.grid_midpoints.copy(),
        delta_j=state.delta_j.copy(),
        log_density_grid=state.log_density_grid.copy(),
        density_grid=state.density_grid.copy(),
        survival_grid=state.survival_grid.copy(),
        edge_survival=state.edge_survival.copy(),
        psi_current=psi_current,
        target_grid_augmented_with_t0=state.target_grid_augmented_with_t0,
        censoring_cache=state.censoring_cache,
        raw_direction=raw_direction,
        centered_direction=centered_direction,
        eif_values=eif_values,
        eif_mean=eif_mean,
        eif_sigma=eif_sigma,
        score_at_zero=score_at_zero,
    )
    return evaluated_state, {"direction_details": direction_details}


def _compute_survival_exact_stage_diagnostics(
    state: PointwiseTMLEState,
    observed_data: pd.DataFrame,
    *,
    survival_clip: float,
    targeting_gbar_floor: float,
    n_obs: int,
    min_score_tol: float,
) -> dict[str, float]:
    exact = _compute_exact_survival_eif(
        target_grid=_state_to_target_grid(state),
        censoring_cache=state.censoring_cache,
        t0=state.t0,
        observed_data=observed_data,
        survival_clip=survival_clip,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    _, threshold = _iterative_tolerance_from_sigma(
        float(exact["eif_sigma"]),
        n_obs,
        min_score_tol,
    )
    return {
        "eif_mean": float(exact["eif_mean"]),
        "sigma": float(exact["eif_sigma"]),
        "threshold": float(threshold),
    }


def _one_local_tmle_update(
    current_state: PointwiseTMLEState,
    observed_data: pd.DataFrame,
    *,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> tuple[PointwiseTMLEState, dict[str, Any]]:
    if current_state.centered_direction is None:
        raise ValueError("current_state must be evaluated before calling _one_local_tmle_update.")

    target_grid = _state_to_target_grid(current_state)
    epsilon_hat, converged_inner, solver_details = _solve_epsilon_t0(
        target_grid,
        current_state.centered_direction,
        observed_data,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
    )
    score_at_solution = float(
        _tmle_score_t0(epsilon_hat, target_grid, current_state.centered_direction, observed_data)
    )
    objective_at_zero = float(
        _tmle_loglik_t0(0.0, target_grid, current_state.centered_direction, observed_data)
    )
    objective_at_solution = float(
        _tmle_loglik_t0(epsilon_hat, target_grid, current_state.centered_direction, observed_data)
    )
    solve_method = str(solver_details["method"])

    finite_update = bool(np.all(np.isfinite([epsilon_hat, score_at_solution, objective_at_solution])))
    accepted_update = bool(finite_update and converged_inner)
    if solve_method == "bounded_fallback":
        score_improved = abs(score_at_solution) <= 0.9 * max(abs(current_state.score_at_zero), 1e-12)
        objective_improved = objective_at_solution >= objective_at_zero - 1e-10
        accepted_update = bool(accepted_update and score_improved and objective_improved)

    if not accepted_update:
        return current_state, {
            "epsilon": float(epsilon_hat),
            "score_at_solution": score_at_solution,
            "objective_at_solution": objective_at_solution,
            "solve_method": solve_method,
            "converged_inner": bool(converged_inner),
            "accepted_update": False,
        }

    density_star, survival_star, edge_survival_star, _ = _tilted_density_and_survival(
        target_grid,
        current_state.centered_direction,
        epsilon_hat,
    )
    updated_state = PointwiseTMLEState(
        t0=current_state.t0,
        grid_edges=current_state.grid_edges.copy(),
        grid_midpoints=current_state.grid_midpoints.copy(),
        delta_j=current_state.delta_j.copy(),
        log_density_grid=np.log(np.clip(density_star, 1e-300, None)),
        density_grid=density_star,
        survival_grid=survival_star,
        edge_survival=edge_survival_star,
        psi_current=float(
            _interp_survival_from_edges(
                current_state.grid_edges, edge_survival_star, np.array([current_state.t0])
            )[0]
        ),
        target_grid_augmented_with_t0=current_state.target_grid_augmented_with_t0,
        censoring_cache=current_state.censoring_cache,
    )
    return updated_state, {
        "epsilon": float(epsilon_hat),
        "score_at_solution": score_at_solution,
        "objective_at_solution": objective_at_solution,
        "solve_method": solve_method,
        "converged_inner": bool(converged_inner),
        "accepted_update": True,
    }


def _iterate_tmle_t0(
    initial_fit: RCInitialFit,
    censoring_cache: RCCensoringCache,
    observed_data: pd.DataFrame,
    t0: float,
    *,
    survival_clip: float,
    targeting_gbar_floor: float,
    max_iter: int,
    min_abs_eps: float,
    min_score_tol: float,
    eps_bracket_start: float,
    eps_bracket_growth: float,
    eps_bracket_max: float,
    eps_fallback_bounds: tuple[float, float],
) -> dict[str, Any]:
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1 for iterative TMLE.")

    state = _initialize_pointwise_state(initial_fit, censoring_cache, t0)
    psi_init = float(state.psi_current)
    history_rows: list[dict[str, Any]] = []
    n_obs = observed_data.shape[0]
    stop_reason = "max_iter"

    for iteration in range(1, max_iter + 1):
        state, _ = _evaluate_pointwise_state(
            state,
            observed_data,
            survival_clip,
            targeting_gbar_floor,
        )
        updated_state, update_info = _one_local_tmle_update(
            state,
            observed_data,
            eps_bracket_start=eps_bracket_start,
            eps_bracket_growth=eps_bracket_growth,
            eps_bracket_max=eps_bracket_max,
            eps_fallback_bounds=eps_fallback_bounds,
        )
        row = {
            "iteration": iteration,
            "psi_before_update": state.psi_current,
            "eif_mean_before_update": state.eif_mean,
            "sigma_before_update": state.eif_sigma,
            "score_at_zero_before_update": state.score_at_zero,
            "epsilon": float("nan"),
            "score_at_solution": float("nan"),
            "objective_at_solution": float("nan"),
            "solve_method": None,
            "converged_inner": None,
            "accepted_update": False,
            "psi": state.psi_current,
            "eif_mean": state.eif_mean,
            "sigma": state.eif_sigma,
            "sigma_over_sqrtnlogn": float("nan"),
            "stop_tolerance": float("nan"),
            "score_at_zero": state.score_at_zero,
            "status": "evaluated",
        }
        row.update(update_info)

        if not update_info["accepted_update"]:
            row["status"] = "solver_failure"
            history_rows.append(row)
            stop_reason = "solver_failure"
            break

        state = updated_state
        state, _ = _evaluate_pointwise_state(
            state,
            observed_data,
            survival_clip,
            targeting_gbar_floor,
        )
        raw_tolerance, tolerance = _iterative_tolerance_from_sigma(
            state.eif_sigma,
            n_obs,
            min_score_tol,
        )
        row.update(
            {
                "psi": state.psi_current,
                "eif_mean": state.eif_mean,
                "sigma": state.eif_sigma,
                "sigma_over_sqrtnlogn": raw_tolerance,
                "stop_tolerance": tolerance,
                "score_at_zero": state.score_at_zero,
            }
        )

        if abs(float(update_info["epsilon"])) < min_abs_eps:
            row["status"] = "epsilon_tiny"
            history_rows.append(row)
            stop_reason = "epsilon_tiny"
            break
        if abs(state.eif_mean) <= tolerance:
            row["status"] = "score_tolerance"
            history_rows.append(row)
            stop_reason = "score_tolerance"
            break
        if iteration == max_iter:
            row["status"] = "max_iter"
            history_rows.append(row)
            stop_reason = "max_iter"
            break

        row["status"] = "continue"
        history_rows.append(row)

    final_state = state
    if final_state.eif_values is None:
        raise RuntimeError("Iterative TMLE final state is missing EIC values.")
    estimand_variance, standard_error = _estimate_eic_variance(final_state.eif_values)
    history = pd.DataFrame(history_rows)
    epsilon_path = (
        history.loc[history["accepted_update"], "epsilon"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    score_at_solution_path = (
        history.loc[history["accepted_update"], "score_at_solution"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    objective_at_solution_path = (
        history.loc[history["accepted_update"], "objective_at_solution"].to_numpy(dtype=float).tolist()
        if len(history) > 0
        else []
    )
    solve_method_path = (
        history.loc[history["accepted_update"], "solve_method"].astype(str).tolist()
        if len(history) > 0
        else []
    )
    raw_tolerance_final, threshold_final = _iterative_tolerance_from_sigma(
        final_state.eif_sigma,
        n_obs,
        min_score_tol,
    )
    return {
        "summary": {
            "t0": float(t0),
            "psi_init": psi_init,
            "psi_star": float(final_state.psi_current),
            "eif_mean": float(final_state.eif_mean),
            "estimand_variance": float(estimand_variance),
            "standard_error": float(standard_error),
            "score_at_zero": float(final_state.score_at_zero),
            "score_at_solution_last": (
                float(score_at_solution_path[-1]) if len(score_at_solution_path) > 0 else float("nan")
            ),
            "objective_at_solution_last": (
                float(objective_at_solution_path[-1])
                if len(objective_at_solution_path) > 0
                else float("nan")
            ),
            "solve_method_last": solve_method_path[-1] if len(solve_method_path) > 0 else "none",
            "n_iterations": int(np.sum(history["accepted_update"])) if len(history) > 0 else 0,
            "stop_reason": stop_reason,
            "epsilon_path": epsilon_path,
            "epsilon_last": float(epsilon_path[-1]) if len(epsilon_path) > 0 else float("nan"),
            "eif_mean_path": history["eif_mean"].to_numpy(dtype=float).tolist()
            if len(history) > 0
            else [float(final_state.eif_mean)],
            "sigma_path": history["sigma"].to_numpy(dtype=float).tolist()
            if len(history) > 0
            else [float(final_state.eif_sigma)],
            "psi_path": [psi_init]
            + (
                history["psi"].to_numpy(dtype=float).tolist()
                if len(history) > 0
                else [float(final_state.psi_current)]
            ),
            "sigma_final": float(final_state.eif_sigma),
            "sigma_over_sqrtnlogn_final": float(raw_tolerance_final),
            "threshold_final": float(threshold_final),
            "n_observations": int(n_obs),
        },
        "history": history,
        "final_state": final_state,
    }


def _summarize_iterative_tmle_step(
    iter_bundle: dict[str, Any],
    step: int,
    *,
    min_score_tol: float,
) -> dict[str, Any]:
    history = iter_bundle["history"]
    step_row = history.loc[history["iteration"] == int(step)]
    if len(step_row) != 1:
        raise ValueError(f"Expected exactly one row for step={step}, found {len(step_row)}.")

    row = step_row.iloc[0]
    n_obs = int(iter_bundle["summary"]["n_observations"])
    raw_tolerance, threshold = _iterative_tolerance_from_sigma(
        float(row["sigma"]),
        n_obs,
        min_score_tol,
    )
    estimand_variance, standard_error = _estimate_eic_variance_from_sigma(
        float(row["sigma"]),
        n_obs,
    )
    return {
        "t0": float(iter_bundle["summary"]["t0"]),
        "step": int(step),
        "psi_star": float(row["psi"]),
        "eif_mean": float(row["eif_mean"]),
        "sigma": float(row["sigma"]),
        "estimand_variance": float(estimand_variance),
        "standard_error": float(standard_error),
        "sigma_over_sqrtnlogn": float(raw_tolerance),
        "threshold": float(threshold),
        "epsilon": float(row["epsilon"]) if np.isfinite(row["epsilon"]) else float("nan"),
        "score_at_zero": float(row["score_at_zero"]),
        "score_at_solution": (
            float(row["score_at_solution"]) if np.isfinite(row["score_at_solution"]) else float("nan")
        ),
        "objective_at_solution": (
            float(row["objective_at_solution"])
            if np.isfinite(row["objective_at_solution"])
            else float("nan")
        ),
        "solve_method": str(row["solve_method"]) if row["solve_method"] is not None else "none",
        "status": str(row["status"]),
    }


class RightCensoredSurvivalTargetLearner:
    """Pointwise HAL-TMLE for right-censored survival targets."""

    def __init__(
        self,
        *,
        clip: float = 1e-6,
        survival_clip: float = 1e-8,
        eps_bracket_start: float = 0.25,
        eps_bracket_growth: float = 2.0,
        eps_bracket_max: float = 16.0,
        eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
        store_pointwise_arrays: bool = False,
    ) -> None:
        self.clip = float(clip)
        self.survival_clip = float(survival_clip)
        self.eps_bracket_start = float(eps_bracket_start)
        self.eps_bracket_growth = float(eps_bracket_growth)
        self.eps_bracket_max = float(eps_bracket_max)
        self.eps_fallback_bounds = (
            float(eps_fallback_bounds[0]),
            float(eps_fallback_bounds[1]),
        )
        self.store_pointwise_arrays = bool(store_pointwise_arrays)

    def run_m_step(
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
        observed_data = last_time_as_event(observed_data)

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
            _target_survival_t0(
                initial_fit,
                censoring_cache,
                observed_data,
                float(t0),
                survival_clip=self.survival_clip,
                eps_bracket_start=self.eps_bracket_start,
                eps_bracket_growth=self.eps_bracket_growth,
                eps_bracket_max=self.eps_bracket_max,
                eps_fallback_bounds=self.eps_fallback_bounds,
                compute_diagnostic_curves=store_arrays,
            )
            for t0 in targeting_points_arr
        ]

        summary = pd.DataFrame(
            {
                "t0": [r.t0 for r in results],
                "psi_init": [r.psi_init for r in results],
                "psi_star": [r.psi_star for r in results],
                "epsilon": [r.epsilon for r in results],
                "estimand_variance": [r.estimand_variance for r in results],
                "standard_error": [r.standard_error for r in results],
                "ci_lower": [r.diagnostics["ci_lower"] for r in results],
                "ci_upper": [r.diagnostics["ci_upper"] for r in results],
                "score_at_zero": [r.score_at_zero for r in results],
                "score_at_solution": [r.score_at_solution for r in results],
                "objective_at_solution": [r.objective_at_solution for r in results],
                "eif_mean": [r.eif_mean for r in results],
                "psi_initial_stage": [r.psi_init for r in results],
                "eif_mean_initial_stage": [
                    r.diagnostics["eif_mean_initial_stage"] for r in results
                ],
                "estimand_variance_initial_stage": [
                    r.diagnostics["estimand_variance_initial_stage"] for r in results
                ],
                "standard_error_initial_stage": [
                    r.diagnostics["standard_error_initial_stage"] for r in results
                ],
                "psi_one_step": [r.psi_star for r in results],
                "eif_mean_one_step": [r.eif_mean for r in results],
                "estimand_variance_one_step": [
                    r.diagnostics["estimand_variance_one_step"] for r in results
                ],
                "standard_error_one_step": [
                    r.diagnostics["standard_error_one_step"] for r in results
                ],
                "score_at_solution_one_step": [r.score_at_solution for r in results],
                "solve_method": [r.solve_method for r in results],
                "converged": [r.converged for r in results],
                "target_grid_augmented_with_t0": [
                    r.diagnostics["direction_details"]["target_grid_augmented_with_t0"]
                    for r in results
                ],
            }
        )

        return {
            "targeting_points": targeting_points_arr,
            "summary": summary,
            "pointwise_fits": [
                _package_pointwise_fit(
                    r,
                    theta_selected=initial_fit.theta_hat,
                    store_pointwise_arrays=store_arrays,
                )
                for r in results
            ],
            "initial_fit": _serialize_initial_fit(initial_fit),
            "censoring_cache": _serialize_censoring_cache(censoring_cache),
            "metadata": {
                "n_targets": int(targeting_points_arr.size),
                "n_observations": int(observed_data.shape[0]),
                "clip": self.clip,
                "survival_clip": self.survival_clip,
                "eps_bracket_start": self.eps_bracket_start,
                "eps_bracket_growth": self.eps_bracket_growth,
                "eps_bracket_max": self.eps_bracket_max,
                "eps_fallback_bounds": self.eps_fallback_bounds,
                "store_pointwise_arrays": store_arrays,
                "pointwise_design": True,
            },
        }

    def get_estimand_variance(
        self,
        targeted_fit: dict[str, Any],
        observed_data: Optional[pd.DataFrame] = None,
        which: str = "default",
        **kwargs,
    ) -> np.ndarray:
        _ = observed_data
        _ = kwargs
        summary = targeted_fit["summary"]
        column = _select_variance_column(summary, which=which)
        return summary[column].to_numpy(dtype=float)


def right_censored_survival_targeting_M_step(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    targeting_points: Union[float, Sequence[float], np.ndarray],
    *,
    km: Optional[KaplanMeier] = None,
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    eps_bracket_start: float = 0.25,
    eps_bracket_growth: float = 2.0,
    eps_bracket_max: float = 16.0,
    eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
    store_pointwise_arrays: bool = False,
) -> dict[str, Any]:
    learner = RightCensoredSurvivalTargetLearner(
        clip=clip,
        survival_clip=survival_clip,
        eps_bracket_start=eps_bracket_start,
        eps_bracket_growth=eps_bracket_growth,
        eps_bracket_max=eps_bracket_max,
        eps_fallback_bounds=eps_fallback_bounds,
        store_pointwise_arrays=store_pointwise_arrays,
    )
    return learner.run_m_step(
        initial_estimator=initial_estimator,
        observed_data=observed_data,
        targeting_points=targeting_points,
        km=km,
    )


def right_censored_survival_estimand_variance(
    targeted_fit: dict[str, Any],
    observed_data: Optional[pd.DataFrame] = None,
    targeting_points: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    which: str = "default",
    **kwargs,
) -> np.ndarray:
    _ = targeting_points
    learner = RightCensoredSurvivalTargetLearner(**kwargs)
    return learner.get_estimand_variance(
        targeted_fit=targeted_fit,
        observed_data=observed_data,
        which=which,
    )


def right_censored_survival_targeting_M_step_v2(
    initial_estimator: RightCensoredInitEstimator,
    observed_data: pd.DataFrame,
    targeting_points: Union[float, Sequence[float], np.ndarray],
    *,
    km: Optional[KaplanMeier] = None,
    mode: str = "iterative",
    one_step_eif_gate: float = 1e-8,
    clip: float = 1e-6,
    survival_clip: float = 1e-8,
    eps_bracket_start: float = 0.25,
    eps_bracket_growth: float = 2.0,
    eps_bracket_max: float = 16.0,
    eps_fallback_bounds: tuple[float, float] = (-20.0, 20.0),
    max_iter: int = 25,
    min_abs_eps: float = 1e-10,
    min_score_tol: float = 1e-8,
    store_pointwise_arrays: bool = False,
    targeting_gbar_floor_scale: Optional[float] = 1.0,
) -> dict[str, Any]:
    if mode not in {"auto", "one_step", "iterative"}:
        raise ValueError("mode must be one of {'auto', 'one_step', 'iterative'}.")
    if not isinstance(observed_data, pd.DataFrame):
        raise TypeError("observed_data must be a pandas DataFrame.")
    required_cols = {"T", "Delta"}
    missing = required_cols.difference(observed_data.columns)
    if missing:
        raise ValueError(f"observed_data is missing required columns: {sorted(missing)}")
    observed_data = last_time_as_event(observed_data)

    requested_mode = mode
    targeting_points_arr = _normalize_targeting_points(targeting_points)
    if km is None:
        km = KaplanMeier().fit(observed_data, time_col="T", delta_col="Delta")

    initial_fit = _build_initial_fit(initial_estimator)
    censoring_cache = _build_censoring_cache(km, clip=clip)
    n_obs = int(observed_data.shape[0])
    targeting_gbar_floor = _compute_targeting_gbar_floor(
        n_obs=n_obs,
        scale=targeting_gbar_floor_scale,
        hard_min=clip,
    )

    summary_rows: list[dict[str, Any]] = []
    pointwise_fits: list[dict[str, Any]] = []
    for t0 in targeting_points_arr:
        init_state = _initialize_pointwise_state(initial_fit, censoring_cache, float(t0))
        init_state, _ = _evaluate_pointwise_state(
            init_state,
            observed_data,
            survival_clip,
            targeting_gbar_floor,
        )
        if init_state.eif_values is None:
            raise RuntimeError("Initial iterative TMLE state is missing EIC values.")
        init_var, init_se = _estimate_eic_variance(init_state.eif_values)
        raw_init_tolerance, threshold_initial = _iterative_tolerance_from_sigma(
            init_state.eif_sigma,
            n_obs,
            min_score_tol,
        )
        passes_initial_threshold = bool(abs(float(init_state.eif_mean)) <= threshold_initial)

        if mode == "auto" and passes_initial_threshold:
            history_df = pd.DataFrame()
            final_state = init_state
            final_summary = None
            final_psi = float(init_state.psi_current)
            final_eif_mean = float(init_state.eif_mean)
            final_var = float(init_var)
            final_se = float(init_se)
            final_score_at_zero = float(init_state.score_at_zero)
            final_stop_reason = "initial_score_tolerance"
            final_n_iterations = 0
            final_epsilon = 0.0
            final_score_at_solution = float(init_state.score_at_zero)
            final_solve_method = "skipped_initial_gate"
            target_grid_init = _state_to_target_grid(init_state)
            final_objective_at_solution = float(
                _tmle_loglik_t0(
                    0.0,
                    target_grid_init,
                    init_state.centered_direction,
                    observed_data,
                )
            )
            final_sigma = float(init_state.eif_sigma)
            final_raw_tolerance = float(raw_init_tolerance)
            final_threshold = float(threshold_initial)
            final_epsilon_path: list[float] = []
            final_eif_mean_path: list[float] = [float(init_state.eif_mean)]
            final_sigma_path: list[float] = [float(init_state.eif_sigma)]
            final_psi_path: list[float] = [float(init_state.psi_current)]
            one_step_state = init_state
            one_step_stage = {
                "psi_star": float(init_state.psi_current),
                "eif_mean": float(init_state.eif_mean),
                "sigma": float(init_state.eif_sigma),
                "estimand_variance": float(init_var),
                "standard_error": float(init_se),
                "sigma_over_sqrtnlogn": float(raw_init_tolerance),
                "threshold": float(threshold_initial),
                "epsilon": 0.0,
                "score_at_zero": float(init_state.score_at_zero),
                "score_at_solution": float(init_state.score_at_zero),
                "objective_at_solution": float(final_objective_at_solution),
                "solve_method": "skipped_initial_gate",
                "status": "skipped_initial_gate",
            }
        else:
            one_step_bundle = _iterate_tmle_t0(
                initial_fit=initial_fit,
                censoring_cache=censoring_cache,
                observed_data=observed_data,
                t0=float(t0),
                survival_clip=survival_clip,
                targeting_gbar_floor=targeting_gbar_floor,
                max_iter=1,
                min_abs_eps=min_abs_eps,
                min_score_tol=min_score_tol,
                eps_bracket_start=eps_bracket_start,
                eps_bracket_growth=eps_bracket_growth,
                eps_bracket_max=eps_bracket_max,
                eps_fallback_bounds=eps_fallback_bounds,
            )
            one_step_state = one_step_bundle["final_state"]
            one_step_stage = _summarize_iterative_tmle_step(
                one_step_bundle,
                step=1,
                min_score_tol=min_score_tol,
            )
            if mode == "one_step":
                final_bundle = one_step_bundle
            else:
                final_bundle = _iterate_tmle_t0(
                    initial_fit=initial_fit,
                    censoring_cache=censoring_cache,
                    observed_data=observed_data,
                    t0=float(t0),
                    survival_clip=survival_clip,
                    targeting_gbar_floor=targeting_gbar_floor,
                    max_iter=max_iter,
                    min_abs_eps=min_abs_eps,
                    min_score_tol=min_score_tol,
                    eps_bracket_start=eps_bracket_start,
                    eps_bracket_growth=eps_bracket_growth,
                    eps_bracket_max=eps_bracket_max,
                    eps_fallback_bounds=eps_fallback_bounds,
                )
            history_df = final_bundle["history"]
            final_summary = final_bundle["summary"]
            final_state = final_bundle["final_state"]
            final_psi = float(final_summary["psi_star"])
            final_eif_mean = float(final_summary["eif_mean"])
            final_var = float(final_summary["estimand_variance"])
            final_se = float(final_summary["standard_error"])
            final_score_at_zero = float(final_summary["score_at_zero"])
            final_stop_reason = str(final_summary["stop_reason"])
            final_n_iterations = int(final_summary["n_iterations"])
            final_epsilon = float(final_summary["epsilon_last"])
            final_score_at_solution = float(final_summary["score_at_solution_last"])
            final_objective_at_solution = float(final_summary["objective_at_solution_last"])
            final_solve_method = str(final_summary["solve_method_last"])
            final_sigma = float(final_summary["sigma_final"])
            final_raw_tolerance = float(final_summary["sigma_over_sqrtnlogn_final"])
            final_threshold = float(final_summary["threshold_final"])
            final_epsilon_path = final_summary["epsilon_path"]
            final_eif_mean_path = final_summary["eif_mean_path"]
            final_sigma_path = final_summary["sigma_path"]
            final_psi_path = final_summary["psi_path"]
        init_exact = _compute_survival_exact_stage_diagnostics(
            init_state,
            observed_data,
            survival_clip=survival_clip,
            targeting_gbar_floor=targeting_gbar_floor,
            n_obs=n_obs,
            min_score_tol=min_score_tol,
        )
        one_step_exact = _compute_survival_exact_stage_diagnostics(
            one_step_state,
            observed_data,
            survival_clip=survival_clip,
            targeting_gbar_floor=targeting_gbar_floor,
            n_obs=n_obs,
            min_score_tol=min_score_tol,
        )
        final_exact = _compute_survival_exact_stage_diagnostics(
            final_state,
            observed_data,
            survival_clip=survival_clip,
            targeting_gbar_floor=targeting_gbar_floor,
            n_obs=n_obs,
            min_score_tol=min_score_tol,
        )
        continued_past_one_step = bool(final_n_iterations > 1)

        default_var = float(init_var)
        default_se = float(init_se)
        ci_lower, ci_upper = _build_wald_interval(final_psi, default_se)
        summary_rows.append(
            {
                "t0": float(t0),
                "psi_init": float(init_state.psi_current),
                "psi_star": final_psi,
                "epsilon": final_epsilon,
                "estimand_variance": default_var,
                "standard_error": default_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "score_at_zero": final_score_at_zero,
                "score_at_solution": final_score_at_solution,
                "objective_at_solution": final_objective_at_solution,
                "eif_mean": final_eif_mean,
                "exact_eif_mean": float(final_exact["eif_mean"]),
                "exact_sigma": float(final_exact["sigma"]),
                "exact_threshold": float(final_exact["threshold"]),
                "exact_minus_implemented_eif_mean": float(
                    final_exact["eif_mean"] - final_eif_mean
                ),
                "solve_method": final_solve_method,
                "converged": bool(final_stop_reason != "solver_failure"),
                "target_grid_augmented_with_t0": bool(init_state.target_grid_augmented_with_t0),
                "psi_initial_stage": float(init_state.psi_current),
                "eif_mean_initial_stage": float(init_state.eif_mean),
                "sigma_initial_stage": float(init_state.eif_sigma),
                "exact_eif_mean_initial_stage": float(init_exact["eif_mean"]),
                "exact_sigma_initial_stage": float(init_exact["sigma"]),
                "sigma_over_sqrtnlogn_initial": float(raw_init_tolerance),
                "threshold_initial": float(threshold_initial),
                "exact_threshold_initial": float(init_exact["threshold"]),
                "exact_minus_implemented_eif_mean_initial_stage": float(
                    init_exact["eif_mean"] - init_state.eif_mean
                ),
                "passes_initial_threshold": passes_initial_threshold,
                "estimand_variance_initial_stage": float(init_var),
                "standard_error_initial_stage": float(init_se),
                "psi_one_step": float(one_step_stage["psi_star"]),
                "eif_mean_one_step": float(one_step_stage["eif_mean"]),
                "sigma_one_step": float(one_step_stage["sigma"]),
                "exact_eif_mean_one_step": float(one_step_exact["eif_mean"]),
                "exact_sigma_one_step": float(one_step_exact["sigma"]),
                "sigma_over_sqrtnlogn_one_step": float(one_step_stage["sigma_over_sqrtnlogn"]),
                "threshold_one_step": float(one_step_stage["threshold"]),
                "exact_threshold_one_step": float(one_step_exact["threshold"]),
                "exact_minus_implemented_eif_mean_one_step": float(
                    one_step_exact["eif_mean"] - one_step_stage["eif_mean"]
                ),
                "epsilon_one_step": float(one_step_stage["epsilon"]),
                "estimand_variance_one_step": float(one_step_stage["estimand_variance"]),
                "standard_error_one_step": float(one_step_stage["standard_error"]),
                "score_at_zero_one_step": float(one_step_stage["score_at_zero"]),
                "score_at_solution_one_step": float(one_step_stage["score_at_solution"]),
                "status_one_step": str(one_step_stage["status"]),
                "psi_final": final_psi,
                "eif_mean_final": final_eif_mean,
                "sigma_final": float(final_sigma),
                "exact_eif_mean_final": float(final_exact["eif_mean"]),
                "exact_sigma_final": float(final_exact["sigma"]),
                "sigma_over_sqrtnlogn_final": float(final_raw_tolerance),
                "threshold_final": float(final_threshold),
                "exact_threshold_final": float(final_exact["threshold"]),
                "exact_minus_implemented_eif_mean_final": float(
                    final_exact["eif_mean"] - final_eif_mean
                ),
                "estimand_variance_final": final_var,
                "standard_error_final": final_se,
                "epsilon_last": final_epsilon,
                "continued_past_one_step": continued_past_one_step,
                "used_iterative": continued_past_one_step,
                "n_iterations": final_n_iterations,
                "stop_reason": final_stop_reason,
                "targeting_gbar_floor": float(targeting_gbar_floor),
                "targeting_gbar_floor_scale": (
                    float(targeting_gbar_floor_scale)
                    if targeting_gbar_floor_scale is not None
                    else float("nan")
                ),
            }
        )

        fit_row: dict[str, Any] = {
            "t0": float(t0),
            "targeting_points": np.asarray([float(t0)], dtype=float),
            "theta_targeting": (
                np.asarray([], dtype=float)
                if mode == "auto" and passes_initial_threshold
                else np.asarray([final_epsilon], dtype=float)
            ),
            "theta_selected": np.asarray(initial_fit.theta_hat, dtype=float).copy(),
            "epsilon": final_epsilon,
            "psi_init": float(init_state.psi_current),
            "psi_star": final_psi,
            "estimand_variance": default_var,
            "standard_error": default_se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "score_at_zero": final_score_at_zero,
            "score_at_solution": final_score_at_solution,
            "objective_at_solution": final_objective_at_solution,
            "eif_mean": final_eif_mean,
            "solve_method": final_solve_method,
            "converged": bool(final_stop_reason != "solver_failure"),
            "target_grid_augmented_with_t0": bool(init_state.target_grid_augmented_with_t0),
            "continued_past_one_step": continued_past_one_step,
            "used_iterative": continued_past_one_step,
            "n_iterations": final_n_iterations,
            "stop_reason": final_stop_reason,
            "targeting_gbar_floor": float(targeting_gbar_floor),
            "targeting_gbar_floor_scale": (
                float(targeting_gbar_floor_scale)
                if targeting_gbar_floor_scale is not None
                else float("nan")
            ),
            "initial_stage": {
                "psi": float(init_state.psi_current),
                "eif_mean": float(init_state.eif_mean),
                "sigma": float(init_state.eif_sigma),
                "sigma_over_sqrtnlogn": float(raw_init_tolerance),
                "threshold": float(threshold_initial),
                "passes_threshold": passes_initial_threshold,
                "estimand_variance": float(init_var),
                "standard_error": float(init_se),
            },
            "one_step_stage": {
                "psi": float(one_step_stage["psi_star"]),
                "epsilon": float(one_step_stage["epsilon"]),
                "eif_mean": float(one_step_stage["eif_mean"]),
                "sigma": float(one_step_stage["sigma"]),
                "sigma_over_sqrtnlogn": float(one_step_stage["sigma_over_sqrtnlogn"]),
                "threshold": float(one_step_stage["threshold"]),
                "estimand_variance": float(one_step_stage["estimand_variance"]),
                "standard_error": float(one_step_stage["standard_error"]),
                "score_at_zero": float(one_step_stage["score_at_zero"]),
                "score_at_solution": float(one_step_stage["score_at_solution"]),
                "status": str(one_step_stage["status"]),
            },
            "final_stage": {
                "psi": final_psi,
                "eif_mean": final_eif_mean,
                "sigma": float(final_sigma),
                "sigma_over_sqrtnlogn": float(final_raw_tolerance),
                "threshold": float(final_threshold),
                "epsilon_last": final_epsilon,
                "estimand_variance": final_var,
                "standard_error": final_se,
            },
            "iteration_history": history_df.to_dict("records") if len(history_df) > 0 else [],
            "epsilon_path": final_epsilon_path,
            "eif_mean_path": final_eif_mean_path,
            "sigma_path": final_sigma_path,
            "psi_path": final_psi_path,
        }

        if store_pointwise_arrays:
            if final_state is not None:
                fit_row.update(
                    {
                        "grid_eval": final_state.grid_edges.copy(),
                        "grid_midpoints": final_state.grid_midpoints.copy(),
                        "delta_j": final_state.delta_j.copy(),
                        "estimated_density": final_state.density_grid.copy(),
                        "estimated_survival": final_state.survival_grid.copy(),
                        "targeted_survival_grid": final_state.survival_grid.copy(),
                        "target_grid_edges": final_state.grid_edges.copy(),
                        "target_grid_midpoints": final_state.grid_midpoints.copy(),
                        "target_delta_j": final_state.delta_j.copy(),
                        "density_before_grid": init_state.density_grid.copy(),
                        "raw_direction": final_state.raw_direction.copy() if final_state.raw_direction is not None else np.empty(0, dtype=float),
                        "centered_direction": final_state.centered_direction.copy() if final_state.centered_direction is not None else np.empty(0, dtype=float),
                        "eic_values": final_state.eif_values.copy() if final_state.eif_values is not None else np.empty(0, dtype=float),
                    }
                )

        pointwise_fits.append(fit_row)

    summary = pd.DataFrame(summary_rows).sort_values("t0").reset_index(drop=True)
    return {
        "targeting_points": targeting_points_arr,
        "summary": summary,
        "pointwise_fits": pointwise_fits,
        "initial_fit": _serialize_initial_fit(initial_fit),
        "censoring_cache": _serialize_censoring_cache(censoring_cache),
        "metadata": {
            "api_version": "v2",
            "mode": requested_mode,
            "requested_mode": requested_mode,
            "one_step_eif_gate": float(one_step_eif_gate),
            "n_targets": int(targeting_points_arr.size),
            "n_observations": int(observed_data.shape[0]),
            "clip": float(clip),
            "survival_clip": float(survival_clip),
            "eps_bracket_start": float(eps_bracket_start),
            "eps_bracket_growth": float(eps_bracket_growth),
            "eps_bracket_max": float(eps_bracket_max),
            "eps_fallback_bounds": tuple(float(x) for x in eps_fallback_bounds),
            "max_iter": int(max_iter),
            "min_abs_eps": float(min_abs_eps),
            "min_score_tol": float(min_score_tol),
            "targeting_gbar_floor": float(targeting_gbar_floor),
            "targeting_gbar_floor_scale": targeting_gbar_floor_scale,
            "store_pointwise_arrays": bool(store_pointwise_arrays),
            "pointwise_design": True,
            "decision_policy": "check the initial EIF score first; target only when the initial score exceeds sigma/(sqrt(n) log n)",
        },
    }
