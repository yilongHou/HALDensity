from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _raw_gbar(cache: Any, t: np.ndarray | float) -> np.ndarray:
    return np.asarray(cache.km.predict(t), dtype=float)


def _clip_active_eic_mask(
    *,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    clip: float,
) -> dict[str, Any]:
    clip = float(clip)
    observed_t = observed_data["T"].to_numpy(dtype=float)
    delta = observed_data["Delta"].to_numpy(dtype=int)
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)

    observed_gbar = _raw_gbar(censoring_cache, observed_t)
    grid_gbar = _raw_gbar(censoring_cache, grid)
    # Flag exactly the regions where the targeter replaces raw KM Gbar by the
    # computational clip. For typical sample sizes this mostly means Gbarhat=0.
    direction_grid_mask = grid_gbar < clip

    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    jump_gbar = np.asarray(censoring_cache.gbar_right, dtype=float)
    bad_jump_times = jump_times[jump_gbar < clip]
    if bad_jump_times.size > 0:
        direction_grid_mask = direction_grid_mask | np.any(
            grid[:, None] >= bad_jump_times[None, :],
            axis=1,
        )

    if bad_jump_times.size > 0:
        event_jump_mask = np.any(observed_t[:, None] >= bad_jump_times[None, :], axis=1)
    else:
        event_jump_mask = np.zeros_like(observed_t, dtype=bool)
    right_idx = np.searchsorted(grid, observed_t, side="left")
    right_idx = np.clip(right_idx, 0, len(grid) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(grid) - 1)
    event_direction_mask = direction_grid_mask[left_idx] | direction_grid_mask[right_idx]
    event_mask = (delta == 1) & ((observed_gbar < clip) | event_jump_mask | event_direction_mask)

    tail_mask = np.array(
        [bool(np.any(direction_grid_mask & (grid > t_i))) for t_i in observed_t],
        dtype=bool,
    )
    censored_mask = (delta == 0) & tail_mask
    mask = event_mask | censored_mask

    return {
        "mask": mask,
        "observed_gbar": observed_gbar,
        "direction_grid_mask": direction_grid_mask,
        "bad_jump_times": bad_jump_times,
        "clip": clip,
    }


def _estimate_positivity_filtered_eic_variance(
    *,
    eic_values: np.ndarray,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    clip: float,
) -> dict[str, Any]:
    eic_values = np.asarray(eic_values, dtype=float)
    mask_info = _clip_active_eic_mask(
        target_grid=target_grid,
        censoring_cache=censoring_cache,
        observed_data=observed_data,
        clip=clip,
    )
    positivity_mask = np.asarray(mask_info["mask"], dtype=bool)
    finite_mask = np.isfinite(eic_values)
    include_mask = finite_mask & ~positivity_mask
    filtered_values = eic_values[include_mask]
    n_included = int(filtered_values.size)

    if n_included <= 1:
        estimand_variance = float("nan")
        standard_error = float("nan")
    else:
        estimand_variance = float(np.var(filtered_values, ddof=1) / n_included)
        standard_error = float(np.sqrt(max(estimand_variance, 0.0)))

    n_total = int(eic_values.size)
    n_excluded = int(np.sum(positivity_mask | ~finite_mask))
    return {
        "estimand_variance": estimand_variance,
        "standard_error": standard_error,
        "included_count": n_included,
        "excluded_count": n_excluded,
        "excluded_rate": float(n_excluded / n_total) if n_total > 0 else float("nan"),
        "positivity_mask": positivity_mask,
        "include_mask": include_mask,
        "observed_gbar": np.asarray(mask_info["observed_gbar"], dtype=float),
        "direction_grid_mask": np.asarray(mask_info["direction_grid_mask"], dtype=bool),
        "bad_jump_times": np.asarray(mask_info["bad_jump_times"], dtype=float),
        "clip": float(clip),
    }


def _build_clipped_wald_interval_allow_nan(center: float, standard_error: float) -> tuple[float, float]:
    if not np.isfinite(standard_error):
        return float("nan"), float("nan")
    ci_lower = float(max(0.0, center - 1.96 * standard_error))
    ci_upper = float(min(1.0, center + 1.96 * standard_error))
    return ci_lower, ci_upper


def _build_unbounded_wald_interval_allow_nan(center: float, standard_error: float) -> tuple[float, float]:
    if not np.isfinite(standard_error):
        return float("nan"), float("nan")
    return float(center - 1.96 * standard_error), float(center + 1.96 * standard_error)


def _interp_midpoint_values(grid: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    return np.interp(pts, grid, values, left=float(values[0]), right=float(values[-1]))


def _interp_edge_values(
    grid_edges: np.ndarray,
    edge_values: np.ndarray,
    points: np.ndarray,
    *,
    left: float,
    right: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.interp(
        pts,
        np.asarray(grid_edges, dtype=float),
        np.asarray(edge_values, dtype=float),
        left=left,
        right=right,
    )


def _evaluate_targeting_gbar(
    censoring_cache: Any,
    t: np.ndarray | float,
    *,
    targeting_gbar_floor: float,
) -> np.ndarray:
    vals = np.asarray(censoring_cache.km.predict(t), dtype=float)
    return np.maximum(vals, float(max(targeting_gbar_floor, censoring_cache.clip)))


def _evaluate_survival_from_target_grid(
    target_grid: Any,
    points: np.ndarray,
    *,
    survival_clip: float,
) -> np.ndarray:
    surv = _interp_edge_values(
        target_grid.grid_edges,
        target_grid.edge_survival,
        np.asarray(points, dtype=float),
        left=1.0,
        right=0.0,
    )
    return np.maximum(surv, float(survival_clip))


def _evaluate_tail_conditional_mean(
    target_grid: Any,
    value_grid: np.ndarray,
    points: np.ndarray,
    *,
    survival_clip: float,
) -> np.ndarray:
    weighted = (
        np.asarray(value_grid, dtype=float)
        * np.asarray(target_grid.density_grid, dtype=float)
        * np.asarray(target_grid.delta_j, dtype=float)
    )
    tail_weighted = np.cumsum(weighted[::-1])[::-1]
    edge_tail_weighted = np.concatenate((tail_weighted, [0.0]))
    tail_num = _interp_edge_values(
        target_grid.grid_edges,
        edge_tail_weighted,
        np.asarray(points, dtype=float),
        left=float(edge_tail_weighted[0]),
        right=0.0,
    )
    tail_surv = _evaluate_survival_from_target_grid(
        target_grid,
        np.asarray(points, dtype=float),
        survival_clip=survival_clip,
    )
    return tail_num / tail_surv


def _evaluate_cumulative_jump_term(
    jump_times: np.ndarray,
    increments: np.ndarray,
    cutoff_points: np.ndarray,
) -> np.ndarray:
    cutoffs = np.asarray(cutoff_points, dtype=float)
    if np.asarray(jump_times, dtype=float).size == 0:
        return np.zeros_like(cutoffs, dtype=float)
    cumulative_increments = np.cumsum(np.asarray(increments, dtype=float))
    cutoff_idx = np.searchsorted(np.asarray(jump_times, dtype=float), cutoffs, side="right") - 1
    return np.where(
        cutoff_idx >= 0,
        cumulative_increments[np.clip(cutoff_idx, 0, len(cumulative_increments) - 1)],
        0.0,
    )


def _compute_raw_phi_direction_grid(
    *,
    target_grid: Any,
    censoring_cache: Any,
    phi_grid: np.ndarray,
    tail_phi_jump: np.ndarray,
    targeting_gbar_floor: float,
    cutoff_points: np.ndarray | None = None,
) -> np.ndarray:
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)
    cutoffs = grid if cutoff_points is None else np.asarray(cutoff_points, dtype=float)
    effective_gbar_floor = float(max(targeting_gbar_floor, censoring_cache.clip))
    gbar_grid = _evaluate_targeting_gbar(
        censoring_cache,
        grid,
        targeting_gbar_floor=effective_gbar_floor,
    )
    first_term = np.asarray(phi_grid, dtype=float) / gbar_grid

    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    jump_masses = np.asarray(censoring_cache.jump_masses, dtype=float)
    if jump_times.size == 0:
        cumulative_jump_term = np.zeros_like(grid)
    else:
        gbar_right_u = _evaluate_targeting_gbar(
            censoring_cache,
            jump_times,
            targeting_gbar_floor=effective_gbar_floor,
        )
        increments = (
            np.asarray(tail_phi_jump, dtype=float)
            * jump_masses
            / np.square(gbar_right_u)
        )
        cumulative_jump_term = _evaluate_cumulative_jump_term(
            jump_times,
            increments,
            cutoffs,
        )
    return first_term - cumulative_jump_term


def _compute_exact_eif_from_raw_direction(
    *,
    target_grid: Any,
    raw_direction_grid: np.ndarray,
    observed_data: pd.DataFrame,
    survival_clip: float,
) -> dict[str, Any]:
    raw_direction_grid = np.asarray(raw_direction_grid, dtype=float)
    raw_mean = float(
        np.sum(
            raw_direction_grid
            * np.asarray(target_grid.density_grid, dtype=float)
            * np.asarray(target_grid.delta_j, dtype=float)
        )
    )
    observed_t = observed_data["T"].to_numpy(dtype=float)
    delta = observed_data["Delta"].to_numpy(dtype=float)
    raw_direction_obs = _interp_midpoint_values(
        np.asarray(target_grid.grid_midpoints, dtype=float),
        raw_direction_grid,
        observed_t,
    )
    tail_mean_obs = _evaluate_tail_conditional_mean(
        target_grid,
        raw_direction_grid,
        observed_t,
        survival_clip=survival_clip,
    )
    eif_values = delta * raw_direction_obs + (1.0 - delta) * tail_mean_obs - raw_mean
    eif_mean = float(np.mean(eif_values))
    eif_sigma = float(np.std(eif_values, ddof=1)) if eif_values.size > 1 else 0.0
    return {
        "raw_direction_grid": raw_direction_grid,
        "raw_mean": raw_mean,
        "eif_values": eif_values,
        "eif_mean": eif_mean,
        "eif_sigma": eif_sigma,
    }


def _compute_exact_mean_eif(
    *,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> dict[str, Any]:
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)
    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    tail_mean_jump = _evaluate_tail_conditional_mean(
        target_grid,
        grid,
        jump_times,
        survival_clip=survival_clip,
    )
    raw_direction = _compute_raw_phi_direction_grid(
        target_grid=target_grid,
        censoring_cache=censoring_cache,
        phi_grid=grid,
        tail_phi_jump=tail_mean_jump,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    return _compute_exact_eif_from_raw_direction(
        target_grid=target_grid,
        raw_direction_grid=raw_direction,
        observed_data=observed_data,
        survival_clip=survival_clip,
    )


def _compute_exact_density_l2_eif(
    *,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> dict[str, Any]:
    density_grid = np.asarray(target_grid.density_grid, dtype=float)
    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    tail_density_jump = _evaluate_tail_conditional_mean(
        target_grid,
        density_grid,
        jump_times,
        survival_clip=survival_clip,
    )
    raw_direction = _compute_raw_phi_direction_grid(
        target_grid=target_grid,
        censoring_cache=censoring_cache,
        phi_grid=2.0 * density_grid,
        tail_phi_jump=2.0 * tail_density_jump,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    return _compute_exact_eif_from_raw_direction(
        target_grid=target_grid,
        raw_direction_grid=raw_direction,
        observed_data=observed_data,
        survival_clip=survival_clip,
    )


def _compute_exact_entropy_eif(
    *,
    target_grid: Any,
    censoring_cache: Any,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> dict[str, Any]:
    phi_grid = -np.log(np.clip(np.asarray(target_grid.density_grid, dtype=float), 1e-300, None))
    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    tail_entropy_jump = _evaluate_tail_conditional_mean(
        target_grid,
        phi_grid,
        jump_times,
        survival_clip=survival_clip,
    )
    raw_direction = _compute_raw_phi_direction_grid(
        target_grid=target_grid,
        censoring_cache=censoring_cache,
        phi_grid=phi_grid,
        tail_phi_jump=tail_entropy_jump,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    return _compute_exact_eif_from_raw_direction(
        target_grid=target_grid,
        raw_direction_grid=raw_direction,
        observed_data=observed_data,
        survival_clip=survival_clip,
    )


def _evaluate_piecewise_linear_edge_integral(
    grid_edges: np.ndarray,
    edge_values: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    edges = np.asarray(grid_edges, dtype=float)
    values = np.asarray(edge_values, dtype=float)
    widths = np.diff(edges)
    trapezoid_areas = 0.5 * (values[:-1] + values[1:]) * widths
    cumulative_area = np.concatenate(([0.0], np.cumsum(trapezoid_areas)))

    clipped_pts = np.clip(pts, float(edges[0]), float(edges[-1]))
    idx = np.searchsorted(edges, clipped_pts, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)

    left = edges[idx]
    right = edges[idx + 1]
    v_left = values[idx]
    v_right = values[idx + 1]
    interval_width = np.maximum(right - left, 1e-15)
    frac = (clipped_pts - left) / interval_width
    v_at_point = v_left + frac * (v_right - v_left)
    partial_area = 0.5 * (v_left + v_at_point) * (clipped_pts - left)

    integral = cumulative_area[idx] + partial_area
    integral = np.where(pts <= edges[0], 0.0, integral)
    integral = np.where(pts >= edges[-1], cumulative_area[-1], integral)
    return integral


def _evaluate_inv_gbar_integral(
    censoring_cache: Any,
    points: np.ndarray,
    *,
    targeting_gbar_floor: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    clipped_pts = np.maximum(pts, 0.0)
    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    if jump_times.size == 0:
        return clipped_pts

    gbar_right = _evaluate_targeting_gbar(
        censoring_cache,
        jump_times,
        targeting_gbar_floor=targeting_gbar_floor,
    )
    interval_starts = np.concatenate(([0.0], jump_times))
    interval_levels = np.concatenate(([1.0], gbar_right))
    cumulative_area = np.concatenate(
        ([0.0], np.cumsum(np.diff(interval_starts) / interval_levels[:-1]))
    )
    interval_idx = np.searchsorted(interval_starts, clipped_pts, side="right") - 1
    interval_idx = np.clip(interval_idx, 0, len(interval_starts) - 1)
    partial = (clipped_pts - interval_starts[interval_idx]) / interval_levels[interval_idx]
    return cumulative_area[interval_idx] + partial


def _compute_exact_rmst_eif(
    *,
    target_grid: Any,
    censoring_cache: Any,
    tau: float,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> dict[str, Any]:
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)
    cutoff_points = np.minimum(grid, float(tau))
    raw_direction = _evaluate_inv_gbar_integral(
        censoring_cache,
        cutoff_points,
        targeting_gbar_floor=targeting_gbar_floor,
    )

    jump_times = np.asarray(censoring_cache.jump_times, dtype=float)
    jump_masses = np.asarray(censoring_cache.jump_masses, dtype=float)
    if jump_times.size > 0:
        tau_integral = _evaluate_piecewise_linear_edge_integral(
            target_grid.grid_edges,
            target_grid.edge_survival,
            np.array([float(tau)], dtype=float),
        )[0]
        tail_num = tau_integral - _evaluate_piecewise_linear_edge_integral(
            target_grid.grid_edges,
            target_grid.edge_survival,
            jump_times,
        )
        tail_num = np.where(jump_times >= float(tau), 0.0, tail_num)
        tail_rmst_jump = np.maximum(tail_num, 0.0) / _evaluate_survival_from_target_grid(
            target_grid,
            jump_times,
            survival_clip=survival_clip,
        )
        gbar_right_u = _evaluate_targeting_gbar(
            censoring_cache,
            jump_times,
            targeting_gbar_floor=targeting_gbar_floor,
        )
        increments = tail_rmst_jump * jump_masses / np.square(gbar_right_u)
        raw_direction = raw_direction - _evaluate_cumulative_jump_term(
            jump_times,
            increments,
            cutoff_points,
        )

    return _compute_exact_eif_from_raw_direction(
        target_grid=target_grid,
        raw_direction_grid=raw_direction,
        observed_data=observed_data,
        survival_clip=survival_clip,
    )


def _compute_exact_survival_eif(
    *,
    target_grid: Any,
    censoring_cache: Any,
    t0: float,
    observed_data: pd.DataFrame,
    survival_clip: float,
    targeting_gbar_floor: float,
) -> dict[str, Any]:
    grid = np.asarray(target_grid.grid_midpoints, dtype=float)
    effective_gbar_floor = float(max(targeting_gbar_floor, censoring_cache.clip))
    gbar_t0_right = float(
        _evaluate_targeting_gbar(
            censoring_cache,
            np.array([float(t0)], dtype=float),
            targeting_gbar_floor=effective_gbar_floor,
        )[0]
    )
    raw_direction = (grid > float(t0)).astype(float) / gbar_t0_right

    jump_times_all = np.asarray(censoring_cache.jump_times, dtype=float)
    jump_masses_all = np.asarray(censoring_cache.jump_masses, dtype=float)
    active_mask = jump_times_all <= float(t0)
    jump_times = jump_times_all[active_mask]
    jump_masses = jump_masses_all[active_mask]
    if jump_times.size > 0:
        s_t0 = float(
            _evaluate_survival_from_target_grid(
                target_grid,
                np.array([float(t0)], dtype=float),
                survival_clip=survival_clip,
            )[0]
        )
        s_u = _evaluate_survival_from_target_grid(
            target_grid,
            jump_times,
            survival_clip=survival_clip,
        )
        gbar_right_u = _evaluate_targeting_gbar(
            censoring_cache,
            jump_times,
            targeting_gbar_floor=effective_gbar_floor,
        )
        increments = s_t0 * jump_masses / (s_u * np.square(gbar_right_u))
        raw_direction = raw_direction - _evaluate_cumulative_jump_term(
            jump_times,
            increments,
            np.minimum(grid, float(t0)),
        )

    return _compute_exact_eif_from_raw_direction(
        target_grid=target_grid,
        raw_direction_grid=raw_direction,
        observed_data=observed_data,
        survival_clip=survival_clip,
    )
