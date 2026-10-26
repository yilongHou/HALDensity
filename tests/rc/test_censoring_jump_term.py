"""The censoring term of the right-censored influence curves uses the jump of 1/G (review R1, 2026-10-25), and of
1/max(G, f) when the floor f binds (review F1)."""
import numpy as np
import pandas as pd
import pytest

from haldensity.rc import KaplanMeier
from haldensity.rc.targeting._common import RCTargetGrid, _build_censoring_cache, _compute_dfg_t0_on_grid
from haldensity.rc.targeting.density_l2 import _compute_density_l2_direction_on_grid
from haldensity.rc.targeting.eic import (
    _censoring_jump_weights,
    _compute_exact_rmst_eif,
    _compute_exact_survival_eif,
    _compute_raw_phi_direction_grid,
    _evaluate_tail_conditional_mean,
)
from haldensity.rc.targeting.entropy import _compute_entropy_direction_on_grid
from haldensity.rc.targeting.mean import _compute_mean_direction_on_grid
from haldensity.rc.targeting.rmst import _compute_rmst_direction_on_grid

JUMPS = [0.25, 0.5, 0.75]           # on cell edges, so no censoring jump falls inside a cell
G_AFTER = [0.6, 0.3, 0.1]           # big jumps: the small risk sets at the end are where the two forms differ
FLOORS = [0.2, 0.35]                # binds at the last jump only; at the last two (the last one then has no jump)
T0 = 0.6                            # a cell edge: survival's time point and the RMST horizon
CLIP = 1e-300
# the eight places that build a censoring term (review R1)
PLACES = ["shared phi direction", "exact survival EIF", "exact RMST EIF", "survival v1/v2 and AIPW", "mean", "RMST",
          "entropy", "density L2"]


def _grid(n_cells=20, seed=0):
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    width = np.diff(edges)
    density = np.random.default_rng(seed).uniform(0.2, 2.0, n_cells)
    density /= np.sum(density * width)
    edge_survival = np.concatenate((np.cumsum((density * width)[::-1])[::-1], [0.0]))
    return RCTargetGrid(t0=float("nan"), grid_edges=edges, grid_midpoints=0.5 * (edges[:-1] + edges[1:]),
                        delta_j=width, density_grid=density, log_density_grid=np.log(density),
                        survival_grid=edge_survival[1:], edge_survival=edge_survival, t0_inserted=False)


def _discrete_censoring():
    km = KaplanMeier()
    km._times, km._surv = np.array(JUMPS), np.array(G_AFTER)
    return _build_censoring_cache(km, clip=1e-12)


def _direction_and_phi(place, grid, cache, floor):
    """The raw direction D on the grid built at ``place``, and the full-data phi it maps."""
    t = grid.grid_midpoints
    exact = dict(target_grid=grid, censoring_cache=cache, survival_clip=CLIP, targeting_gbar_floor=floor,
                 observed_data=pd.DataFrame({"T": [0.1, 0.9], "Delta": [1, 0]}))
    if place == "shared phi direction":
        tail = _evaluate_tail_conditional_mean(grid, t, cache.jump_times, survival_clip=CLIP)
        return _compute_raw_phi_direction_grid(target_grid=grid, censoring_cache=cache, phi_grid=t,
                                               tail_phi_jump=tail, targeting_gbar_floor=floor), t
    if place == "exact survival EIF":
        return _compute_exact_survival_eif(t0=T0, **exact)["raw_direction_grid"], (t > T0).astype(float)
    if place == "exact RMST EIF":
        return _compute_exact_rmst_eif(tau=T0, **exact)["raw_direction_grid"], np.minimum(t, T0)
    if place == "survival v1/v2 and AIPW":
        return _compute_dfg_t0_on_grid(grid, cache, T0, CLIP, floor)[0], (t > T0).astype(float)
    if place == "mean":
        return _compute_mean_direction_on_grid(grid, cache, CLIP, floor)[0], t
    if place == "RMST":
        return _compute_rmst_direction_on_grid(grid, cache, T0, CLIP, floor)[0], np.minimum(t, T0)
    if place == "entropy":
        return _compute_entropy_direction_on_grid(grid, cache, CLIP, floor)[0], -np.log(grid.density_grid)
    return _compute_density_l2_direction_on_grid(grid, cache, CLIP, floor)[0], 2.0 * grid.density_grid


def _conditional_mean(grid, cache, direction, floor):
    """E[D(O) | T* = t] at the cell midpoints for D(O) = Delta D(T) + (1 - Delta) E[D(T*) | T* > T] - P D, with
    max(G, f) as the censoring survival: given T* = t the event is seen when C >= t, and censored at each jump
    c < t, so E[D(O) | T* = t] = D(t) P(C >= t) + sum_{c < t} P(C = c) E[D(T*) | T* > c] - P D."""
    t, mass = grid.grid_midpoints, grid.density_grid * grid.delta_j
    jumps = np.maximum(cache.gbar_left, floor) - np.maximum(cache.gbar_right, floor)
    tail = _evaluate_tail_conditional_mean(grid, direction, cache.jump_times, survival_clip=CLIP)
    censored = np.array([np.sum((jumps * tail)[cache.jump_times < x]) for x in t])
    return direction * np.maximum(cache.km.predict(t), floor) + censored - np.sum(direction * mass)


def test_the_censoring_weight_is_the_jump_of_one_over_g():
    weights, denominators = _censoring_jump_weights(_discrete_censoring())
    left, right = np.array([1.0] + G_AFTER[:-1]), np.array(G_AFTER)
    np.testing.assert_allclose(weights, 1.0 / right - 1.0 / left, rtol=1e-14)
    np.testing.assert_allclose(denominators, left * right, rtol=1e-14)


@pytest.mark.parametrize("floor", FLOORS)
def test_with_the_floor_binding_the_weight_is_the_jump_of_one_over_the_floored_g(floor):
    weights, _ = _censoring_jump_weights(_discrete_censoring(), floor)
    left, right = np.maximum([1.0] + G_AFTER[:-1], floor), np.maximum(G_AFTER, floor)
    np.testing.assert_allclose(weights, 1.0 / right - 1.0 / left, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("floor", [0.0] + FLOORS)
@pytest.mark.parametrize("place", PLACES)
def test_the_influence_curve_has_conditional_mean_phi_minus_psi(place, floor):
    # E[D(O) | T* = t] = phi(t) - P phi, with max(G, f) as the censoring survival (f = 0: G itself). With
    # dG / G(u)^2 instead of dG / (G(u-) G(u)) the error for the mean is up to 4.3 (R1); with the floor binding and
    # dG unfloored (before F1), up to 0.57 for the mean and 1.46 for density L2
    grid, cache = _grid(), _discrete_censoring()
    direction, phi = _direction_and_phi(place, grid, cache, floor)
    mass = grid.density_grid * grid.delta_j
    np.testing.assert_allclose(_conditional_mean(grid, cache, direction, floor), phi - np.sum(phi * mass),
                               atol=1e-12)


@pytest.mark.parametrize("floor, t0", [(0.2, 0.6), (0.35, 0.3)])
def test_with_the_floor_binding_survival_keeps_its_closed_form(floor, t0):
    # survival's direction I(t > t0) / G(t0) - sum_{u <= min(t, t0)} S(t0) / S(u) dG(u) / (G(u-) G(u)) equals the
    # general direction for phi = I(t > t0) because the jumps of 1/G after t0 add up to 1/G(t) - 1/G(t0); with
    # the floor binding after t0 that needs the jumps of 1/max(G, f) (before F1 the two differed by up to 1.87)
    grid = _grid()
    cache = _discrete_censoring()
    phi = (grid.grid_midpoints > t0).astype(float)
    general = _compute_raw_phi_direction_grid(
        target_grid=grid, censoring_cache=cache, phi_grid=phi, targeting_gbar_floor=floor,
        tail_phi_jump=_evaluate_tail_conditional_mean(grid, phi, cache.jump_times, survival_clip=CLIP))
    closed_form = _compute_exact_survival_eif(
        target_grid=grid, censoring_cache=cache, t0=t0, survival_clip=CLIP, targeting_gbar_floor=floor,
        observed_data=pd.DataFrame({"T": [0.1, 0.9], "Delta": [1, 0]}))["raw_direction_grid"]
    np.testing.assert_allclose(general, closed_form, rtol=1e-13, atol=1e-13)
