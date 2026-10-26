"""The censoring term of the right-censored influence curves uses the jump of 1/G (review R1, 2026-10-25)."""
import numpy as np

from haldensity.rc import KaplanMeier
from haldensity.rc.targeting._common import RCTargetGrid, _build_censoring_cache
from haldensity.rc.targeting.eic import (
    _censoring_jump_weights,
    _compute_raw_phi_direction_grid,
    _evaluate_tail_conditional_mean,
)

JUMPS = [0.25, 0.5, 0.75]           # on cell edges, so no censoring jump falls inside a cell
G_AFTER = [0.6, 0.3, 0.1]           # big jumps: the small risk sets at the end are where the two forms differ


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


def test_the_censoring_weight_is_the_jump_of_one_over_g():
    weights, denominators = _censoring_jump_weights(_discrete_censoring())
    left, right = np.array([1.0] + G_AFTER[:-1]), np.array(G_AFTER)
    np.testing.assert_allclose(weights, 1.0 / right - 1.0 / left, rtol=1e-14)
    np.testing.assert_allclose(denominators, left * right, rtol=1e-14)


def test_the_influence_curve_has_conditional_mean_phi_minus_psi():
    # D(O) = Delta D(T) + (1 - Delta) E[D(T*) | T* > T] - P D with D the code's raw direction for the mean
    # (phi(t) = t). Given T* = t, the event is seen when C >= t and censored at each jump c < t, so
    # E[D(O) | T* = t] = D(t) P(C >= t) + sum_{c < t} P(C = c) E[D(T*) | T* > c] - P D, which must be
    # phi(t) - psi. With dG / G(u)^2 instead of dG / (G(u-) G(u)) the error here is up to 4.3.
    grid = _grid()
    cache = _discrete_censoring()
    t, mass = grid.grid_midpoints, grid.density_grid * grid.delta_j
    tail_phi = _evaluate_tail_conditional_mean(grid, t, cache.jump_times, survival_clip=1e-300)
    direction = _compute_raw_phi_direction_grid(target_grid=grid, censoring_cache=cache, phi_grid=t,
                                                tail_phi_jump=tail_phi, targeting_gbar_floor=0.0)
    tail_direction = _evaluate_tail_conditional_mean(grid, direction, cache.jump_times, survival_clip=1e-300)
    censored_part = np.array([np.sum((cache.jump_masses * tail_direction)[cache.jump_times < x]) for x in t])
    conditional_mean = direction * cache.km.predict(t) + censored_part - np.sum(direction * mass)
    np.testing.assert_allclose(conditional_mean, t - np.sum(t * mass), atol=1e-13)
