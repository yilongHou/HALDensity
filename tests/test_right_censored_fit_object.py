"""Smoke tests for the frozen right-censored fitted-distribution object."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring.right import (
    KaplanMeier,
    RightCensoredHALFit,
    RightCensoredInitEstimator,
    RightCensoredObservedL1MLE,
    compute_ipcw_weights,
)
from haldensity.censoring.right.comparison import simulate_beta_uniform_right_censored
from haldensity.censoring.tuners.right_tuners import RightCensoredObservedL1Tuner


def _make_data(n: int = 80, seed: int = 11) -> pd.DataFrame:
    return simulate_beta_uniform_right_censored(n=n, seed=seed)["observed_data"]


def _fit_ipcw(data: pd.DataFrame, *, n_grid_points: int = 80) -> RightCensoredInitEstimator:
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    t_vals = np.asarray(data["T"].values, dtype=float)
    delta_vals = np.asarray(data["Delta"].values, dtype=int)
    weights = compute_ipcw_weights(t_vals, delta_vals, lambda t: np.atleast_1d(km.predict(t)))
    uncensored = delta_vals == 1
    est = RightCensoredInitEstimator(
        norm_constraint=8.0,
        n_grid_points=n_grid_points,
        basis_order=0,
        include_intercept_in_constraint=False,
        use_secondary_solver=True,
    )
    fitted = est.fit(
        pd.DataFrame({"W1": t_vals[uncensored]}),
        sample_weights=weights[uncensored],
    )
    assert fitted is est
    return est


def _compress_theta_to_selected_support(estimator: RightCensoredInitEstimator) -> np.ndarray:
    support = np.asarray(estimator.grid_points_hal_selected, dtype=float)
    all_knots = np.asarray(estimator._grid_points_hal, dtype=float)
    theta_full = np.asarray(estimator.theta_hat, dtype=float)
    knot_start = 1
    theta_selected = np.zeros(knot_start + support.size, dtype=float)
    theta_selected[:knot_start] = theta_full[:knot_start]
    for i, knot in enumerate(support):
        matches = np.where(np.isclose(all_knots, knot, atol=1e-10, rtol=0.0))[0]
        theta_selected[knot_start + i] = theta_full[knot_start + int(matches[0])]
    return theta_selected


def _fit_l1mle(data: pd.DataFrame) -> RightCensoredObservedL1MLE:
    stage1 = _fit_ipcw(data)
    support = np.asarray(stage1.grid_points_hal_selected, dtype=float)
    if support.size == 0:
        raise AssertionError("test setup produced an empty Stage 1 support")
    est = RightCensoredObservedL1MLE(
        working_grid_points=support,
        norm_constraint=float(stage1.norm_constraint),
        basis_order=0,
        n_grid_points=80,
        learning_rate=0.1,
        n_iterations=150,
        ll_change_tol=1e-4,
        include_intercept_in_constraint=False,
        warm_start_theta=_compress_theta_to_selected_support(stage1),
    )
    fitted = est.fit(data)
    assert fitted is est
    return est


def test_from_state_roundtrip_matches_hand_built_object() -> None:
    theta = np.array([0.25, 0.0, -0.4, 0.1], dtype=float)
    knots = np.array([0.0, 0.4, 1.0], dtype=float)
    original = RightCensoredHALFit(
        theta=theta,
        knots=knots,
        selected_knots=np.array([0.4], dtype=float),
        basis_order=0,
        norm_shift=0.25,
        norm_Z=1.2,
        last_failure=0.7,
        normalizer_method="midpoint",
        n_cells=80,
    )
    restored = RightCensoredHALFit.from_state(original.to_state())
    points = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(restored.density(points), original.density(points), atol=1e-12)
    np.testing.assert_allclose(restored.survival(points), original.survival(points), atol=1e-12)
    np.testing.assert_allclose(restored.rmst(0.75), original.rmst(0.75), atol=1e-12)
    assert restored.last_failure == pytest.approx(0.7)
    assert restored.beyond_last_failure(0.8).item()
    assert not restored.beyond_last_failure(0.7).item()


def test_ipcw_get_fit_matches_live_density_and_survives_reload() -> None:
    data = _make_data()
    est = _fit_ipcw(data)
    fitted = est.get_fit()
    points = np.linspace(0.05, 0.95, 25)
    live = np.asarray(est.get_density_at_points(points), dtype=float)
    np.testing.assert_allclose(fitted.density(points), live, atol=1e-12)

    reloaded = RightCensoredHALFit.from_state(fitted.to_state())
    np.testing.assert_allclose(reloaded.density(points), live, atol=1e-12)
    np.testing.assert_allclose(reloaded.survival(points), fitted.survival(points), atol=1e-12)
    np.testing.assert_allclose(reloaded.rmst([0.5, 0.75]), fitted.rmst([0.5, 0.75]), atol=1e-12)

    last_failure = float(np.max(data.loc[data["Delta"] == 1, "T"]))
    assert fitted.last_failure == pytest.approx(last_failure)
    assert np.all(fitted.survival(points) >= -1e-12)
    assert fitted.survival(1.0) == pytest.approx(0.0, abs=1e-12)


def test_l1mle_get_fit_preserves_predictions_and_has_no_tail_warning() -> None:
    data = _make_data(n=96, seed=3)
    est = _fit_l1mle(data)
    fitted = est.get_fit()
    points = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    live = np.asarray(est.get_density_at_points(points), dtype=float)
    np.testing.assert_allclose(fitted.density(points), live, atol=1e-12)

    last_failure = float(np.max(data.loc[data["Delta"] == 1, "T"]))
    query = np.array([last_failure - 1e-3, last_failure, min(last_failure + 0.05, 1.0), 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = fitted.density(query)
        survival = fitted.survival(query)
        rmst = fitted.rmst(0.99)
    assert values.shape == query.shape
    assert np.all(np.isfinite(survival))
    assert np.isfinite(rmst)
    assert caught == []
    np.testing.assert_array_equal(fitted.beyond_last_failure(query), query > last_failure)


def test_cvl1mle_tuner_get_fit_matches_selected_estimator() -> None:
    data = _make_data(n=72, seed=19)
    stage1 = _fit_ipcw(data)
    if np.asarray(stage1.grid_points_hal_selected).size == 0:
        pytest.skip("Stage 1 selected an empty support")
    tuner = RightCensoredObservedL1Tuner(
        data,
        stage1_estimator=stage1,
        cv_folds=2,
        norm_constraint_factors=[1.0],
        random_state=4,
        silent=True,
        l1_kwargs={
            "n_grid_points": 80,
            "learning_rate": 0.1,
            "n_iterations": 80,
            "ll_change_tol": 1e-3,
        },
    )
    result = tuner.optimize(n_trials=1)
    fitted = tuner.get_fit()
    points = np.array([0.25, 0.5, 0.75], dtype=float)
    np.testing.assert_allclose(
        fitted.density(points),
        result.estimator.get_density_at_points(points),
        atol=1e-12,
    )
    assert result.estimator.get_fit() is result.estimator.get_fit()
