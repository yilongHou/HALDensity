import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import interp1d

from haldensity.full_uni import CVXPYEstimator, OptunaHyperparameterTuner


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=150)})


@pytest.fixture(scope="module")
def fit(data):
    return CVXPYEstimator(norm_constraint=20.0, basis_order=0, log_frequency=-1, solver="ECOS",
                          use_secondary_solver=False).fit(data)


def _tuner(data, **kwargs):
    return OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=2, **kwargs)


def test_exact_scoring_evaluates_the_fit_at_the_validation_points(data, fit):
    points = np.random.default_rng(1).beta(2.0, 5.0, size=50)
    score = _tuner(data)._compute_log_likelihood_from_density(fit, points)
    assert score == pytest.approx(float(np.mean(fit.get_log_density_at_points(points))), abs=1e-12)


def test_interpolated_scoring_keeps_the_old_score(data, fit):
    points = np.random.default_rng(1).beta(2.0, 5.0, size=50)
    grid, density = fit.get_density()
    old = np.mean(np.log(np.maximum(
        interp1d(grid, density, kind="linear", bounds_error=False, fill_value=(density[0], density[-1]))(points),
        1e-10)))
    score = _tuner(data, cv_scoring="interpolated")._compute_log_likelihood_from_density(fit, points)
    assert score == pytest.approx(float(old), abs=1e-12)


def test_scorings_differ_where_interpolation_blurs_a_step(data, fit):
    # k = 0 fits are step functions; near a knot the linear interpolation mixes two levels
    knots = np.asarray(fit.grid_points_hal_selected, dtype=float)
    points = np.clip(np.concatenate([knots + 1e-4, knots - 1e-4]), 0.0, 1.0)
    exact = _tuner(data)._compute_log_likelihood_from_density(fit, points)
    interpolated = _tuner(data, cv_scoring="interpolated")._compute_log_likelihood_from_density(fit, points)
    assert abs(exact - interpolated) > 1e-3


def test_unknown_cv_scoring_is_rejected(data):
    with pytest.raises(ValueError, match="cv_scoring must be one of"):
        _tuner(data, cv_scoring="midpoint")
