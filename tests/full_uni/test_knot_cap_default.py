"""Default knot cap (Yilong, 2026-10-24): K = 1,000 for k >= 1, max(1,000, ceil(n^(2/3))) for k = 0."""

import numpy as np
import pandas as pd
import pytest

from haldensity.core.univariate.objective import candidate_knots, default_max_knots
from haldensity.core.univariate.support import refit_support_from_estimator
from haldensity.full_uni import CVXPYEstimator


def _sample(n, seed=0):
    return pd.DataFrame({"W1": np.random.default_rng(seed).beta(2.0, 5.0, size=n)})


def test_the_rule():
    assert [default_max_knots(n, 1) for n in (10, 5_000, 10**6)] == [1000, 1000, 1000]
    assert default_max_knots(10**6, 2) == 1000
    # k = 0: ceil(n^(2/3)) once it passes 1,000, i.e. above n = 31,622
    assert [default_max_knots(n, 0) for n in (500, 31_622, 31_623, 40_000, 10**6)] == [1000, 1000, 1001, 1170, 10_000]


def test_samples_up_to_1000_keep_every_knot():
    data = _sample(600)
    default = CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1).fit(data)
    every = CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, max_knots=None).fit(data)
    np.testing.assert_array_equal(default._grid_points_hal, every._grid_points_hal)
    np.testing.assert_array_equal(default.optimized_theta_raw, every.optimized_theta_raw)


def test_larger_samples_are_capped():
    data = _sample(1500)
    fit = CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1).fit(data)
    results = fit.get_results()
    assert results["max_knots"] == "auto" and results["max_knots_used"] == 1000
    assert results["n_candidate_knots"] == 1000 and np.all(np.isin(fit._grid_points_hal, data["W1"]))
    # basis cleaning rebuilds the same knots from the same setting
    assert np.array_equal(refit_support_from_estimator(fit, data).knots, fit._grid_points_hal)
    # all knots on request
    assert CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, max_knots=None).fit(data) \
        .get_results()["n_candidate_knots"] == 1500


def test_k0_cap_grows_with_n():
    x = np.random.default_rng(1).uniform(size=40_000)
    assert candidate_knots(x, "auto", basis_order=0).size == 1170       # ceil(40,000^(2/3))
    assert candidate_knots(x, "auto", basis_order=1).size == 1000
    assert candidate_knots(x, None).size == 40_000


def test_invalid_settings():
    with pytest.raises(ValueError, match="max_knots must be"):
        CVXPYEstimator(max_knots="all")
    with pytest.raises(ValueError, match="basis order"):
        candidate_knots(np.arange(5.0), "auto")
