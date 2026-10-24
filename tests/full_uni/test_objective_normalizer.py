import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.solvers.cvxpy import solve_cvxpy_problem
from haldensity.core.univariate.normalizer import log_normalizer, log_unnormalized_density
from haldensity.core.univariate.objective import candidate_knots
from haldensity.core.univariate.support import refit_support_from_estimator
from haldensity.full_uni import CVXPYEstimator

ECOS = dict(_solver="ECOS", _use_secondary_solver=False, log_frequency=-1)
@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=300)})


def test_default_matches_the_pre_option_problem(data):
    # the objective exactly as CVXPYEstimator built it before objective_normalizer / max_knots, with the
    # n x m float32 data basis; the fit now uses the exact float64 column sums of that basis, so the
    # solutions agree to the float32 rounding (identical at k = 0, whose basis is 0/1)
    for k in (0, 1):
        knots = np.unique(data["W1"].dropna())
        b_ik = create_basis_functions(data, knots, order=k)[0]
        grid = np.linspace(0, 1, 200)
        b_jk = create_basis_functions(pd.DataFrame({"W1": (grid[:-1] + grid[1:]) / 2}), knots, order=k)[0]
        theta = cp.Variable(b_ik.shape[1])
        loss = -cp.sum(b_ik @ theta) + len(data) * cp.log_sum_exp(np.log(grid[1:] - grid[:-1]) + b_jk @ theta)
        problem = cp.Problem(cp.Minimize(loss), [cp.norm1(theta[1:]) <= 5.0])
        solve_cvxpy_problem(problem=problem, primary_solver="ECOS", use_secondary_solver=False)
        fit = CVXPYEstimator(norm_constraint=5.0, basis_order=k, max_knots=None, _solver_tolerance="default",
                             _objective_normalizer="midpoint", **ECOS).fit(data)
        # theta[0] is not identified (a constant added to f cancels between the data term and n log Z)
        np.testing.assert_allclose(fit.optimized_theta_raw[1:], theta.value[1:], rtol=0, atol=1e-5)
        assert fit.objective_value_ == pytest.approx(problem.value, rel=1e-7)


def test_k0_knot_aware_objective_is_the_exact_integral(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=0, _objective_normalizer="knot_aware",
                         log_frequency=-1).fit(data)
    theta, knots = fit.optimized_theta_raw, fit._grid_points_hal
    exact = log_normalizer(theta, knots, 0).log_z
    assert fit.objective_log_normalizer() == pytest.approx(exact, abs=1e-9)
    # the value the solver minimized is -sum f(x_i) + n * (exact log Z)
    f_data = log_unnormalized_density(data["W1"].to_numpy(), theta, knots, 0)
    assert fit.objective_value_ == pytest.approx(-f_data.sum() + len(data) * exact, rel=1e-6)


def test_knot_aware_removes_the_k0_spike(data):
    x = data["W1"].to_numpy()
    train_ll, flagged = {}, {}
    for name in ("midpoint", "knot_aware"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = CVXPYEstimator(norm_constraint=50.0, basis_order=0, _objective_normalizer=name,
                                 log_frequency=-1).fit(data)
        train_ll[name] = float(np.mean(fit.get_log_density_at_points(x)))
        flagged[name] = any("Internal check failed" in str(w.message) for w in caught)
    assert train_ll["knot_aware"] > 0.5 and train_ll["knot_aware"] > train_ll["midpoint"]
    # the forced old midpoint rule spikes at k = 0 and the internal safeguard says so; the exact objective does not
    assert flagged == {"midpoint": True, "knot_aware": False}


def test_k1_knot_aware_sees_more_of_the_mass_than_the_midpoint_grid(data):
    gaps = {}
    for name in ("midpoint", "knot_aware"):
        fit = CVXPYEstimator(norm_constraint=1e4, basis_order=1, _objective_normalizer=name, log_frequency=-1).fit(data)
        accurate = log_normalizer(fit.optimized_theta_raw, fit._grid_points_hal, 1).log_z
        gaps[name] = abs(accurate - fit.objective_log_normalizer())
    assert gaps["knot_aware"] <= gaps["midpoint"]


def test_max_knots_places_knots_at_data_quantiles(data):
    x = data["W1"].to_numpy()
    knots = candidate_knots(x, 25)
    assert 2 <= knots.size <= 25 and np.all(np.isin(knots, x))
    assert knots[0] == x.min() and knots[-1] == x.max()
    np.testing.assert_array_equal(candidate_knots(x, None), np.unique(x))
    np.testing.assert_array_equal(candidate_knots(x, 10_000), np.unique(x))

    capped = CVXPYEstimator(norm_constraint=5.0, basis_order=1, max_knots=25, **ECOS).fit(data)
    np.testing.assert_array_equal(capped._grid_points_hal, knots)
    assert capped.theta_hat.size == 2 + knots.size
    uncapped = CVXPYEstimator(norm_constraint=5.0, basis_order=1, **ECOS).fit(data)
    huge = CVXPYEstimator(norm_constraint=5.0, basis_order=1, max_knots=10_000, **ECOS).fit(data)
    np.testing.assert_array_equal(uncapped.optimized_theta_raw, huge.optimized_theta_raw)


@pytest.mark.parametrize("settings", [dict(basis_order=0, _objective_normalizer="knot_aware", max_knots=40),
                                      dict(basis_order=1, _objective_normalizer="knot_aware", max_knots=40)])
def test_basis_cleaning_rebuilds_the_same_problem(data, settings):
    fit = CVXPYEstimator(norm_constraint=10.0, log_frequency=-1, **settings).fit(data)
    refit = refit_support_from_estimator(fit, data)
    assert refit.theta.shape == fit.theta_hat.shape and refit.kkt["n_knots"] == fit._grid_points_hal.size


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="_objective_normalizer must be None or one of"):
        CVXPYEstimator(_objective_normalizer="exact")
    with pytest.raises(ValueError, match="max_knots must be"):
        CVXPYEstimator(max_knots=0)
