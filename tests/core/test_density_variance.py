import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from haldensity.core.univariate.normalizer import log_unnormalized_density, quadrature_rule
from haldensity.core.variance import density_confidence_interval, estimate_covariance_beta
from haldensity.full_uni.estimators.cvxpy import CVXPYEstimator


def _fit(k, norm_constraint=8.0, seed=1):
    data = pd.DataFrame({"W1": np.random.default_rng(seed).beta(2.0, 5.0, size=200)})
    return data, CVXPYEstimator(norm_constraint=norm_constraint, basis_order=k, log_frequency=-1).fit(data)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_band_centre_is_the_estimator_density(k):
    # the centre and E[phi] use the accurate normalizer, as get_density_at_points does (before: a
    # 200-point trapezoid, up to 5.1% off the plotted density on Galaxies)
    data, estimator = _fit(k)
    results = estimator.get_results()
    x = np.linspace(0.02, 0.9, 7)
    cov = estimate_covariance_beta(data, results, basis_order=k)
    ci = density_confidence_interval(x, results, cov, basis_order=k)
    np.testing.assert_allclose(ci["density"], estimator.get_density_at_points(x), rtol=1e-10)


@pytest.mark.parametrize("k", [0, 2])
def test_quadrature_rule_integrates_the_fitted_density(k):
    _, estimator = _fit(k)
    theta, knots = estimator.theta_hat, estimator._grid_points_hal
    points, weights, result = quadrature_rule(theta, knots, k)
    mass = weights * np.exp(log_unnormalized_density(points, theta, knots, k) - result.log_z)
    assert mass.sum() == pytest.approx(1.0, abs=1e-12)
    # E[X] by the rule against adaptive quadrature with breakpoints at the selected knots
    density = lambda t: float(np.exp(log_unnormalized_density(np.array([t]), theta, knots, k)[0] - result.log_z))
    selected = knots[theta[(1 if k == 0 else k + 1):] != 0]
    reference = quad(lambda t: t * density(t), 0.0, 1.0, points=selected[(selected > 0) & (selected < 1)],
                     limit=1000, epsabs=1e-13, epsrel=1e-12)[0]
    assert mass @ points == pytest.approx(reference, rel=1e-10)


def test_delta_method_confidence_interval_runs_end_to_end():
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})
    estimator = CVXPYEstimator(
        norm_constraint=5.0, basis_order=1, log_frequency=-1, _solver="ECOS", _use_secondary_solver=False
    ).fit(data)
    results = estimator.get_results()

    cov = estimate_covariance_beta(data, results, basis_order=1)
    ci = density_confidence_interval(np.linspace(0.05, 0.95, 5), results, cov, basis_order=1)

    assert cov.shape[0] == cov.shape[1]
    assert np.all(np.isfinite(ci[["density", "se", "lower", "upper"]].to_numpy()))
    assert np.all(ci["se"] >= 0)
    assert np.all((ci["lower"] <= ci["density"]) & (ci["density"] <= ci["upper"]))


def test_relative_cutoff_and_model_information_are_internal_candidates():
    # candidates under study (Yilong, 2026-10-24); the default stays the empirical information + ridge 1e-6
    data, estimator = _fit(1)
    results = estimator.get_results()
    x = np.linspace(0.05, 0.6, 5)
    default = estimate_covariance_beta(data, results, basis_order=1)
    np.testing.assert_array_equal(default, estimate_covariance_beta(data, results, basis_order=1, _inverse="ridge",
                                                                    _ridge_param=1e-6, _information="empirical"))
    for options in ({"_inverse": "cutoff"}, {"_inverse": "cutoff", "_information": "model"},
                    {"_information": "model"}):
        cov = estimate_covariance_beta(data, results, basis_order=1, **options)
        assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > -1e-12 * np.abs(cov).max()
        se = density_confidence_interval(x, results, cov, basis_order=1)["se"].to_numpy()
        assert np.all(np.isfinite(se) & (se > 0))
    # the centred intercept is zero, so the cutoff drops its direction: no variance there
    cut = estimate_covariance_beta(data, results, basis_order=1, _inverse="cutoff")
    assert np.abs(cut[0]).max() < 1e-8 * np.abs(cut).max()
    with pytest.raises(ValueError, match="_inverse"):
        estimate_covariance_beta(data, results, basis_order=1, _inverse="pinv")
    with pytest.raises(ValueError, match="_information"):
        estimate_covariance_beta(data, results, basis_order=1, _information="observed")


def _squeezed(results, data, k, s):
    """The same fit in units squeezed by s: x' = s x on [0, s], so p'(x') = p(x) / s exactly."""
    theta = np.asarray(results["theta_hat"], dtype=float).copy()
    first = 1 if k == 0 else k + 1
    for r in range(1, k + 1):
        theta[r] /= s ** r                                # x'^r = s^r x^r
    theta[first:] /= s ** k                               # (x' - s xi)_+^k = s^k (x - xi)_+^k; I(.) unchanged at k = 0
    knots = np.unique(np.asarray(results["data_points"], dtype=float)) * s
    return {"theta_hat": theta, "data_points": knots}, pd.DataFrame({"W1": data["W1"] * s})


@pytest.mark.parametrize("s", [0.08, 0.001])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_the_scaled_floor_does_not_depend_on_units(k, s):
    # round 2 (Yilong, 2026-10-24): a ridge of floor * (each column's own score variance) has no units; s = 0.001
    # also checks that the intercept is dropped whatever the units (its centred score is zero only to rounding)
    data, estimator = _fit(k)
    results = estimator.get_results()
    x = np.linspace(0.05, 0.6, 5)
    squeezed, squeezed_data = _squeezed(results, data, k, s)

    def se(res, d, points, support, **options):
        cov = estimate_covariance_beta(d, res, basis_order=k, support_refit=None, _support=support, **options)
        return density_confidence_interval(points, res, cov, basis_order=k, support_refit=None, _support=support)

    for options in ({"_inverse": "scaled_floor", "_floor": 1e-6}, {"_inverse": "scaled_floor", "_floor": 1e-3},
                    {"_inverse": "scaled_floor", "_floor": 1e-6, "_information": "model"}):
        original = se(results, data, x, (0.0, 1.0), **options)
        rescaled = se(squeezed, squeezed_data, s * x, (0.0, s), **options)
        np.testing.assert_allclose(rescaled["density"] * s, original["density"], rtol=1e-9)
        np.testing.assert_allclose(rescaled["se"] * s, original["se"], rtol=1e-8)
    if k >= 1:                                            # the absolute ridge depends on the units
        original = se(results, data, x, (0.0, 1.0), _ridge_param=1e-6)
        rescaled = se(squeezed, squeezed_data, s * x, (0.0, s), _ridge_param=1e-6)
        assert np.max(np.abs(rescaled["se"] * s / original["se"] - 1)) > 1e-3
    cov = estimate_covariance_beta(data, results, basis_order=k, support_refit=None, _inverse="scaled_floor")
    assert np.all(cov[0] == 0)                            # the intercept's centred score is zero: dropped


def test_constant_columns_are_found_relative_to_each_column():
    from haldensity.core.variance import _constant_columns
    rng = np.random.default_rng(5)
    basis = np.column_stack([np.ones(50), rng.uniform(size=50), 1e-6 * rng.uniform(size=50)])
    scores = basis - basis.mean(axis=0)
    scores[:, 0] = 1e-16                                  # the intercept's centred score: rounding only
    assert _constant_columns(basis, scores).tolist() == [True, False, False]       # a tiny column is not constant
    scaled = basis * np.array([1.0, 1e-4, 1e-4])
    assert _constant_columns(scaled, scores * np.array([1.0, 1e-4, 1e-4])).tolist() == [True, False, False]
