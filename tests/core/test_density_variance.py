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
