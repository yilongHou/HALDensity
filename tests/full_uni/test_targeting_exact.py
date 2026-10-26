"""Full-data targeting integrates exactly (Yilong, 2026-10-24): cells with edges at the knots and the targeting
points, Gauss-Legendre inside, so the score equations hold at the targeting points themselves."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.full_uni import CVXPYEstimator
from haldensity.full_uni.targeting import (MedianTargetLearner, MomentsTargetLearner, SurvivalTargetLearner,
                                           median_estimand_variance, moments_estimand_variance,
                                           survival_estimand_variance)
from haldensity.full_uni.targeting.base import BaseTargetLearner
from haldensity.full_uni.targeting.exact import ExactDensity, cell_edges, converged_rule

TARGETS = np.array([0.15, 0.3, 0.5])


def _inputs(k, norm_constraint=20.0):
    data = pd.DataFrame({"W1": np.random.default_rng(4).beta(2.0, 5.0, size=200)})
    fit = CVXPYEstimator(norm_constraint=norm_constraint, basis_order=k, log_frequency=-1).fit(data)
    first = 1 if k == 0 else k + 1
    keep = fit.theta_hat[first:] != 0
    old_theta = np.concatenate([fit.theta_hat[:first], fit.theta_hat[first:][keep]])
    return data, dict(uncensored_augmented=data, grid_points_hal_selected=fit._grid_points_hal[keep], old_theta=old_theta)


def test_exact_density_matches_adaptive_quadrature():
    theta, knots = np.array([0.3, 1.0, -2.0, 3.0, -6.0]), np.array([0.25, 0.6])   # k = 2: 1, x, x^2, two knots
    log_f = lambda x: log_unnormalized_density(x, theta, knots, 2)
    density = ExactDensity(log_f, converged_rule(log_f, cell_edges(np.append(knots, 0.4))))
    z = quad(lambda x: np.exp(log_f(np.array([x]))[0]), 0, 1, points=[0.25, 0.4, 0.6], epsabs=1e-14, epsrel=1e-13)[0]
    assert density.log_z == pytest.approx(np.log(z), abs=1e-12)
    for t in (0.1, 0.25, 0.33, 0.4, 0.9):
        reference = quad(lambda x: np.exp(log_f(np.array([x]))[0]) / z, 0, t, epsabs=1e-14, epsrel=1e-13,
                         points=[p for p in (0.25, 0.4, 0.6) if p < t] or None)[0]
        assert density.cdf(t)[0] == pytest.approx(reference, abs=1e-12)
    mean = quad(lambda x: x * np.exp(log_f(np.array([x]))[0]) / z, 0, 1, points=[0.25, 0.4, 0.6], epsabs=1e-14)[0]
    assert density.moment(1) == pytest.approx(mean, abs=1e-12)
    assert density.cdf(density.quantile(0.5))[0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("k", [0, 2])
def test_survival_and_mean_score_equations_hold_at_the_targets(k):
    data, inputs = _inputs(k)
    x = data["W1"].to_numpy()
    surv = SurvivalTargetLearner(norm_constraint=100, basis_order=k).run_m_step(**inputs, targeting_points=TARGETS)
    np.testing.assert_allclose(surv["estimate"], [np.mean(x > t) for t in TARGETS], atol=1e-9)
    assert surv["score_residual"] < 1e-9
    assert surv["plug_in"].shape == (3,) and np.all(np.diff(surv["plug_in"]) < 0)
    variance = survival_estimand_variance(targeted_fit=surv, uncensored_data=data, targeting_points=TARGETS)
    assert np.all(np.isfinite(variance) & (variance > 0))
    mean = MomentsTargetLearner(norm_constraint=100, basis_order=k).run_m_step(**inputs, x_moment=1)
    assert mean["estimate"][0] == pytest.approx(x.mean(), abs=1e-9)
    assert moments_estimand_variance(targeted_fit=mean, uncensored_data=data, x_moment=1)[0] > 0
    model = BaseTargetLearner.to_results_model(surv)
    assert model.estimate == pytest.approx(list(surv["estimate"])) and model.integration_order >= 32


def test_median_score_equation_holds_at_the_initial_median():
    data, inputs = _inputs(2)
    x = data["W1"].to_numpy()
    learner = MedianTargetLearner(norm_constraint=100, basis_order=2)
    med = learner.run_m_step(**inputs)
    m0, alpha = learner._median_cache[1], med["theta_targeting"][0]
    theta, knots = inputs["old_theta"], inputs["grid_points_hal_selected"]
    log_f = lambda z: log_unnormalized_density(z, theta, knots, 2) + alpha * (z < m0)
    targeted = ExactDensity(log_f, converged_rule(log_f, cell_edges(np.append(knots, m0))))
    assert targeted.cdf(m0)[0] == pytest.approx(np.mean(x < m0), abs=1e-9)
    assert med["estimate"][0] == pytest.approx(targeted.quantile(0.5), abs=1e-10)
    assert med["density_at_estimate"][0] == pytest.approx(targeted.density(med["estimate"])[0], rel=1e-10)
    assert median_estimand_variance(targeted_fit=med, uncensored_data=data)[0] > 0


def test_a_fit_without_exact_estimates_falls_back_with_a_warning():
    data, inputs = _inputs(2)
    surv = SurvivalTargetLearner(norm_constraint=100, basis_order=2).run_m_step(**inputs, targeting_points=TARGETS)
    old = {key: value for key, value in surv.items() if key != "estimate"}
    with pytest.warns(UserWarning, match="no exact 'estimate'"):
        survival_estimand_variance(targeted_fit=old, uncensored_data=data, targeting_points=TARGETS)


def test_targeting_works_when_the_fit_selects_no_knots():
    # review T3: a k = 0 fit with no selected knots crashed in the M-step (an empty basis); its TMLEs are still exact
    import pandas as pd

    from haldensity.full_uni import CVXPYEstimator
    from haldensity.full_uni.targeting import MomentsTargetLearner, SurvivalTargetLearner, targeting_inputs

    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2, 5, 100)})
    inputs = targeting_inputs(CVXPYEstimator(norm_constraint=1e-6, basis_order=0).fit(data))
    assert inputs["grid_points_hal_selected"].size == 0
    x = data["W1"].to_numpy()
    survival = SurvivalTargetLearner(norm_constraint=100, basis_order=0).run_m_step(
        uncensored_augmented=data, **inputs, targeting_points=np.array([0.3]))
    mean = MomentsTargetLearner(norm_constraint=100, basis_order=0).run_m_step(
        uncensored_augmented=data, **inputs, x_moment=1)
    np.testing.assert_allclose(survival["estimate"], [np.mean(x > 0.3)], atol=1e-12)
    np.testing.assert_allclose(mean["estimate"], [x.mean()], atol=1e-12)
