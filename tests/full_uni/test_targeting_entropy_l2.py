"""Full-data entropy and density-L2 targets (Yilong, 2026-10-25): exact functionals, the score equations, the
plug-in at large n, and the influence-curve variance against a small simulation."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import beta

from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.full_uni import CVXPYEstimator
from haldensity.full_uni.targeting import (DensityL2TargetLearner, EntropyTargetLearner, density_l2_estimand_variance,
                                           density_l2_targeting_M_step, entropy_estimand_variance,
                                           entropy_targeting_M_step, targeting_inputs)
from haldensity.full_uni.targeting.base import BaseTargetLearner
from haldensity.full_uni.targeting.exact import ExactDensity, cell_edges, converged_rule

TRUTH = beta(2.0, 3.0)
ENTROPY = float(TRUTH.entropy())
DENSITY_L2 = quad(lambda t: TRUTH.pdf(t) ** 2, 0, 1)[0]


def _fit(n, k, norm_constraint, seed):
    data = pd.DataFrame({"W1": np.random.default_rng(seed).beta(2.0, 3.0, size=n)})
    return data, targeting_inputs(CVXPYEstimator(norm_constraint=norm_constraint, basis_order=k).fit(data))


def test_exact_entropy_and_density_l2_match_closed_forms():
    # p(x) = c exp(a x) on [0, 1] (order 1, no knots): log Z = log((e^a - 1) / a), E X = 1 / (1 - e^-a) - 1 / a,
    # entropy = log Z - a E X, integral of p^2 = a (e^a + 1) / (2 (e^a - 1))
    a = 2.5
    log_f = lambda x: log_unnormalized_density(x, np.array([0.3, a]), np.zeros(0), 1)
    density = ExactDensity(log_f, converged_rule(log_f, cell_edges(np.zeros(0))))
    mean = 1 / (1 - np.exp(-a)) - 1 / a
    assert density.entropy() == pytest.approx(np.log((np.exp(a) - 1) / a) - a * mean, abs=1e-12)
    assert density.density_l2() == pytest.approx(a * (np.exp(a) + 1) / (2 * (np.exp(a) - 1)), abs=1e-12)
    # order 2 with knots, against adaptive quadrature
    theta, knots = np.array([0.3, 1.0, -2.0, 3.0, -6.0]), np.array([0.25, 0.6])
    log_g = lambda x: log_unnormalized_density(x, theta, knots, 2)
    density = ExactDensity(log_g, converged_rule(log_g, cell_edges(knots)))
    p = lambda t: np.exp(log_g(np.array([t]))[0] - density.log_z)
    assert density.entropy() == pytest.approx(
        -quad(lambda t: p(t) * np.log(p(t)), 0, 1, points=[0.25, 0.6], epsabs=1e-14)[0], abs=1e-12)
    assert density.density_l2() == pytest.approx(quad(lambda t: p(t) ** 2, 0, 1, points=[0.25, 0.6],
                                                      epsabs=1e-14)[0], abs=1e-12)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_one_entropy_step_solves_the_influence_curve_equation(k):
    # along p^(1 - alpha) / Z the influence curve -log p - psi is affine in log p, so the step's score equation is
    # P_n D* = 0; a second step changes nothing
    data, inputs = _fit(300, k, 30.0, seed=301 + k)
    x = data["W1"].to_numpy()
    learner = EntropyTargetLearner(norm_constraint=100, basis_order=k, max_steps=5)
    fit = learner.run_m_step(uncensored_augmented=data, **inputs)
    assert len(fit["theta_targeting_path"]) == 1
    assert abs(fit["eic_mean_path"][-1]) < 1e-12 and fit["score_residual"] < 1e-12
    assert abs(np.mean(learner.eic_values(fit, x))) < 1e-12
    assert fit["estimate"][0] != fit["plug_in"][0]
    variance = entropy_estimand_variance(targeted_fit=fit, uncensored_data=data, basis_order=k)
    assert variance.shape == (1,) and variance[0] > 0
    assert BaseTargetLearner.to_results_model(fit).theta_targeting_path == pytest.approx(
        list(fit["theta_targeting_path"]))


@pytest.mark.parametrize("k", [0, 1, 2])
def test_density_l2_steps_repeat_until_the_influence_curve_equation_holds(k):
    data, inputs = _fit(300, k, 30.0, seed=301 + k)
    x = data["W1"].to_numpy()
    one = density_l2_targeting_M_step(data, **inputs, norm_constraint=100, basis_order=k)
    assert one["score_residual"] < 1e-12                       # the step's own score equation P_n b = E b
    assert len(one["theta_targeting_path"]) == 1
    learner = DensityL2TargetLearner(norm_constraint=100, basis_order=k, max_steps=25)
    repeated = learner.run_m_step(uncensored_augmented=data, **inputs)
    assert abs(repeated["eic_mean_path"][-1]) <= 1e-10
    assert abs(np.mean(learner.eic_values(repeated, x))) <= 1e-10
    assert 1 < len(repeated["theta_targeting_path"]) < 25
    # one step leaves P_n D* of second order; the steps after it move the estimate by about that much
    assert abs(repeated["estimate"][0] - one["estimate"][0]) <= 2 * abs(one["eic_mean_path"][0]) + 1e-12
    assert repeated["plug_in"][0] == one["plug_in"][0]
    assert density_l2_estimand_variance(targeted_fit=repeated, uncensored_data=data, basis_order=k)[0] > 0


@pytest.mark.parametrize("learner_class", [EntropyTargetLearner, DensityL2TargetLearner])
def test_a_uniform_start_leaves_nothing_to_target(learner_class):
    # a fit with no knots and no slope (e.g. k = 0 at M = 0) is uniform: D* is constant, the coefficient is 0
    data = pd.DataFrame({"W1": np.random.default_rng(0).uniform(size=50)})
    learner = learner_class(norm_constraint=10, basis_order=0, max_steps=3)
    fit = learner.run_m_step(uncensored_augmented=data, grid_points_hal_selected=np.zeros(0), old_theta=np.zeros(1))
    assert list(fit["theta_targeting_path"]) == [0.0] and fit["solver_status"] == "constant influence curve"
    assert fit["estimate"][0] == pytest.approx(fit["plug_in"][0], abs=1e-14)
    assert fit["estimate"][0] == pytest.approx(0.0 if learner_class is EntropyTargetLearner else 1.0, abs=1e-12)
    assert learner.get_estimand_variance(fit, data)[0] == pytest.approx(0.0, abs=1e-24)


def test_plug_in_and_tmle_recover_a_beta_density_at_large_n():
    data, inputs = _fit(2000, 2, 60.0, seed=7)
    entropy = entropy_targeting_M_step(data, **inputs, norm_constraint=100, basis_order=2)
    l2 = density_l2_targeting_M_step(data, **inputs, norm_constraint=100, basis_order=2)
    se_entropy = np.sqrt(entropy_estimand_variance(targeted_fit=entropy, uncensored_data=data, basis_order=2)[0])
    se_l2 = np.sqrt(density_l2_estimand_variance(targeted_fit=l2, uncensored_data=data, basis_order=2)[0])
    for value in (entropy["plug_in"][0], entropy["estimate"][0]):
        assert abs(value - ENTROPY) < 3 * se_entropy
    for value in (l2["plug_in"][0], l2["estimate"][0]):
        assert abs(value - DENSITY_L2) < 3 * se_l2


def test_influence_curve_variance_against_a_small_simulation():
    # 60 samples of n = 400, order 1, M fixed at 20: mean Var(D*) / n over the Monte Carlo variance of the TMLE
    # was 1.53 for the entropy and 1.48 for density L2 when this test was written (order 2: 1.25 and 1.18); the
    # check is that they agree within a factor of 2
    estimates, variances = {"entropy": [], "l2": []}, {"entropy": [], "l2": []}
    for rep in range(60):
        data, inputs = _fit(400, 1, 20.0, seed=100 + rep)
        for name, learner in (("entropy", EntropyTargetLearner(norm_constraint=100, basis_order=1)),
                              ("l2", DensityL2TargetLearner(norm_constraint=100, basis_order=1))):
            fit = learner.run_m_step(uncensored_augmented=data, **inputs)
            estimates[name].append(fit["estimate"][0])
            variances[name].append(learner.get_estimand_variance(fit, data)[0])
    for name in estimates:
        ratio = np.mean(variances[name]) / np.var(estimates[name], ddof=1)
        assert 0.5 < ratio < 2.0, (name, ratio)
