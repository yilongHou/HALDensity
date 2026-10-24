"""The objective's normalizer is the package's choice, by order k and dimension d (Yilong, 2026-10-24)."""

import numpy as np
import pandas as pd
import pytest

from haldensity.core.objective_rules import (
    EXACT,
    EXACT_OR_SOBOL,
    MIDPOINT,
    MIDPOINT_KNOTS_ADDED,
    SOBOL,
    objective_rule,
)
from haldensity.core.univariate.objective import resolve_objective_normalizer
from haldensity.core.univariate.support import FullDataHALProblem, refit_support, refit_support_from_estimator
from haldensity.core.variance import _auto_refit
from haldensity.full_uni import CVXPYEstimator, OptunaHyperparameterTuner


def _data(n=150, seed=0):
    return pd.DataFrame({"W1": np.random.default_rng(seed).beta(2.0, 5.0, size=n)})


def test_rule_for_each_order_and_dimension():
    assert [objective_rule(k, 1) for k in (0, 1, 2, 3)] == [EXACT, MIDPOINT_KNOTS_ADDED, MIDPOINT, MIDPOINT]
    assert [objective_rule(k, d) for k in (0, 1, 2) for d in (2, 3)] == [EXACT_OR_SOBOL] * 2 + [SOBOL] * 4
    with pytest.raises(ValueError):
        objective_rule(-1, 1)
    with pytest.raises(ValueError):
        objective_rule(0, 0)


def test_univariate_resolution():
    assert [resolve_objective_normalizer("auto", k) for k in (0, 1, 2)] == [
        ("knot_aware", EXACT), ("knot_aware", MIDPOINT_KNOTS_ADDED), ("midpoint", MIDPOINT)]
    assert resolve_objective_normalizer("midpoint", 0) == ("midpoint", MIDPOINT)
    assert resolve_objective_normalizer("knot_aware", 2) == ("knot_aware", MIDPOINT_KNOTS_ADDED)
    with pytest.raises(ValueError, match="objective_normalizer"):
        resolve_objective_normalizer("exact", 0)


@pytest.mark.parametrize("k,explicit,rule", [(0, "knot_aware", EXACT), (1, "knot_aware", MIDPOINT_KNOTS_ADDED),
                                             (2, "midpoint", MIDPOINT)])
def test_the_default_is_the_rule_of_the_order(k, explicit, rule):
    data = _data()
    default = CVXPYEstimator(norm_constraint=10.0, basis_order=k, log_frequency=-1).fit(data)
    given = CVXPYEstimator(norm_constraint=10.0, basis_order=k, log_frequency=-1, _objective_normalizer=explicit).fit(data)
    np.testing.assert_array_equal(default.optimized_theta_raw, given.optimized_theta_raw)
    results = default.get_results()
    assert results["objective_rule"] == rule and results["objective_normalizer"] == explicit


def test_the_choice_is_not_a_user_option():
    import inspect
    public = [name for name in inspect.signature(CVXPYEstimator).parameters if not name.startswith("_")]
    assert "objective_normalizer" not in public and not hasattr(CVXPYEstimator, "knot_aware_formulation")
    # the internal hook still forces a rule (tests, benchmarks, experiment repos)
    forced = CVXPYEstimator(norm_constraint=10.0, basis_order=0, log_frequency=-1, _objective_normalizer="midpoint")
    assert forced.fit(_data()).get_results()["objective_rule"] == MIDPOINT
    with pytest.raises(ValueError, match="_objective_normalizer"):
        CVXPYEstimator(_objective_normalizer="auto")


def test_a_joint_search_records_each_trials_objective():
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", _data(n=120), cv_folds=3, random_state=0,
                                      param_overrides={"basis_order": [0, 1, 2],
                                                       "norm_constraint": {"low": 1.0, "high": 50.0, "log": True}},
                                      use_conservative_adjustment=False)
    tuner.optimize(n_trials=9, show_progress=False)
    seen = {}
    for trial in tuner.study.trials:
        k = trial.params.get("basis_order", tuner._trial_estimator_params[trial.number]["basis_order"])
        assert trial.user_attrs["objective_rule"] == objective_rule(k, 1)
        seen[k] = trial.user_attrs["objective_rule"]
    assert len(seen) >= 2                                  # more than one order was tried


def test_basis_cleaning_and_the_se_rebuild_the_same_problem():
    data = _data(seed=3)
    tight = CVXPYEstimator(norm_constraint=8.0, basis_order=0, log_frequency=-1).fit(data)
    refit = refit_support_from_estimator(tight, data)
    np.testing.assert_array_equal(refit.knots, tight._grid_points_hal)
    np.testing.assert_allclose(refit.theta[1:], tight.theta_hat[1:], atol=1e-6)
    assert FullDataHALProblem(data, 0).objective_rule == EXACT          # the problem cleaning rebuilds
    # a loosely solved fit's SE refit resolves "auto" the same way as an explicit exact objective
    loose = CVXPYEstimator(norm_constraint=8.0, basis_order=0, log_frequency=-1, solver_tolerance="default").fit(data)
    cleaned = _auto_refit(data, loose.get_results(), 0)
    explicit = refit_support(data, 0, 8.0, objective_normalizer="knot_aware")
    np.testing.assert_array_equal(cleaned.theta, explicit.theta)


def test_the_multivariate_estimator_records_its_rule():
    from haldensity.datasets.bivariate_truncated_normal import BivariateTruncatedNormal
    from haldensity.full_multi.estimator import MultvarHAL
    dgp = BivariateTruncatedNormal(mean=(0.45, 0.60), covariance=[[0.015, 0.006], [0.006, 0.020]])
    data = dgp.generate_samples(24, seed=11)
    model = MultvarHAL(k=1, norm_constraint=25.0, support=dgp.bounds, quadrature_points_per_dim=10).fit(data)
    assert model.get_results()["objective_rule"] == SOBOL and model.summary()["normalizer_method_used"] == "sobol"
