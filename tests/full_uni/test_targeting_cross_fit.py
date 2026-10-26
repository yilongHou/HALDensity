"""Cross-fitted full-data targeting (Yilong, 2026-10-26): cross_fit="tuner" (default), "refit", "nested", "none"."""
import warnings

import numpy as np
import pandas as pd
import pytest

from haldensity.core.crossfit import FoldFits
from haldensity.full_uni import CVXPYEstimator, OptunaHyperparameterTuner
from haldensity.full_uni.targeting import (CDFTargetLearner, DensityL2TargetLearner, EntropyTargetLearner,
                                           MedianTargetLearner, MomentsTargetLearner, SurvivalTargetLearner,
                                           survival_estimand_variance, targeting_inputs)
from haldensity.full_uni.targeting.base import BaseTargetLearner

POINTS = np.array([0.2, 0.4, 0.7])
LEARNERS = {
    "survival": (SurvivalTargetLearner, dict(targeting_points=POINTS)),
    "cdf": (CDFTargetLearner, dict(targeting_points=POINTS)),
    "mean": (MomentsTargetLearner, dict(x_moment=1)),
    "median": (MedianTargetLearner, {}),
    "entropy": (EntropyTargetLearner, {}),
    "density L2": (DensityL2TargetLearner, {}),
}


def _data(n=200, seed=1):
    return pd.DataFrame({"W1": np.random.default_rng(seed).beta(2.0, 3.0, size=n)})


def _tuned(data, k=2, trials=4):
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, random_state=0, silent=True, show_progress=False,
                                      param_overrides={"basis_order": k})
    tuner.optimize(n_trials=trials, show_progress=False)
    return tuner.fit_best_model()


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("name", list(LEARNERS))
def test_one_fold_reproduces_the_single_step_exactly(name, k):
    # check (i): the pooled M-step with one fold (the sample's own fit, scored on the whole sample) is the
    # full-sample step, operation for operation
    data = _data(seed=10 + k)
    est = CVXPYEstimator(norm_constraint=30.0, basis_order=k).fit(data)
    one = FoldFits(splits=[(np.arange(len(data)), np.arange(len(data)))], estimators=[est], source="tuner")
    cls, extra = LEARNERS[name]
    inputs = targeting_inputs(est)
    single = cls(norm_constraint=100, basis_order=k).run_m_step(uncensored_augmented=data, **inputs, **extra,
                                                                 cross_fit="none")
    inputs["folds"] = one
    pooled = cls(norm_constraint=100, basis_order=k).run_m_step(uncensored_augmented=data, **inputs, **extra)
    assert pooled["cross_fit"] == "tuner" and pooled["n_folds"] == 1
    np.testing.assert_array_equal(pooled["theta_targeting"], single["theta_targeting"])
    np.testing.assert_array_equal(pooled["estimate"], single["estimate"])
    np.testing.assert_array_equal(pooled["plug_in_cross_fitted"], single["plug_in"])
    np.testing.assert_array_equal(pooled["plug_in"], single["plug_in"])
    assert pooled["score_residual"] == single["score_residual"]


@pytest.mark.parametrize("k", [0, 2])
def test_survival_cdf_and_mean_are_the_empirical_values(k):
    # check (ii): with the tuner's 5 folds, one alpha for all folds solves sum_v sum_{i in v} (b(x_i) - E_v b) = 0,
    # so the fold-weighted targeted value is the empirical one, as for the full-sample TMLE
    data = _data(seed=20 + k)
    est = _tuned(data, k=k)
    inputs = targeting_inputs(est)
    assert inputs["folds"] is est.fold_fits_ and est.fold_fits_.n_folds == 5
    x = data["W1"].to_numpy()
    surv = SurvivalTargetLearner(norm_constraint=100, basis_order=k).run_m_step(
        uncensored_augmented=data, **inputs, targeting_points=POINTS)
    cdf = CDFTargetLearner(norm_constraint=100, basis_order=k).run_m_step(
        uncensored_augmented=data, **inputs, targeting_points=POINTS)
    mean = MomentsTargetLearner(norm_constraint=100, basis_order=k).run_m_step(
        uncensored_augmented=data, **inputs, x_moment=1)
    for fit in (surv, cdf, mean):
        assert fit["cross_fit"] == "tuner" and fit["n_folds"] == 5
        assert fit["score_residual"] < 1e-11                     # check (iii): the pooled score equations
    np.testing.assert_allclose(surv["estimate"], [np.mean(x > t) for t in POINTS], atol=1e-11)
    np.testing.assert_allclose(cdf["estimate"], [np.mean(x <= t) for t in POINTS], atol=1e-11)
    assert mean["estimate"][0] == pytest.approx(x.mean(), abs=1e-11)
    variance = survival_estimand_variance(targeted_fit=surv, uncensored_data=data, targeting_points=POINTS)
    np.testing.assert_allclose(variance, np.var(surv["eic_values"], axis=0, ddof=1) / len(data), rtol=1e-12)
    assert BaseTargetLearner.to_results_model(surv).cross_fit == "tuner"


def test_entropy_and_l2_pooled_score_is_solved():
    # check (iii) for the targets whose influence curve depends on the density: the pooled P_n D* after the steps
    data = _data(seed=30)
    est = _tuned(data, k=2)
    inputs = targeting_inputs(est)
    entropy = EntropyTargetLearner(norm_constraint=100, basis_order=2).run_m_step(uncensored_augmented=data, **inputs)
    assert entropy["cross_fit"] == "tuner" and len(entropy["theta_targeting_path"]) == 1
    assert abs(entropy["eic_mean_path"][-1]) < 1e-12 and abs(np.mean(entropy["eic_values"])) < 1e-12
    l2 = DensityL2TargetLearner(norm_constraint=100, basis_order=2, max_steps=25).run_m_step(
        uncensored_augmented=data, **inputs)
    assert abs(l2["eic_mean_path"][-1]) <= 1e-10 and abs(np.mean(l2["eic_values"])) <= 1e-10
    one_step = DensityL2TargetLearner(norm_constraint=100, basis_order=2).run_m_step(uncensored_augmented=data, **inputs)
    assert one_step["score_residual"] < 1e-11                    # the step's own pooled score equation
    assert l2["plug_in_cross_fitted"][0] == one_step["plug_in_cross_fitted"][0]


def test_refit_fallback_cache_nested_and_the_warning():
    data = _data(n=120, seed=40)
    est = CVXPYEstimator(norm_constraint=20.0, basis_order=1).fit(data)       # no tuner: no fold fits
    inputs = targeting_inputs(est)
    assert "folds" not in inputs and inputs["fit"] is est
    surv = SurvivalTargetLearner(norm_constraint=100, basis_order=1).run_m_step(uncensored_augmented=data, **inputs,
                                                                               targeting_points=POINTS)
    assert surv["cross_fit"] == "refit" and surv["n_folds"] == 5
    cached = est._cross_fit_cache
    MomentsTargetLearner(norm_constraint=100, basis_order=1).run_m_step(uncensored_augmented=data, **inputs, x_moment=1)
    assert est._cross_fit_cache is cached and len(cached) == 1               # the second target reused the refits
    tuned = _tuned(data, k=1, trials=3)
    nested = MomentsTargetLearner(norm_constraint=100, basis_order=1).run_m_step(
        uncensored_augmented=data, **targeting_inputs(tuned), x_moment=1, cross_fit="nested")
    assert nested["cross_fit"] == "nested" and nested["n_folds"] == 5
    assert nested["estimate"][0] == pytest.approx(data["W1"].mean(), abs=1e-10)
    with pytest.raises(ValueError, match="nested"):
        MomentsTargetLearner(norm_constraint=100, basis_order=1).run_m_step(
            uncensored_augmented=data, **inputs, x_moment=1, cross_fit="nested")
    by_hand = {key: inputs[key] for key in ("old_theta", "grid_points_hal_selected")}
    with pytest.warns(UserWarning, match="No fitted estimator to cross-fit with"):
        fit = MomentsTargetLearner(norm_constraint=100, basis_order=1).run_m_step(uncensored_augmented=data, **by_hand,
                                                                                 x_moment=1)
    assert fit["cross_fit"] == "none"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert MomentsTargetLearner(norm_constraint=100, basis_order=1).run_m_step(
            uncensored_augmented=data, **by_hand, x_moment=1, cross_fit=False)["cross_fit"] == "none"
