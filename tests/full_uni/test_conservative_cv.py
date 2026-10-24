import numpy as np
import pandas as pd
import pytest

from haldensity.core.cv.conservative import bisection_search, conservative_threshold
from haldensity.full_uni import OptunaHyperparameterTuner


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})


def test_bisection_finds_the_smallest_bound_above_the_threshold():
    calls = []

    def is_above(m, step):
        calls.append((step, m))
        return m >= 3.0              # the score falls below the threshold for m < 3

    found = bisection_search(is_above, 400.0, bracket_factor=0.5, min_norm_constraint=1e-6, rel_tol=0.02,
                             max_steps=50)
    assert 3.0 <= found <= 3.0 * 1.03
    assert calls[0] == (0, 400.0)
    assert [step for step, _ in calls] == list(range(len(calls)))


def test_thresholds():
    legacy, _ = conservative_threshold(best_mean=1.0, best_sd=0.4, n_valid=4, selection_rule="legacy_percent_sd",
                                       k_percent=0.05, se_multiplier=1.0)
    one_se, details = conservative_threshold(best_mean=1.0, best_sd=0.4, n_valid=4, selection_rule="one_se",
                                             k_percent=0.05, se_multiplier=1.0)
    assert legacy == pytest.approx(0.98) and one_se == pytest.approx(0.8)
    assert details["best_standard_error"] == pytest.approx(0.2)


def test_full_data_cv_moves_to_the_smallest_bound_above_the_threshold(data):
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=3, random_state=0,
                                      param_overrides={"basis_order": 1, "norm_constraint": (1.0, 200.0)})
    result = tuner.optimize(n_trials=6, show_progress=False)

    best_m = tuner.max_cv_params["norm_constraint"]
    chosen = tuner.conservative_params["norm_constraint"]
    threshold = tuner.adjustment_results[0]["threshold"]
    assert chosen <= best_m
    assert result["best_params"]["norm_constraint"] == chosen
    assert tuner.best_estimator_params["norm_constraint"] == chosen
    chosen_row = [r for r in tuner.adjustment_results if r["norm_constraint"] == chosen]
    assert chosen_row and chosen_row[0]["cv_ll"] >= threshold
    assert tuner.fit_best_model().norm_constraint == chosen


def test_adjustment_can_be_switched_off(data):
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=3, random_state=0,
                                      param_overrides={"basis_order": 1, "norm_constraint": (1.0, 200.0)},
                                      use_conservative_adjustment=False)
    tuner.optimize(n_trials=4, show_progress=False)
    assert tuner.conservative_params is None
    assert tuner.best_params["norm_constraint"] == tuner.max_cv_params["norm_constraint"]


def test_penalty_tuned_estimators_are_not_adjusted(data):
    tuner = OptunaHyperparameterTuner("FISTAEstimator", data, cv_folds=2, random_state=0, max_iter=50,
                                      param_overrides={"basis_order": 1})
    tuner.optimize(n_trials=2, show_progress=False)
    assert tuner.conservative_params is None
    assert "not tuned by norm_constraint" in tuner.adjustment_note
