"""Decision 6 (Yilong, 2026-10-24): the default order is fixed k = 2; CV over the order on request."""

import inspect

import numpy as np
import pandas as pd

from haldensity.core.defaults import DEFAULT_BASIS_ORDER
from haldensity.full_uni import CVXPYEstimator, FISTAEstimator, OptunaHyperparameterTuner
from haldensity.full_uni.targeting.base import BaseTargetLearner
from haldensity.full_uni.targeting.survival import survival_targeting_M_step


def test_estimators_and_targeting_default_to_order_2():
    assert DEFAULT_BASIS_ORDER == 2
    assert CVXPYEstimator().basis_order == 2 and FISTAEstimator().basis_order == 2
    assert BaseTargetLearner().basis_order == 2
    assert inspect.signature(survival_targeting_M_step).parameters["basis_order"].default == 2


def test_the_tuner_fixes_the_order_unless_asked():
    data = pd.DataFrame({"W1": np.random.default_rng(2).beta(2.0, 5.0, size=100)})
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=3, random_state=0,
                                      param_overrides={"norm_constraint": {"low": 1.0, "high": 50.0, "log": True}})
    tuner.optimize(n_trials=3, show_progress=False)
    assert all(params["basis_order"] == 2 for params in tuner._trial_estimator_params.values())
    assert "basis_order" not in tuner.study.best_params                 # not searched
    assert tuner.fit_best_model().basis_order == 2
    asked = OptunaHyperparameterTuner("CVXPYEstimator", data, param_overrides={"basis_order": [0, 1, 2]})
    assert asked.param_overrides["basis_order"] == [0, 1, 2]
