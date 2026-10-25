"""Public vs internal options (Yilong, 2026-10-24): numerical choices are the package's job."""

import inspect

import numpy as np
import pandas as pd
import pytest

from haldensity.core.variance import estimate_covariance_beta
from haldensity.full_uni import CVXPYEstimator, OptunaHyperparameterTuner


def _public(f):
    return [name for name in inspect.signature(f).parameters if not name.startswith("_") and name != "deprecated"]


def _data(n=120):
    return pd.DataFrame({"W1": np.random.default_rng(6).beta(2.0, 5.0, size=n)})


def test_public_signatures():
    # n_grid_points is public again (Yilong, 2026-10-24): whether the data are too concentrated is the user's call
    assert _public(CVXPYEstimator) == ["norm_constraint", "basis_order", "max_knots", "n_grid_points",
                                       "solver_threads", "log_dir", "log_frequency"]
    assert _public(OptunaHyperparameterTuner) == [
        "estimator_name", "data", "cv_folds", "metric", "random_state", "max_iter", "tolerance",
        "silent", "show_progress", "param_overrides", "estimator_kwargs", "use_conservative_adjustment",
        "conservative_selection_rule", "conservative_k_percent", "conservative_se_multiplier"]
    assert _public(estimate_covariance_beta) == ["data", "estimation_results", "basis_order", "support_refit"]


@pytest.mark.parametrize("old,value,attribute", [("tol", 1e-3, "tol"),
                                                 ("solver", "CLARABEL", "solver"),
                                                 ("use_secondary_solver", False, "use_secondary_solver"),
                                                 ("solver_waterfall", ["ECOS"], "solver_waterfall")])
def test_estimator_options_of_0_1_0_warn_and_still_work(old, value, attribute):
    with pytest.warns(DeprecationWarning, match=f"_{old}=") as caught:
        est = CVXPYEstimator(**{old: value})
    assert "0.3.0" in str(caught[0].message)
    assert getattr(est, attribute) == value


def test_tuner_and_variance_options_of_0_1_0_warn_and_still_work():
    with pytest.warns(DeprecationWarning, match="estimator_kwargs"):
        tuner = OptunaHyperparameterTuner("CVXPYEstimator", _data(), n_grid_points=300)
    assert tuner.estimator_kwargs == {"n_grid_points": 300}       # passed on (it had no effect in 0.1.0)
    data = _data()
    fit = CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1).fit(data)
    results = fit.get_results()
    with pytest.warns(DeprecationWarning, match="_ridge_param="):
        old = estimate_covariance_beta(data, results, 1, ridge_param=1e-4)
    # the 0.1.0 option keeps the 0.1.0 rule, an absolute ridge (the default is the scaled floor since 0.2.0)
    np.testing.assert_array_equal(old, estimate_covariance_beta(data, results, 1, _inverse="ridge", _ridge_param=1e-4))


def test_options_new_in_0_2_0_are_internal_only():
    with pytest.raises(TypeError, match="solver_tolerance"):
        CVXPYEstimator(solver_tolerance="default")
    with pytest.raises(TypeError, match="normalizer"):
        CVXPYEstimator(normalizer="midpoint")
    with pytest.raises(TypeError, match="cv_scoring"):
        OptunaHyperparameterTuner("CVXPYEstimator", _data(), cv_scoring="interpolated")
    with pytest.raises(TypeError):
        CVXPYEstimator(3.0)                                   # keyword-only, so old positional calls fail loudly


def test_private_hooks_work():
    data = _data()
    loose = CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, _solver_tolerance="default",
                           _normalizer="midpoint", _tol=1e-3).fit(data)
    assert loose.certified_ is None and loose.normalizer == "midpoint" and loose.tol == 1e-3
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=3, _cv_solver_tolerance="tight",
                                      _conservative_max_steps=4)
    assert tuner.cv_solver_tolerance == "tight" and tuner.conservative_max_steps == 4
