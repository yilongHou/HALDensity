import numpy as np
import pytest

from haldensity.datasets import make_right_censored
from haldensity.rc import (
    METHODS,
    RightCensoredHALFit,
    RightCensoredPointEstimate,
    fit_cvl1mle,
    fit_right_censored,
    fit_undersmoothed,
)

POINTS = np.array([0.5, 0.25])
KWARGS = {
    "cv_folds": 2,
    "n_trials": 2,
    "random_state": 7,
    "n_grid_points": 80,
    "stage1_param_overrides": {"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
    "norm_constraint_factors": [1.0, 1.5, 2.0],
    "l1_kwargs": {"n_grid_points": 80, "learning_rate": 0.1, "n_iterations": 120, "ll_change_tol": 1e-3},
}


@pytest.fixture(scope="module")
def data():
    return make_right_censored(64, seed=17)


@pytest.fixture(scope="module")
def estimates(data):
    return {method: fit_right_censored(data, POINTS, method=method, **KWARGS) for method in METHODS}


def test_all_methods_share_inputs_and_outputs(estimates):
    for method, est in estimates.items():
        assert isinstance(est, RightCensoredPointEstimate) and est.method == method
        assert list(est.summary.columns) == ["t0", "survival", "density", "norm_factor", "norm_constraint"]
        np.testing.assert_allclose(est.summary["t0"], np.sort(POINTS))
        assert np.all((est.summary["survival"] >= 0.0) & (est.summary["survival"] <= 1.0))
        assert np.all(est.summary["density"] >= 0.0)
        np.testing.assert_allclose(est.summary["norm_constraint"], est.summary["norm_factor"] * est.stage1_norm_constraint)
        for t0 in POINTS:
            assert isinstance(est.fit_at(t0), RightCensoredHALFit)
    # all four share the same Stage-1 fit, hence the same M1
    assert len({est.stage1_norm_constraint for est in estimates.values()}) == 1


def test_ipcw_hal_and_l1mle_m1_use_bound_m1(estimates):
    for method in ("ipcw_hal", "l1mle_m1"):
        assert set(estimates[method].fits) == {1.0}
    stage1 = estimates["ipcw_hal"].details["stage1_tuning_result"].estimator
    np.testing.assert_allclose(estimates["ipcw_hal"].summary["density"], stage1.get_density_at_points(np.sort(POINTS)))


def test_cvl1mle_uses_the_cross_validated_bound(estimates):
    est = estimates["cvl1mle"]
    chosen = est.details["cv_tuning_result"].best_params["norm_constraint_factor"]
    assert set(est.fits) == {chosen}
    assert np.all(est.summary["norm_factor"] == chosen)


def test_undersmoothed_rule_and_consistency_with_l1mle_m1(estimates):
    est = estimates["undersmoothed"]
    selection = est.details["selection"]
    long = est.details["targeting_summary_long"]
    for _, row in selection.iterrows():
        group = long[np.isclose(long["t0"], row["t0"])].sort_values("norm_factor")
        passing = group[group["passes_final"]]
        expected = passing["norm_factor"].iloc[0] if not passing.empty else \
            group["norm_factor"].iloc[int(np.argmin(np.abs(group["eif_mean_final"].to_numpy())))]
        assert row["norm_factor"] == expected
    # the path's factor-1.0 fit is the l1mle_m1 fit
    np.testing.assert_array_equal(
        est.details["path_estimators"][1.0].get_density_at_points(np.sort(POINTS)),
        estimates["l1mle_m1"].summary["density"].to_numpy(),
    )


def test_named_functions_and_default_method(data):
    assert fit_cvl1mle.__name__ == "fit_cvl1mle" and fit_undersmoothed.__name__ == "fit_undersmoothed"
    with pytest.raises(TypeError, match="method is fixed"):
        fit_cvl1mle(data, POINTS, method="ipcw_hal")
    with pytest.raises(ValueError, match="method must be one of"):
        fit_right_censored(data, POINTS, method="km")
    with pytest.raises(ValueError, match="fixed by the method"):
        fit_right_censored(data, POINTS, l1_kwargs={"norm_constraint": 2.0})
    import inspect
    assert inspect.signature(fit_right_censored).parameters["method"].default == "cvl1mle"
