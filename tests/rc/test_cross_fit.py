"""Cross-fitted right-censored targeting (Yilong, 2026-10-26): cross_fit="tuner" (default), "refit", "nested", "none"."""
import numpy as np
import pandas as pd
import pytest

from haldensity.core.crossfit import FoldFits, resolve_cross_fit
from haldensity.datasets import make_right_censored
from haldensity.rc import KaplanMeier, RightCensoredInitEstimator, compute_ipcw_weights, fit_right_censored
from haldensity.rc import last_time_as_event
from haldensity.rc import targeting as T
from haldensity.rc.targeting import crossfit as C

FAST = dict(cv_folds=2, n_trials=2, random_state=7, n_grid_points=80,
            stage1_param_overrides={"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
            norm_constraint_factors=[1.0, 1.5],
            l1_kwargs={"n_grid_points": 80, "learning_rate": 0.1, "n_iterations": 60, "ll_change_tol": 1e-3})


def _pipeline(data):
    fit = fit_right_censored(data, [0.5], method="cvl1mle", **FAST)
    (estimator,) = fit.details["estimators"].values()
    return estimator


def test_cross_fit_values():
    assert resolve_cross_fit(True) == "tuner" and resolve_cross_fit(False) == "none"
    assert [resolve_cross_fit(v) for v in ("tuner", "refit", "nested", "none")] == ["tuner", "refit", "nested", "none"]
    with pytest.raises(ValueError):
        resolve_cross_fit("folds")


def test_default_uses_the_tuner_fold_fits():
    data = make_right_censored(80, seed=3)
    estimator = _pipeline(data)
    assert isinstance(estimator.fold_fits_, FoldFits) and estimator.fold_fits_.source == "tuner"
    assert estimator.fold_fits_.n_folds == 2                          # the tuner's own folds (cv_folds=2 here)
    fit = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data)
    row = fit["summary"].iloc[0]
    assert row["cross_fit"] == "tuner" and row["n_folds"] == 2
    assert abs(row["eif_mean"]) <= row["threshold"]                   # the pooled score is solved to the rule
    assert row["standard_error"] ** 2 == pytest.approx(row["estimand_variance"])
    variance = T.right_censored_mean_estimand_variance(targeted_fit=fit, observed_data=data)
    assert variance[0] == pytest.approx(row["standard_error"] ** 2, rel=1e-12)
    assert fit["cross_fit"]["eif_values"].shape == (len(data), 1)
    assert np.mean(fit["cross_fit"]["eif_values"]) == pytest.approx(row["eif_mean"], abs=1e-15)
    full = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data, cross_fit=False)
    assert row["psi_init"] == pytest.approx(full["summary"]["psi_init"].iloc[0], abs=1e-12)  # the full-sample plug-in
    assert "cross_fit" not in full["summary"].columns
    same = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data, cross_fit=True)
    assert same["summary"]["psi_star"].iloc[0] == row["psi_star"]


def test_tuner_falls_back_to_refit_without_tuner_fold_fits():
    data = make_right_censored(80, seed=4)
    converted = last_time_as_event(data)
    km = KaplanMeier().fit(converted)
    weights = compute_ipcw_weights(converted["T"].to_numpy(), converted["Delta"].to_numpy(), km.predict)
    events = converted["Delta"].to_numpy() == 1
    estimator = RightCensoredInitEstimator(norm_constraint=10.0, basis_order=0, n_grid_points=80).fit(
        pd.DataFrame({"W1": converted["T"].to_numpy()[events]}), sample_weights=weights[events])
    fit = T.right_censored_survival_targeting_M_step_v2(initial_estimator=estimator, observed_data=data,
                                                        targeting_points=[0.3, 0.6])
    assert fit["summary"]["cross_fit"].tolist() == ["refit", "refit"]
    assert fit["summary"]["n_folds"].tolist() == [5, 5]
    # a pipeline estimator used with other data than it was fitted on also refits
    other = make_right_censored(80, seed=5)
    estimator = _pipeline(data)
    assert T.right_censored_entropy_targeting_M_step(initial_estimator=estimator, observed_data=other)[
        "summary"]["cross_fit"].iloc[0] == "refit"


@pytest.mark.parametrize("target", ["mean", "entropy", "density_l2", "rmst", "survival"])
def test_one_fold_is_the_full_sample_step(target):
    # the cross-fitted loop with a single fold holding every observation (its own fit, its own Kaplan-Meier) is the
    # full-sample step: same estimates and SEs at every stage
    data = make_right_censored(120, seed=6)
    estimator = _pipeline(data)
    one = FoldFits(splits=[(np.arange(len(data)), np.arange(len(data)))], estimators=[estimator], source="tuner")
    converted = last_time_as_event(data)
    folds = C._fold_parts(one, converted, clip=1e-6, floor_scale=1.0, use_floor=True)
    spec = {"mean": C.mean_spec(1e-8), "entropy": C.entropy_spec(1e-8), "density_l2": C.density_l2_spec(1e-8),
            "rmst": C.rmst_spec(0.8, 1e-8), "survival": C.survival_spec(0.4, 1e-8)}[target]
    stages, _, _ = C._path(spec, folds, len(data), mode="iterative", max_iter=25, min_abs_eps=1e-10,
                           min_score_tol=1e-8, eps_bracket_start=0.25, eps_bracket_growth=2.0, eps_bracket_max=16.0,
                           eps_fallback_bounds=(-20.0, 20.0))
    common = dict(initial_estimator=estimator, observed_data=data, cross_fit="none")
    full = {"mean": lambda: T.right_censored_mean_targeting_M_step(**common),
            "entropy": lambda: T.right_censored_entropy_targeting_M_step(**common),
            "density_l2": lambda: T.right_censored_density_l2_targeting_M_step(**common),
            "rmst": lambda: T.right_censored_rmst_targeting_M_step(targeting_points=[0.8], **common),
            "survival": lambda: T.right_censored_survival_targeting_M_step_v2(targeting_points=[0.4], **common)}[target]()
    row = full["summary"].iloc[0]
    assert stages[0]["psi"] == pytest.approx(row["psi_init"], abs=1e-12)
    assert stages[-1]["psi"] == pytest.approx(row["psi_star"], abs=1e-12)
    assert stages[0]["se"] == pytest.approx(row["standard_error"], abs=1e-12)   # the full-sample default SE: initial


def test_uncensored_survival_and_mean_are_the_empirical_values():
    rng = np.random.default_rng(8)
    data = pd.DataFrame({"T": rng.beta(2.2, 2.5, size=100), "Delta": np.ones(100, dtype=int)})
    estimator = _pipeline(data)
    surv = T.right_censored_survival_targeting_M_step_v2(initial_estimator=estimator, observed_data=data,
                                                         targeting_points=[0.3, 0.5], min_score_tol=1e-12,
                                                         max_iter=50)["summary"]
    assert surv["cross_fit"].tolist() == ["tuner", "tuner"]
    np.testing.assert_allclose(surv["psi_star"], [np.mean(data["T"] > t) for t in (0.3, 0.5)], atol=1e-9)


def test_refit_and_nested():
    data = make_right_censored(60, seed=9)
    estimator = _pipeline(data)
    refit = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data, cross_fit="refit")
    nested = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data,
                                                    cross_fit="nested")
    assert refit["summary"]["cross_fit"].iloc[0] == "refit" and nested["summary"]["cross_fit"].iloc[0] == "nested"
    assert refit["summary"]["n_folds"].iloc[0] == 2 == nested["summary"]["n_folds"].iloc[0]   # the tuner's splits
    tuner = T.right_censored_mean_targeting_M_step(initial_estimator=estimator, observed_data=data)
    for fit in (refit, nested, tuner):
        assert np.isfinite(fit["summary"]["psi_star"].iloc[0]) and fit["summary"]["standard_error"].iloc[0] > 0
    with pytest.raises(ValueError, match="nested"):
        plain = RightCensoredInitEstimator(norm_constraint=10.0, basis_order=0, n_grid_points=80).fit(
            pd.DataFrame({"W1": data["T"].to_numpy()[data["Delta"].to_numpy() == 1]}))
        T.right_censored_mean_targeting_M_step(initial_estimator=plain, observed_data=data, cross_fit="nested")
