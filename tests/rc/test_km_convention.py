"""The Kaplan-Meier convention: when the largest observed time is censored, it is treated as an event."""
import numpy as np
import pandas as pd
import pytest

from haldensity.datasets import make_right_censored
from haldensity.rc import (
    KaplanMeier,
    RightCensoredInitEstimator,
    compute_ipcw_weights,
    fit_right_censored,
    last_time_as_event,
)
from haldensity.rc.targeting import right_censored_mean_targeting_M_step


def _censored_last(n, start=0):
    """The first ``make_right_censored`` sample (seed >= start) whose largest observed time is censored."""
    for seed in range(start, start + 500):
        data = make_right_censored(n, seed=seed)
        if data.loc[data["T"].idxmax(), "Delta"] == 0:
            return data
    raise RuntimeError("no sample with a censored largest time")


def test_last_time_as_event_changes_only_a_censored_largest_time():
    data = pd.DataFrame({"T": [0.3, 0.9, 0.5], "Delta": [1, 0, 0]})
    out = last_time_as_event(data)
    assert out["Delta"].tolist() == [1, 1, 0]
    assert data["Delta"].tolist() == [1, 0, 0]  # the input is not modified
    assert out["Delta"].dtype == data["Delta"].dtype
    assert last_time_as_event(out) is out  # applying it twice changes nothing
    ties = pd.DataFrame({"T": [0.9, 0.2, 0.9], "Delta": [0, 1, 1]})
    assert last_time_as_event(ties)["Delta"].tolist() == [1, 1, 1]
    assert last_time_as_event(data.iloc[:0]).empty


def test_censoring_survival_stays_positive_with_no_jump_at_the_largest_time():
    data = _censored_last(200)
    t_max = float(data["T"].max())
    km = KaplanMeier().fit(data)
    assert km.predict(t_max) > 0.0
    assert km.predict(t_max) == km.predict(np.nextafter(t_max, 0.0))
    assert km.predict(1.0) == km.predict(t_max)
    np.testing.assert_array_equal(KaplanMeier().fit(last_time_as_event(data)).stepwise_survival_()[1],
                                  km.stepwise_survival_()[1])


def test_ipcw_weights_add_up_to_the_sample_size():
    # sum_i Delta_i / G(T_i-) = n F_KM(T_max) (Satten and Datta 2001), and F_KM(T_max) = 1 once the
    # largest time is an event
    data = last_time_as_event(_censored_last(300))
    km = KaplanMeier().fit(data)
    weights = compute_ipcw_weights(data["T"].to_numpy(), data["Delta"].to_numpy(), km.predict)
    assert weights.sum() == pytest.approx(len(data), rel=1e-12)


def test_mean_targeting_uses_every_observation_when_the_largest_time_is_censored():
    # the rc/02 tutorial's sample: before the convention the influence curve reached -2,595 at the
    # censored largest time, the SE over all observations was 7.6 and the default SE left out every
    # censored observation
    data = make_right_censored(400, seed=2)
    assert data.loc[data["T"].idxmax(), "Delta"] == 0
    flipped = last_time_as_event(data)
    km = KaplanMeier().fit(data)
    weights = compute_ipcw_weights(flipped["T"].to_numpy(), flipped["Delta"].to_numpy(), km.predict)
    events = flipped["Delta"].to_numpy() == 1
    initial = RightCensoredInitEstimator(norm_constraint=30.0, basis_order=1).fit(
        pd.DataFrame({"W1": flipped["T"].to_numpy()[events]}), sample_weights=weights[events])
    summary = right_censored_mean_targeting_M_step(initial_estimator=initial, observed_data=data, km=km)["summary"]
    row = summary.iloc[0]
    assert row["clip_active_eic_excluded_count"] == 0
    assert row["standard_error"] == pytest.approx(row["standard_error_all_eic"], rel=1e-12)
    assert row["standard_error"] < 0.05


def test_pipeline_counts_a_censored_largest_time_as_the_last_failure():
    data = _censored_last(64)
    est = fit_right_censored(
        data, [0.5], method="ipcw_hal", cv_folds=2, n_trials=2, random_state=7, n_grid_points=80,
        stage1_param_overrides={"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
    )
    assert est.fit_at(0.5).last_failure == pytest.approx(float(data["T"].max()))
