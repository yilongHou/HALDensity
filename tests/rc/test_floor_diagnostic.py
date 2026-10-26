"""gbar_floor_active looks only where each target's influence curve uses the censoring survival (review R3, R4)."""
import numpy as np
import pandas as pd

from haldensity.datasets import make_right_censored
from haldensity.rc import KaplanMeier, RightCensoredInitEstimator, compute_ipcw_weights, last_time_as_event
from haldensity.rc import targeting as T

CLIP = 0.4      # the censoring survival is 0.71 at t = 0.3 and 0.245 from t = 0.7 on (make_right_censored seed 0)


def _setup():
    data = make_right_censored(400, seed=0)
    flipped = last_time_as_event(data)
    km = KaplanMeier().fit(data)
    weights = compute_ipcw_weights(flipped["T"].to_numpy(), flipped["Delta"].to_numpy(), km.predict)
    events = flipped["Delta"].to_numpy() == 1
    initial = RightCensoredInitEstimator(norm_constraint=30.0, basis_order=1).fit(
        pd.DataFrame({"W1": flipped["T"].to_numpy()[events]}), sample_weights=weights[events])
    return dict(initial_estimator=initial, observed_data=data, km=km, clip=CLIP)


def test_the_floor_is_active_only_where_the_influence_curve_reaches():
    common = _setup()
    points = np.array([0.3, 0.8])
    v1 = T.right_censored_survival_targeting_M_step(targeting_points=points, **common)["summary"]
    v2 = T.right_censored_survival_targeting_M_step_v2(targeting_points=points, **common)["summary"]
    assert v1["gbar_floor_active"].tolist() == [False, True]     # survival v1 now reports it too (R4)
    assert v2["gbar_floor_active"].tolist() == [False, True]
    for tau, active in ((0.3, False), (0.8, True)):
        rmst = T.right_censored_rmst_targeting_M_step(targeting_points=[tau], **common)["summary"]
        assert bool(rmst["gbar_floor_active"].iloc[0]) is active
    mean = T.right_censored_mean_targeting_M_step(**common)["summary"]
    assert bool(mean["gbar_floor_active"].iloc[0])                 # the mean's curve uses the whole range
