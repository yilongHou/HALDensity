import numpy as np
import pandas as pd
import pytest

from haldensity.datasets import interval_censor, load_galaxies, make_right_censored


def test_load_galaxies_returns_the_classic_82_velocities():
    data = load_galaxies()
    assert list(data.columns) == ["id", "velocity"]
    assert len(data) == 82
    assert data["velocity"].min() == 9172 and data["velocity"].max() == 34279
    # kept as in Roeder (1990) / MASS::galaxies, including the documented typo
    assert int(data.loc[data["id"] == 78, "velocity"].iloc[0]) == 26690


@pytest.mark.parametrize("event_clip", [None, (1e-3, 0.999)])
def test_make_right_censored_reproduces_the_legacy_draws(event_clip):
    rng = np.random.default_rng(17)
    event = rng.beta(2.2, 2.5, size=64)
    if event_clip is not None:
        event = np.clip(event, *event_clip)
    censor = rng.uniform(0.0, 1.2, size=64)
    expected = pd.DataFrame({"T": np.minimum(event, censor), "Delta": (event <= censor).astype(int)})

    pd.testing.assert_frame_equal(make_right_censored(64, seed=17, event_clip=event_clip), expected)


def test_interval_censor_dispatch_has_no_informative_mechanism():
    t_event = np.random.default_rng(0).beta(2.0, 3.0, size=20)
    intervals = interval_censor(t_event, mechanism="inspection_uniform", random_state=0)
    assert np.all(intervals["L"] <= intervals["R"])
    with pytest.raises(ValueError, match="Unsupported interval censoring mechanism"):
        interval_censor(t_event, mechanism="width_uniform")
