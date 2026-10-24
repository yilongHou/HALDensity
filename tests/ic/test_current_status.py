import numpy as np
import pandas as pd
import pytest

from haldensity.datasets import interval_censor_current_status_uniform
from haldensity.ic.current_status import current_status_to_interval


def test_adapter_gives_the_simulator_intervals():
    t_event = np.random.default_rng(3).beta(2.0, 3.0, size=50)
    inspection = np.random.default_rng(11).uniform(0.0, 1.0, size=50)
    observed = pd.DataFrame({"C": inspection, "Delta": (t_event <= inspection).astype(int)})

    expected = interval_censor_current_status_uniform(t_event, random_state=11)
    pd.testing.assert_frame_equal(current_status_to_interval(observed), expected)


def test_adapter_validates_input():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        current_status_to_interval(pd.DataFrame({"C": [0.5], "Delta": [2]}))
    with pytest.raises(ValueError, match="must lie in"):
        current_status_to_interval(pd.DataFrame({"C": [1.5], "Delta": [1]}))
