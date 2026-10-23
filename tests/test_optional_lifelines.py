import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from haldensity.censoring.interval.estimators import (
    IntervalCensoredFISTAEstimator,
    IntervalCensoredInitEstimator,
)


def _interval_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.0, 0.8, size=n)
    return pd.DataFrame({"L": left, "R": left + 0.1})


def test_package_imports_without_lifelines():
    code = (
        "import sys; sys.modules['lifelines'] = None; "
        "import haldensity, haldensity.censoring, haldensity.targeting"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_turnbull_knots_without_lifelines_point_to_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "lifelines", None)
    data = _interval_data()

    with pytest.raises(ImportError, match=r"haldensity\[turnbull\]"):
        IntervalCensoredInitEstimator().fit(data, knot_strategy="turnbull")

    x_mid = 0.5 * (data["L"].values + data["R"].values)
    with pytest.raises(ImportError, match=r"haldensity\[turnbull\]"):
        IntervalCensoredFISTAEstimator._build_knot_grid(data, x_mid, "turnbull", 1e-5, None, "L", "R")
