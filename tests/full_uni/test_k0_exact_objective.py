import numpy as np
import pandas as pd
import pytest

from haldensity.full_uni import CVXPYEstimator


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=300)})


def test_exact_k0_objective_avoids_the_spike_between_close_knots(data):
    # At M = 50 the midpoint objective can put mass in a cell between two close knots that holds no
    # grid midpoint; the exact objective sees every cell.
    x = data["W1"].to_numpy()
    fits = {
        name: CVXPYEstimator(norm_constraint=50.0, basis_order=0, log_frequency=-1, objective_normalizer=name).fit(data)
        for name in ("midpoint", "exact")
    }
    train_ll = {name: float(np.mean(fit.get_log_density_at_points(x))) for name, fit in fits.items()}
    assert train_ll["exact"] > 0.5
    assert train_ll["exact"] > train_ll["midpoint"]


def test_exact_objective_is_k0_only():
    with pytest.raises(ValueError, match="basis_order=0 only"):
        CVXPYEstimator(basis_order=1, objective_normalizer="exact")
    with pytest.raises(ValueError, match="objective_normalizer must be"):
        CVXPYEstimator(objective_normalizer="gauss_legendre")
