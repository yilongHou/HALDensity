"""Internal k = 0 safeguard (Yilong, 2026-10-24): warn and ask for a report if the final fit shows the CVLL imbalance."""

import warnings

import numpy as np
import pandas as pd
import pytest

from haldensity.core.univariate.imbalance import check_imbalance, imbalance_diagnostics
from haldensity.full_uni import CVXPYEstimator


def _spiky():
    knots = np.linspace(0.05, 0.95, 50)
    knots[20] = knots[19] + 5e-4                          # two knots 5e-4 apart
    theta = np.zeros(1 + knots.size)
    theta[1 + 19], theta[1 + 20] = 40.0, -40.0            # huge opposite steps that cancel
    theta[1 + 5], theta[1 + 30] = 0.3, -0.4
    return theta, knots


def test_flags_a_constructed_spiky_fit():
    theta, knots = _spiky()
    with pytest.warns(UserWarning, match="please report it"):
        diagnostics = check_imbalance(theta, knots)
    assert diagnostics["flag_a"] and diagnostics["flag_b"]
    pair = diagnostics["flag_a_pairs"][0]
    assert pair["coefficients"] == (40.0, -40.0) and pair["gap"] < 1e-3
    assert diagnostics["share_l1_above_10"] == pytest.approx(80.0 / 80.7)


def test_flag_b_alone():
    theta, knots = _spiky()
    theta[1 + 20] = 0.0                                   # one huge coefficient, no opposite partner
    diagnostics = imbalance_diagnostics(theta, knots)
    assert not diagnostics["flag_a"] and diagnostics["flag_b"]


def test_a_normal_fit_is_not_flagged_and_records_the_diagnostics():
    data = pd.DataFrame({"W1": np.random.default_rng(5).beta(2.0, 5.0, size=300)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)          # no imbalance warning (nor any other)
        fit = CVXPYEstimator(norm_constraint=10.0, basis_order=0, log_frequency=-1).fit(data)
    recorded = fit.get_results()["imbalance"]
    assert recorded is not None and not recorded["flag_a"] and not recorded["flag_b"]
    assert recorded["max_abs_theta"] < 10 and 0 < recorded["max_cell_mass_share"] < 1
    assert CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1).fit(data).get_results()["imbalance"] is None
