"""The fitting grid n_grid_points is public (Yilong, 2026-10-24): the tuner passes it to every fold and to the
final fit, and every fit records the missed-mass diagnostic log_z_gap and warns above 0.01 nats."""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from haldensity.datasets import load_galaxies
from haldensity.full_uni import CVXPYEstimator, OptunaHyperparameterTuner
from haldensity.full_uni.estimators.cvxpy import MISSED_MASS_WARNING

GALAXIES_M = 596793.0          # the case study's CV choice at k = 2


def _galaxies():
    return pd.DataFrame({"W1": load_galaxies()["velocity"].to_numpy() / 3e5})


def _missed_mass_warnings(caught):
    return [str(w.message) for w in caught if str(w.message).startswith(MISSED_MASS_WARNING)]


def test_spread_data_have_no_gap_and_no_warning():
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = CVXPYEstimator(norm_constraint=10.0, basis_order=2, log_frequency=-1).fit(data)
    assert abs(fit.get_results()["log_z_gap"]) < 1e-3
    assert not _missed_mass_warnings(caught)


def test_concentrated_data_warn_with_a_grid_that_removes_the_gap():
    data = _galaxies()                                # 8% of [0, 1]; the main peak is narrower than a cell
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coarse = CVXPYEstimator(norm_constraint=GALAXIES_M, basis_order=2, log_frequency=-1).fit(data)
    messages = _missed_mass_warnings(caught)
    assert coarse.get_results()["log_z_gap"] > 0.01 and len(messages) == 1
    suggested = int(re.search(r"pass n_grid_points=(\d+) to the estimator", messages[0]).group(1))
    assert f"estimator_kwargs={{'n_grid_points': {suggested}}}" in messages[0]      # the tuner's route
    assert suggested == 2400                          # 200 cells across the data range (span 0.0837)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fine = CVXPYEstimator(norm_constraint=GALAXIES_M, basis_order=2, n_grid_points=suggested,
                              log_frequency=-1).fit(data)
    assert fine.get_results()["log_z_gap"] < 0.01 and not _missed_mass_warnings(caught)


def test_n_grid_points_is_public_and_validated():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert CVXPYEstimator(n_grid_points=500).n_grid_points == 500
    with pytest.raises(ValueError, match="n_grid_points"):
        CVXPYEstimator(n_grid_points=1)


def test_the_tuner_passes_the_grid_to_every_fold_and_the_final_fit():
    seen = []

    class Recording(CVXPYEstimator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            seen.append(self.n_grid_points)

    data = pd.DataFrame({"W1": np.random.default_rng(2).beta(2.0, 5.0, size=120)})
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=2, silent=True, show_progress=False,
                                      estimator_kwargs={"n_grid_points": 300, "log_frequency": -1})
    tuner.estimator_class = Recording
    tuner.optimize(n_trials=2, show_progress=False)
    assert seen and set(seen) == {300}
    assert tuner.fit_best_model().n_grid_points == 300


def test_the_0_1_0_tuner_option_is_passed_on_with_a_warning():
    data = pd.DataFrame({"W1": np.random.default_rng(2).beta(2.0, 5.0, size=60)})
    with pytest.warns(DeprecationWarning, match="estimator_kwargs"):
        tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, n_grid_points=300)
    assert tuner.estimator_kwargs["n_grid_points"] == 300
    with pytest.raises(ValueError, match="twice"):
        OptunaHyperparameterTuner("CVXPYEstimator", data, n_grid_points=300, estimator_kwargs={"n_grid_points": 400})


def test_the_warning_is_silent_in_cv_folds_and_shown_on_the_final_fit():
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", _galaxies(), cv_folds=2, silent=True, show_progress=False,
                                      param_overrides={"norm_constraint": GALAXIES_M},
                                      estimator_kwargs={"log_frequency": -1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuner.optimize(n_trials=1, show_progress=False)
    assert not _missed_mass_warnings(caught)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuner.fit_best_model()
    assert _missed_mass_warnings(caught)
