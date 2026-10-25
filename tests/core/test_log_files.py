"""Estimators that log must not keep log files open (decision 9, 2026-10-24: ~250 open files per CV task)."""

import os

import numpy as np
import pandas as pd
import pytest

from haldensity.full_uni import CVXPYEstimator

FD_DIR = "/dev/fd" if os.path.isdir("/dev/fd") else None


@pytest.mark.skipif(FD_DIR is None, reason="needs /dev/fd to count open files")
def test_repeated_fits_do_not_grow_the_number_of_open_files(tmp_path):
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=100)})
    log = str(tmp_path / "logs" / "cvxpy.log")

    def fit_once():
        est = CVXPYEstimator(norm_constraint=5.0, basis_order=0, log_dir=log, log_frequency=1, _solver="CLARABEL",
                             _solver_waterfall=["CLARABEL", "ECOS"])
        est.fit(data)
        est.get_results()                               # logs the number of selected knots
        return est

    fit_once()                                          # warm-up: solver libraries may open files once
    before = len(os.listdir(FD_DIR))
    kept = [fit_once() for _ in range(20)]              # estimators kept alive: their loggers must not hold files
    assert len(os.listdir(FD_DIR)) <= before
    assert kept[-1].do_log and os.path.getsize(log) > 0  # logging still writes


def test_each_estimator_starts_its_log_afresh(tmp_path):
    log = tmp_path / "est.log"
    log.write_text("old run\n")
    data = pd.DataFrame({"W1": np.random.default_rng(1).beta(2.0, 5.0, size=80)})
    est = CVXPYEstimator(norm_constraint=5.0, basis_order=0, log_dir=str(log), log_frequency=1, _solver="CLARABEL",
                         _solver_waterfall=["CLARABEL", "ECOS"]).fit(data)
    est.get_results()
    text = log.read_text()
    assert "old run" not in text and "selected knots" in text


def test_no_log_file_unless_log_dir_is_given(tmp_path, monkeypatch):
    # quiet default (Yilong, 2026-10-24): before, CVXPYEstimator and the weighted MLE wrote ./local/logs/cvxpy.log
    # in every CV fold
    from haldensity.full_uni import OptunaHyperparameterTuner
    from haldensity.rc.estimators import RightCensoredInitEstimator

    monkeypatch.chdir(tmp_path)
    data = pd.DataFrame({"W1": np.random.default_rng(2).beta(2.0, 5.0, size=80)})
    CVXPYEstimator(norm_constraint=5.0, basis_order=1).fit(data).get_results()
    RightCensoredInitEstimator(norm_constraint=5.0, basis_order=0).fit(data).get_results()
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, cv_folds=2, silent=True, show_progress=False)
    tuner.optimize(n_trials=2, show_progress=False)
    tuner.fit_best_model()
    assert os.listdir(tmp_path) == []


def test_the_tuner_log_frequency_only_warns():
    from haldensity.full_uni import OptunaHyperparameterTuner

    data = pd.DataFrame({"W1": np.random.default_rng(3).beta(2.0, 5.0, size=40)})
    with pytest.warns(DeprecationWarning, match="estimator_kwargs"):
        tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, log_frequency=5)
    assert "log_frequency" not in tuner.estimator_kwargs


def test_log_frequency_zero_turns_logging_off_in_first_order_estimators(tmp_path):
    from haldensity.full_uni.estimators.fista import FISTAEstimator

    data = pd.DataFrame({"W1": np.random.default_rng(4).beta(2.0, 5.0, size=60)})
    est = FISTAEstimator(lam=1.0, n_iterations=30, basis_order=0, log_dir=str(tmp_path / "f.log"), log_frequency=0)
    est.fit(data)                                       # 0 divided the iteration counter before 0.2.0
    assert not est.do_log
