import numpy as np

from haldensity.datasets import make_right_censored
from haldensity.rc.observed_mle import (
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedFPGDEstimator,
)
from haldensity.rc.tuners import RightCensoredInitTuner


def _make_data(n: int = 48, seed: int = 17):
    return make_right_censored(n, seed=seed), None


def test_right_censored_observed_estimators_fit_valid_density():
    data, _ = _make_data()

    fista = RightCensoredObservedFISTAEstimator(
        lam=0.02,
        n_iterations=60,
        ll_change_tol=1e-4,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)
    fpgd = RightCensoredObservedFPGDEstimator(
        norm_constraint=8.0,
        n_iterations=80,
        ll_change_tol=1e-4,
        learning_rate=0.1,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)

    for estimator in (fista, fpgd):
        _, density = estimator.get_density()
        density_mass = float(np.sum(density * estimator.delta_j))
        assert np.isclose(density_mass, 1.0, atol=1e-5)
        results = estimator.get_results()
        assert int(results["n_iterations_run"]) > 0
        assert "recovery_count" in results
        assert "optimization_history" in results


def test_right_censored_observed_estimators_recover_from_one_unstable_step():
    data, _ = _make_data(seed=19)

    class OneRecoveryFISTA(RightCensoredObservedFISTAEstimator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._forced_recovery_done = False

        def _objective_is_unstable(self, obj: float) -> bool:
            if not self._forced_recovery_done and np.isfinite(obj):
                self._forced_recovery_done = True
                return True
            return super()._objective_is_unstable(obj)

    class OneRecoveryFPGD(RightCensoredObservedFPGDEstimator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._forced_recovery_done = False

        def _objective_is_unstable(self, obj: float) -> bool:
            if not self._forced_recovery_done and np.isfinite(obj):
                self._forced_recovery_done = True
                return True
            return super()._objective_is_unstable(obj)

    fista = OneRecoveryFISTA(
        lam=0.02,
        n_iterations=60,
        ll_change_tol=1e-4,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)
    fpgd = OneRecoveryFPGD(
        norm_constraint=8.0,
        n_iterations=80,
        ll_change_tol=1e-4,
        learning_rate=0.1,
        n_grid_points=80,
        basis_order=0,
    ).fit(data)

    for estimator in (fista, fpgd):
        results = estimator.get_results()
        assert int(results["n_iterations_run"]) > 0
        assert int(results["recovery_count"]) >= 1
        _, density = estimator.get_density()
        density_mass = float(np.sum(density * estimator.delta_j))
        assert np.isclose(density_mass, 1.0, atol=1e-5)


def test_right_censored_init_tuner_accepts_one_se_conservative_rule():
    data, _ = _make_data(seed=27)
    tuner = RightCensoredInitTuner(
        data=data,
        cv_folds=2,
        random_state=5,
        n_grid_points=80,
        param_overrides={"basis_order": [0], "norm_constraint": {"low": 3.0, "high": 6.0, "log": False}},
        validation_metric="ipcw_loglik",
        silent=True,
        use_conservative_adjustment=True,
        conservative_selection_rule="one_se",
        conservative_se_multiplier=1.0,
    )
    result = tuner.optimize(n_trials=2)

    assert result.metadata["validation_metric"] == "ipcw_loglik"
    assert result.metadata["conservative_selection_rule"] == "one_se"
    assert result.metadata["conservative_se_multiplier"] == 1.0
    assert result.metadata["conservative_params"] is not None
