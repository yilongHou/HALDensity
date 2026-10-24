import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from haldensity.full_uni.tuner import OptunaHyperparameterTuner


_ORIGINAL_SOLVE = cp.Problem.solve


@pytest.fixture
def mosek_unavailable(monkeypatch):
    """Make every MOSEK solve fail as it does when MOSEK is not installed."""

    def solve_without_mosek(self, *args, **kwargs):
        if str(kwargs.get("solver", "")).upper() == "MOSEK":
            raise cp.error.SolverError("The solver MOSEK is not installed.")
        return _ORIGINAL_SOLVE(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", solve_without_mosek)


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"W1": rng.beta(2.0, 5.0, size=n)})


def _tuner(**kwargs):
    return OptunaHyperparameterTuner(
        "CVXPYEstimator",
        _data(),
        cv_folds=3,
        random_state=1,
        param_overrides={"basis_order": 1, "norm_constraint": (1.0, 30.0)},
        **kwargs,
    )


def test_cv_falls_back_like_the_final_fit_when_mosek_is_unavailable(mosek_unavailable):
    tuner = _tuner()
    tuner.optimize(n_trials=3, show_progress=False)

    values = [trial.value for trial in tuner.study.trials]
    assert all(np.isfinite(v) for v in values), values

    best = tuner.fit_best_model()
    assert best.solver_used_ != "MOSEK"
    assert best.problem_status_ in {"optimal", "optimal_inaccurate"}


def test_cv_ecos_only_option(mosek_unavailable):
    tuner = _tuner(estimator_kwargs={"solver": "ECOS", "use_secondary_solver": False})
    tuner.optimize(n_trials=2, show_progress=False)

    assert all(np.isfinite(trial.value) for trial in tuner.study.trials)
    assert tuner.fit_best_model().solver_used_ == "ECOS"


def test_cv_raises_when_every_trial_fails(monkeypatch):
    def failing_solve(self, *args, **kwargs):
        raise cp.error.SolverError("solver unavailable")

    monkeypatch.setattr(cp.Problem, "solve", failing_solve)
    tuner = _tuner()

    with pytest.raises(RuntimeError, match="All 2 CV trials failed"):
        tuner.optimize(n_trials=2, show_progress=False)
    assert tuner.best_params is None


def test_estimator_kwargs_must_not_overlap_tuned_parameters():
    tuner = _tuner(estimator_kwargs={"norm_constraint": 5.0})

    with pytest.raises(ValueError, match="overlap tuned parameters"):
        tuner.optimize(n_trials=1, show_progress=False)


def test_final_fit_uses_the_best_trials_full_settings(mosek_unavailable):
    tuner = _tuner(estimator_kwargs={"solver": "ECOS", "use_secondary_solver": False})
    tuner.optimize(n_trials=2, show_progress=False)

    assert "basis_order" not in tuner.best_params  # fixed override, not suggested by Optuna
    assert tuner.best_estimator_params["basis_order"] == 1
    best = tuner.fit_best_model()
    assert best.basis_order == 1
    assert best.norm_constraint == pytest.approx(tuner.best_params["norm_constraint"])


def test_final_fit_accepts_composite_arguments():
    tuner = OptunaHyperparameterTuner(
        "ProximalNewtonLBFGSEstimator", _data(), cv_folds=2, random_state=1, max_iter=50
    )
    tuner.optimize(n_trials=1, show_progress=False)

    best = tuner.fit_best_model()  # lbfgs_gamma_clip_min/max must be rebuilt into one tuple
    assert isinstance(tuner.best_estimator_params["lbfgs_gamma_clip_range"], tuple)
    assert best.n_iterations == 50
