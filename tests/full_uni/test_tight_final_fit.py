"""The final fit is one tight, certified solve: its solution is the point estimate and the SE basis."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from haldensity.core.solvers.cvxpy import MOSEK_TIGHT_PARAMS, thread_options, tight_ladder
from haldensity.core.univariate.support import FullDataHALProblem, kkt_report, refit_support_from_estimator
from haldensity.core.variance import _REFIT_CACHE, density_confidence_interval, estimate_covariance_beta
from haldensity.full_uni import CVXPYEstimator

CASES = [(0, "midpoint", 5.0), (0, "midpoint", 20.0), (1, "midpoint", 20.0), (1, "midpoint", 50.0),
         (2, "midpoint", 20.0), (2, "midpoint", 100.0), (0, "knot_aware", 20.0), (1, "knot_aware", 50.0)]
FREE_SOLVERS = dict(_solver="CLARABEL", _solver_waterfall=["CLARABEL", "ECOS", "SCS"])
@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(200).beta(2.0, 5.0, size=200)})


def test_tight_tolerances_reach_every_solver(monkeypatch, data):
    calls = []

    def record(self, *args, **kwargs):
        calls.append(kwargs)
        raise cp.error.SolverError("recorded")

    monkeypatch.setattr(cp.Problem, "solve", record)
    with pytest.raises(RuntimeError, match="No solver configuration"):
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, _objective_normalizer="midpoint").fit(data)
    assert [c["solver"] for c in calls] == ["MOSEK", "CLARABEL", "CLARABEL", "ECOS", "ECOS", "SCS"]
    # the tolerances, plus the single-thread setting (haldensity.core.solvers.thread_options)
    assert calls[0]["mosek_params"] == {**MOSEK_TIGHT_PARAMS, "MSK_IPAR_NUM_THREADS": 1}
    assert calls[0]["accept_unknown"] is True
    assert all(calls[0]["mosek_params"][f"MSK_DPAR_INTPNT_CO_TOL_{name}"] == 1e-12
               for name in ("REL_GAP", "PFEAS", "DFEAS"))
    clarabel_threads = thread_options("CLARABEL")
    assert calls[1] == dict(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=400,
                            **clarabel_threads)
    assert calls[2] == dict(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300,
                            **clarabel_threads)
    assert calls[3] == dict(solver="ECOS", abstol=1e-12, reltol=1e-12, feastol=1e-12, max_iters=500)
    assert calls[4] == dict(solver="ECOS", abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=300)
    assert calls[5] == dict(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000, time_limit_secs=600.0)

    # solver_tolerance="default" passes no tolerance (v0.1.0), only the thread setting
    calls.clear()
    with pytest.raises(RuntimeError):
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, _solver_tolerance="default",
                       _objective_normalizer="midpoint").fit(data)
    assert calls and all(set(c) == {"solver"} | set(thread_options(c["solver"])) for c in calls)


def test_tight_ladder_follows_the_solver_settings():
    assert [s for s, _, _ in tight_ladder(["ECOS", "MOSEK"])] == ["ECOS", "ECOS", "MOSEK"]
    assert tight_ladder(["OSQP"]) == (("OSQP", "solver defaults", {}),)


@pytest.mark.parametrize("k,normalizer,M", CASES)
def test_the_final_fit_is_the_old_cleaned_basis(data, k, normalizer, M):
    # v0.1.0 fit + separate cleaning refit (free solvers, as before MOSEK joined the ladder) ...
    old = CVXPYEstimator(norm_constraint=M, basis_order=k, _objective_normalizer=normalizer, log_frequency=-1,
                         _solver_tolerance="default").fit(data)
    cleaned = refit_support_from_estimator(old, data, ladder=tight_ladder(["CLARABEL", "ECOS", "SCS"]))
    # ... equals the single tight fit with the same solvers: point estimate and SE basis at once
    new = CVXPYEstimator(norm_constraint=M, basis_order=k, _objective_normalizer=normalizer, log_frequency=-1,
                         **FREE_SOLVERS).fit(data)
    np.testing.assert_array_equal(new.theta_hat, cleaned.theta)
    assert new.certified_ and cleaned.certified

    # certification before / after: the loosely solved basis is never smaller than the certified one
    first = 1 if k == 0 else k + 1
    assert np.count_nonzero(old.theta_hat[first:]) >= np.count_nonzero(new.theta_hat[first:])
    assert new.kkt_["n_kept_violating"] == 0


def test_the_se_uses_the_fits_own_certified_basis(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1).fit(data)
    results = fit.get_results()
    cached = len(_REFIT_CACHE)
    cov = estimate_covariance_beta(data, results, 1)
    assert cov.support_refit is fit.support_ and len(_REFIT_CACHE) == cached   # no separate refit
    ci = density_confidence_interval(np.linspace(0.05, 0.95, 5), results, cov, 1)
    assert ci.attrs["basis"] == "cleaned" and ci.attrs["basis_certified"]
    np.testing.assert_array_equal(fit.support_.theta, fit.theta_hat)


def test_a_slack_bound_is_certified_by_a_vanishing_gradient(data):
    fit = CVXPYEstimator(norm_constraint=1000.0, basis_order=0, _objective_normalizer="knot_aware",
                         log_frequency=-1).fit(data)
    assert fit.kkt_["bound"] == "slack" and fit.certified_
    assert fit.kkt_["max_gradient_per_obs"] <= 1e-5


def test_the_bound_is_classified_by_the_l1_norm(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, **FREE_SOLVERS).fit(data)
    problem, theta = FullDataHALProblem(data, 1), fit.optimized_theta_raw
    l1 = np.abs(theta[1:]).sum()
    # a solver stopping slightly inside the ball (as MOSEK's near-optimal stop does, 0.99978 M) is binding
    near = kkt_report(problem, theta, norm_constraint=l1 / 0.9998)
    assert near["bound"] == "active" and near["certified"]
    # far inside the ball the bound is not binding, so the gradient must vanish; here it does not
    far = kkt_report(problem, theta, norm_constraint=2 * l1)
    assert far["bound"] == "slack" and not far["certified"]


def test_the_scs_step_is_capped(data):
    from haldensity.core.solvers.cvxpy import SCS_TIME_LIMIT_S, TIGHT_LADDER
    from haldensity.core.univariate.support import _solve_quietly
    assert SCS_TIME_LIMIT_S == 600.0
    assert [opts["time_limit_secs"] for solver, _, opts in TIGHT_LADDER if solver == "SCS"] == [600.0]
    # SCS stops at its time limit: a short one on a problem it cannot finish to 1e-12 in time
    built = FullDataHALProblem(data, 1).build(50.0)
    _solve_quietly(built.problem, "SCS", dict(eps_abs=1e-12, eps_rel=1e-12, max_iters=10**7, time_limit_secs=0.05))
    assert built.problem.solver_stats.solve_time < 0.05 + 0.25


def test_the_best_attempt_is_kept_and_flagged(monkeypatch, data):
    import haldensity.core.univariate.support as support
    real, failing = support.kkt_report, iter([5, 2, 7, 3, 9])   # one value per attempt, none certified

    def nothing_certifies(*args, **kwargs):
        return {**real(*args, **kwargs), "certified": False, "n_failing": next(failing)}

    monkeypatch.setattr(support, "kkt_report", nothing_certifies)
    with pytest.warns(UserWarning, match="Basis cleaning failed"):
        fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, **FREE_SOLVERS).fit(data)
    assert [(a["solver"], a["option"]) for a in fit.solve_attempts_] == [
        ("CLARABEL", "tol 1e-12"), ("CLARABEL", "tol 1e-10"), ("ECOS", "tol 1e-12"), ("ECOS", "tol 1e-10"),
        ("SCS", "eps 1e-9, 600 s cap")]                           # every solver tried
    assert (fit.solver_used_, fit.solver_option_) == ("CLARABEL", "tol 1e-10")   # the fewest failing (2)
    assert fit.certified_ is False and fit.get_results()["certified"] is False
    assert fit.kkt_["n_failing"] == 2


def test_failing_coordinates_are_counted_for_either_bound(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, **FREE_SOLVERS).fit(data)
    problem, theta = FullDataHALProblem(data, 1), fit.optimized_theta_raw.copy()
    assert kkt_report(problem, theta, norm_constraint=20.0)["n_failing"] == 0
    theta[2 + int(np.flatnonzero(np.abs(theta[2:]) <= 1e-8)[0])] = 1e-3        # a kept coefficient that should be 0
    assert kkt_report(problem, theta, norm_constraint=20.0)["n_failing"] >= 1
    slack = kkt_report(problem, fit.optimized_theta_raw, norm_constraint=40.0)  # far inside the ball
    assert slack["bound"] == "slack" and slack["n_failing"] > 0 and not slack["certified"]


def test_invalid_solver_tolerance(data):
    with pytest.raises(ValueError, match="solver_tolerance"):
        CVXPYEstimator(_solver_tolerance="loose")
    assert FullDataHALProblem(data, 1, objective_normalizer="midpoint").build(10.0).formulation == "dense"
    assert FullDataHALProblem(data, 1).build(10.0).formulation == "structured"     # "auto" at k = 1
