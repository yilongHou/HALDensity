"""The final fit is one tight, certified solve: its solution is the point estimate and the SE basis."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from haldensity.core.solvers.cvxpy import MOSEK_TIGHT_PARAMS, tight_ladder
from haldensity.core.univariate.support import FullDataHALProblem, kkt_report, refit_support_from_estimator
from haldensity.core.variance import _REFIT_CACHE, density_confidence_interval, estimate_covariance_beta
from haldensity.full_uni import CVXPYEstimator

CASES = [(0, "midpoint", 5.0), (0, "midpoint", 20.0), (1, "midpoint", 20.0), (1, "midpoint", 50.0),
         (2, "midpoint", 20.0), (2, "midpoint", 100.0), (0, "knot_aware", 20.0), (1, "knot_aware", 50.0)]
FREE_SOLVERS = dict(solver="CLARABEL", solver_waterfall=["CLARABEL", "ECOS", "SCS"])


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
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1).fit(data)
    assert [c["solver"] for c in calls] == ["MOSEK", "CLARABEL", "CLARABEL", "ECOS", "ECOS", "SCS"]
    assert calls[0]["mosek_params"] == MOSEK_TIGHT_PARAMS and calls[0]["accept_unknown"] is True
    assert all(calls[0]["mosek_params"][f"MSK_DPAR_INTPNT_CO_TOL_{name}"] == 1e-12
               for name in ("REL_GAP", "PFEAS", "DFEAS"))
    assert calls[1] == dict(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=400)
    assert calls[2] == dict(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
    assert calls[3] == dict(solver="ECOS", abstol=1e-12, reltol=1e-12, feastol=1e-12, max_iters=500)
    assert calls[4] == dict(solver="ECOS", abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=300)
    assert calls[5] == dict(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)

    # solver_tolerance="default" passes no tolerance (v0.1.0)
    calls.clear()
    with pytest.raises(RuntimeError):
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, solver_tolerance="default").fit(data)
    assert calls and all(set(c) == {"solver"} for c in calls)


def test_tight_ladder_follows_the_solver_settings():
    assert [s for s, _, _ in tight_ladder(["ECOS", "MOSEK"])] == ["ECOS", "ECOS", "MOSEK"]
    assert tight_ladder(["OSQP"]) == (("OSQP", "solver defaults", {}),)


@pytest.mark.parametrize("k,normalizer,M", CASES)
def test_the_final_fit_is_the_old_cleaned_basis(data, k, normalizer, M):
    # v0.1.0 fit + separate cleaning refit (free solvers, as before MOSEK joined the ladder) ...
    old = CVXPYEstimator(norm_constraint=M, basis_order=k, objective_normalizer=normalizer, log_frequency=-1,
                         solver_tolerance="default").fit(data)
    cleaned = refit_support_from_estimator(old, data, ladder=tight_ladder(["CLARABEL", "ECOS", "SCS"]))
    # ... equals the single tight fit with the same solvers: point estimate and SE basis at once
    new = CVXPYEstimator(norm_constraint=M, basis_order=k, objective_normalizer=normalizer, log_frequency=-1,
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
    fit = CVXPYEstimator(norm_constraint=1000.0, basis_order=0, objective_normalizer="knot_aware",
                         log_frequency=-1).fit(data)
    assert fit.kkt_["bound"] == "slack" and fit.certified_
    assert fit.kkt_["max_gradient_per_obs"] <= 1e-5


def test_the_bound_is_classified_by_its_multiplier(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, **FREE_SOLVERS).fit(data)
    problem, theta = FullDataHALProblem(data, 1), fit.optimized_theta_raw
    l1 = np.abs(theta[1:]).sum()
    # a solver stopping slightly inside the ball (as MOSEK's near-optimal stop does) is still binding
    near = kkt_report(problem, theta, norm_constraint=l1 / 0.9998)
    assert near["bound"] == "active" and near["certified"]
    # lambda > 0 with the bound far from binding is not optimal
    far = kkt_report(problem, theta, norm_constraint=2 * l1)
    assert far["bound"] == "active" and not far["certified"]


def test_an_uncertified_fit_is_flagged_and_warned(monkeypatch, data):
    import haldensity.core.univariate.support as support

    monkeypatch.setattr(support, "kkt_report",
                        lambda *a, **kw: {**kkt_report(*a, **kw), "certified": False, "n_kept_violating": 1})
    with pytest.warns(UserWarning, match="could not be certified"):
        fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, **FREE_SOLVERS).fit(data)
    assert fit.certified_ is False and len(fit.solve_attempts_) == 5      # every configuration tried
    assert fit.get_results()["certified"] is False


def test_invalid_solver_tolerance(data):
    with pytest.raises(ValueError, match="solver_tolerance"):
        CVXPYEstimator(solver_tolerance="loose")
    problem = FullDataHALProblem(data, 1)
    assert problem.build(10.0).formulation == "dense"
