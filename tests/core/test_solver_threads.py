"""Every solver call is single-threaded by default (Yilong, 2026-10-24); solver_threads raises it."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from haldensity.core.solvers.cvxpy import (
    _CLARABEL_THREADS,
    limited_threads,
    solve_cvxpy_problem,
    thread_options,
    with_threads,
)
from haldensity.full_uni import CVXPYEstimator


@pytest.fixture
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=150)})


def test_thread_options_per_solver():
    assert thread_options("MOSEK") == {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}}
    assert thread_options("clarabel", 3) == ({"max_threads": 3} if _CLARABEL_THREADS else {})
    assert thread_options("SCS") == {} and thread_options("ECOS") == {}       # single-threaded solvers
    merged = with_threads("MOSEK", {"mosek_params": {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12}}, 2)
    assert merged["mosek_params"] == {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12, "MSK_IPAR_NUM_THREADS": 2}
    with pytest.raises(ValueError, match="solver_threads"):
        thread_options("MOSEK", 0)
    with pytest.raises(ValueError, match="solver_threads"):
        CVXPYEstimator(solver_threads=0)


def _record(monkeypatch):
    calls = []

    def record(self, *args, **kwargs):
        from threadpoolctl import threadpool_info
        calls.append(dict(kwargs, _blas=[pool["num_threads"] for pool in threadpool_info()]))
        raise cp.error.SolverError("recorded")

    monkeypatch.setattr(cp.Problem, "solve", record)
    return calls


@pytest.mark.parametrize("tolerance", ["tight", "default"])
def test_the_thread_limit_reaches_every_solver(monkeypatch, data, tolerance):
    calls = _record(monkeypatch)
    with pytest.raises(RuntimeError):
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, solver_tolerance=tolerance).fit(data)
    by_solver = {c["solver"]: c for c in calls}
    assert set(by_solver) == {"MOSEK", "CLARABEL", "ECOS", "SCS"}
    assert by_solver["MOSEK"]["mosek_params"]["MSK_IPAR_NUM_THREADS"] == 1
    if _CLARABEL_THREADS:
        assert by_solver["CLARABEL"]["max_threads"] == 1
    assert all(threads == 1 for c in calls for threads in c["_blas"])       # BLAS/OpenMP pools during the call

    calls.clear()
    with pytest.raises(RuntimeError):
        CVXPYEstimator(norm_constraint=10.0, basis_order=1, log_frequency=-1, solver_tolerance=tolerance,
                       solver_threads=4).fit(data)
    assert {c["solver"]: c for c in calls}["MOSEK"]["mosek_params"]["MSK_IPAR_NUM_THREADS"] == 4


def test_the_shared_helper_is_single_threaded_by_default(monkeypatch):
    calls = _record(monkeypatch)
    x = cp.Variable(2)
    with pytest.raises(RuntimeError):
        solve_cvxpy_problem(cp.Problem(cp.Minimize(cp.sum_squares(x))), "MOSEK", use_secondary_solver=True,
                            solver_waterfall=["CLARABEL"])
    assert calls[0]["mosek_params"] == {"MSK_IPAR_NUM_THREADS": 1}
    assert calls[1].get("max_threads", 1) == 1


def test_blas_pools_are_restored_after_the_call():
    from threadpoolctl import threadpool_info
    before = [pool["num_threads"] for pool in threadpool_info()]
    with limited_threads(1):
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
    assert [pool["num_threads"] for pool in threadpool_info()] == before
