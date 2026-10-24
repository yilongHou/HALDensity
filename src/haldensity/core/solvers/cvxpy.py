"""Shared CVXPY solver selection and fallback utilities.

Threads. Every solver call is single-threaded by default (``DEFAULT_SOLVER_THREADS``; Yilong,
2026-10-24, so that runs on shared servers do not take other people's cores). Per call, through each
solver's own parameter: MOSEK ``MSK_IPAR_NUM_THREADS``; CLARABEL ``max_threads`` (its Rayon pool;
versions without the setting are single-threaded). SCS and ECOS are single-threaded. The BLAS/OpenMP
pools of numpy and scipy are limited for the duration of the call with ``threadpoolctl``. On a
256-core server an uncapped CLARABEL solve used 109 cores and took 4 times as long as with one thread;
MOSEK used 5.6. Raise the count with ``solver_threads`` when one fit runs alone on a dedicated machine.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import cvxpy as cp


DEFAULT_ACCEPTABLE_STATUSES: frozenset[str] = frozenset({"optimal", "optimal_inaccurate"})
DEFAULT_SOLVER_THREADS = 1


def _clarabel_has_max_threads() -> bool:
    try:
        import clarabel
        return hasattr(clarabel.DefaultSettings(), "max_threads")
    except Exception:  # not installed: CVXPY reports it when CLARABEL is asked for
        return False


_CLARABEL_THREADS = _clarabel_has_max_threads()


def thread_options(solver_name: str, threads: int = DEFAULT_SOLVER_THREADS) -> dict[str, Any]:
    """Per-call options that limit ``solver_name``'s own threads to ``threads`` (empty if it has none)."""
    threads = int(threads)
    if threads < 1:
        raise ValueError("solver_threads must be a positive integer")
    name = str(solver_name).strip().upper()
    if name == "MOSEK":
        return {"mosek_params": {"MSK_IPAR_NUM_THREADS": threads}}
    if name == "CLARABEL" and _CLARABEL_THREADS:
        return {"max_threads": threads}
    return {}


def with_threads(solver_name: str, options: dict[str, Any], threads: int = DEFAULT_SOLVER_THREADS) -> dict[str, Any]:
    """``options`` plus the thread limit for ``solver_name`` (MOSEK parameters are merged, not replaced)."""
    merged = copy.deepcopy(dict(options))
    for key, value in thread_options(solver_name, threads).items():
        if key == "mosek_params":
            merged["mosek_params"] = {**dict(merged.get("mosek_params") or {}), **value}
        else:
            merged[key] = value
    return merged


@contextlib.contextmanager
def limited_threads(threads: int = DEFAULT_SOLVER_THREADS) -> Iterator[None]:
    """Limit the BLAS/OpenMP pools of numpy, scipy and the solvers to ``threads`` while inside."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl comes with scikit-learn
        yield
        return
    with threadpool_limits(limits=int(threads)):
        yield


@dataclass(frozen=True)
class CVXPYSolveResult:
    """Result metadata for a successful CVXPY solve attempt."""

    solver_used: str
    status: str
    objective_value: Optional[float]
    attempted_solvers: tuple[str, ...]


def _normalize_solver_name(solver_name: str) -> str:
    normalized = str(solver_name).strip().upper()
    if normalized == "":
        raise ValueError("Solver name must be non-empty")
    return normalized


def build_solver_sequence(
    primary_solver: str,
    *,
    use_secondary_solver: bool,
    solver_waterfall: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Build an ordered, de-duplicated solver candidate sequence."""
    seen: set[str] = set()
    candidates: list[str] = []

    def _append(solver_name: str) -> None:
        normalized = _normalize_solver_name(solver_name)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    _append(primary_solver)
    if use_secondary_solver and solver_waterfall is not None:
        for solver_name in solver_waterfall:
            _append(solver_name)

    return tuple(candidates)


def solve_cvxpy_problem(
    problem: cp.Problem,
    primary_solver: str,
    *,
    use_secondary_solver: bool,
    solver_waterfall: Sequence[str] | None = None,
    build_solve_kwargs: Callable[[str], dict[str, Any]] | None = None,
    before_attempt: Callable[[str], None] | None = None,
    acceptable_statuses: Iterable[str] = DEFAULT_ACCEPTABLE_STATUSES,
    solver_threads: int = DEFAULT_SOLVER_THREADS,
) -> CVXPYSolveResult:
    """Solve a CVXPY problem with a primary solver and optional fallback sequence.

    Each call runs with ``solver_threads`` threads (default 1; see the module docstring).
    """
    candidates = build_solver_sequence(
        primary_solver=primary_solver,
        use_secondary_solver=use_secondary_solver,
        solver_waterfall=solver_waterfall,
    )
    if len(candidates) == 0:
        raise ValueError("At least one solver candidate is required")

    accepted = {str(status) for status in acceptable_statuses}
    attempted: list[str] = []
    failure_details: list[str] = []
    last_exception: Exception | None = None

    for solver_name in candidates:
        attempted.append(solver_name)

        if before_attempt is not None:
            before_attempt(solver_name)

        kwargs = {} if build_solve_kwargs is None else dict(build_solve_kwargs(solver_name))
        kwargs = with_threads(solver_name, kwargs, solver_threads)
        kwargs["solver"] = solver_name

        try:
            with limited_threads(solver_threads):
                problem.solve(**kwargs)
        except Exception as exc:  # pragma: no cover - solver availability varies by environment
            last_exception = exc
            failure_details.append(f"{solver_name}: {exc}")
            continue

        status = str(problem.status)
        if status in accepted and problem.value is not None:
            return CVXPYSolveResult(
                solver_used=solver_name,
                status=status,
                objective_value=float(problem.value),
                attempted_solvers=tuple(attempted),
            )

        failure_details.append(f"{solver_name}: status={status}, value={problem.value}")

    if len(candidates) == 1 and last_exception is not None:
        raise RuntimeError(f"CVXPY optimization failed: {last_exception}") from last_exception

    attempt_msg = ", ".join(candidates)
    detail_msg = "; ".join(failure_details) if failure_details else "no solver details captured"
    raise RuntimeError(
        f"CVXPY optimization failed with solvers [{attempt_msg}]; details: {detail_msg}"
    ) from last_exception


# Tight settings for the final fit at the chosen bound (CVXPYEstimator(solver_tolerance="tight") and
# basis cleaning): per solver, the configurations tried in order. MOSEK's interior-point tolerances
# (relative gap, primal and dual feasibility) are 1e-12; NEAR_REL = 1 stops MOSEK from reporting a
# solution that only meets 1000 x those tolerances as "optimal", so such a stop shows up as
# "optimal_inaccurate" (accept_unknown) and the optimality check decides whether it is good enough.
# The SCS step is capped at SCS_TIME_LIMIT_S (Yilong, 2026-10-24: without a cap it ran past 17 min at
# n = 20,000; with it, 7 of the 9 problems that reached it were certified, docs/dev/benchmarks/final_fit_repair).
SCS_TIME_LIMIT_S = 600.0
MOSEK_TIGHT_PARAMS: dict[str, float] = {
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1.0,
}
TIGHT_STEPS: dict[str, tuple[tuple[str, str, dict[str, Any]], ...]] = {
    "MOSEK": (("MOSEK", "tol 1e-12", dict(mosek_params=MOSEK_TIGHT_PARAMS, accept_unknown=True)),),
    "CLARABEL": (
        ("CLARABEL", "tol 1e-12", dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=400)),
        ("CLARABEL", "tol 1e-10", dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)),
    ),
    "ECOS": (
        ("ECOS", "tol 1e-12", dict(abstol=1e-12, reltol=1e-12, feastol=1e-12, max_iters=500)),
        ("ECOS", "tol 1e-10", dict(abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=300)),
    ),
    "SCS": (("SCS", "eps 1e-9, 600 s cap",
             dict(eps_abs=1e-9, eps_rel=1e-9, max_iters=200000, time_limit_secs=SCS_TIME_LIMIT_S)),),
}


def tight_ladder(solvers: Iterable[str]) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """The tight configurations of ``solvers`` in order (a solver without tight settings: its defaults)."""
    steps: list[tuple[str, str, dict[str, Any]]] = []
    for solver_name in solvers:
        name = _normalize_solver_name(solver_name)
        steps.extend(TIGHT_STEPS.get(name, ((name, "solver defaults", {}),)))
    return tuple(steps)


TIGHT_LADDER = tight_ladder(("MOSEK", "CLARABEL", "ECOS", "SCS"))
