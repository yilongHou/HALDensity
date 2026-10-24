"""Solver helpers (CVXPY primary solver plus fallback waterfall; tight settings for the final fit)."""

from haldensity.core.solvers.cvxpy import (
    DEFAULT_SOLVER_THREADS,
    TIGHT_LADDER,
    CVXPYSolveResult,
    build_solver_sequence,
    solve_cvxpy_problem,
    tight_ladder,
    thread_options,
)

__all__ = [
    "CVXPYSolveResult",
    "DEFAULT_SOLVER_THREADS",
    "TIGHT_LADDER",
    "build_solver_sequence",
    "solve_cvxpy_problem",
    "thread_options",
    "tight_ladder",
]
