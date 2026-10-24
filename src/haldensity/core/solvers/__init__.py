"""Solver helpers (CVXPY primary solver plus fallback waterfall)."""

from haldensity.core.solvers.cvxpy import (
    CVXPYSolveResult,
    build_solver_sequence,
    solve_cvxpy_problem,
)

__all__ = [
    "CVXPYSolveResult",
    "build_solver_sequence",
    "solve_cvxpy_problem",
]
