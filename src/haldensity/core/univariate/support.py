"""Basis cleaning before the delta-method SE of the full-data HAL-MLE (optional step).

Why. At an exact solution of the L1-constrained HAL-MLE every inactive
coefficient is exactly zero. Interior-point solvers return a nearby point whose
inactive coefficients are only small, and many stay above the 1e-4 pruning
cutoff, so the basis passed to the delta method is much larger than the
solution's (e.g. 186-717 functions against ~8 at n = 1280), which inflates the
SE (2.4-14 times the Monte Carlo sd in that study).

What. :func:`refit_support` rebuilds the full-data problem of
``CVXPYEstimator`` exactly (summed negative log-likelihood with its midpoint
normalizer on ``n_grid_points``, knots at the unique data values,
``||theta[1:]||_1 <= M``), re-solves it at the selected M with tight
tolerances (CLARABEL 1e-12, 1e-10; ECOS 1e-12, 1e-10; SCS 1e-9; first
certified solution wins, else the least violating), prunes at 1e-4 and
certifies the result with :func:`kkt_report`:

* lambda is estimated by the median |gradient_j| over kept knot coordinates
  with |theta_j| >= 0.1 max|theta| (the largest |gradient_j| if none);
* a kept coordinate is violating when |gradient_j| < 0.99 lambda;
* the support is certified when none is violating and no knot coordinate has
  |gradient_j| > 1.01 lambda.

Use the result only for the SE (``support_refit=`` of the functions in
:mod:`haldensity.core.variance`); the point estimate stays the original fit. A
refit that fails certification is kept and flagged (``certified=False``). The
cleaned support is a property of the optimization problem, not of the true
density, and it is not unique (nearby knots give nearly the same function).

Reference implementation: uni_sieve_MLE-phase1 ``hal_support.refit_support``
and ``kkt_report``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from haldensity.core.basis.univariate import create_basis_functions

LADDER: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("CLARABEL", "tol 1e-12", dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=400)),
    ("CLARABEL", "tol 1e-10", dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)),
    ("ECOS", "tol 1e-12", dict(abstol=1e-12, reltol=1e-12, feastol=1e-12, max_iters=500)),
    ("ECOS", "tol 1e-10", dict(abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=300)),
    ("SCS", "eps 1e-9", dict(eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)),
)
PRUNE = 1e-4
RESIDUE_FLOOR = 1e-8
KKT_SLACK = 0.99
MAX_RATIO = 1.01
BIG_FRACTION = 0.1
_OK_STATUSES = ("optimal", "optimal_inaccurate")


class FullDataHALProblem:
    """``CVXPYEstimator``'s optimization problem for one sample."""

    def __init__(self, data: pd.DataFrame, basis_order: int, n_grid_points: int = 200):
        x = np.asarray(data["W1"].dropna(), dtype=float)
        self.basis_order = int(basis_order)
        self.n = int(x.size)
        self.knots = np.unique(x)
        grid = np.linspace(0, 1, int(n_grid_points))
        midpoints = (grid[:-1] + grid[1:]) / 2
        self.data_basis = create_basis_functions(pd.DataFrame({"W1": x}), self.knots, order=self.basis_order)[0]
        self.grid_basis = create_basis_functions(pd.DataFrame({"W1": midpoints}), self.knots, order=self.basis_order)[0]
        self.log_delta = np.log(np.diff(grid))

    def solve(self, norm_constraint: float, solver: str, options: dict[str, Any]) -> tuple[Optional[np.ndarray], str]:
        theta = cp.Variable(self.data_basis.shape[1])
        loss = -cp.sum(self.data_basis @ theta) + self.n * cp.log_sum_exp(self.log_delta + self.grid_basis @ theta)
        problem = cp.Problem(cp.Minimize(loss), [cp.norm1(theta[1:]) <= float(norm_constraint)])
        with warnings.catch_warnings():
            # the ladder records each status and the optimality check decides; CVXPY's notice is noise here
            warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
            problem.solve(solver=solver, **options)
        value = None if theta.value is None else np.asarray(theta.value, dtype=float).copy()
        return value, str(problem.status)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Gradient of the summed negative log-likelihood (midpoint normalizer) at ``theta``."""
        data_basis = self.data_basis.astype(float)
        grid_basis = self.grid_basis.astype(float)
        z = self.log_delta + grid_basis @ theta
        weights = np.exp(z - logsumexp(z))
        return -data_basis.sum(axis=0) + self.n * (weights @ grid_basis)


def kkt_report(problem: FullDataHALProblem, theta: np.ndarray, prune: float = PRUNE,
               slack: float = KKT_SLACK) -> dict[str, Any]:
    """Optimality diagnostics of a raw (unpruned) solution; see the module docstring."""
    k = problem.basis_order
    first = 1 if k == 0 else k + 1
    gradient = problem.gradient(theta)
    knot_gradient, knot_theta = gradient[first:], theta[first:]
    kept = np.abs(knot_theta) > prune
    big = kept & (np.abs(knot_theta) >= BIG_FRACTION * np.max(np.abs(knot_theta), initial=0.0))
    lam = float(np.median(np.abs(knot_gradient[big]))) if big.any() else float(np.max(np.abs(gradient[1:]), initial=0.0))
    ratio = np.abs(knot_gradient) / lam if lam > 0 else np.full(knot_gradient.shape, np.nan)
    violating = int(np.sum(kept & (ratio < slack)))
    max_ratio = float(np.nanmax(ratio)) if ratio.size and np.any(np.isfinite(ratio)) else float("nan")
    return dict(
        lambda_hat=lam,
        n_knots=int(problem.knots.size),
        n_kept=int(kept.sum()),
        n_kept_violating=violating,
        n_residue=int(np.sum((np.abs(knot_theta) > RESIDUE_FLOOR) & ~kept)),
        n_near_active=int(np.sum(ratio >= slack)),
        max_ratio=max_ratio,
        l1_raw=float(np.abs(theta[1:]).sum()),
        certified=bool(violating == 0 and (not np.isfinite(max_ratio) or max_ratio <= MAX_RATIO)),
    )


@dataclass(frozen=True)
class SupportRefit:
    """Cleaned basis for the delta-method SE (see the module docstring).

    ``theta`` holds the pruned refit coefficients over the same knots as the
    original fit (zeros off the cleaned support); ``certified`` says whether
    the optimality check passed. ``attempts`` lists every solver configuration
    tried.
    """

    theta: np.ndarray
    theta_raw: np.ndarray
    knots: np.ndarray
    basis_order: int
    norm_constraint: float
    certified: bool
    kkt: dict[str, Any]
    solver_used: str
    solver_status: str
    option: str
    seconds: float
    attempts: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def n_selected(self) -> int:
        first = 1 if self.basis_order == 0 else self.basis_order + 1
        return int(np.count_nonzero(self.theta[first:]))


def refit_support(
    data: pd.DataFrame,
    basis_order: int,
    norm_constraint: float,
    *,
    n_grid_points: int = 200,
    ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER,
    prune: float = PRUNE,
) -> SupportRefit:
    """Re-solve the full-data HAL-MLE at ``norm_constraint`` with tight tolerances and certify the support."""
    started = perf_counter()
    problem = FullDataHALProblem(data, basis_order, n_grid_points)
    attempts: list[dict[str, Any]] = []
    best = None
    for solver, label, options in ladder:
        call: dict[str, Any] = dict(solver=solver, option=label)
        began = perf_counter()
        try:
            theta, status = problem.solve(norm_constraint, solver, options)
        except Exception as exc:  # solver availability and failures vary by platform
            attempts.append(dict(call, outcome=f"{type(exc).__name__}: {exc}"[:300], seconds=perf_counter() - began))
            continue
        call["status"] = status
        if theta is None or not np.all(np.isfinite(theta)) or status not in _OK_STATUSES:
            attempts.append(dict(call, outcome="no usable solution", seconds=perf_counter() - began))
            continue
        report = kkt_report(problem, theta, prune)
        attempts.append(dict(call, outcome="solved", n_kept=report["n_kept"],
                             n_kept_violating=report["n_kept_violating"], max_ratio=report["max_ratio"],
                             certified=report["certified"], seconds=perf_counter() - began))
        rank = (not report["certified"], report["n_kept_violating"], abs(report["max_ratio"] - 1.0))
        if best is None or rank < best[0]:
            best = (rank, solver, label, status, theta, report)
        if report["certified"]:
            break
    if best is None:
        raise RuntimeError(f"No solver configuration returned a usable solution: {attempts}")
    _, solver, label, status, theta_raw, report = best
    k = int(basis_order)
    first = 1 if k == 0 else k + 1
    theta = theta_raw.copy()
    theta[first:] = np.where(np.abs(theta[first:]) > prune, theta[first:], 0.0)
    return SupportRefit(
        theta=theta, theta_raw=theta_raw, knots=problem.knots, basis_order=k,
        norm_constraint=float(norm_constraint), certified=bool(report["certified"]), kkt=report,
        solver_used=solver, solver_status=status, option=label, seconds=perf_counter() - started,
        attempts=tuple(attempts),
    )


def refit_support_from_estimator(estimator: Any, data: pd.DataFrame, **kwargs: Any) -> SupportRefit:
    """:func:`refit_support` with the bound, basis order and grid of a fitted ``CVXPYEstimator``."""
    if not getattr(estimator, "is_fitted", False):
        raise ValueError("Estimator must be fitted first")
    refit = refit_support(data, int(estimator.basis_order), float(estimator.norm_constraint),
                          n_grid_points=int(getattr(estimator, "n_grid_points", 200)), **kwargs)
    if not np.array_equal(refit.knots, np.asarray(estimator._grid_points_hal, dtype=float)):
        raise ValueError("The data do not match the estimator's knots; pass the data it was fitted on")
    return refit
