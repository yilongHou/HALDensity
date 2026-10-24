"""Basis cleaning before the delta-method SE of the full-data HAL-MLE (optional step).

Why. At an exact solution of the L1-constrained HAL-MLE every inactive
coefficient is exactly zero. Interior-point solvers return a nearby point whose
inactive coefficients are only small, and many stay above the 1e-4 pruning
cutoff, so the basis passed to the delta method is much larger than the
solution's (e.g. 186-717 functions against ~8 at n = 1280), which inflates the
SE (2.4-14 times the Monte Carlo sd in that study).

What. :func:`refit_support` rebuilds the full-data problem of
``CVXPYEstimator`` exactly (summed negative log-likelihood with the fit's
objective normalizer, midpoint or knot-aware, on ``n_grid_points``; the fit's
candidate knots; ``||theta[1:]||_1 <= M``; for knot-aware with k <= 1 the same
structured form as the fit, from :mod:`haldensity.core.univariate.objective`),
re-solves it at the selected M with tight
tolerances (MOSEK 1e-12; CLARABEL 1e-12, 1e-10; ECOS 1e-12, 1e-10; SCS 1e-9 capped
at 600 s; first certified solution wins, else the attempt with the fewest
coordinates failing the optimality conditions, flagged), prunes at 1e-4 and
certifies the result with :func:`kkt_report`:

* lambda is estimated by the median |gradient_j| over kept knot coordinates
  with |theta_j| >= 0.1 max|theta| (the largest |gradient_j| if none);
* a kept coordinate is violating when |gradient_j| < 0.99 lambda;
* the support is certified when none is violating and no knot coordinate has
  |gradient_j| > 1.01 lambda;
* if ||theta[1:]||_1 < (1 - 1e-3) M the bound is not binding (lambda = 0) and
  the solution is certified when max_j |gradient_j| / n <= 1e-5 (the 1e-3
  covers MOSEK's near-optimal stop, e.g. at 0.99978 M); at the bound, a
  solution passing either test is certified (lambda may be ~0 there).

The same routine (:func:`certified_solve`, MOSEK 1e-12 first) is the final fit
of ``CVXPYEstimator(solver_tolerance="tight")``, its default: there the
certified solution is both the point estimate and the SE basis, and no
separate refit is made.

A separate refit (:func:`refit_support`, for a fit made with
``solver_tolerance="default"``) is used only for the SE (``support_refit=`` of
the functions in :mod:`haldensity.core.variance`); the point estimate stays that
fit. A solution that fails certification is kept and flagged
(``certified=False``). The
cleaned support is a property of the optimization problem, not of the true
density, and it is not unique (nearby knots give nearly the same function).

Reference implementation: uni_sieve_MLE-phase1 ``hal_support.refit_support``
and ``kkt_report``.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Optional, Union

import cvxpy as cp
import numpy as np
import pandas as pd

from haldensity.core.solvers.cvxpy import DEFAULT_SOLVER_THREADS, TIGHT_LADDER, limited_threads, with_threads
from haldensity.core.univariate.objective import (
    STRUCTURED_ORDERS,
    candidate_knots,
    dense_problem,
    normalizer_terms,
    objective_gradient,
    structured_problem,
)

LADDER: tuple[tuple[str, str, dict[str, Any]], ...] = TIGHT_LADDER
PRUNE = 1e-4
RESIDUE_FLOOR = 1e-8
KKT_SLACK = 0.99
MAX_RATIO = 1.01
BIG_FRACTION = 0.1
ACTIVE_BOUND = 1e-3          # ||theta[1:]||_1 below (1 - this) * M: the bound is not binding ...
SLACK_GRADIENT_TOL = 1e-5    # ... and the solution is certified when max |gradient_j| / n is at most this
_OK_STATUSES = ("optimal", "optimal_inaccurate")


@dataclass
class BuiltProblem:
    """A CVXPY problem for one bound; ``theta()`` gives HAL coefficients, the scales undo solver scaling."""

    problem: cp.Problem
    l1_constraint: cp.Constraint
    theta: Callable[[], Optional[np.ndarray]]
    formulation: str
    objective_scale: float = 1.0
    dual_scale: float = 1.0


class FullDataHALProblem:
    """``CVXPYEstimator``'s optimization problem for one sample (same knots, cells and formulation)."""

    def __init__(self, data: pd.DataFrame, basis_order: int, n_grid_points: int = 200,
                 objective_normalizer: str = "midpoint", max_knots: Union[int, str, None] = "auto",
                 structured: bool = True):
        self.x = np.asarray(data["W1"].dropna(), dtype=float)
        self.basis_order = int(basis_order)
        self.n = int(self.x.size)
        self.n_grid_points = int(n_grid_points)
        self.objective_normalizer = objective_normalizer
        self.knots = candidate_knots(self.x, max_knots, basis_order=self.basis_order)
        self.structured = (bool(structured) and objective_normalizer == "knot_aware"
                           and self.basis_order in STRUCTURED_ORDERS)
        self.terms = normalizer_terms(self.x, self.knots, self.basis_order, self.n_grid_points, objective_normalizer,
                                      build_basis=False)

    def build(self, norm_constraint: float, dense: bool = False) -> BuiltProblem:
        """The problem at ``norm_constraint``: structured form when available (unless ``dense``)."""
        if self.structured and not dense:
            sparse = structured_problem(self.x, self.knots, self.basis_order, self.n_grid_points, norm_constraint)
            return BuiltProblem(sparse.problem, sparse.l1_constraint, sparse.theta, "structured",
                                sparse.objective_scale, sparse.dual_scale)
        dense = dense_problem(self.x, self.knots, self.basis_order, self.n_grid_points, self.objective_normalizer,
                              norm_constraint)
        return BuiltProblem(dense.problem, dense.l1_constraint, dense.theta, "dense")

    def solve(self, norm_constraint: float, solver: str, options: dict[str, Any],
              threads: int = DEFAULT_SOLVER_THREADS) -> tuple[Optional[np.ndarray], str]:
        built = self.build(norm_constraint)
        _solve_quietly(built.problem, solver, options, threads)
        value = built.theta()
        return (None if value is None else np.asarray(value, dtype=float).copy()), str(built.problem.status)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Gradient of the summed negative log-likelihood (the objective's own normalizer) at ``theta``."""
        return objective_gradient(theta, self.x, self.knots, self.basis_order, self.terms)


def _solve_quietly(problem: cp.Problem, solver: str, options: dict[str, Any],
                   threads: int = DEFAULT_SOLVER_THREADS) -> None:
    with warnings.catch_warnings(), limited_threads(threads):
        # the ladder records each status and the optimality check decides; CVXPY's notice is noise here
        warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
        problem.solve(solver=solver, **with_threads(solver, options, threads))


def kkt_report(problem: FullDataHALProblem, theta: np.ndarray, prune: float = PRUNE,
               slack: float = KKT_SLACK, norm_constraint: Optional[float] = None) -> dict[str, Any]:
    """Optimality diagnostics of a raw (unpruned) solution; see the module docstring."""
    k = problem.basis_order
    first = 1 if k == 0 else k + 1
    gradient = problem.gradient(theta)
    knot_gradient, knot_theta = gradient[first:], theta[first:]
    kept = np.abs(knot_theta) > prune
    l1_raw = float(np.abs(theta[1:]).sum())
    max_gradient = float(np.max(np.abs(gradient), initial=0.0)) / max(problem.n, 1)
    big = kept & (np.abs(knot_theta) >= BIG_FRACTION * np.max(np.abs(knot_theta), initial=0.0))
    lam = float(np.median(np.abs(knot_gradient[big]))) if big.any() else float(np.max(np.abs(gradient[1:]), initial=0.0))
    common = dict(n_knots=int(problem.knots.size), n_kept=int(kept.sum()),
                  n_residue=int(np.sum((np.abs(knot_theta) > RESIDUE_FLOOR) & ~kept)), l1_raw=l1_raw,
                  max_gradient_per_obs=max_gradient)
    # coordinates whose gradient does not vanish (the condition when lambda = 0)
    gradient_failing = int(np.sum(np.abs(gradient) / max(problem.n, 1) > SLACK_GRADIENT_TOL))
    if norm_constraint is not None and l1_raw < (1.0 - ACTIVE_BOUND) * float(norm_constraint):
        # the bound is not binding: lambda = 0, so the whole gradient must vanish
        return dict(bound="slack", lambda_hat=lam, n_kept_violating=0, n_near_active=0, max_ratio=float("nan"),
                    n_failing=gradient_failing, certified=bool(gradient_failing == 0), **common)
    ratio = np.abs(knot_gradient) / lam if lam > 0 else np.full(knot_gradient.shape, np.nan)
    violating = int(np.sum(kept & (ratio < slack)))
    max_ratio = float(np.nanmax(ratio)) if ratio.size and np.any(np.isfinite(ratio)) else float("nan")
    certified = violating == 0 and (not np.isfinite(max_ratio) or max_ratio <= MAX_RATIO)
    # kept coefficients below 0.99 lambda plus coordinates above 1.01 lambda
    failing = violating + int(np.sum(ratio > MAX_RATIO))
    if norm_constraint is not None:
        # at the bound with lambda ~ 0 a vanishing gradient is also optimal
        certified = certified or gradient_failing == 0
        failing = 0 if certified else min(failing, gradient_failing)
    return dict(bound="active" if norm_constraint is not None else "assumed active", lambda_hat=lam,
                n_kept_violating=violating, n_near_active=int(np.sum(ratio >= slack)), max_ratio=max_ratio,
                n_failing=failing, certified=bool(certified), **common)


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
    formulation: str = "dense"
    objective_value: Optional[float] = None
    dual_value: Optional[float] = None

    @property
    def n_selected(self) -> int:
        first = 1 if self.basis_order == 0 else self.basis_order + 1
        return int(np.count_nonzero(self.theta[first:]))


def certified_solve(
    problem: FullDataHALProblem,
    norm_constraint: float,
    ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER,
    prune: float = PRUNE,
    solver_threads: int = DEFAULT_SOLVER_THREADS,
) -> SupportRefit:
    """Solve ``problem`` at ``norm_constraint`` down ``ladder`` until the optimality check certifies.

    The first certified solution wins. If none is certified, the best attempt from any configuration
    is kept: the fewest coordinates failing the optimality conditions (``n_failing``), then the max
    ratio closest to 1 (or the smallest gradient when the bound is not binding); the caller flags it.
    The problem is built once (the structured form when available; the dense form only if no
    configuration returns a usable solution of it). Every configuration tried is recorded in
    ``attempts``.
    """
    started = perf_counter()
    attempts: list[dict[str, Any]] = []
    best = None
    forms = [False, True] if problem.structured else [True]
    for dense in forms:
        built = problem.build(norm_constraint, dense=dense)
        for solver, label, options in ladder:
            call: dict[str, Any] = dict(solver=solver, option=label, formulation=built.formulation)
            began = perf_counter()
            try:
                _solve_quietly(built.problem, solver, options, solver_threads)
            except Exception as exc:  # solver availability and failures vary by platform
                attempts.append(dict(call, outcome=f"{type(exc).__name__}: {exc}"[:300], seconds=perf_counter() - began))
                continue
            status = str(built.problem.status)
            call["status"] = status
            theta = built.theta()
            theta = None if theta is None else np.asarray(theta, dtype=float).copy()
            if theta is None or not np.all(np.isfinite(theta)) or status not in _OK_STATUSES:
                attempts.append(dict(call, outcome="no usable solution", seconds=perf_counter() - began))
                continue
            report = kkt_report(problem, theta, prune, norm_constraint=norm_constraint)
            attempts.append(dict(call, outcome="solved", n_kept=report["n_kept"],
                                 n_kept_violating=report["n_kept_violating"], n_failing=report["n_failing"],
                                 max_ratio=report["max_ratio"],
                                 bound=report["bound"], max_gradient_per_obs=report["max_gradient_per_obs"],
                                 certified=report["certified"], seconds=perf_counter() - began))
            closeness = abs(report["max_ratio"] - 1.0) if np.isfinite(report["max_ratio"]) else report["max_gradient_per_obs"]
            rank = (not report["certified"], report["n_failing"], closeness)
            if best is None or rank < best[0]:
                dual = built.l1_constraint.dual_value
                best = (rank, solver, label, status, theta, report, built.formulation,
                        float(built.problem.value) * built.objective_scale,
                        None if dual is None else float(np.asarray(dual).ravel()[0]) * built.dual_scale)
            if report["certified"]:
                break
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"No solver configuration returned a usable solution: {attempts}")
    _, solver, label, status, theta_raw, report, formulation, objective_value, dual_value = best
    k = problem.basis_order
    first = 1 if k == 0 else k + 1
    theta = theta_raw.copy()
    theta[first:] = np.where(np.abs(theta[first:]) > prune, theta[first:], 0.0)
    return SupportRefit(
        theta=theta, theta_raw=theta_raw, knots=problem.knots, basis_order=k,
        norm_constraint=float(norm_constraint), certified=bool(report["certified"]), kkt=report,
        solver_used=solver, solver_status=status, option=label, seconds=perf_counter() - started,
        attempts=tuple(attempts), formulation=formulation, objective_value=objective_value, dual_value=dual_value,
    )


def refit_support(
    data: pd.DataFrame,
    basis_order: int,
    norm_constraint: float,
    *,
    n_grid_points: int = 200,
    objective_normalizer: str = "midpoint",
    max_knots: Union[int, str, None] = "auto",
    ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER,
    prune: float = PRUNE,
    solver_threads: int = DEFAULT_SOLVER_THREADS,
) -> SupportRefit:
    """Re-solve the full-data HAL-MLE at ``norm_constraint`` with tight tolerances and certify the support."""
    problem = FullDataHALProblem(data, basis_order, n_grid_points, objective_normalizer, max_knots)
    return certified_solve(problem, norm_constraint, ladder, prune, solver_threads)


def refit_support_from_estimator(estimator: Any, data: pd.DataFrame, **kwargs: Any) -> SupportRefit:
    """:func:`refit_support` with the bound, basis order and grid of a fitted ``CVXPYEstimator``."""
    if not getattr(estimator, "is_fitted", False):
        raise ValueError("Estimator must be fitted first")
    refit = refit_support(data, int(estimator.basis_order), float(estimator.norm_constraint),
                          n_grid_points=int(getattr(estimator, "n_grid_points", 200)),
                          objective_normalizer=getattr(estimator, "objective_normalizer", "midpoint"),
                          max_knots=getattr(estimator, "max_knots", None),
                          solver_threads=getattr(estimator, "solver_threads", DEFAULT_SOLVER_THREADS), **kwargs)
    if not np.array_equal(refit.knots, np.asarray(estimator._grid_points_hal, dtype=float)):
        raise ValueError("The data do not match the estimator's knots; pass the data it was fitted on")
    return refit
