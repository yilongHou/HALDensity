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
tolerances (MOSEK 1e-12; CLARABEL 1e-12, 1e-10; ECOS 1e-12, 1e-10; SCS 1e-9; first
certified solution wins, else the least violating), prunes at 1e-4 and
certifies the result with :func:`kkt_report`:

* lambda is estimated by the median |gradient_j| over kept knot coordinates
  with |theta_j| >= 0.1 max|theta| (the largest |gradient_j| if none);
* a kept coordinate is violating when |gradient_j| < 0.99 lambda;
* the support is certified when none is violating and no knot coordinate has
  |gradient_j| > 1.01 lambda;
* if ||theta[1:]||_1 < (1 - 1e-3) M the bound is not binding (lambda = 0) and
  the solution is certified when max_j |gradient_j| / n <= 1e-5 (the 1e-3
  covers MOSEK's near-optimal stop, e.g. at 0.99978 M).

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
from typing import Any, Callable, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.solvers.cvxpy import TIGHT_LADDER
from haldensity.core.univariate.objective import (
    STRUCTURED_ORDERS,
    _tail_sums,
    candidate_knots,
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
# Proposed repair of an uncertified final fit (repair_support; not used by default)
REPAIR_MAX_SHARE = 0.01      # the coefficients set to zero may hold at most this share of ||theta[1:]||_1
REPAIR_MAX_ROUNDS = 10       # polish: re-solve on the support at most this often
REPAIR_START_SIZE = 200      # polish: start from at most this many of the largest near-active coefficients
REPAIR_MAX_SUPPORT = 2000    # polish: largest support solved directly
REPAIR_ADD_PER_ROUND = 100   # polish: most coordinates added to the support per round
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
                 objective_normalizer: str = "midpoint", max_knots: Optional[int] = None,
                 structured: bool = True):
        self.x = np.asarray(data["W1"].dropna(), dtype=float)
        self.basis_order = int(basis_order)
        self.n = int(self.x.size)
        self.n_grid_points = int(n_grid_points)
        self.objective_normalizer = objective_normalizer
        self.knots = candidate_knots(self.x, max_knots)
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
        data_basis = create_basis_functions(pd.DataFrame({"W1": self.x}), self.knots, order=self.basis_order)[0]
        terms = normalizer_terms(self.x, self.knots, self.basis_order, self.n_grid_points, self.objective_normalizer)
        theta = cp.Variable(data_basis.shape[1])
        data_log_density = data_basis @ theta
        if terms.basis is not None:
            cells = terms.basis @ theta
        else:
            # k = 0 knot-aware: the level on the cell starting at a knot is the data term's value there
            parts = [theta[0:1]] if terms.leading_intercept_cell else []
            if terms.data_rows.size:
                parts.append(data_log_density[terms.data_rows])
            cells = cp.hstack(parts) if len(parts) > 1 else parts[0]
        loss = -cp.sum(data_log_density) + self.n * cp.log_sum_exp(terms.log_widths + cells)
        l1 = cp.norm1(theta[1:]) <= float(norm_constraint)
        problem = cp.Problem(cp.Minimize(loss), [l1])
        return BuiltProblem(problem, l1, lambda: None if theta.value is None else theta.value, "dense")

    def solve(self, norm_constraint: float, solver: str, options: dict[str, Any]) -> tuple[Optional[np.ndarray], str]:
        built = self.build(norm_constraint)
        _solve_quietly(built.problem, solver, options)
        value = built.theta()
        return (None if value is None else np.asarray(value, dtype=float).copy()), str(built.problem.status)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Gradient of the summed negative log-likelihood (the objective's own normalizer) at ``theta``."""
        return objective_gradient(theta, self.x, self.knots, self.basis_order, self.terms)

    def objective(self, theta: np.ndarray) -> float:
        """The summed negative log-likelihood with the objective's own normalizer at ``theta``."""
        from scipy.special import logsumexp

        from haldensity.core.univariate.normalizer import log_unnormalized_density
        theta = np.asarray(theta, dtype=float)
        cells = self.terms.log_density(theta, self.knots, self.basis_order)
        data = log_unnormalized_density(self.x, theta, self.knots, self.basis_order)
        return float(-data.sum() + self.n * logsumexp(self.terms.log_widths + cells))

    def solve_on_support(self, support: np.ndarray, norm_constraint: float,
                         ladder: tuple[tuple[str, str, dict[str, Any]], ...]) -> tuple[Optional[np.ndarray], str]:
        """The problem restricted to the knot coordinates ``support`` (plus the polynomial ones).

        The data term is linear in theta (its coefficients are the basis column sums), so the restricted
        problem has only ``cells x |support|`` entries. Returns the full-length solution and the solver.
        """
        k, n = self.basis_order, self.n
        first = 1 if k == 0 else k + 1
        knots = self.knots[support]
        sums = np.concatenate(([float(n)], [float(np.sum(self.x ** r)) for r in range(1, k + 1)],
                               _tail_sums(self.x, np.ones(n), knots, k)))
        points = self.terms.points
        columns = [np.ones_like(points)] + [points ** r for r in range(1, k + 1)]
        if knots.size:
            columns.append((points[:, None] >= knots[None, :]).astype(float) if k == 0
                           else np.maximum(points[:, None] - knots[None, :], 0.0) ** k)
        cells = np.column_stack(columns)
        theta = cp.Variable(cells.shape[1])
        loss = -(sums / n) @ theta + cp.log_sum_exp(self.terms.log_widths + cells @ theta)     # objective / n
        restricted = cp.Problem(cp.Minimize(loss), [cp.norm1(theta[1:]) <= float(norm_constraint)])
        for solver, label, options in ladder:
            try:
                _solve_quietly(restricted, solver, options)
            except Exception:  # solver availability and failures vary by platform
                continue
            value = theta.value
            if str(restricted.status) in _OK_STATUSES and value is not None and np.all(np.isfinite(value)):
                full = np.zeros(first + self.knots.size)
                full[:first] = value[:first]
                full[first + np.asarray(support, dtype=int)] = value[first:]
                return full, f"{solver} {label}"
        return None, ""


def _solve_quietly(problem: cp.Problem, solver: str, options: dict[str, Any]) -> None:
    with warnings.catch_warnings():
        # the ladder records each status and the optimality check decides; CVXPY's notice is noise here
        warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
        problem.solve(solver=solver, **copy.deepcopy(options))


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
    if norm_constraint is not None and l1_raw < (1.0 - ACTIVE_BOUND) * float(norm_constraint):
        # the bound is not binding: lambda = 0, so the whole gradient must vanish
        return dict(bound="slack", lambda_hat=lam, n_kept_violating=0, n_near_active=0, max_ratio=float("nan"),
                    certified=bool(max_gradient <= SLACK_GRADIENT_TOL), **common)
    ratio = np.abs(knot_gradient) / lam if lam > 0 else np.full(knot_gradient.shape, np.nan)
    violating = int(np.sum(kept & (ratio < slack)))
    max_ratio = float(np.nanmax(ratio)) if ratio.size and np.any(np.isfinite(ratio)) else float("nan")
    certified = violating == 0 and (not np.isfinite(max_ratio) or max_ratio <= MAX_RATIO)
    return dict(bound="active" if norm_constraint is not None else "assumed active", lambda_hat=lam,
                n_kept_violating=violating, n_near_active=int(np.sum(ratio >= slack)), max_ratio=max_ratio,
                certified=bool(certified), **common)


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


def repair_support(problem: FullDataHALProblem, theta: np.ndarray, norm_constraint: float, *,
                   polish: bool = True, prune: float = PRUNE, max_share: Optional[float] = None,
                   ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER) -> dict[str, Any]:
    """Proposed rule (not the default) for a solution that fails the optimality check at a binding bound.

    The kept coefficients whose gradient is below 0.99 lambda (the conditions say they are zero) are set
    to zero. ``polish=False`` ("zero"): only if together they hold at most ``max_share`` (default
    ``REPAIR_MAX_SHARE`` = 1%) of ||theta[1:]||_1; the check is re-run on that vector. ``polish=True``:
    the problem is re-solved on the remaining support (the near-active coefficients) and checked against
    the full gradient (active set: start from at most ``REPAIR_START_SIZE`` of the largest of them;
    after each re-solve keep the coefficients that stay nonzero and pass, add up to
    ``REPAIR_ADD_PER_ROUND`` that the check says are missing, gradient above 1.01 lambda; at most
    ``REPAIR_MAX_ROUNDS`` re-solves). A certified result is optimal however
    much was removed, so no share limit applies unless ``max_share`` is given. Returns the repaired raw
    solution, its check, and how many coefficients were removed or added (and their size).
    """
    if max_share is None and not polish:
        max_share = REPAIR_MAX_SHARE
    started = perf_counter()
    theta = np.asarray(theta, dtype=float)
    M = float(norm_constraint)
    k = problem.basis_order
    first = 1 if k == 0 else k + 1
    report = kkt_report(problem, theta, prune, norm_constraint=M)
    out: dict[str, Any] = dict(applied=False, certified=report["certified"], theta_raw=theta, kkt=report,
                               removed=0, removed_l1=0.0, removed_share=0.0, added=0, rounds=0, reason="",
                               polish_solver="")
    if report["certified"] or report["bound"] != "active":
        out.update(reason="already certified" if report["certified"] else "bound not binding",
                   seconds=perf_counter() - started)
        return out
    knot_theta = theta[first:]
    ratio = np.abs(problem.gradient(theta)[first:]) / report["lambda_hat"]
    kept = np.abs(knot_theta) > prune
    low = kept & (ratio < KKT_SLACK)
    l1 = float(np.abs(theta[1:]).sum())
    share = float(np.abs(knot_theta[low]).sum()) / l1 if l1 > 0 else 0.0
    out["removed_share"] = share
    if max_share is not None and share > max_share:
        out.update(reason=f"the coefficients to zero hold {share:.3g} of the L1 norm", removed_share=share,
                   seconds=perf_counter() - started)
        return out
    candidate = theta.copy()
    candidate[first:][~kept | low] = 0.0
    if not polish:
        rep = kkt_report(problem, candidate, prune, norm_constraint=M)
        out.update(applied=True, certified=rep["certified"], theta_raw=candidate, kkt=rep)
    else:
        support = np.flatnonzero(kept & ~low)
        if support.size > REPAIR_START_SIZE:            # active-set start: the largest near-active coefficients
            support = support[np.argsort(-np.abs(knot_theta[support]))[:REPAIR_START_SIZE]]
        for rounds in range(1, REPAIR_MAX_ROUNDS + 1):
            if support.size > REPAIR_MAX_SUPPORT:
                out["reason"] = f"support of {support.size} knots too large to polish"
                break
            solved, label = problem.solve_on_support(support, M, ladder)
            if solved is None:
                out["reason"] = "restricted solve failed"
                break
            rep = kkt_report(problem, solved, prune, norm_constraint=M)
            out.update(applied=True, certified=rep["certified"], theta_raw=solved, kkt=rep, rounds=rounds,
                       polish_solver=label)
            if rep["certified"]:
                break
            lam = rep["lambda_hat"]
            new_ratio = np.abs(problem.gradient(solved)[first:]) / lam if lam > 0 else np.zeros(knot_theta.size)
            nonzero = np.abs(solved[first:]) > prune
            keep = support[nonzero[support] & (new_ratio[support] >= KKT_SLACK)]
            outside = np.setdiff1d(np.arange(knot_theta.size), support)
            missing = outside[new_ratio[outside] > MAX_RATIO]
            add = missing[np.argsort(-new_ratio[missing])][:REPAIR_ADD_PER_ROUND]
            if not add.size and keep.size == support.size:
                out["reason"] = "uncertified with nothing to add or drop"
                break
            support = np.union1d(keep, add)
    final = np.abs(out["theta_raw"][first:]) > prune
    out.update(removed=int(np.sum(kept & ~final)), removed_l1=float(np.abs(knot_theta[kept & ~final]).sum()),
               added=int(np.sum(~kept & final)), seconds=perf_counter() - started)
    out["removed_share"] = out["removed_l1"] / l1 if l1 > 0 else 0.0
    return out


def certified_solve(
    problem: FullDataHALProblem,
    norm_constraint: float,
    ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER,
    prune: float = PRUNE,
    repair: Optional[str] = None,
) -> SupportRefit:
    """Solve ``problem`` at ``norm_constraint`` down ``ladder`` until the optimality check certifies.

    The first certified solution wins, else the least violating one. The problem is built once (the
    structured form when available; the dense form only if no configuration returns a usable
    solution of it). Every configuration tried is recorded in ``attempts``. ``repair`` (proposed, off
    by default): "zero" or "polish" applies :func:`repair_support` to each uncertified solution before
    the next configuration is tried.
    """
    if repair not in (None, "zero", "polish"):
        raise ValueError(f"repair must be None, 'zero' or 'polish', got {repair!r}")
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
                _solve_quietly(built.problem, solver, options)
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
                                 n_kept_violating=report["n_kept_violating"], max_ratio=report["max_ratio"],
                                 bound=report["bound"], max_gradient_per_obs=report["max_gradient_per_obs"],
                                 certified=report["certified"], seconds=perf_counter() - began))
            closeness = abs(report["max_ratio"] - 1.0) if np.isfinite(report["max_ratio"]) else report["max_gradient_per_obs"]
            rank = (not report["certified"], report["n_kept_violating"], closeness)
            if best is None or rank < best[0]:
                dual = built.l1_constraint.dual_value
                best = (rank, solver, label, status, theta, report, built.formulation,
                        float(built.problem.value) * built.objective_scale,
                        None if dual is None else float(np.asarray(dual).ravel()[0]) * built.dual_scale)
            if report["certified"]:
                break
            if repair is not None:
                began = perf_counter()
                # the restricted problem is small: always solved with the tight ladder
                fixed = repair_support(problem, theta, norm_constraint, polish=repair == "polish", prune=prune)
                attempts.append(dict(solver=solver, option=f"{label} + repair {repair}", formulation=built.formulation,
                                     outcome="repaired" if fixed["applied"] else f"not repaired: {fixed['reason']}",
                                     certified=fixed["certified"], removed=fixed["removed"],
                                     removed_share=fixed["removed_share"], added=fixed["added"],
                                     seconds=perf_counter() - began))
                if fixed["certified"]:
                    best = ((False, 0, 0.0), solver, f"{label} + repair {repair}", status, fixed["theta_raw"],
                            fixed["kkt"], built.formulation, problem.objective(fixed["theta_raw"]),
                            fixed["kkt"]["lambda_hat"])
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
    max_knots: Optional[int] = None,
    ladder: tuple[tuple[str, str, dict[str, Any]], ...] = LADDER,
    prune: float = PRUNE,
) -> SupportRefit:
    """Re-solve the full-data HAL-MLE at ``norm_constraint`` with tight tolerances and certify the support."""
    problem = FullDataHALProblem(data, basis_order, n_grid_points, objective_normalizer, max_knots)
    return certified_solve(problem, norm_constraint, ladder, prune)


def refit_support_from_estimator(estimator: Any, data: pd.DataFrame, **kwargs: Any) -> SupportRefit:
    """:func:`refit_support` with the bound, basis order and grid of a fitted ``CVXPYEstimator``."""
    if not getattr(estimator, "is_fitted", False):
        raise ValueError("Estimator must be fitted first")
    refit = refit_support(data, int(estimator.basis_order), float(estimator.norm_constraint),
                          n_grid_points=int(getattr(estimator, "n_grid_points", 200)),
                          objective_normalizer=getattr(estimator, "objective_normalizer", "midpoint"),
                          max_knots=getattr(estimator, "max_knots", None), **kwargs)
    if not np.array_equal(refit.knots, np.asarray(estimator._grid_points_hal, dtype=float)):
        raise ValueError("The data do not match the estimator's knots; pass the data it was fitted on")
    return refit
