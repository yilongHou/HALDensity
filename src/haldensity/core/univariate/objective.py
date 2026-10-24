"""Pieces of the full-data HAL-MLE objective shared by the estimator and the basis-cleaning refit.

The objective is ``-sum_i f(x_i) + n log Z(theta)`` with ``f = theta' phi`` and an approximate
normalizer ``log Z = log sum_c w_c exp(f(t_c))`` over cells c with widths w_c:

* ``objective_normalizer="midpoint"`` (default): the ``n_grid_points - 1`` equal cells of
  ``linspace(0, 1, n_grid_points)``, evaluated at their midpoints. A cell between two close knots
  that contains no grid midpoint is invisible to it, which lets the solver put a spike there.
* ``objective_normalizer="knot_aware"``: the candidate knots are breakpoints of the cells.
  For k = 0 the cells are exactly those between consecutive breakpoints {0, knots, 1}, on which
  ``exp(f)`` is constant, so the normalizer is exact; the level on the cell starting at knot xi is
  ``f(xi)``, already computed for the data term (every knot is a data value), so no new matrix is
  built. For k >= 1 the breakpoints are the knots together with the equal grid, with one midpoint
  per cell (not exact, but no gap between knots goes unseen).

``max_knots`` caps the number of candidate knots: K quantiles of the data (actual observations,
de-duplicated), as hal9001's ``num_knots``; ``None`` keeps every unique data value. ``"auto"`` (the
estimator's default; Yilong, 2026-10-24) is :func:`default_max_knots`: K = 1,000 for k >= 1 and
K = max(1,000, ceil(n^(2/3))) for k = 0. A rough rule backed by the benchmark
(docs/dev/benchmarks/knot_aware_normalizer): with K knots a jump can only be placed to within about
1/K, which costs about h^2 / K in squared L2 for a jump of height h; that must stay below the
statistical error of k = 0, n^(-2/3), so K must be at least about n^(2/3). At k >= 1 the same count
needs far fewer knots (about n^(4/15) for k = 1), so 1,000 is ample. On the jump density at
n = 20,000 (n^(2/3) ~ 740), K = 1,000 cost 5-13% in L2 against all knots, K = 250 up to 43% and
K = 100 up to 88%. The k = 0 rule exceeds 1,000 only above n of about 31,600.

Structured formulation (knot-aware, k = 0 and 1). The dense formulation multiplies an n x m basis
by theta. The same problem has a sparse form in the values of f on the cells (a linear, invertible
change of variables, so the optimum is the same):

* k = 0: levels L_c on the cells [0, xi_1), [xi_1, xi_2), ..., [xi_m, 1]; theta_0 = L_0,
  theta_j = L_j - L_{j-1}, so ||theta[1:]||_1 = ||diff(L)||_1 (a total-variation constraint), the
  data term is sum_c count_c L_c and the normalizer is log sum_c w_c exp(L_c).
* k = 1: values u_j of f at the interval starts {0, xi_1, ..., xi_m} and one slope s_j per interval,
  with u_{j+1} = u_j + s_j (xi_{j+1} - xi_j); theta_0 = u_0, theta_1 = s_0, theta_{1+j} = s_j - s_{j-1},
  so ||theta[1:]||_1 = |s_0| + ||diff(s)||_1; f at a cell midpoint or at an observation is
  u_j + s_j (t - xi_j) for the interval j that contains it.

Both have O(n + cells) nonzeros instead of O(n m), see :func:`structured_problem`. They are scaled
for the solvers (objective / n; for k = 1 slopes / max(1, M)): without the scaling MOSEK stopped up
to 0.1 short of the optimum (objective summed over n = 1,000-5,000 observations) and CLARABEL up to
2.5; with it MOSEK's shortfall was at most 3e-5 over k = 0, 1, n = 500-5,000, M reasonable and 100x.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from haldensity.core.basis.univariate import create_basis_functions

OBJECTIVE_NORMALIZERS = ("midpoint", "knot_aware")      # what the objective implements
OBJECTIVE_SETTINGS = ("auto",) + OBJECTIVE_NORMALIZERS    # what a user may ask for


def resolve_objective_normalizer(setting: str, basis_order: int) -> tuple[str, str]:
    """(implemented normalizer, rule identifier) for a univariate fit of order ``basis_order``.

    ``"auto"`` follows :func:`haldensity.core.objective_rules.objective_rule` with d = 1: the exact
    objective at k = 0 and the knots-added grid at k = 1 (both ``"knot_aware"``), the old midpoint rule
    at k >= 2. An explicit ``"midpoint"`` or ``"knot_aware"`` is used as given.
    """
    from haldensity.core.objective_rules import EXACT, MIDPOINT, MIDPOINT_KNOTS_ADDED, objective_rule
    if setting not in OBJECTIVE_SETTINGS:
        raise ValueError(f"objective_normalizer must be one of {OBJECTIVE_SETTINGS}, got {setting!r}")
    k = int(basis_order)
    if setting == "auto":
        rule = objective_rule(k, 1)
        return ("midpoint" if rule == MIDPOINT else "knot_aware"), rule
    if setting == "midpoint":
        return "midpoint", MIDPOINT
    return "knot_aware", EXACT if k == 0 else MIDPOINT_KNOTS_ADDED
STRUCTURED_ORDERS = (0, 1)
KNOT_CAP = 1000                  # default number of candidate knots (k >= 1; the floor for k = 0)


def default_max_knots(n: int, basis_order: int) -> int:
    """The default knot cap: 1,000 for k >= 1, max(1,000, ceil(n^(2/3))) for k = 0 (module docstring)."""
    if int(basis_order) != 0:
        return KNOT_CAP
    n = int(n)
    k = max(1, int(round(n ** (2.0 / 3.0))))          # exact integer ceil(n^(2/3)): least k with k^3 >= n^2
    while k ** 3 < n ** 2:
        k += 1
    while k > 1 and (k - 1) ** 3 >= n ** 2:
        k -= 1
    return max(KNOT_CAP, k)


def resolve_max_knots(max_knots, n: int, basis_order: Optional[int]) -> Optional[int]:
    """The cap K for a sample of n observations: ``"auto"`` -> :func:`default_max_knots`; None (all
    knots) or a positive integer as given."""
    if isinstance(max_knots, str):
        if max_knots != "auto":
            raise ValueError(f"max_knots must be 'auto', None or a positive integer, got {max_knots!r}")
        if basis_order is None:
            raise ValueError("max_knots='auto' needs the basis order")
        return default_max_knots(n, basis_order)
    if max_knots is None:
        return None
    if int(max_knots) < 1:
        raise ValueError("max_knots must be 'auto', None or a positive integer")
    return int(max_knots)


def candidate_knots(x: np.ndarray, max_knots=None, basis_order: Optional[int] = None) -> np.ndarray:
    """All unique data values, or K data quantiles (observations) when capped at K < #unique values.

    ``max_knots``: None (all), a positive integer K, or ``"auto"`` (:func:`default_max_knots`, which
    needs ``basis_order``; n is the number of finite observations).
    """
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    cap = resolve_max_knots(max_knots, values.size, basis_order)
    if cap is None or cap >= unique.size:
        return unique
    probs = np.linspace(0.0, 1.0, cap)
    return np.unique(np.quantile(values, probs, method="inverted_cdf"))


@dataclass(frozen=True)
class NormalizerTerms:
    """Cells of the objective's normalizer.

    ``log_widths[c]`` is the log width of cell c and ``points[c]`` the point at which the objective
    evaluates f for that cell (the midpoint; the left end for k = 0 knot-aware, where f is constant on
    the cell). In the dense formulation the log-density on the cells is either ``basis @ theta``
    (``basis`` set) or, for k = 0 knot-aware, ``[theta[0]] (if leading_intercept_cell)`` followed by
    ``(data_basis @ theta)[data_rows]``.
    """

    log_widths: np.ndarray
    points: np.ndarray
    basis: Optional[np.ndarray] = None
    data_rows: Optional[np.ndarray] = None
    leading_intercept_cell: bool = False

    @property
    def n_cells(self) -> int:
        return int(self.log_widths.size)

    def log_density(self, theta: np.ndarray, knots: np.ndarray, basis_order: int) -> np.ndarray:
        """Log-density on the cells for a numeric ``theta`` (float64)."""
        from haldensity.core.univariate.normalizer import log_unnormalized_density
        return log_unnormalized_density(self.points, theta, knots, basis_order)


def normalizer_terms(
    x: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    n_grid_points: int,
    objective_normalizer: str = "midpoint",
    build_basis: bool = True,
) -> NormalizerTerms:
    """The cells of the objective's normalizer; ``x`` are the data values the basis rows belong to.

    ``build_basis=False`` skips the dense basis on the cells (the structured formulation and the
    gradient need only widths and points).
    """
    if objective_normalizer not in OBJECTIVE_NORMALIZERS:
        raise ValueError(f"objective_normalizer must be one of {OBJECTIVE_NORMALIZERS}, got {objective_normalizer!r}")
    knots = np.asarray(knots, dtype=float).ravel()
    grid = np.linspace(0, 1, int(n_grid_points))
    if objective_normalizer == "midpoint":
        mids = (grid[:-1] + grid[1:]) / 2
        basis = create_basis_functions(pd.DataFrame({"W1": mids}), knots, order=basis_order)[0] if build_basis else None
        return NormalizerTerms(log_widths=np.log(np.diff(grid)), points=mids, basis=basis)

    if int(basis_order) == 0:
        x = np.asarray(x, dtype=float).ravel()
        order = np.argsort(x, kind="stable")
        sorted_x = x[order]
        starts = knots[(knots >= 0.0) & (knots < 1.0)]
        if starts.size == 0:      # exp(f) = exp(theta[0]) on all of [0, 1)
            return NormalizerTerms(log_widths=np.zeros(1), points=np.zeros(1), data_rows=np.zeros(0, dtype=int),
                                   leading_intercept_cell=True)
        position = np.searchsorted(sorted_x, starts, side="left")
        if np.any(position >= sorted_x.size) or not np.array_equal(sorted_x[np.minimum(position, sorted_x.size - 1)], starts):
            raise ValueError("knot_aware with k = 0 needs every knot to be a data value")
        rows = order[position]
        ends = np.append(starts[1:], 1.0)
        widths = ends - starts
        leading = bool(starts[0] > 0.0)
        points = starts
        if leading:
            widths = np.concatenate(([starts[0]], widths))
            points = np.concatenate(([0.0], starts))
        return NormalizerTerms(log_widths=np.log(widths), points=points, data_rows=rows, leading_intercept_cell=leading)

    edges = knot_aware_edges(knots, n_grid_points)
    mids = (edges[:-1] + edges[1:]) / 2
    basis = create_basis_functions(pd.DataFrame({"W1": mids}), knots, order=basis_order)[0] if build_basis else None
    return NormalizerTerms(log_widths=np.log(np.diff(edges)), points=mids, basis=basis)


def knot_aware_edges(knots: np.ndarray, n_grid_points: int) -> np.ndarray:
    """Breakpoints of the k >= 1 knot-aware cells: the equal grid together with the knots in (0, 1)."""
    knots = np.asarray(knots, dtype=float).ravel()
    return np.unique(np.concatenate((np.linspace(0, 1, int(n_grid_points)), knots[(knots > 0.0) & (knots < 1.0)])))


@dataclass
class StructuredProblem:
    """The sparse knot-aware problem; ``theta()`` maps its solution back to HAL coefficients.

    The problem is scaled for the conic solvers (objective divided by n; for k = 1 the slopes by
    S = max(1, M)): its optimal value times ``objective_scale`` is the objective of the dense
    formulation, and the L1 constraint's dual value times ``dual_scale`` is its dual.
    """

    problem: cp.Problem
    l1_constraint: cp.Constraint
    theta: Callable[[], Optional[np.ndarray]]
    n_variables: int
    objective_scale: float
    dual_scale: float


def structured_problem(
    x: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    n_grid_points: int,
    norm_constraint: float,
) -> StructuredProblem:
    """The knot-aware HAL-MLE for k = 0 or 1 in sparse form (see the module docstring)."""
    k = int(basis_order)
    if k not in STRUCTURED_ORDERS:
        raise ValueError(f"the structured formulation covers basis_order in {STRUCTURED_ORDERS}")
    x = np.asarray(x, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    n, m, M = x.size, knots.size, float(norm_constraint)

    if k == 0:
        # cell c = [0, xi_1) for c = 0, [xi_c, xi_{c+1}) for c >= 1; observation x is in cell #(knots <= x)
        share = np.bincount(np.searchsorted(knots, x, side="right"), minlength=m + 1) / n
        widths = np.diff(np.concatenate(([0.0], np.clip(knots, 0.0, 1.0), [1.0])))
        keep = np.flatnonzero(widths > 0)
        L = cp.Variable(m + 1)
        loss = -share @ L + cp.log_sum_exp(np.log(widths[keep]) + L[keep])
        l1 = cp.norm1(cp.diff(L)) <= M
        problem = cp.Problem(cp.Minimize(loss), [l1])

        def theta() -> Optional[np.ndarray]:
            if L.value is None:
                return None
            return np.concatenate(([L.value[0]], np.diff(L.value)))

        return StructuredProblem(problem, l1, theta, m + 1, objective_scale=float(n), dual_scale=float(n))

    # k = 1: u_j = f(a_j) at the interval starts a = [0, xi_1, ..., xi_m]; slope s_j = S * sigma_j on
    # [a_j, a_{j+1}); u_{j+1} = u_j + s_j (a_{j+1} - a_j). Cells of the knot-aware grid lie inside one
    # interval, so f at a cell midpoint is u_j + s_j (mid - a_j), and likewise at each observation.
    S = max(1.0, M)
    starts = np.concatenate(([0.0], knots))
    gaps = np.diff(np.concatenate((starts, [1.0])))[:-1]
    edges = knot_aware_edges(knots, n_grid_points)
    widths, mids = np.diff(edges), (edges[:-1] + edges[1:]) / 2
    j_cell = np.searchsorted(knots, edges[:-1], side="right")              # #(knots <= left end)
    j_x = np.searchsorted(knots, x, side="right")
    share_u = np.bincount(j_x, minlength=m + 1) / n
    share_s = np.bincount(j_x, weights=x - starts[j_x], minlength=m + 1) / n
    u, sigma = cp.Variable(m + 1), cp.Variable(m + 1)
    f_mid = u[j_cell] + cp.multiply(S * (mids - starts[j_cell]), sigma[j_cell])
    loss = -(share_u @ u + S * (share_s @ sigma)) + cp.log_sum_exp(np.log(widths) + f_mid)
    links = u[1:] - u[:-1] == cp.multiply(S * gaps, sigma[:-1])
    l1 = cp.abs(sigma[0]) + cp.norm1(cp.diff(sigma)) <= M / S
    problem = cp.Problem(cp.Minimize(loss), [links, l1])

    def theta() -> Optional[np.ndarray]:
        if u.value is None or sigma.value is None:
            return None
        return np.concatenate(([u.value[0], S * sigma.value[0]], S * np.diff(sigma.value)))

    return StructuredProblem(problem, l1, theta, 2 * (m + 1), objective_scale=float(n), dual_scale=float(n) / S)


def basis_matrix(points: np.ndarray, knots: np.ndarray, basis_order: int) -> np.ndarray:
    """The HAL basis at ``points`` in float64: 1, t, ..., t^k, then I(t >= xi) (k = 0) or (t - xi)_+^k."""
    t = np.asarray(points, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    k = int(basis_order)
    columns = [np.ones_like(t)] + [t ** r for r in range(1, k + 1)]
    if knots.size:
        columns.append((t[:, None] >= knots[None, :]).astype(float) if k == 0
                       else np.maximum(t[:, None] - knots[None, :], 0.0) ** k)
    return np.column_stack(columns)


def data_basis_sums(x: np.ndarray, knots: np.ndarray, basis_order: int) -> np.ndarray:
    """sum_i phi(x_i) for every basis function, exactly in float64 from sorted data and prefix sums
    (O(n log n)): the data term -sum_i f(x_i) is -data_basis_sums @ theta, without the n x p basis."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    k = int(basis_order)
    head = [float(x.size)] + [float(np.sum(x ** r)) for r in range(1, k + 1)]
    return np.concatenate((head, _tail_sums(x, np.ones(x.size), np.asarray(knots, dtype=float).ravel(), k)))


@dataclass
class DenseProblem:
    """The HAL-MLE in theta (all normalizers); ``theta()`` returns the solution."""

    problem: cp.Problem
    l1_constraint: cp.Constraint
    theta: Callable[[], Optional[np.ndarray]]


def dense_problem(
    x: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    n_grid_points: int,
    objective_normalizer: str,
    norm_constraint: float,
) -> DenseProblem:
    """``-data_basis_sums @ theta + n log sum_c w_c exp(phi(t_c)' theta)`` with ``||theta[1:]||_1 <= M``.

    The data term is linear in theta, so only the cells x p normalizer basis is a matrix (199 x p for
    the old midpoint rule; one row per cell of the knots-added grid otherwise). Before 0.2.0 the
    problem also held the n x p data basis (17 GB at n = 20,000 with every knot).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    k = int(basis_order)
    sums = data_basis_sums(x, knots, k)
    terms = normalizer_terms(x, knots, k, n_grid_points, objective_normalizer, build_basis=False)
    cells = basis_matrix(terms.points, knots, k)
    theta = cp.Variable(sums.size)
    loss = -sums @ theta + x.size * cp.log_sum_exp(terms.log_widths + cells @ theta)
    l1 = cp.norm1(theta[1:]) <= float(norm_constraint)
    return DenseProblem(cp.Problem(cp.Minimize(loss), [l1]), l1,
                        lambda: None if theta.value is None else np.asarray(theta.value, dtype=float).copy())


def _tail_sums(points: np.ndarray, weights: np.ndarray, knots: np.ndarray, basis_order: int) -> np.ndarray:
    """sum_p weights_p * phi_knot(points_p) for every knot: I(p >= xi) for k = 0, (p - xi)_+^k for k >= 1."""
    order = np.argsort(points, kind="stable")
    p, w = points[order], weights[order]
    k = int(basis_order)
    if k == 0:
        tail = np.concatenate((np.cumsum(w[::-1])[::-1], [0.0]))
        return tail[np.searchsorted(p, knots, side="left")]
    start = np.searchsorted(p, knots, side="right")
    out = np.zeros(knots.size)
    for r in range(k + 1):
        moments = np.concatenate((np.cumsum((w * p ** r)[::-1])[::-1], [0.0]))
        out += comb(k, r) * (-knots) ** (k - r) * moments[start]
    return out


def objective_gradient(theta: np.ndarray, x: np.ndarray, knots: np.ndarray, basis_order: int,
                       terms: NormalizerTerms) -> np.ndarray:
    """Gradient of -sum_i f(x_i) + n log sum_c w_c exp(f(t_c)) in theta, without a dense basis.

    The same function for the dense and the structured formulation (it is defined on theta).
    """
    from haldensity.core.univariate.normalizer import log_unnormalized_density
    theta = np.asarray(theta, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    k, n = int(basis_order), x.size
    z = terms.log_widths + log_unnormalized_density(terms.points, theta, knots, k)
    cell_weights = n * np.exp(z - logsumexp(z))                   # n * softmax over the cells
    grad = np.empty(theta.size)
    grad[0] = -n + cell_weights.sum()
    for r in range(1, k + 1):
        grad[r] = -np.sum(x ** r) + np.sum(cell_weights * terms.points ** r)
    first = 1 if k == 0 else k + 1
    grad[first:] = -_tail_sums(x, np.ones(n), knots, k) + _tail_sums(terms.points, cell_weights, knots, k)
    return grad
