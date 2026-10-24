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
de-duplicated), as hal9001's ``num_knots``; ``None`` keeps every unique data value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from haldensity.core.basis.univariate import create_basis_functions

OBJECTIVE_NORMALIZERS = ("midpoint", "knot_aware")


def candidate_knots(x: np.ndarray, max_knots: Optional[int] = None) -> np.ndarray:
    """All unique data values, or ``max_knots`` data quantiles (observations) when capped."""
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    if max_knots is None or int(max_knots) >= unique.size:
        return unique
    if int(max_knots) < 1:
        raise ValueError("max_knots must be a positive integer or None")
    probs = np.linspace(0.0, 1.0, int(max_knots))
    return np.unique(np.quantile(values, probs, method="inverted_cdf"))


@dataclass(frozen=True)
class NormalizerTerms:
    """Cells of the objective's normalizer.

    ``log_widths[c]`` is the log width of cell c. The log-density on the cells is either
    ``basis @ theta`` (``basis`` set) or, for k = 0 knot-aware, ``[theta[0]] (if leading_intercept_cell)
    followed by (data_basis @ theta)[data_rows]``.
    """

    log_widths: np.ndarray
    basis: Optional[np.ndarray] = None
    data_rows: Optional[np.ndarray] = None
    leading_intercept_cell: bool = False

    @property
    def n_cells(self) -> int:
        return int(self.log_widths.size)

    def log_density(self, theta: np.ndarray, data_log_density: Optional[np.ndarray] = None) -> np.ndarray:
        """Log-density on the cells for a numeric ``theta`` (numpy)."""
        if self.basis is not None:
            return self.basis.astype(float) @ np.asarray(theta, dtype=float)
        if data_log_density is None:
            raise ValueError("data_log_density is required when the cells reuse the data rows")
        cells = np.asarray(data_log_density, dtype=float)[self.data_rows]
        return np.concatenate(([float(theta[0])], cells)) if self.leading_intercept_cell else cells


def normalizer_terms(
    x: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    n_grid_points: int,
    objective_normalizer: str = "midpoint",
) -> NormalizerTerms:
    """The cells of the objective's normalizer; ``x`` are the data values the basis rows belong to."""
    if objective_normalizer not in OBJECTIVE_NORMALIZERS:
        raise ValueError(f"objective_normalizer must be one of {OBJECTIVE_NORMALIZERS}, got {objective_normalizer!r}")
    knots = np.asarray(knots, dtype=float).ravel()
    grid = np.linspace(0, 1, int(n_grid_points))
    if objective_normalizer == "midpoint":
        mids = (grid[:-1] + grid[1:]) / 2
        basis = create_basis_functions(pd.DataFrame({"W1": mids}), knots, order=basis_order)[0]
        return NormalizerTerms(log_widths=np.log(np.diff(grid)), basis=basis)

    if int(basis_order) == 0:
        x = np.asarray(x, dtype=float).ravel()
        order = np.argsort(x, kind="stable")
        sorted_x = x[order]
        starts = knots[(knots >= 0.0) & (knots < 1.0)]
        if starts.size == 0:      # exp(f) = exp(theta[0]) on all of [0, 1)
            return NormalizerTerms(log_widths=np.zeros(1), data_rows=np.zeros(0, dtype=int), leading_intercept_cell=True)
        position = np.searchsorted(sorted_x, starts, side="left")
        if np.any(position >= sorted_x.size) or not np.array_equal(sorted_x[np.minimum(position, sorted_x.size - 1)], starts):
            raise ValueError("knot_aware with k = 0 needs every knot to be a data value")
        rows = order[position]
        ends = np.append(starts[1:], 1.0)
        widths = ends - starts
        leading = bool(starts[0] > 0.0)
        if leading:
            widths = np.concatenate(([starts[0]], widths))
        return NormalizerTerms(log_widths=np.log(widths), data_rows=rows, leading_intercept_cell=leading)

    edges = np.unique(np.concatenate((grid, knots[(knots > 0.0) & (knots < 1.0)])))
    mids = (edges[:-1] + edges[1:]) / 2
    basis = create_basis_functions(pd.DataFrame({"W1": mids}), knots, order=basis_order)[0]
    return NormalizerTerms(log_widths=np.log(np.diff(edges)), basis=basis)
