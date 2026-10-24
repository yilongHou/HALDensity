"""Which normalizer the fitting objective uses, by basis order k and dimension d (``"auto"``).

Decided by Yilong, 2026-10-24 ("we can tailor different rules for different k, and in future
different d"). :func:`objective_rule` returns a rule identifier; core does not implement the rules,
each data-type package does (core never imports a data-type package):

* d = 1 (``haldensity.full_uni``):

  - k = 0: ``"exact"``, the exact objective: one term per cell between consecutive knots;
  - k = 1: ``"midpoint_knots_added"``, the midpoint rule with the knots added to the grid
    (both through the step-height rewrite);
  - k >= 2: ``"midpoint"``, the old midpoint rule (199 fixed cells). With the knots added, k = 2 cost
    107-415 s per 5-fold CV trial at n = 1,280 against 3.4-4.3 s, with about the same CV score.

* d >= 2 (``haldensity.full_multi``, its existing rule): k = 0: ``"exact_or_sobol"`` (exact below the
  cell quota, the Sobol bank above); k >= 1: ``"sobol"``.

Evidence for d = 1: docs/dev/benchmarks/knot_aware_normalizer (the old midpoint rule spiked at k = 0
and ran to 17 GB at n = 20,000; the exact objective and the knots-added grid never spiked).
"""

from __future__ import annotations

EXACT = "exact"
MIDPOINT_KNOTS_ADDED = "midpoint_knots_added"
MIDPOINT = "midpoint"
EXACT_OR_SOBOL = "exact_or_sobol"
SOBOL = "sobol"
OBJECTIVE_RULES = (EXACT, MIDPOINT_KNOTS_ADDED, MIDPOINT, EXACT_OR_SOBOL, SOBOL)


def objective_rule(basis_order: int, dimension: int = 1) -> str:
    """The objective's normalizer rule for basis order ``basis_order`` in ``dimension`` dimensions."""
    k, d = int(basis_order), int(dimension)
    if k != basis_order or k < 0:
        raise ValueError(f"basis_order must be a nonnegative integer, got {basis_order!r}")
    if d != dimension or d < 1:
        raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
    if d >= 2:
        return EXACT_OR_SOBOL if k == 0 else SOBOL
    if k == 0:
        return EXACT
    if k == 1:
        return MIDPOINT_KNOTS_ADDED
    return MIDPOINT
