"""Internal safeguard: the coefficient imbalance of a k = 0 fit (Yilong, 2026-10-24).

The pattern (Yilong's CVLL note, docs/dev/cvll-multicollinearity): two neighbouring knots, less than
1e-3 apart, carry huge coefficients of opposite sign that cancel on the data and use up the L1 budget.
With the package's objective at k = 0 (the exact objective) it should not occur: in session 69's
check it appeared in 0 of 800 fits (largest |theta| 1.45). If it does, it signals a numerical problem
inside the package, so the fit warns and asks for a report; it is not a user setting and nothing is
refitted or adjusted.

Flags on the final fit's coefficients (theta[1:], over the selected knots in order):

* A: adjacent selected knots less than ``PAIR_GAP`` = 1e-3 apart with opposite-sign coefficients,
  both larger than ``BIG`` = 10 in absolute value;
* B: coefficients with |theta| > ``BIG`` carry more than ``BIG_SHARE`` = 1/2 of ||theta[1:]||_1.

Reference implementation: uni_sieve_MLE ``imbalance.py::imbalance_diagnostics`` (branch local-target).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
from scipy.special import logsumexp

PAIR_GAP = 1e-3
BIG = 10.0
BIG_SHARE = 0.5
REPORT_URL = "https://github.com/yilongHou/HALDensity/issues"


def imbalance_diagnostics(theta: np.ndarray, knots: np.ndarray, data: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Diagnostics and flags of a k = 0 fit: ``theta`` = (intercept, one coefficient per candidate knot)."""
    theta = np.asarray(theta, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    order = np.argsort(knots, kind="stable")
    knots, coef = knots[order], theta[1:][order]
    selected = np.flatnonzero(coef != 0.0)
    absolute = np.abs(coef)
    l1 = float(absolute.sum())
    out: dict[str, Any] = dict(n_selected=int(selected.size), l1_norm=l1,
                               max_abs_theta=float(absolute.max()) if absolute.size else 0.0)
    # adjacent selected knots with opposite-sign coefficients: the worst pair, and pairs that flag A
    worst, worst_gap, flagged = 0.0, math.nan, []
    for left, right in zip(selected[:-1], selected[1:]):
        if coef[left] * coef[right] < 0:
            size, gap = float(min(absolute[left], absolute[right])), float(knots[right] - knots[left])
            if size > worst:
                worst, worst_gap = size, gap
            if gap < PAIR_GAP and size > BIG:
                pair = dict(knots=(float(knots[left]), float(knots[right])),
                            coefficients=(float(coef[left]), float(coef[right])), gap=gap)
                if data is not None:
                    d = np.asarray(data, dtype=float)
                    pair["points_between"] = int(np.count_nonzero((d > knots[left]) & (d < knots[right])))
                flagged.append(pair)
    out.update(opposite_pair_min_abs=worst, opposite_pair_gap=worst_gap)
    for c in (BIG, 1.0):
        out[f"share_l1_above_{c:g}"] = float(absolute[absolute > c].sum() / l1) if l1 > 0 else 0.0
    # the largest share of the fitted mass in one cell between consecutive selected knots (exact at k = 0)
    active = knots[selected]
    edges = np.unique(np.concatenate(([0.0], active[(active > 0.0) & (active < 1.0)], [1.0])))
    levels = theta[0] + np.concatenate(([0.0], np.cumsum(coef)))[np.searchsorted(knots, edges[:-1], side="right")]
    log_mass = levels + np.log(np.diff(edges))
    share = np.exp(log_mass - logsumexp(log_mass))
    i = int(np.argmax(share))
    out.update(max_cell_mass_share=float(share[i]), max_cell_width=float(edges[i + 1] - edges[i]))
    out.update(flag_a=bool(flagged), flag_a_pairs=flagged[:5], flag_b=bool(out[f"share_l1_above_{BIG:g}"] > BIG_SHARE))
    return out


def check_imbalance(theta: np.ndarray, knots: np.ndarray, data: Optional[np.ndarray] = None,
                    stacklevel: int = 2) -> dict[str, Any]:
    """:func:`imbalance_diagnostics`, with a warning asking for a report if flag A or B fires."""
    diagnostics = imbalance_diagnostics(theta, knots, data)
    if diagnostics["flag_a"] or diagnostics["flag_b"]:
        shown = {key: diagnostics[key] for key in ("flag_a", "flag_b", "flag_a_pairs", "share_l1_above_10",
                                                   "max_abs_theta", "opposite_pair_min_abs", "opposite_pair_gap",
                                                   "max_cell_mass_share", "max_cell_width", "n_selected")}
        warnings.warn(
            "Internal check failed: the k = 0 fit shows a coefficient imbalance (huge opposite-sign coefficients "
            "on neighbouring knots, or most of the L1 norm in coefficients above 10). This indicates a numerical "
            f"problem inside HALDensity, not a property of your data; please report it at {REPORT_URL} with "
            f"these diagnostics: {shown}",
            UserWarning,
            stacklevel=stacklevel,
        )
    return diagnostics
