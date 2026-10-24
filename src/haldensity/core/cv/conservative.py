"""Conservative adjustment of a cross-validated L1 bound (core, used by every data type).

After cross-validation picks the bound M* with mean score S(M*) and fold SD sd(M*),
the conservative bound is the smallest M whose mean CV score is still at least
the threshold

* ``"legacy_percent_sd"`` (default): S(M*) - k_percent * sd(M*), or
* ``"one_se"``: S(M*) - se_multiplier * sd(M*) / sqrt(n_valid_folds).

It is found by geometric bracketing below M* (multiply by ``bracket_factor``
until the score drops below the threshold) followed by bisection in log M to a
relative tolerance. Why: with many nearly collinear basis functions the CV
score can be flat in M over a wide range, so the CV maximiser lands on a huge M;
the adjustment moves to the smallest M that is not worse than the best by more
than the threshold. See docs/dev/cvll-multicollinearity.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

SELECTION_RULES = ("legacy_percent_sd", "one_se")


def cv_standard_error(sd_score: float, n_valid: int) -> float:
    """Convert a fold-level SD into a standard error for the mean CV score."""
    n = int(max(n_valid, 0))
    if n <= 0 or not np.isfinite(sd_score):
        return 0.0
    return float(sd_score) / float(np.sqrt(n))


def conservative_threshold(
    *,
    best_mean: float,
    best_sd: float,
    n_valid: int,
    selection_rule: str,
    k_percent: float,
    se_multiplier: float,
) -> tuple[float, dict[str, Any]]:
    """The conservative-selection threshold and its diagnostics."""
    rule = str(selection_rule).strip().lower()
    if rule == "legacy_percent_sd":
        threshold = float(best_mean - float(k_percent) * float(best_sd))
        return threshold, {
            "selection_rule": "legacy_percent_sd",
            "k_percent": float(k_percent),
            "best_sd": float(best_sd),
            "best_standard_error": cv_standard_error(best_sd, n_valid),
            "n_valid": int(n_valid),
        }
    if rule == "one_se":
        best_se = cv_standard_error(best_sd, n_valid)
        threshold = float(best_mean - float(se_multiplier) * best_se)
        return threshold, {
            "selection_rule": "one_se",
            "se_multiplier": float(se_multiplier),
            "best_sd": float(best_sd),
            "best_standard_error": float(best_se),
            "n_valid": int(n_valid),
        }
    raise ValueError("selection_rule must be one of {'legacy_percent_sd', 'one_se'}")


def bisection_search(
    is_above: Callable[[float, int], bool],
    start: float,
    *,
    bracket_factor: float,
    min_norm_constraint: float,
    rel_tol: float,
    max_steps: int,
) -> float:
    """Smallest bound with ``is_above(bound, step)`` true, by bracketing then log-bisection.

    ``is_above(start, 0)`` is evaluated first (the CV-selected bound, above the
    threshold by construction). Each later call gets the next step number; the
    search stops after ``max_steps`` evaluations beyond the start.
    """
    step = 0
    is_above(start, step)
    step += 1

    best_good = float(start)
    bad: Optional[float] = None
    while step <= max_steps:
        candidate = best_good * bracket_factor
        if candidate <= min_norm_constraint:
            best_good = float(min_norm_constraint)
            break
        above = is_above(candidate, step)
        step += 1
        if above:
            best_good = float(candidate)
            continue
        bad = float(candidate)
        break

    if bad is None:
        return float(best_good)

    good = float(best_good)
    tol = float(max(rel_tol, 1e-6))
    while step <= max_steps:
        if good <= min_norm_constraint:
            good = float(min_norm_constraint)
            break
        if (good - bad) / max(good, min_norm_constraint) <= tol:
            break
        mid = float(np.exp((np.log(bad) + np.log(good)) / 2.0))
        above = is_above(mid, step)
        step += 1
        if above:
            good = mid
        else:
            bad = mid
    return float(good)
