"""Cross-fitting for the targeting steps, shared by the full-data and right-censored targets (Yilong, 2026-10-26).

``cross_fit`` names where the fold fits come from:

- ``"tuner"`` (default): the tuner's own fold fits at the parameters it chose, on its own splits ("cross-fitted
  with tuning on all folds": the bound M, the order and, for right-censored data, Stage 1's knots were chosen with
  every fold, including the one a fit is later scored on). An estimator fitted without a tuner, at a fixed bound,
  has none, and then ``"tuner"`` falls back to ``"refit"``; the result says which ran.
- ``"refit"``: the estimator's own settings refitted on each training fold (the tuner's splits when it has them,
  else 5 shuffled folds, seed 0).
- ``"nested"``: the whole tuning repeated inside each training fold, so no fold fit has seen its validation fold
  (about 5 times the cost of the initial fit).
- ``"none"``: no cross-fitting: the full-sample targeting step.

``True`` means ``"tuner"`` and ``False`` ``"none"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

CROSS_FIT_VALUES = ("tuner", "refit", "nested", "none")
DEFAULT_CROSS_FIT = "tuner"
DEFAULT_FOLDS = 5
DEFAULT_FOLD_SEED = 0


def resolve_cross_fit(value: Union[str, bool, None]) -> str:
    """``cross_fit`` as one of :data:`CROSS_FIT_VALUES` (True: "tuner", False: "none")."""
    if value is True:
        return "tuner"
    if value is False:
        return "none"
    if value is None:
        return DEFAULT_CROSS_FIT
    if not isinstance(value, str) or value not in CROSS_FIT_VALUES:
        raise ValueError(f"cross_fit must be one of {CROSS_FIT_VALUES} (or True / False), got {value!r}")
    return value


@dataclass
class FoldFits:
    """Fold fits for cross-fitted targeting: ``splits`` holds (training rows, validation rows) of the data the fits
    came from, ``estimators`` the fit trained on each training fold; ``source`` is "tuner", "refit" or "nested";
    ``data_key`` identifies that data, so the fits are not used with other data."""

    splits: list[tuple[np.ndarray, np.ndarray]]
    estimators: list[Any]
    source: str
    data_key: Optional[tuple] = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.estimators)

    def matches(self, key: tuple) -> bool:
        return self.data_key is None or self.data_key == key


def data_key(*columns: np.ndarray) -> tuple:
    """A key for the data the fold fits were made on: its length and a digest of its values (stable across
    processes, unlike ``hash``)."""
    import hashlib

    arrays = [np.ascontiguousarray(np.asarray(c, dtype=float)) for c in columns]
    return (int(arrays[0].size),) + tuple(hashlib.sha1(a.tobytes()).hexdigest() for a in arrays)


def default_splits(n_obs: int, n_splits: int = DEFAULT_FOLDS,
                   random_state: int = DEFAULT_FOLD_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold splits, as the tuners make them (sklearn KFold)."""
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    return [(train.copy(), validation.copy()) for train, validation in kfold.split(np.arange(int(n_obs)))]
