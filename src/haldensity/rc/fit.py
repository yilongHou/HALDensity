"""Fitted right-censored HAL distribution.

A :class:`~haldensity.core.univariate.fit.HALFit` that also records the last observed
failure time, so queries beyond it can be flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from haldensity.core.univariate.fit import HALFit, _as_1d


def last_failure_from_frame(
    data: pd.DataFrame,
    *,
    time_col: str = "T",
    delta_col: str = "Delta",
    fallback_col: str = "W1",
) -> Optional[float]:
    """Return the last observed failure time, if any."""
    if time_col in data.columns and delta_col in data.columns:
        times = np.asarray(data[time_col].values, dtype=float).ravel()
        delta = np.asarray(data[delta_col].values, dtype=int).ravel()
        events = times[delta == 1]
        if events.size:
            return float(np.max(events))
        return None
    if fallback_col in data.columns:
        values = np.asarray(data[fallback_col].values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            return float(np.max(values))
    return None


@dataclass(frozen=True)
class RightCensoredHALFit(HALFit):
    """Fitted right-censored HAL density with frozen ``(m, Z)`` and the last observed failure."""

    last_failure: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.last_failure is not None:
            object.__setattr__(self, "last_failure", float(self.last_failure))
            if not np.isfinite(self.last_failure):
                raise ValueError("last_failure must be finite when provided")

    @classmethod
    def from_estimator(
        cls,
        estimator: Any,
        *,
        last_failure: Optional[float] = None,
    ) -> "RightCensoredHALFit":
        """Build a fit from a live RC estimator without recomputing ``Z``."""
        fields = cls._estimator_fields(estimator)
        recorded = last_failure
        if recorded is None:
            recorded = getattr(estimator, "_last_failure", None)
        return cls(**fields, last_failure=None if recorded is None else float(recorded))

    def to_state(self) -> dict[str, Any]:
        """Return a JSON/npz-friendly state dictionary (same keys and order as before the split)."""
        state: dict[str, Any] = {}
        for key, value in super().to_state().items():
            state[key] = value
            if key == "norm_Z":
                state["last_failure"] = None if self.last_failure is None else float(self.last_failure)
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RightCensoredHALFit":
        """Reload a fit from :meth:`to_state` without recomputing ``Z``."""
        return cls(**cls._state_fields(state), last_failure=state.get("last_failure"))

    def beyond_last_failure(self, t: np.ndarray | float) -> np.ndarray:
        """Return a boolean mask for queries strictly after the last failure."""
        points = _as_1d(t)
        if self.last_failure is None:
            return np.zeros(points.shape, dtype=bool)
        return points > float(self.last_failure)
