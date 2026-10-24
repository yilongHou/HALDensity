"""Convert current-status observations into intervals for the interval-censored estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def current_status_to_interval(
    data: pd.DataFrame,
    *,
    inspection_col: str = "C",
    delta_col: str = "Delta",
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> pd.DataFrame:
    """Turn current-status data ``(C, Delta = 1{T <= C})`` into intervals ``(L, R)``.

    ``Delta = 1`` (event by the inspection) gives ``[t_min, C]``; ``Delta = 0``
    gives ``[C, t_max]``. These are the intervals produced by
    :func:`haldensity.datasets.interval_censor_current_status_uniform`.

    Returns
    -------
    pd.DataFrame
        Columns ``L`` and ``R``, ready for :mod:`haldensity.ic` estimators.
    """
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")
    for col in (inspection_col, delta_col):
        if col not in data.columns:
            raise ValueError(f"data must contain column {col!r}")
    inspection = np.asarray(data[inspection_col].values, dtype=float)
    delta = np.asarray(data[delta_col].values, dtype=int)
    if not np.all(np.isin(delta, (0, 1))):
        raise ValueError(f"{delta_col} must be 0 or 1")
    if np.any((inspection < t_min) | (inspection > t_max)):
        raise ValueError(f"{inspection_col} must lie in [{t_min}, {t_max}]")
    is_event = delta == 1
    return pd.DataFrame({
        "L": np.where(is_event, t_min, inspection),
        "R": np.where(is_event, inspection, t_max),
    })
