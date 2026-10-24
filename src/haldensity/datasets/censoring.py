"""Simple simulators for censored data, for tests and tutorials.

Right-censored data: :func:`make_right_censored` draws event and censoring
times and returns the observed ``(T, Delta)``.

Interval-censored data: lightweight utilities that turn latent event times T
into ``(L, R)`` intervals under non-informative inspection schemes (current
status is the one-inspection case).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


IntervalMechanism = Literal["inspection_uniform", "current_status_uniform"]


def _as_rng(random_state: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(random_state)


def interval_censor_inspection_uniform(
    event_times: np.ndarray,
    *,
    n_inspections: int = 8,
    t_min: float = 0.0,
    t_max: float = 1.0,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inspection-time bracketing with inspection times independent of T.

    For each i, draw `n_inspections` inspection times uniformly on [t_min, t_max],
    sort them, add boundaries (t_min, t_max), and return the bracketing interval
    [L_i, R_i] such that L_i < T_i <= R_i (ties go to the right bin).
    """
    if n_inspections <= 0:
        raise ValueError("n_inspections must be positive")
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")

    T = np.asarray(event_times, dtype=float)
    n = int(T.shape[0])
    g = _as_rng(random_state, rng)

    inspections = g.uniform(t_min, t_max, size=(n, n_inspections))
    inspections.sort(axis=1)

    C = np.concatenate(
        [
            np.full((n, 1), t_min, dtype=float),
            inspections,
            np.full((n, 1), t_max, dtype=float),
        ],
        axis=1,
    )

    # Bracket index j with C_j < T <= C_{j+1}. Use <= with side="right" behavior.
    idx = np.sum(C <= T[:, None], axis=1) - 1
    idx = np.clip(idx, 0, C.shape[1] - 2)

    L = C[np.arange(n), idx]
    R = C[np.arange(n), idx + 1]
    return pd.DataFrame({"L": L, "R": R})


def interval_censor_current_status_uniform(
    event_times: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Current-status interval censoring with independent uniform inspection time.

    Draw one inspection time C ⟂ T uniformly on [t_min, t_max]. Observe:
    - [t_min, C] if T <= C
    - [C, t_max] otherwise
    """
    if not (t_min < t_max):
        raise ValueError("require t_min < t_max")

    T = np.asarray(event_times, dtype=float)
    n = int(T.shape[0])
    g = _as_rng(random_state, rng)

    C = g.uniform(t_min, t_max, size=n)
    is_event = T <= C
    L = np.where(is_event, t_min, C)
    R = np.where(is_event, C, t_max)
    return pd.DataFrame({"L": L, "R": R})



def interval_censor(
    event_times: np.ndarray,
    *,
    mechanism: IntervalMechanism = "inspection_uniform",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch helper for interval censoring mechanisms."""
    if mechanism == "inspection_uniform":
        return interval_censor_inspection_uniform(event_times, **kwargs)
    if mechanism == "current_status_uniform":
        return interval_censor_current_status_uniform(event_times, **kwargs)
    raise ValueError(f"Unsupported interval censoring mechanism: {mechanism}")


def make_right_censored(
    n: int,
    *,
    seed: int = 0,
    event_beta_shape: tuple[float, float] = (2.2, 2.5),
    censor_low: float = 0.0,
    censor_high: float = 1.2,
    event_clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Simulate right-censored data with Beta event times and uniform censoring.

    Event times are Beta(a, b) on ``[0, 1]`` and censoring times are
    Unif[censor_low, censor_high]. The default Unif[0, 1.2] keeps the censoring
    survival positive on ``[0, 1]``. ``event_clip=(lo, hi)`` clips the event
    times before censoring (the package tests use ``(1e-3, 0.999)``).

    Returns
    -------
    pd.DataFrame
        Columns ``T`` (observed time, min of event and censoring) and ``Delta``
        (1 if the event is observed).
    """
    rng = np.random.default_rng(seed)
    a, b = event_beta_shape
    event_times = rng.beta(a, b, size=int(n))
    if event_clip is not None:
        event_times = np.clip(event_times, event_clip[0], event_clip[1])
    censor_times = rng.uniform(float(censor_low), float(censor_high), size=int(n))
    observed = np.minimum(event_times, censor_times)
    delta = (event_times <= censor_times).astype(int)
    return pd.DataFrame({"T": observed, "Delta": delta})
