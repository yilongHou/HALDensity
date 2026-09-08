"""Lightweight fitted right-censored HAL distribution.

The object stores coefficients, knots, and the frozen normalization constants
from ``fit()``. Queries do not recompute ``Z``. A targeting update should build
a new instance rather than mutating this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from haldensity.utils.basis import create_basis_functions

SCHEMA_VERSION = 1
SUPPORT = (0.0, 1.0)


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


def _as_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


@dataclass
class RightCensoredHALFit:
    """Fitted right-censored HAL density with frozen ``(m, Z)``."""

    theta: np.ndarray
    knots: np.ndarray
    selected_knots: np.ndarray
    basis_order: int
    norm_shift: float
    norm_Z: float
    last_failure: Optional[float]
    normalizer_method: str
    n_cells: int
    support: tuple[float, float] = SUPPORT
    min_knot_spacing: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.theta = _as_1d(self.theta).copy()
        self.knots = _as_1d(self.knots).copy()
        self.selected_knots = _as_1d(self.selected_knots).copy()
        self.basis_order = int(self.basis_order)
        self.norm_shift = float(self.norm_shift)
        self.norm_Z = float(self.norm_Z)
        if self.norm_Z <= 0.0:
            raise ValueError("norm_Z must be positive")
        self.n_cells = int(self.n_cells)
        if self.n_cells < 1:
            raise ValueError("n_cells must be at least 1")
        self.support = (float(self.support[0]), float(self.support[1]))
        self.schema_version = int(self.schema_version)

    @classmethod
    def from_estimator(
        cls,
        estimator: Any,
        *,
        last_failure: Optional[float] = None,
    ) -> "RightCensoredHALFit":
        """Build a fit from a live RC estimator without recomputing ``Z``."""
        if not bool(getattr(estimator, "is_fitted", False)):
            raise RuntimeError("Estimator must be fitted before get_fit()")
        if getattr(estimator, "theta_hat", None) is None:
            raise RuntimeError("Estimator is missing theta_hat")
        if getattr(estimator, "_norm_shift", None) is None or getattr(estimator, "_norm_Z", None) is None:
            raise RuntimeError("Estimator is missing frozen normalization constants")
        knots = getattr(estimator, "_grid_points_hal", None)
        if knots is None:
            raise RuntimeError("Estimator is missing working knots")

        grid = getattr(estimator, "grid_points", None)
        if grid is not None and np.asarray(grid).size >= 2:
            n_cells = int(np.asarray(grid).size - 1)
        else:
            n_edges = int(getattr(estimator, "n_grid_points", 200))
            n_cells = max(n_edges - 1, 1)

        selected = getattr(estimator, "grid_points_hal_selected", None)
        if selected is None:
            selected = np.array([], dtype=float)

        recorded = last_failure
        if recorded is None:
            recorded = getattr(estimator, "_last_failure", None)

        return cls(
            theta=estimator.theta_hat,
            knots=knots,
            selected_knots=selected,
            basis_order=int(estimator.basis_order),
            norm_shift=float(estimator._norm_shift),
            norm_Z=float(estimator._norm_Z),
            last_failure=None if recorded is None else float(recorded),
            normalizer_method="midpoint",
            n_cells=n_cells,
        )

    def to_state(self) -> dict[str, Any]:
        """Return a JSON/npz-friendly state dictionary."""
        return {
            "schema_version": int(self.schema_version),
            "theta": self.theta.copy(),
            "knots": self.knots.copy(),
            "selected_knots": self.selected_knots.copy(),
            "basis_order": int(self.basis_order),
            "norm_shift": float(self.norm_shift),
            "norm_Z": float(self.norm_Z),
            "last_failure": None if self.last_failure is None else float(self.last_failure),
            "normalizer_method": str(self.normalizer_method),
            "n_cells": int(self.n_cells),
            "support": [float(self.support[0]), float(self.support[1])],
            "min_knot_spacing": self.min_knot_spacing,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RightCensoredHALFit":
        """Reload a fit from :meth:`to_state` without recomputing ``Z``."""
        support = state.get("support", SUPPORT)
        return cls(
            theta=state["theta"],
            knots=state["knots"],
            selected_knots=state.get("selected_knots", np.array([], dtype=float)),
            basis_order=int(state["basis_order"]),
            norm_shift=float(state["norm_shift"]),
            norm_Z=float(state["norm_Z"]),
            last_failure=state.get("last_failure"),
            normalizer_method=str(state.get("normalizer_method", "midpoint")),
            n_cells=int(state["n_cells"]),
            support=(float(support[0]), float(support[1])),
            min_knot_spacing=state.get("min_knot_spacing"),
            schema_version=int(state.get("schema_version", SCHEMA_VERSION)),
        )

    def density(self, t: np.ndarray | float) -> np.ndarray:
        """Evaluate the frozen fitted density at ``t``."""
        points = _as_1d(t)
        if points.size == 0:
            return np.array([], dtype=float)
        df_pts = pd.DataFrame({"W1": points})
        basis_eval, _ = create_basis_functions(
            df_pts,
            self.knots,
            order=self.basis_order,
            include_intercept=True,
        )
        if basis_eval.shape[1] != self.theta.size:
            raise ValueError("Theta length does not match the stored knot basis")
        log_eval = basis_eval @ self.theta
        shifted = np.clip(log_eval - self.norm_shift, -700.0, 700.0)
        return np.exp(shifted) / self.norm_Z

    def log_density(self, t: np.ndarray | float) -> np.ndarray:
        values = self.density(t)
        return np.log(np.maximum(values, np.finfo(float).tiny))

    def beyond_last_failure(self, t: np.ndarray | float) -> np.ndarray:
        """Return a boolean mask for queries strictly after the last failure."""
        points = _as_1d(t)
        if self.last_failure is None:
            return np.zeros(points.shape, dtype=bool)
        return points > float(self.last_failure)

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        """Survival of the frozen density. Does not recompute ``Z``."""
        points = _as_1d(t)
        if self.basis_order == 0:
            return self._survival_k0(points)
        return self._survival_midpoint(points)

    def rmst(self, tau: np.ndarray | float) -> np.ndarray:
        """Restricted mean of the frozen density. Does not recompute ``Z``."""
        horizons = _as_1d(tau)
        if self.basis_order == 0:
            return self._rmst_k0(horizons)
        return self._rmst_midpoint(horizons)

    def _knot_partition(self) -> np.ndarray:
        lo, hi = self.support
        return np.unique(
            np.concatenate(([lo], self.knots, [hi])).astype(float)
        )

    def _piece_density(self) -> tuple[np.ndarray, np.ndarray]:
        knots = self._knot_partition()
        midpoints = 0.5 * (knots[:-1] + knots[1:])
        return knots, self.density(midpoints)

    def _survival_k0(self, t: np.ndarray) -> np.ndarray:
        knots, density = self._piece_density()
        left = knots[:-1]
        right = knots[1:]
        remain = np.maximum(0.0, right - np.maximum(t[:, None], left))
        return np.sum(remain * density, axis=1)

    def _rmst_k0(self, tau: np.ndarray) -> np.ndarray:
        knots, density = self._piece_density()
        left = knots[:-1]
        right = knots[1:]
        widths = right - left
        masses = density * widths
        tail = np.zeros(masses.size + 1, dtype=float)
        tail[:-1] = np.cumsum(masses[::-1])[::-1]
        out = np.zeros(tau.size, dtype=float)
        for i, horizon in enumerate(tau):
            total = 0.0
            for j, (a, b, p_j, h_j) in enumerate(zip(left, right, density, widths)):
                tail_right = tail[j + 1]
                if horizon <= a:
                    continue
                if horizon >= b:
                    total += p_j * (h_j * h_j) / 2.0 + tail_right * h_j
                else:
                    delta = horizon - a
                    total += p_j * (h_j * delta - 0.5 * delta * delta) + tail_right * delta
            out[i] = total
        return out

    def _midpoint_bank(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.support
        edges = np.linspace(lo, hi, int(self.n_cells) + 1)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        return midpoints, widths, self.density(midpoints)

    def _survival_midpoint(self, t: np.ndarray) -> np.ndarray:
        midpoints, widths, density = self._midpoint_bank()
        masses = density * widths
        return np.sum((midpoints[None, :] > t[:, None]) * masses, axis=1)

    def _rmst_midpoint(self, tau: np.ndarray) -> np.ndarray:
        midpoints, widths, density = self._midpoint_bank()
        masses = density * widths
        return np.sum(np.minimum(midpoints[None, :], tau[:, None]) * masses, axis=1)
