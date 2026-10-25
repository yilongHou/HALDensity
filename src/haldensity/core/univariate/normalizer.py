"""Accurate log normalizer of a univariate HAL density, for evaluation, reporting and CV scoring.

A HAL log-density on [0, 1] is ``f(x) = theta' phi(x)`` with the basis of
:func:`~haldensity.core.basis.univariate.create_basis_functions`: an intercept,
``x, ..., x^k`` and one term per knot (``I(x >= xi)`` for k = 0,
``(x - xi)_+^k`` for k >= 1). Between consecutive knots with a nonzero
coefficient, ``f`` is a polynomial of degree k, so

    log Z = log integral_0^1 exp(f(x)) dx

is computed cell by cell with breakpoints at those knots: Gauss-Legendre with
32 nodes per cell, doubled until log Z changes by less than 1e-12. For k = 0,
``f`` is constant on each cell and the sum ``sum_c width_c exp(level_c)`` is
exact.

The fitting objectives keep their own (midpoint-rule) normalizers; this module
only evaluates a fitted density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

DEFAULT_ORDER = 32
DEFAULT_TOL = 1e-12
DEFAULT_MAX_ORDER = 1024
SUPPORT = (0.0, 1.0)


@dataclass(frozen=True)
class LogNormalizer:
    """Result of :func:`log_normalizer`.

    Attributes
    ----------
    log_z : float
        log of the integral of ``exp(f)`` over the support.
    order : int
        Gauss-Legendre nodes per cell of the reported value (0 for k = 0, exact).
    change : float
        |log Z at ``order`` - log Z at ``order / 2``| (0 for k = 0).
    converged : bool
        Whether ``change <= tol`` before ``max_order`` was reached.
    n_cells : int
        Number of cells between consecutive breakpoints.
    """

    log_z: float
    order: int
    change: float
    converged: bool
    n_cells: int


def _active_knots(theta: np.ndarray, knots: np.ndarray, basis_order: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(basis_order)
    first = 1 if k == 0 else k + 1
    coef = theta[first:]
    if coef.size != knots.size:
        raise ValueError("Theta length does not match the knots and basis order")
    keep = coef != 0.0
    return knots[keep], coef[keep]


def log_unnormalized_density(
    points: np.ndarray,
    theta: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    chunk: int = 4096,
) -> np.ndarray:
    """``f(x) = theta' phi(x)`` in float64, without building the basis matrix."""
    x = np.asarray(points, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    k = int(basis_order)
    active, coef = _active_knots(theta, np.asarray(knots, dtype=float).ravel(), k)
    value = np.full(x.shape, theta[0])
    for r in range(1, k + 1):
        value += theta[r] * x ** r
    if active.size:
        for lo in range(0, x.size, chunk):
            part = x[lo:lo + chunk, None]
            if k == 0:
                terms = (part >= active[None, :]).astype(float)
            else:
                terms = np.maximum(part - active[None, :], 0.0) ** k
            value[lo:lo + chunk] += terms @ coef
    return value


def _breakpoints(active: np.ndarray, support: tuple[float, float]) -> np.ndarray:
    lo, hi = float(support[0]), float(support[1])
    inside = active[(active > lo) & (active < hi)]
    return np.unique(np.concatenate(([lo], inside, [hi])))


def _log_z_gauss_legendre(theta, knots, basis_order, edges, order) -> float:
    nodes, weights = leggauss(int(order))
    a, b = edges[:-1, None], edges[1:, None]
    points = (a + b) / 2 + (b - a) / 2 * nodes
    log_w = np.log((b - a) / 2 * weights)
    values = log_unnormalized_density(points.ravel(), theta, knots, basis_order).reshape(points.shape)
    return float(logsumexp(values + log_w))


def log_normalizer(
    theta: np.ndarray,
    knots: np.ndarray,
    basis_order: int,
    support: tuple[float, float] = SUPPORT,
    *,
    order: int = DEFAULT_ORDER,
    tol: float = DEFAULT_TOL,
    max_order: int = DEFAULT_MAX_ORDER,
) -> LogNormalizer:
    """Accurate ``log integral exp(theta' phi(x)) dx`` over ``support``; see the module docstring."""
    theta = np.asarray(theta, dtype=float).ravel()
    knots = np.asarray(knots, dtype=float).ravel()
    active, _ = _active_knots(theta, knots, basis_order)
    edges = _breakpoints(active, support)
    n_cells = int(edges.size - 1)
    if int(basis_order) == 0:
        mids = (edges[:-1] + edges[1:]) / 2
        levels = log_unnormalized_density(mids, theta, knots, 0)
        log_z = float(logsumexp(levels + np.log(np.diff(edges))))
        return LogNormalizer(log_z=log_z, order=0, change=0.0, converged=True, n_cells=n_cells)
    q = int(order)
    previous = _log_z_gauss_legendre(theta, knots, basis_order, edges, q)
    while True:
        refined = _log_z_gauss_legendre(theta, knots, basis_order, edges, 2 * q)
        change = abs(refined - previous)
        if change <= tol or 2 * q >= max_order:
            break
        q, previous = 2 * q, refined
    return LogNormalizer(log_z=refined, order=2 * q, change=float(change), converged=bool(change <= tol),
                         n_cells=n_cells)
