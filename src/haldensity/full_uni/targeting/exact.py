"""Exact integration for full-data targeting (Yilong, 2026-10-24).

The targeting step integrates over cells whose edges are the HAL knots and the points where the targeting basis
jumps (the targeting points, or the median). Inside a cell the log-density is a polynomial plus a smooth or
constant targeting term, so Gauss-Legendre per cell converges fast; the number of nodes per cell is doubled until
log Z changes by less than 1e-12, as the accurate normalizer does (for the targeted density, after the M-step's
solve, until it changes by less than 1e-10: ``base.TARGETED_RULE_TOL``)
(:mod:`haldensity.core.univariate.normalizer`). The same rule gives the functionals of a fitted density exactly:
cumulative probabilities (whole cells, plus Gauss-Legendre on the part of a cell), moments and quantiles. Before
0.2.0 the step summed over a 5e-4 grid, so the score equations held at the nearest grid cell instead of the
targeting point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import logsumexp

from haldensity.core.univariate.normalizer import DEFAULT_MAX_ORDER, DEFAULT_ORDER, DEFAULT_TOL, SUPPORT

LogDensity = Callable[[np.ndarray], np.ndarray]


def cell_edges(breakpoints: np.ndarray, support: tuple[float, float] = SUPPORT) -> np.ndarray:
    """The support's end points and the breakpoints inside it, sorted and unique."""
    lo, hi = float(support[0]), float(support[1])
    points = np.asarray(breakpoints, dtype=float).ravel()
    return np.unique(np.concatenate(([lo], points[(points > lo) & (points < hi)], [hi])))


@dataclass(frozen=True)
class ExactRule:
    """Gauss-Legendre points and weights, ``order`` nodes in each cell between consecutive ``edges``."""

    edges: np.ndarray
    order: int
    points: np.ndarray
    weights: np.ndarray

    @classmethod
    def build(cls, edges: np.ndarray, order: int) -> "ExactRule":
        nodes, weights = leggauss(int(order))
        a, b = edges[:-1, None], edges[1:, None]
        return cls(np.asarray(edges, dtype=float), int(order), ((a + b) / 2 + (b - a) / 2 * nodes).ravel(),
                   ((b - a) / 2 * weights).ravel())

    def finer(self) -> "ExactRule":
        return ExactRule.build(self.edges, 2 * self.order)


def log_integral(log_density: LogDensity, rule: ExactRule) -> float:
    """log of the integral of exp(log_density) by the rule."""
    return float(logsumexp(log_density(rule.points) + np.log(rule.weights)))


def converged_rule(log_density: LogDensity, edges: np.ndarray, order: int = DEFAULT_ORDER, tol: float = DEFAULT_TOL,
                   max_order: int = DEFAULT_MAX_ORDER) -> ExactRule:
    """The rule whose log integral of exp(log_density) changes by at most ``tol`` when the nodes double."""
    rule = ExactRule.build(edges, order)
    value = log_integral(log_density, rule)
    while 2 * rule.order <= max_order:
        finer = rule.finer()
        finer_value = log_integral(log_density, finer)
        if abs(finer_value - value) <= tol:
            return finer
        rule, value = finer, finer_value
    return rule


class ExactDensity:
    """``p(x) = exp(log_density(x) - log Z)`` on the rule's cells, with exact functionals."""

    def __init__(self, log_density: LogDensity, rule: ExactRule):
        self.log_density = log_density
        self.rule = rule
        self.log_z = log_integral(log_density, rule)
        self._mass = rule.weights * np.exp(log_density(rule.points) - self.log_z)
        self._cumulative = np.concatenate(([0.0], np.cumsum(self._mass.reshape(-1, rule.order).sum(axis=1))))

    def density(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_density(np.atleast_1d(np.asarray(x, dtype=float))) - self.log_z)

    def cdf(self, t: np.ndarray) -> np.ndarray:
        """P(X <= t): the whole cells left of t plus Gauss-Legendre on the part of the cell that holds t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        edges = self.rule.edges
        nodes, weights = leggauss(self.rule.order)
        out = np.empty(t.shape)
        for i, value in enumerate(t):
            if value <= edges[0] or value >= edges[-1]:
                out[i] = 0.0 if value <= edges[0] else 1.0
                continue
            cell = int(np.searchsorted(edges, value, side="right")) - 1
            a = edges[cell]
            partial = 0.0
            if value > a:
                x = (a + value) / 2 + (value - a) / 2 * nodes
                partial = float(np.sum((value - a) / 2 * weights * self.density(x)))
            out[i] = self._cumulative[cell] + partial
        return out

    def survival(self, t: np.ndarray) -> np.ndarray:
        return 1.0 - self.cdf(t)

    def moment(self, power: int) -> float:
        return float(self._mass @ self.rule.points ** power)

    def quantile(self, probability: float = 0.5) -> float:
        cell = int(np.clip(np.searchsorted(self._cumulative, probability, side="left") - 1, 0, self.rule.edges.size - 2))
        a, b = self.rule.edges[cell], self.rule.edges[cell + 1]
        return float(brentq(lambda m: self.cdf(m)[0] - probability, a, b, xtol=1e-15, rtol=4 * np.finfo(float).eps))
