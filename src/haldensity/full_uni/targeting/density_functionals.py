"""Full-data targets whose influence curve depends on the density itself: entropy and density L2 (Yilong,
2026-10-25).

For psi(p) = -integral p log p the influence curve is -log p(x) - psi; for psi(p) = integral p^2 it is
2 (p(x) - psi). A step's targeting basis is the influence curve's x-part at the density the step starts from
(-log p or 2 p), so the step moves along the least favourable path through that density, solved as in every
full-data target: the exact rule, the solve within the L1 bound and the Newton polish of the score equation
P_n b = E_alpha b (:class:`~haldensity.full_uni.targeting.base.BaseTargetLearner`).

- Entropy: the path is p^(1 - alpha) / Z, and the influence curve there, -log p_alpha - psi, is affine in log p, so
  the step's score equation is P_n D*(p_alpha) = 0: one step solves it.
- Density L2: after one step P_n D* is of second order, not 0. ``max_steps`` > 1 repeats the step from each
  targeted density until |P_n D*| <= ``score_tol``.

``plug_in`` is the target under the initial fit and ``estimate`` under the last targeted density. The results add
the coefficient of every step (``theta_targeting_path``), log Z of the initial and of each targeted density
(``log_normalizer_path``) and P_n D* after each step (``eic_mean_path``), from which :meth:`eic_values` rebuilds the
last targeted density at any points (the SE, or the influence curve on held-out data).
"""
from types import SimpleNamespace
import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from haldensity.core.defaults import DEFAULT_BASIS_ORDER
from haldensity.core.crossfit import FoldFits, resolve_cross_fit
from haldensity.full_uni.targeting.base import NO_CROSS_FIT_WARNING, BaseTargetLearner
from haldensity.full_uni.targeting.exact import ExactDensity, cell_edges, converged_rule, log_integral

# stop repeating the step when |P_n D*| is at most this
EIC_SCORE_TOL = 1e-10
# point sets each step's log-density remembers (the data, the rule's nodes at two orders, the plotting grid)
CACHED_POINT_SETS = 8


class _Cached:
    """A step's log-density, evaluated once per point set. A step's targeted log-density calls the start's twice
    (itself and the targeting basis), so without this the cost of evaluating it doubles with every step."""

    def __init__(self, log_density: Callable[[np.ndarray], np.ndarray]):
        self.log_density = log_density
        self._points: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        for i, points in enumerate(self._points):
            if points.shape == x.shape and np.array_equal(points, x):
                return self._values[i]
        value = self.log_density(x)
        self._points, self._values = ([x.copy()] + self._points)[:CACHED_POINT_SETS], \
            ([value] + self._values)[:CACHED_POINT_SETS]
        return value


class FitDependentTargetLearner(BaseTargetLearner):
    """Shared by :class:`EntropyTargetLearner` and :class:`DensityL2TargetLearner`; subclasses give the influence
    curve (:meth:`influence_curve`) and the target (:meth:`functional`)."""

    def __init__(
        self,
        norm_constraint: float = 20,
        basis_order: int = DEFAULT_BASIS_ORDER,
        max_steps: int = 1,
        score_tol: float = EIC_SCORE_TOL,
        solver: str = "MOSEK",
        use_secondary_solver: bool = True,
        solver_waterfall: Sequence[str] = ("CLARABEL", "ECOS", "SCS"),
    ):
        """``max_steps``: targeting steps at most (1, the default, is the single-step TMLE); each repeat starts
        from the last targeted density, until |P_n D*| <= ``score_tol``. The rest as in
        :class:`~haldensity.full_uni.targeting.base.BaseTargetLearner`."""
        super().__init__(norm_constraint=norm_constraint, basis_order=basis_order, solver=solver,
                         use_secondary_solver=use_secondary_solver, solver_waterfall=solver_waterfall)
        if int(max_steps) < 1:
            raise ValueError("max_steps must be at least 1")
        self.max_steps = int(max_steps)
        self.score_tol = float(score_tol)

    def influence_curve(self, log_density: np.ndarray, psi: float) -> np.ndarray:
        """D*(x) from the normalized log-density at points x and the target psi; with psi = 0, its x-part, the
        targeting basis."""
        raise NotImplementedError

    def functional(self, density: ExactDensity) -> float:
        """The target under ``density``, exactly."""
        raise NotImplementedError

    def target_values(self, density: ExactDensity, **kwargs) -> tuple[np.ndarray, None]:
        return np.array([self.functional(density)]), None

    def _basis(self, x: np.ndarray, kwargs: dict) -> np.ndarray:
        """The influence curve's x-part at the start of the step (kwargs['_start'] = (log-density up to its
        constant, its log Z)); shape (n, 1)."""
        log_density, log_z = kwargs["_start"]
        return self.influence_curve(log_density(np.asarray(x, dtype=float)) - log_z, 0.0)[:, None]

    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._basis(uncensored_data["W1"].to_numpy(dtype=float), kwargs)

    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._basis(data_grid["W1"].to_numpy(dtype=float), kwargs)

    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._basis(data_grid_full["W1"].to_numpy(dtype=float), kwargs)

    def _solve(self, fixed_data, b_ik_targeting, fixed_nodes, b_nodes, weights, n_samples):
        """As the base class's, except that a constant influence curve (a uniform start density) leaves nothing to
        target: the coefficient is 0, not wherever the solver stops along a direction that changes nothing."""
        if np.ptp(b_nodes) == 0 and np.ptp(b_ik_targeting) == 0:
            return np.zeros(b_ik_targeting.shape[1]), SimpleNamespace(solver_used="none",
                                                                      status="constant influence curve")
        return super()._solve(fixed_data, b_ik_targeting, fixed_nodes, b_nodes, weights, n_samples)

    def run_m_step(
        self,
        uncensored_augmented: pd.DataFrame,
        grid_points_hal_selected: Union[list, np.ndarray],
        old_theta: Union[list, np.ndarray],
        *,
        cross_fit: Union[str, bool] = "tuner",
        fit: Any = None,
        folds: Optional[FoldFits] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """The targeting step(s) from the initial fit; the results of
        :meth:`BaseTargetLearner.run_m_step` (for the last step) plus ``theta_targeting_path``,
        ``log_normalizer_path`` and ``eic_mean_path`` (see the module docstring). ``cross_fit``, ``fit`` and
        ``folds`` as in :meth:`BaseTargetLearner.run_m_step`; cross-fitted, every step is pooled over the
        folds (one alpha), each fold restarting from its own targeted density, until the pooled
        |P_n D*| <= ``score_tol``."""
        old_theta, hal_knots = self._check_inputs(grid_points_hal_selected, old_theta)
        kwargs = {**kwargs, "_old_theta": old_theta, "_grid_points_hal_selected": hal_knots}
        cross_fit = resolve_cross_fit(cross_fit)
        if cross_fit != "none":
            fold_fits = self._fold_fits_for(uncensored_augmented, fit, folds, cross_fit)
            if fold_fits is not None:
                return self._run_cross_fitted(uncensored_augmented.copy(), old_theta, hal_knots,
                                              grid_points_hal_selected, fold_fits, kwargs)
            warnings.warn(NO_CROSS_FIT_WARNING, UserWarning, stacklevel=2)
        fixed = _Cached(self._hal_log_density(old_theta, hal_knots))
        data = uncensored_augmented.copy()
        x = data["W1"].to_numpy(dtype=float)
        start = (fixed, log_integral(fixed, converged_rule(fixed, cell_edges(hal_knots))))
        alphas, log_zs, eic_means = [], [start[1]], []
        for _ in range(self.max_steps):
            step_kwargs = {**kwargs, "_start": start}
            step = self._targeting_step(start[0], hal_knots, data, step_kwargs)
            start = (_Cached(step.targeted), step.density.log_z)
            alphas.append(float(step.alpha[0]))
            log_zs.append(step.density.log_z)
            eic_means.append(float(np.mean(self.influence_curve(start[0](x) - step.density.log_z,
                                                                self.functional(step.density)))))
            if abs(eic_means[-1]) <= self.score_tol:
                break
        results = self._results(step, fixed, old_theta, grid_points_hal_selected, hal_knots, step_kwargs)
        results.update(theta_targeting_path=np.array(alphas), log_normalizer_path=np.array(log_zs),
                       eic_mean_path=np.array(eic_means), cross_fit="none")
        return results

    def _run_cross_fitted(self, data: pd.DataFrame, old_theta: np.ndarray, hal_knots: np.ndarray,
                          grid_points_hal_selected, fold_fits: FoldFits, kwargs: dict) -> dict[str, Any]:
        """The steps pooled over the folds: each fold starts from its own fit, then from its own targeted
        density; one alpha per step; stop when the pooled P_n D* is within score_tol."""
        folds = self._fold_inputs(data, fold_fits, kwargs)
        n = float(sum(fold[2].shape[0] for fold in folds))
        starts = []
        for fixed_v, knots_v, _, _, _ in folds:
            cached = _Cached(fixed_v)
            starts.append((cached, log_integral(cached, converged_rule(cached, cell_edges(knots_v)))))
        alphas, log_zs, eic_means = [], [[start[1] for start in starts]], []
        for _ in range(self.max_steps):
            step_folds = [(start[0], knots_v, data_v, {**kwargs_v, "_start": start})
                          for start, (_, knots_v, data_v, kwargs_v, _) in zip(starts, folds)]
            steps = self._pooled_steps(step_folds)
            last_kwargs = [fold[3] for fold in step_folds]
            starts = [(_Cached(step.targeted), step.density.log_z) for step in steps]
            alphas.append(float(steps[0].alpha[0]))
            log_zs.append([step.density.log_z for step in steps])
            eic_means.append(float(sum(
                (data_v.shape[0] / n) * np.mean(self.influence_curve(start[0](data_v["W1"].to_numpy(dtype=float))
                                                                      - step.density.log_z,
                                                                      self.functional(step.density)))
                for start, step, (_, _, data_v, _, _) in zip(starts, steps, folds))))
            if abs(eic_means[-1]) <= self.score_tol:
                break
        final_folds = [(fixed_v, knots_v, data_v, kwargs_last, rows)
                       for (fixed_v, knots_v, data_v, _, rows), kwargs_last in zip(folds, last_kwargs)]
        return self._results_cross_fitted(
            steps, final_folds, fold_fits, data, old_theta, hal_knots, grid_points_hal_selected, kwargs,
            extra={"theta_targeting_path": np.array(alphas), "log_normalizer_path": None,
                   "fold_log_normalizer_paths": [list(path) for path in zip(*log_zs)],
                   "eic_mean_path": np.array(eic_means)})

    def influence_curve_at(self, density: ExactDensity, x: np.ndarray, **kwargs) -> np.ndarray:
        log_density = density.log_density(np.asarray(x, dtype=float)) - density.log_z
        return self.influence_curve(log_density, self.functional(density))[:, None]

    def last_log_density(self, targeted_fit: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
        """The last targeted density's log-density (normalized), rebuilt from ``targeted_fit``'s path."""
        old_theta, knots = self._check_inputs(targeted_fit["grid_points_hal_selected"], targeted_fit["old_theta"])
        log_density = self._hal_log_density(old_theta, knots)
        log_zs = np.asarray(targeted_fit["log_normalizer_path"], dtype=float)
        for alpha, log_z in zip(np.asarray(targeted_fit["theta_targeting_path"], dtype=float), log_zs[:-1]):
            def step(t: np.ndarray, previous=log_density, a=float(alpha), z=float(log_z)) -> np.ndarray:
                start = previous(t)
                return start + a * self.influence_curve(start - z, 0.0)
            log_density = step
        return (lambda unnormalized, z: lambda t: unnormalized(np.asarray(t, dtype=float)) - z)(log_density,
                                                                                               float(log_zs[-1]))

    def eic_values(self, targeted_fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
        """D*(x) under the last targeted density, at its estimate."""
        if targeted_fit.get("cross_fit", "none") != "none":
            values = np.asarray(targeted_fit["eic_values"], dtype=float)[:, 0]
            if len(np.atleast_1d(x)) != values.size:
                raise ValueError("a cross-fitted fit has influence-curve values at its own data only")
            return values
        estimate = float(self._exact_estimate(targeted_fit)[0])
        return self.influence_curve(self.last_log_density(targeted_fit)(x), estimate)

    def get_estimand_variance(self, targeted_fit: dict[str, Any], uncensored_data: pd.DataFrame,
                              **kwargs) -> np.ndarray:
        """Var(D*) / n, D* the influence curve under the last targeted density at the data (one-element array)."""
        cross_fitted = self._cross_fitted_variance(targeted_fit, uncensored_data)
        if cross_fitted is not None:
            return cross_fitted
        eic = self.eic_values(targeted_fit, uncensored_data["W1"].to_numpy(dtype=float))
        return np.array([np.var(eic, ddof=1) / eic.size])


class EntropyTargetLearner(FitDependentTargetLearner):
    """TMLE of the entropy psi = -integral p log p; influence curve -log p(x) - psi. One step solves
    P_n D* = 0 (the path p^(1 - alpha) / Z), so ``max_steps`` > 1 changes nothing."""

    def influence_curve(self, log_density: np.ndarray, psi: float) -> np.ndarray:
        return -np.asarray(log_density, dtype=float) - psi

    def functional(self, density: ExactDensity) -> float:
        return density.entropy()


class DensityL2TargetLearner(FitDependentTargetLearner):
    """TMLE of psi = integral p^2; influence curve 2 (p(x) - psi). One step leaves P_n D* of second order;
    ``max_steps`` > 1 repeats it."""

    def influence_curve(self, log_density: np.ndarray, psi: float) -> np.ndarray:
        return 2.0 * (np.exp(np.asarray(log_density, dtype=float)) - psi)

    def functional(self, density: ExactDensity) -> float:
        return density.density_l2()


def entropy_targeting_M_step(
    uncensored_augmented: pd.DataFrame,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: float = 20,
    basis_order: int = DEFAULT_BASIS_ORDER,
    max_steps: int = 1,
    cross_fit: Union[str, bool] = "tuner",
    fit: Any = None,
    folds: Any = None,
) -> dict:
    """:class:`EntropyTargetLearner`'s M-step in one call."""
    learner = EntropyTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order, max_steps=max_steps)
    return learner.run_m_step(uncensored_augmented=uncensored_augmented,
                              grid_points_hal_selected=grid_points_hal_selected, old_theta=old_theta,
                              cross_fit=cross_fit, fit=fit, folds=folds)


def entropy_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    norm_constraint: float = 20,
    basis_order: int = DEFAULT_BASIS_ORDER,
) -> np.ndarray:
    """Var(D*) / n for the entropy TMLE, D* = -log p(x) - psi under the targeted density."""
    learner = EntropyTargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(targeted_fit=targeted_fit, uncensored_data=uncensored_data)


def density_l2_targeting_M_step(
    uncensored_augmented: pd.DataFrame,
    grid_points_hal_selected: Union[list, np.ndarray],
    old_theta: Union[list, np.ndarray],
    norm_constraint: float = 20,
    basis_order: int = DEFAULT_BASIS_ORDER,
    max_steps: int = 1,
    cross_fit: Union[str, bool] = "tuner",
    fit: Any = None,
    folds: Any = None,
) -> dict:
    """:class:`DensityL2TargetLearner`'s M-step in one call."""
    learner = DensityL2TargetLearner(norm_constraint=norm_constraint, basis_order=basis_order, max_steps=max_steps)
    return learner.run_m_step(uncensored_augmented=uncensored_augmented,
                              grid_points_hal_selected=grid_points_hal_selected, old_theta=old_theta,
                              cross_fit=cross_fit, fit=fit, folds=folds)


def density_l2_estimand_variance(
    targeted_fit: dict[str, Any],
    uncensored_data: pd.DataFrame,
    norm_constraint: float = 20,
    basis_order: int = DEFAULT_BASIS_ORDER,
) -> np.ndarray:
    """Var(D*) / n for the density-L2 TMLE, D* = 2 (p(x) - psi) under the targeted density."""
    learner = DensityL2TargetLearner(norm_constraint=norm_constraint, basis_order=basis_order)
    return learner.get_estimand_variance(targeted_fit=targeted_fit, uncensored_data=uncensored_data)
