import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Union, Any, Callable, Optional, Sequence
from pydantic import BaseModel
from haldensity.core.crossfit import FoldFits, data_key, default_splits, resolve_cross_fit
from haldensity.core.defaults import DEFAULT_BASIS_ORDER
from haldensity.core.basis.univariate import create_basis_functions
from scipy.special import logsumexp
from haldensity.core.solvers.cvxpy import solve_cvxpy_problem
from haldensity.core.univariate.normalizer import DEFAULT_MAX_ORDER, log_unnormalized_density
from haldensity.full_uni.targeting.exact import ExactDensity, ExactRule, cell_edges, converged_rule, log_integral

# the targeted density's log Z may change by at most this much when the nodes per cell double; otherwise the
# M-step is solved again on the finer rule
TARGETED_RULE_TOL = 1e-10
# Newton polishing of the targeting coefficients (bound not binding): stop when every score equation
# P_n b = E_alpha b holds to SCORE_TOL; interior-point solvers stop near 1e-5 on this problem, and their
# tight settings rarely report "optimal" here although the solution is good (SCS then ran for minutes)
SCORE_TOL = 1e-12
MAX_POLISH_STEPS = 50
# above this largest score-equation error the estimate is not the exact TMLE, and run_m_step warns
SCORE_WARN = 1e-8


FALLBACK_WARNING = ("targeted_fit has no exact 'estimate' (made before 0.2.0?); using sums over its fine grid, "
                    "which are inaccurate where the density changes quickly. Rerun run_m_step.")
NO_CROSS_FIT_WARNING = ("No fitted estimator to cross-fit with (the M-step got old_theta and the knots only), so the "
                        "full-sample targeting step ran (cross_fit='none'). Pass **targeting_inputs(estimator) to "
                        "cross-fit, or cross_fit='none' to silence this.")


def _estimator_params(estimator: Any) -> dict[str, Any]:
    """The settings an estimator was made with: fit_params_ (set by the tuner), else its constructor arguments found
    as attributes."""
    import inspect

    if getattr(estimator, "fit_params_", None) is not None:
        return dict(estimator.fit_params_)
    params = {}
    for name, parameter in inspect.signature(type(estimator).__init__).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in ("log_dir", "log_frequency"):
            continue
        if hasattr(estimator, name):
            params[name] = getattr(estimator, name)
    return params


def _cached_fold_fits(estimator: Any, source: str, key: tuple) -> Optional[FoldFits]:
    return (getattr(estimator, "_cross_fit_cache", None) or {}).get((source, key))


def _cache_fold_fits(estimator: Any, fold_fits: FoldFits) -> FoldFits:
    """Keep refitted or nested fold fits on the estimator, so the next target on the same data reuses them."""
    try:
        cache = estimator.__dict__.setdefault("_cross_fit_cache", {})
        cache[(fold_fits.source, fold_fits.data_key)] = fold_fits
    except AttributeError:
        pass
    return fold_fits


def _refit_full(estimator: Any, train: pd.DataFrame) -> Any:
    """cross_fit="refit": the estimator's own settings on a training fold."""
    fold = type(estimator)(**_estimator_params(estimator))
    fold.fit(train)
    fold.fit_params_ = getattr(estimator, "fit_params_", None)
    return fold


def _nested_full(estimator: Any, train: pd.DataFrame) -> Any:
    """cross_fit="nested": the tuner that chose the estimator, run again on a training fold, and its fit."""
    recipe = getattr(estimator, "tuning_recipe_", None)
    if recipe is None:
        raise ValueError("cross_fit='nested' repeats the tuning inside each training fold, so it needs an estimator "
                         "from OptunaHyperparameterTuner.fit_best_model()")
    from haldensity.full_uni.tuner import OptunaHyperparameterTuner

    tuner = OptunaHyperparameterTuner(recipe["estimator_name"], train, **recipe["tuner_kwargs"])
    tuner.optimize(n_trials=recipe["n_trials"], show_progress=False)
    return tuner.fit_best_model(_fold_fits=False)


class TargetingResults(BaseModel):
    """Standardized targeting M-step results."""
    old_theta: list[float]
    theta_targeting: list[float]
    theta_intercept: float
    theta_selected: list[float]
    estimated_density: list[float]
    grid_midpoints: list[float]
    grid_points_hal_selected: Optional[list[float]] = None
    b_jk: list[list[float]]
    b_jk_targeting: list[list[float]]
    delta_j: list[float]
    grid_eval: list[float]
    # exact functionals (0.2.0): the target under the targeted density (the TMLE) and under the initial fit
    estimate: Optional[list[float]] = None
    plug_in: Optional[list[float]] = None
    density_at_estimate: Optional[list[float]] = None
    log_normalizer: Optional[float] = None
    integration_order: Optional[int] = None
    # how the targeting step was solved: the largest score-equation error, the solver and its status
    score_residual: Optional[float] = None
    solver_used: Optional[str] = None
    solver_status: Optional[str] = None
    # targets whose influence curve depends on the density (entropy, density L2): one coefficient per step, log Z
    # of the initial and of each targeted density, and P_n D* after each step
    theta_targeting_path: Optional[list[float]] = None
    log_normalizer_path: Optional[list[float]] = None
    eic_mean_path: Optional[list[float]] = None
    # cross-fitting (2026-10-26): which version ran, the folds, the fold fits' cross-fitted plug-in
    cross_fit: Optional[str] = None
    n_folds: Optional[int] = None
    plug_in_cross_fitted: Optional[list[float]] = None


@dataclass
class TargetingStep:
    """One targeting step from a start log-density: its coefficients, the solve, the integration rule, the targeted
    log-density (up to its constant), the targeted density and the largest score-equation error."""
    alpha: np.ndarray
    solve_result: Any
    rule: ExactRule
    targeted: Callable[[np.ndarray], np.ndarray]
    density: ExactDensity
    score_residual: float


def targeting_inputs(fit: Any) -> dict[str, Any]:
    """The M-step's inputs from a fitted full-data estimator (or its ``get_results()``).

    Returns {"old_theta": the intercept, the powers x, ..., x^k and the nonzero knot coefficients,
    "grid_points_hal_selected": those knots}, so that
    ``learner.run_m_step(uncensored_augmented=data, **targeting_inputs(est), ...)`` works at every order;
    build the learner with the fit's ``basis_order``. (``get_results()["theta_hat"]`` has one coefficient
    per candidate knot, zeros included, so it cannot be passed as it is.) For an estimator it also returns
    "fit" (the estimator, which cross-fitting refits when it has to) and, when the tuner attached them
    (:meth:`OptunaHyperparameterTuner.fit_best_model`), "folds" (its fold fits), so the M-step is cross-fitted.
    """
    results = fit.get_results() if hasattr(fit, "get_results") else fit
    theta = np.asarray(results["theta_hat"], dtype=float).ravel()
    knots = np.asarray(results["data_points"], dtype=float).ravel()
    first = theta.size - knots.size                      # the intercept and the k powers
    if first < 1:
        raise ValueError("theta_hat is shorter than the knots: not the results of a full-data HAL fit")
    keep = theta[first:] != 0
    inputs: dict[str, Any] = {"old_theta": np.concatenate([theta[:first], theta[first:][keep]]),
                              "grid_points_hal_selected": knots[keep]}
    if hasattr(fit, "get_results"):
        inputs["fit"] = fit
        if getattr(fit, "fold_fits_", None) is not None:
            inputs["folds"] = fit.fold_fits_
    return inputs


class BaseTargetLearner:
    """Shared M-step logic; subclasses supply the targeting basis, its jump points, the target and its variance."""

    def __init__(
        self,
        norm_constraint: int = 20,
        basis_order: int = DEFAULT_BASIS_ORDER,
        solver: str = "MOSEK",
        use_secondary_solver: bool = True,
        solver_waterfall: Sequence[str] = ("CLARABEL", "ECOS", "SCS"),
    ):
        """
        Initialize the base target learner.

        Parameters
        ----------
        norm_constraint : int, optional
            L1-norm constraint for the targeting coefficient(s) (default is 20).
        basis_order : int, optional
            Order of the basis functions (default 2, DEFAULT_BASIS_ORDER).
        solver : str, optional
            Primary CVXPY solver for the targeting M-step.
        use_secondary_solver : bool, optional
            Whether to use fallback solvers when the primary solver fails.
        solver_waterfall : Sequence[str], optional
            Ordered fallback solver list.
        """
        self.norm_constraint = norm_constraint
        self.basis_order = basis_order
        self.solver = str(solver).upper()
        self.use_secondary_solver = bool(use_secondary_solver)
        self.solver_waterfall = tuple(str(name).upper() for name in solver_waterfall)

    def get_b_ik_targeting(self, uncensored_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at data points; shape (n_data, n_targeting_basis)."""
        raise NotImplementedError

    def get_b_jk_targeting(self, data_grid: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at integration points; shape (n_points, n_targeting_basis)."""
        raise NotImplementedError

    def get_b_jk_targeting_full(self, data_grid_full: pd.DataFrame, **kwargs) -> np.ndarray:
        """Targeting basis at fine grid; shape (n_grid_full, n_targeting_basis)."""
        raise NotImplementedError

    def targeting_breakpoints(self, **kwargs) -> np.ndarray:
        """Points where the targeting basis jumps; the integration rule puts cell edges there (none by default)."""
        return np.zeros(0)

    def target_values(self, density: ExactDensity, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """(the target under ``density``, the density at the target or None), computed exactly."""
        raise NotImplementedError

    def _targeting_at(self, x: np.ndarray, kwargs: dict) -> np.ndarray:
        values = np.asarray(self.get_b_jk_targeting(pd.DataFrame({'W1': x}), **kwargs), dtype=float)
        return values[:, None] if values.ndim == 1 else values

    def _solve(self, fixed_data, b_ik_targeting, fixed_nodes, b_nodes, weights, n_samples):
        """The targeting coefficients: -sum_i L(x_i) + n log sum_q w_q exp(L(x_q)), ||alpha||_1 <= bound.

        The solvers' own tolerances, then Newton polishing (:meth:`_polish`) while the bound is not binding."""
        alpha = cp.Variable(b_ik_targeting.shape[1])
        loss = (-cp.sum(fixed_data + b_ik_targeting @ alpha)
                + n_samples * cp.log_sum_exp(np.log(weights) + fixed_nodes + b_nodes @ alpha))
        problem = cp.Problem(cp.Minimize(loss), [cp.norm1(alpha) <= self.norm_constraint])
        alpha.value = np.zeros(b_ik_targeting.shape[1])
        result = solve_cvxpy_problem(
            problem=problem,
            primary_solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            solver_waterfall=self.solver_waterfall,
            build_solve_kwargs=lambda _solver_name: {"warm_start": True},
        )
        alpha_value = self._polish(np.asarray(alpha.value, dtype=float), b_ik_targeting.mean(axis=0), fixed_nodes,
                                   b_nodes, np.log(weights))
        return alpha_value, result

    def _polish(self, alpha, data_mean, fixed_nodes, b_nodes, log_weights) -> np.ndarray:
        """Newton steps on the score equations E_alpha[b] = P_n b while ||alpha||_1 stays inside the bound.

        The objective's gradient is n (E_alpha[b] - P_n b) and its Hessian n Cov_alpha(b), both exact by the
        rule, so a few steps from the solver's point reach SCORE_TOL. At the bound the equations cannot all
        hold (the constraint's multiplier is in the optimality conditions): the solver's point is kept."""
        def residual(a):
            logits = log_weights + fixed_nodes + b_nodes @ a
            p = np.exp(logits - logsumexp(logits))
            mean = p @ b_nodes
            return mean - data_mean, p, mean

        gradient, p, mean = residual(alpha)
        for _ in range(MAX_POLISH_STEPS):
            if np.max(np.abs(gradient)) <= SCORE_TOL or np.abs(alpha).sum() >= self.norm_constraint * (1 - 1e-9):
                break
            centred = b_nodes - mean
            step = -np.linalg.lstsq((centred * p[:, None]).T @ centred, gradient, rcond=None)[0]
            scale = 1.0
            while scale > 1e-8:
                candidate = alpha + scale * step
                if np.abs(candidate).sum() <= self.norm_constraint:
                    candidate_gradient, candidate_p, candidate_mean = residual(candidate)
                    if np.max(np.abs(candidate_gradient)) < np.max(np.abs(gradient)):
                        alpha, gradient, p, mean = candidate, candidate_gradient, candidate_p, candidate_mean
                        break
                scale /= 2
            else:
                break
        return alpha

    def run_m_step(
        self,
        uncensored_augmented: pd.DataFrame,
        grid_points_hal_selected: Union[list, np.ndarray],
        old_theta: Union[list, np.ndarray],
        *,
        cross_fit: Union[str, bool] = "tuner",
        fit: Any = None,
        folds: Optional[FoldFits] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Perform an M-step update for a density model.

        This function performs an M-step update for a density model that has the form
           f(x) = exp(linear_predictor) / Z,
        where the linear predictor is the sum of two parts:
          (i) a "hal‐basis" part with an intercept and coefficients given by old_theta,
         (ii) a new "targeting‐basis" part with free coefficients (α) that are optimized.

        Only the targeting basis coefficients are updated; the intercept and the hal‐basis
        coefficients (old_theta) are kept fixed. The normalizing integral is computed exactly
        (:mod:`haldensity.full_uni.targeting.exact`): cells with edges at the HAL knots and at the
        points where the targeting basis jumps, Gauss-Legendre inside each cell. So at an optimum
        inside the L1 bound the score equations hold at the targeting points themselves (e.g. the
        targeted survival at t equals the empirical survival at t).

        Parameters
        ----------
        uncensored_augmented : pd.DataFrame
            A pandas DataFrame containing the uncensored data points.
        grid_points_hal_selected : Union[list, np.ndarray]
            The knots with nonzero coefficients.
        old_theta : Union[list, np.ndarray]
            The current (fixed) parameter vector for the intercept and hal basis functions:
            old_theta[0] is the intercept, then the powers x, ..., x^k (k >= 1) and one
            coefficient per knot in grid_points_hal_selected.
        cross_fit : str or bool, optional
            Cross-fitting (:mod:`haldensity.core.crossfit`): "tuner" (default; True) uses the tuner's fold fits,
            ``folds`` or ``fit.fold_fits_`` (cross-fitted with tuning on all folds), and falls back to "refit"
            without them; "refit" refits ``fit``'s settings on each training fold; "nested" repeats the tuning
            inside each training fold; "none" (False) is the full-sample step. Cross-fitted: one alpha for all
            folds, each fold's fit trained on the other folds and its score equations taken over its own
            validation fold; the estimate is the fold-size-weighted mean of the targeted fold fits' values.
            Without an estimator to cross-fit with (inputs given by hand), the full-sample step runs, with a
            warning.
        fit, folds
            The fitted estimator and its tuner fold fits, as :func:`targeting_inputs` returns them.
        **kwargs
            Additional method-specific parameters passed to targeting basis methods.

        Returns
        -------
        dict[str, Any]
            The targeting coefficients, the targeted density on a fine grid, and the exact
            functionals: ``estimate`` (the target under the targeted density, the TMLE),
            ``plug_in`` (under the initial fit), ``density_at_estimate`` (median only),
            ``log_normalizer``, ``integration_order`` and ``score_residual`` (the largest
            |E[b] - P_n b| over the targeting basis; about 1e-12 unless the L1 bound binds);
            ``cross_fit`` (which version ran). Cross-fitted, also ``plug_in_cross_fitted``, ``n_folds``,
            ``eic_values`` (each observation's influence curve under its own fold's targeted fit),
            ``fold_estimates`` and ``validation_rows``; ``estimated_density`` is then the fold-size-weighted
            mix of the targeted fold densities and ``log_normalizer`` is None.
        """
        old_theta, hal_knots = self._check_inputs(grid_points_hal_selected, old_theta)
        # Backward-compat: some learners (e.g., Median) expect underscored keys
        kwargs = {**kwargs, '_old_theta': old_theta, '_grid_points_hal_selected': hal_knots}
        cross_fit = resolve_cross_fit(cross_fit)
        if cross_fit != "none":
            fold_fits = self._fold_fits_for(uncensored_augmented, fit, folds, cross_fit)
            if fold_fits is not None:
                return self._run_cross_fitted(uncensored_augmented.copy(), old_theta, hal_knots,
                                              grid_points_hal_selected, fold_fits, kwargs)
            warnings.warn(NO_CROSS_FIT_WARNING, UserWarning, stacklevel=2)
        fixed = self._hal_log_density(old_theta, hal_knots)
        step = self._targeting_step(fixed, hal_knots, uncensored_augmented.copy(), kwargs)
        results = self._results(step, fixed, old_theta, grid_points_hal_selected, hal_knots, kwargs)
        results["cross_fit"] = "none"
        return results

    # ------------------------------------------------------------ cross-fitting --------------------------------------

    def _fold_fits_for(self, data: pd.DataFrame, fit: Any, folds: Optional[FoldFits],
                       cross_fit: str) -> Optional[FoldFits]:
        """The fold fits for ``cross_fit``; "tuner" falls back to "refit" without usable tuner fold fits; None
        when there is no estimator to refit."""
        key = data_key(np.asarray(data["W1"], dtype=float))
        tuner = folds if folds is not None else getattr(fit, "fold_fits_", None)
        usable = tuner is not None and tuner.source == "tuner" and tuner.matches(key)
        if cross_fit == "tuner" and usable:
            return tuner
        if fit is None:
            if cross_fit in ("refit", "nested"):
                raise ValueError(f"cross_fit={cross_fit!r} needs the fitted estimator: pass "
                                 "**targeting_inputs(estimator) or fit=estimator")
            return None
        if cross_fit == "tuner":
            cross_fit = "refit"
        cached = _cached_fold_fits(fit, cross_fit, key)       # several targets on one fit refit once
        if cached is not None:
            return cached
        splits = ([(train.copy(), validation.copy()) for train, validation in tuner.splits] if usable
                  else default_splits(len(data)))
        make = _refit_full if cross_fit == "refit" else _nested_full
        whole = data.reset_index(drop=True)
        estimators = [make(fit, whole.iloc[train].reset_index(drop=True)) for train, _ in splits]
        return _cache_fold_fits(fit, FoldFits(splits=splits, estimators=estimators, source=cross_fit, data_key=key))

    def _fold_inputs(self, data: pd.DataFrame, fold_fits: FoldFits, kwargs: dict) -> list[tuple]:
        """Per fold: (its fit's log-density, its knots, its validation rows of the data, its kwargs, the rows)."""
        whole = data.reset_index(drop=True)
        out = []
        for (_, validation), estimator in zip(fold_fits.splits, fold_fits.estimators):
            inputs = targeting_inputs(estimator)
            theta_v, knots_v = self._check_inputs(inputs["grid_points_hal_selected"], inputs["old_theta"])
            kwargs_v = {**kwargs, '_old_theta': theta_v, '_grid_points_hal_selected': knots_v}
            out.append((self._hal_log_density(theta_v, knots_v), knots_v,
                        whole.iloc[validation].reset_index(drop=True), kwargs_v, np.asarray(validation)))
        return out

    def _run_cross_fitted(self, data: pd.DataFrame, old_theta: np.ndarray, hal_knots: np.ndarray,
                          grid_points_hal_selected, fold_fits: FoldFits, kwargs: dict) -> dict[str, Any]:
        folds = self._fold_inputs(data, fold_fits, kwargs)
        steps = self._pooled_steps([fold[:4] for fold in folds])
        return self._results_cross_fitted(steps, folds, fold_fits, data, old_theta, hal_knots,
                                          grid_points_hal_selected, kwargs)

    def _solve_pooled(self, parts: list[tuple]) -> tuple[np.ndarray, Any]:
        """:meth:`_solve` with one alpha for all folds: the sum over folds of -sum_i L_v(x_i) + n_v log sum_q w_q
        exp(L_v(x_q)), then Newton polishing of the pooled score. With one fold it is :meth:`_solve`, operation
        for operation."""
        alpha = cp.Variable(parts[0][1].shape[1])
        loss = None
        for fixed_data, b_ik, fixed_nodes, b_nodes, weights, n_samples, _ in parts:
            term = (-cp.sum(fixed_data + b_ik @ alpha)
                    + n_samples * cp.log_sum_exp(np.log(weights) + fixed_nodes + b_nodes @ alpha))
            loss = term if loss is None else loss + term
        problem = cp.Problem(cp.Minimize(loss), [cp.norm1(alpha) <= self.norm_constraint])
        alpha.value = np.zeros(parts[0][1].shape[1])
        result = solve_cvxpy_problem(
            problem=problem,
            primary_solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            solver_waterfall=self.solver_waterfall,
            build_solve_kwargs=lambda _solver_name: {"warm_start": True},
        )
        polish = [(b_ik.mean(axis=0), fixed_nodes, b_nodes, np.log(weights), share)
                  for _, b_ik, fixed_nodes, b_nodes, weights, _, share in parts]
        return self._polish_pooled(np.asarray(alpha.value, dtype=float), polish), result

    def _polish_pooled(self, alpha, parts: list[tuple]) -> np.ndarray:
        """:meth:`_polish` on the pooled score sum_v (n_v / n)(E_v,alpha[b_v] - P_n,v b_v); its Hessian is
        sum_v (n_v / n) Cov_v,alpha(b_v)."""
        def residual(a):
            total, ps, means = None, [], []
            for data_mean, fixed_nodes, b_nodes, log_weights, share in parts:
                logits = log_weights + fixed_nodes + b_nodes @ a
                p = np.exp(logits - logsumexp(logits))
                mean = p @ b_nodes
                term = share * (mean - data_mean)
                total = term if total is None else total + term
                ps.append(p)
                means.append(mean)
            return total, ps, means

        gradient, ps, means = residual(alpha)
        for _ in range(MAX_POLISH_STEPS):
            if np.max(np.abs(gradient)) <= SCORE_TOL or np.abs(alpha).sum() >= self.norm_constraint * (1 - 1e-9):
                break
            hessian = None
            for (_, _, b_nodes, _, share), p, mean in zip(parts, ps, means):
                centred = b_nodes - mean
                term = share * ((centred * p[:, None]).T @ centred)
                hessian = term if hessian is None else hessian + term
            step = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            scale = 1.0
            while scale > 1e-8:
                candidate = alpha + scale * step
                if np.abs(candidate).sum() <= self.norm_constraint:
                    candidate_gradient, candidate_ps, candidate_means = residual(candidate)
                    if np.max(np.abs(candidate_gradient)) < np.max(np.abs(gradient)):
                        alpha, gradient, ps, means = candidate, candidate_gradient, candidate_ps, candidate_means
                        break
                scale /= 2
            else:
                break
        return alpha

    def _pooled_steps(self, folds: list[tuple]) -> list[TargetingStep]:
        """:meth:`_targeting_step` for several folds with one alpha: folds = [(the fold fit's log-density, its knots,
        its validation data, its kwargs)]; one TargetingStep per fold (they share alpha, the solve and the pooled
        score residual)."""
        prepared = []
        for fixed, hal_knots, data, kwargs in folds:
            fixed_data = fixed(data['W1'].to_numpy(dtype=float))
            b_ik = np.asarray(self.get_b_ik_targeting(data, **kwargs), dtype=float)
            if b_ik.ndim == 1:
                b_ik = b_ik[:, None]
            edges = cell_edges(np.concatenate([hal_knots, np.ravel(self.targeting_breakpoints(**kwargs))]))
            prepared.append({"fixed": fixed, "kwargs": kwargs, "fixed_data": fixed_data, "b_ik": b_ik,
                             "n": len(data), "rule": converged_rule(fixed, edges)})
        total = float(sum(p["n"] for p in prepared))
        while True:
            alpha, solve_result = self._solve_pooled([
                (p["fixed_data"], p["b_ik"], p["fixed"](p["rule"].points), self._targeting_at(p["rule"].points, p["kwargs"]),
                 p["rule"].weights, p["n"], p["n"] / total) for p in prepared])
            refined = False
            for p in prepared:
                def targeted(x: np.ndarray, fixed=p["fixed"], kwargs=p["kwargs"], alpha=alpha) -> np.ndarray:
                    return fixed(x) + self._targeting_at(x, kwargs) @ alpha
                p["targeted"] = targeted
                if 2 * p["rule"].order > DEFAULT_MAX_ORDER:
                    continue
                finer = p["rule"].finer()
                if abs(log_integral(targeted, finer) - log_integral(targeted, p["rule"])) > TARGETED_RULE_TOL:
                    p["rule"] = finer
                    refined = True
            if not refined:
                break
        residual = None
        for p in prepared:
            density = ExactDensity(p["targeted"], p["rule"])
            mass = p["rule"].weights * np.exp(p["targeted"](p["rule"].points) - density.log_z)
            term = (p["n"] / total) * (mass @ self._targeting_at(p["rule"].points, p["kwargs"]) - p["b_ik"].mean(axis=0))
            residual = term if residual is None else residual + term
            p["density"] = density
        score_residual = float(np.max(np.abs(residual)))
        if score_residual > SCORE_WARN:
            at_bound = np.abs(alpha).sum() >= self.norm_constraint * (1 - 1e-6)
            warnings.warn(f"The cross-fitted targeting step's pooled score equations hold only to {score_residual:.1e}, "
                          "so the estimate is not the exact CV-TMLE: "
                          + (f"the targeting coefficients reached their L1 bound (norm_constraint={self.norm_constraint}); "
                             "raise it" if at_bound else "the solver stopped early") + ".", UserWarning, stacklevel=3)
        return [TargetingStep(alpha, solve_result, p["rule"], p["targeted"], p["density"], score_residual)
                for p in prepared]

    def influence_curve_at(self, density: ExactDensity, x: np.ndarray, **kwargs) -> np.ndarray:
        """The target's influence curve under ``density`` at the points ``x``; shape (len(x), n_targets)."""
        raise NotImplementedError

    def _results_cross_fitted(self, steps: list[TargetingStep], folds: list[tuple], fold_fits: FoldFits,
                              data: pd.DataFrame, old_theta: np.ndarray, hal_knots: np.ndarray,
                              grid_points_hal_selected, kwargs: dict, extra: Optional[dict] = None) -> dict[str, Any]:
        """run_m_step's results for a cross-fitted run (see run_m_step)."""
        n = float(sum(fold[2].shape[0] for fold in folds))
        shares = [fold[2].shape[0] / n for fold in folds]
        fold_estimates, densities_at, plug_ins = [], [], []
        for step, (fixed_v, _, _, kwargs_v, _) in zip(steps, folds):
            value, at = self.target_values(step.density, **kwargs_v)
            fold_estimates.append(np.atleast_1d(np.asarray(value, dtype=float)))
            densities_at.append(None if at is None else np.atleast_1d(np.asarray(at, dtype=float)))
            plug_ins.append(np.atleast_1d(np.asarray(self.target_values(ExactDensity(fixed_v, step.rule), **kwargs_v)[0],
                                                     dtype=float)))
        estimate = sum(share * value for share, value in zip(shares, fold_estimates))
        eic = np.empty((data.shape[0], fold_estimates[0].size))
        for step, (_, _, data_v, kwargs_v, rows) in zip(steps, folds):
            values = np.asarray(self.influence_curve_at(step.density, data_v['W1'].to_numpy(dtype=float), **kwargs_v),
                                dtype=float)
            eic[rows] = values.reshape(len(rows), -1)
        fixed = self._hal_log_density(old_theta, hal_knots)
        full_rule = converged_rule(fixed, cell_edges(np.concatenate([hal_knots, np.ravel(self.targeting_breakpoints(**kwargs))])))
        plug_in, _ = self.target_values(ExactDensity(fixed, full_rule), **kwargs)
        grid_eval_full = np.unique(np.concatenate((np.linspace(0, 1, 2001), hal_knots[hal_knots >= 0])))
        grid_midpoints_full = (grid_eval_full[:-1] + grid_eval_full[1:]) / 2
        data_grid_full = pd.DataFrame({'W1': grid_midpoints_full})
        b_jk_full, _ = create_basis_functions(data_grid_full, hal_knots, order=self.basis_order, include_intercept=False)
        mixed_density = sum(share * step.density.density(grid_midpoints_full) for share, step in zip(shares, steps))
        results = {
            "old_theta": old_theta,
            "theta_targeting": steps[0].alpha,
            "theta_intercept": old_theta[0],
            "theta_selected": old_theta[1:],
            "estimated_density": mixed_density,
            "grid_midpoints": grid_midpoints_full,
            "grid_points_hal_selected": grid_points_hal_selected,
            "b_jk": b_jk_full,
            "b_jk_targeting": self.get_b_jk_targeting_full(data_grid_full, **folds[0][3]),
            "delta_j": np.diff(grid_eval_full),
            "grid_eval": grid_eval_full,
            "estimate": np.asarray(estimate, dtype=float),
            "plug_in": np.asarray(plug_in, dtype=float),
            "plug_in_cross_fitted": np.asarray(sum(share * value for share, value in zip(shares, plug_ins)),
                                               dtype=float),
            "density_at_estimate": (None if densities_at[0] is None else
                                    np.asarray(sum(share * value for share, value in zip(shares, densities_at)),
                                               dtype=float)),
            "log_normalizer": None,
            "fold_log_normalizers": [step.density.log_z for step in steps],
            "integration_order": max(step.rule.order for step in steps),
            "solver_used": steps[0].solve_result.solver_used,
            "solver_status": steps[0].solve_result.status,
            "score_residual": steps[0].score_residual,
            "cross_fit": fold_fits.source,
            "n_folds": fold_fits.n_folds,
            "eic_values": eic,
            "fold_estimates": np.vstack(fold_estimates),
            "validation_rows": [fold[4].copy() for fold in folds],
        }
        results.update(extra or {})
        return results

    @staticmethod
    def _cross_fitted_variance(targeted_fit: dict[str, Any], uncensored_data: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        """Var(D*) / n over the cross-fitted influence curve of a cross-fitted fit; None for a full-sample fit."""
        if targeted_fit.get("cross_fit", "none") == "none" or targeted_fit.get("eic_values") is None:
            return None
        eic = np.asarray(targeted_fit["eic_values"], dtype=float)
        if uncensored_data is not None and len(uncensored_data) != eic.shape[0]:
            raise ValueError("a cross-fitted fit's variance uses the data it was targeted on")
        return np.var(eic, axis=0, ddof=1) / eic.shape[0]

    def _check_inputs(self, grid_points_hal_selected, old_theta) -> tuple[np.ndarray, np.ndarray]:
        """(old_theta, knots) as float arrays, after checking they fit together at the learner's order."""
        old_theta = np.asarray(old_theta, dtype=float).ravel()
        hal_knots = np.asarray(grid_points_hal_selected, dtype=float).ravel()
        k = self.basis_order
        expected = (1 if k == 0 else k + 1) + hal_knots.size
        if old_theta.size != expected:
            raise ValueError(
                f"old_theta has {old_theta.size} entries, but an order-{k} fit with {hal_knots.size} selected knots "
                f"needs {expected}: the intercept, {k if k >= 1 else 'no'} powers and one coefficient per selected "
                f"knot. Build both from the fit with haldensity.full_uni.targeting.targeting_inputs(estimator) "
                f"(get_results()['theta_hat'] also holds the zero coefficients), and give the learner the fit's "
                f"basis_order.")
        return old_theta, hal_knots

    def _hal_log_density(self, old_theta: np.ndarray, hal_knots: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def fixed(x: np.ndarray) -> np.ndarray:
            """The initial fit's log-density up to its constant: intercept, powers and selected knots."""
            return log_unnormalized_density(x, old_theta, hal_knots, self.basis_order)
        return fixed

    def _targeting_step(self, fixed: Callable[[np.ndarray], np.ndarray], hal_knots: np.ndarray,
                        uncensored_data: pd.DataFrame, kwargs: dict) -> TargetingStep:
        """One targeting step from the log-density ``fixed`` (up to its constant): the coefficients of the
        targeting basis that maximize the likelihood within the L1 bound, on the exact rule."""
        n_samples = len(uncensored_data)
        fixed_data = fixed(uncensored_data['W1'].to_numpy(dtype=float))
        b_ik_targeting = np.asarray(self.get_b_ik_targeting(uncensored_data, **kwargs), dtype=float)
        if b_ik_targeting.ndim == 1:
            b_ik_targeting = b_ik_targeting[:, None]

        # exact integration: cells at the knots and the targeting jumps, nodes doubled until log Z settles
        edges = cell_edges(np.concatenate([hal_knots, np.ravel(self.targeting_breakpoints(**kwargs))]))
        rule = converged_rule(fixed, edges)
        while True:
            theta_targeting, solve_result = self._solve(fixed_data, b_ik_targeting, fixed(rule.points),
                                                        self._targeting_at(rule.points, kwargs), rule.weights,
                                                        n_samples)

            def targeted(x: np.ndarray, alpha: np.ndarray = theta_targeting) -> np.ndarray:
                return fixed(x) + self._targeting_at(x, kwargs) @ alpha

            if 2 * rule.order > DEFAULT_MAX_ORDER:
                break
            finer = rule.finer()
            if abs(log_integral(targeted, finer) - log_integral(targeted, rule)) <= TARGETED_RULE_TOL:
                break
            rule = finer

        targeted_density = ExactDensity(targeted, rule)
        # the largest score-equation error, |E_alpha[b] - P_n b| (about 1e-12 unless the L1 bound binds)
        mass = rule.weights * np.exp(targeted(rule.points) - targeted_density.log_z)
        score_residual = float(np.max(np.abs(mass @ self._targeting_at(rule.points, kwargs)
                                             - b_ik_targeting.mean(axis=0))))
        if score_residual > SCORE_WARN:
            at_bound = np.abs(theta_targeting).sum() >= self.norm_constraint * (1 - 1e-6)
            reason = (f"the targeting coefficients reached their L1 bound (norm_constraint={self.norm_constraint}); "
                      "raise it. This happens when an interval cut by the targeting points (below the first point, "
                      "between two consecutive points or above the last) holds no observation, so the targeted "
                      "density must put almost no mass there" if at_bound else "the solver stopped early")
            warnings.warn(f"The targeting step's score equations hold only to {score_residual:.1e}, so the estimate "
                          f"is not the exact TMLE: {reason}.", UserWarning, stacklevel=3)
        return TargetingStep(theta_targeting, solve_result, rule, targeted, targeted_density, score_residual)

    def _results(self, step: TargetingStep, fixed: Callable[[np.ndarray], np.ndarray], old_theta: np.ndarray,
                 grid_points_hal_selected, hal_knots: np.ndarray, kwargs: dict) -> dict[str, Any]:
        """run_m_step's results from its (last) step; ``fixed`` is the initial fit, for the plug-in."""
        estimate, density_at_estimate = self.target_values(step.density, **kwargs)
        plug_in, _ = self.target_values(ExactDensity(fixed, step.rule), **kwargs)

        # the targeted density on a fine grid (for plots), normalized exactly
        grid_eval_full = np.unique(np.concatenate((np.linspace(0, 1, 2001), hal_knots[hal_knots >= 0])))
        grid_midpoints_full = (grid_eval_full[:-1] + grid_eval_full[1:]) / 2
        data_grid_full = pd.DataFrame({'W1': grid_midpoints_full})
        b_jk_full, _ = create_basis_functions(data_grid_full, hal_knots, order=self.basis_order,
                                              include_intercept=False)

        return {
            "old_theta": old_theta,               # fixed intercept and hal-basis coefficients
            "theta_targeting": step.alpha,          # newly optimized targeting coefficients
            "theta_intercept": old_theta[0],
            "theta_selected": old_theta[1:],
            "estimated_density": step.density.density(grid_midpoints_full),
            "grid_midpoints": grid_midpoints_full,
            "grid_points_hal_selected": grid_points_hal_selected,
            "b_jk": b_jk_full,
            "b_jk_targeting": self.get_b_jk_targeting_full(data_grid_full, **kwargs),
            "delta_j": np.diff(grid_eval_full),
            "grid_eval": grid_eval_full,
            "estimate": np.asarray(estimate, dtype=float),
            "plug_in": np.asarray(plug_in, dtype=float),
            "density_at_estimate": None if density_at_estimate is None else np.asarray(density_at_estimate, dtype=float),
            "log_normalizer": step.density.log_z,
            "integration_order": step.rule.order,
            "solver_used": step.solve_result.solver_used,
            "solver_status": step.solve_result.status,
            "score_residual": step.score_residual,
        }

    def get_estimand_variance(self, targeted_fit: dict[str, Any], uncensored_data: pd.DataFrame, **kwargs) -> Union[float, np.ndarray]:
        """Return variance(s) of estimand(s) via EIC; to be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _exact_estimate(targeted_fit: dict[str, Any]) -> Optional[np.ndarray]:
        """The fit's exact estimate, or None (with a warning) for a fit made before 0.2.0."""
        estimate = targeted_fit.get("estimate")
        if estimate is None:
            warnings.warn(FALLBACK_WARNING, UserWarning, stacklevel=3)
            return None
        return np.atleast_1d(np.asarray(estimate, dtype=float))

    @staticmethod
    def to_results_model(results: dict[str, Any]) -> TargetingResults:
        """Convert a targeting results dict to a `TargetingResults` model."""
        def as_list(x):
            if x is None:
                return None
            return np.atleast_1d(x).tolist() if isinstance(x, np.ndarray) else list(x)

        return TargetingResults(
            old_theta=as_list(results["old_theta"]),
            theta_targeting=as_list(results["theta_targeting"]),
            theta_intercept=float(results["theta_intercept"]),
            theta_selected=as_list(results["theta_selected"]),
            estimated_density=as_list(results["estimated_density"]),
            grid_midpoints=as_list(results["grid_midpoints"]),
            grid_points_hal_selected=as_list(results.get("grid_points_hal_selected")),
            b_jk=[list(map(float, row)) for row in np.atleast_2d(results["b_jk"]).tolist()],
            b_jk_targeting=[list(map(float, row)) for row in np.atleast_2d(results["b_jk_targeting"]).tolist()],
            delta_j=as_list(results["delta_j"]),
            grid_eval=as_list(results["grid_eval"]),
            estimate=as_list(results.get("estimate")),
            plug_in=as_list(results.get("plug_in")),
            density_at_estimate=as_list(results.get("density_at_estimate")),
            log_normalizer=None if results.get("log_normalizer") is None else float(results["log_normalizer"]),
            integration_order=None if results.get("integration_order") is None else int(results["integration_order"]),
            score_residual=None if results.get("score_residual") is None else float(results["score_residual"]),
            solver_used=None if results.get("solver_used") is None else str(results["solver_used"]),
            solver_status=None if results.get("solver_status") is None else str(results["solver_status"]),
            theta_targeting_path=as_list(results.get("theta_targeting_path")),
            log_normalizer_path=as_list(results.get("log_normalizer_path")),
            eic_mean_path=as_list(results.get("eic_mean_path")),
            cross_fit=None if results.get("cross_fit") is None else str(results["cross_fit"]),
            n_folds=None if results.get("n_folds") is None else int(results["n_folds"]),
            plug_in_cross_fitted=as_list(results.get("plug_in_cross_fitted")),
        )
