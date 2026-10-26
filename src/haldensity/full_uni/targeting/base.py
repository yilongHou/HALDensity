import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Union, Any, Optional, Sequence
from pydantic import BaseModel
from haldensity.core.defaults import DEFAULT_BASIS_ORDER
from haldensity.core.basis.univariate import create_basis_functions
from scipy.special import logsumexp
from haldensity.core.solvers.cvxpy import solve_cvxpy_problem
from haldensity.core.univariate.normalizer import DEFAULT_MAX_ORDER, log_unnormalized_density
from haldensity.full_uni.targeting.exact import ExactDensity, cell_edges, converged_rule, log_integral

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


def targeting_inputs(fit: Any) -> dict[str, np.ndarray]:
    """The M-step's inputs from a fitted full-data estimator (or its ``get_results()``).

    Returns {"old_theta": the intercept, the powers x, ..., x^k and the nonzero knot coefficients,
    "grid_points_hal_selected": those knots}, so that
    ``learner.run_m_step(uncensored_augmented=data, **targeting_inputs(est), ...)`` works at every order;
    build the learner with the fit's ``basis_order``. (``get_results()["theta_hat"]`` has one coefficient
    per candidate knot, zeros included, so it cannot be passed as it is.)
    """
    results = fit.get_results() if hasattr(fit, "get_results") else fit
    theta = np.asarray(results["theta_hat"], dtype=float).ravel()
    knots = np.asarray(results["data_points"], dtype=float).ravel()
    first = theta.size - knots.size                      # the intercept and the k powers
    if first < 1:
        raise ValueError("theta_hat is shorter than the knots: not the results of a full-data HAL fit")
    keep = theta[first:] != 0
    return {"old_theta": np.concatenate([theta[:first], theta[first:][keep]]), "grid_points_hal_selected": knots[keep]}


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
        **kwargs
            Additional method-specific parameters passed to targeting basis methods.

        Returns
        -------
        dict[str, Any]
            The targeting coefficients, the targeted density on a fine grid, and the exact
            functionals: ``estimate`` (the target under the targeted density, the TMLE),
            ``plug_in`` (under the initial fit), ``density_at_estimate`` (median only),
            ``log_normalizer``, ``integration_order`` and ``score_residual`` (the largest
            |E[b] - P_n b| over the targeting basis; about 1e-12 unless the L1 bound binds).
        """
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
        # Backward-compat: some learners (e.g., Median) expect underscored keys
        kwargs = {**kwargs, '_old_theta': old_theta, '_grid_points_hal_selected': hal_knots}

        def fixed(x: np.ndarray) -> np.ndarray:
            """The initial fit's log-density up to its constant: intercept, powers and selected knots."""
            return log_unnormalized_density(x, old_theta, hal_knots, k)

        uncensored_data = uncensored_augmented.copy()
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
                          f"is not the exact TMLE: {reason}.", UserWarning, stacklevel=2)
        estimate, density_at_estimate = self.target_values(targeted_density, **kwargs)
        plug_in, _ = self.target_values(ExactDensity(fixed, rule), **kwargs)

        # the targeted density on a fine grid (for plots), normalized exactly
        grid_eval_full = np.unique(np.concatenate((np.linspace(0, 1, 2001), hal_knots[hal_knots >= 0])))
        grid_midpoints_full = (grid_eval_full[:-1] + grid_eval_full[1:]) / 2
        data_grid_full = pd.DataFrame({'W1': grid_midpoints_full})
        b_jk_full, _ = create_basis_functions(data_grid_full, hal_knots, order=k, include_intercept=False)

        return {
            "old_theta": old_theta,               # fixed intercept and hal-basis coefficients
            "theta_targeting": theta_targeting,     # newly optimized targeting coefficients
            "theta_intercept": old_theta[0],
            "theta_selected": old_theta[1:],
            "estimated_density": targeted_density.density(grid_midpoints_full),
            "grid_midpoints": grid_midpoints_full,
            "grid_points_hal_selected": grid_points_hal_selected,
            "b_jk": b_jk_full,
            "b_jk_targeting": self.get_b_jk_targeting_full(data_grid_full, **kwargs),
            "delta_j": np.diff(grid_eval_full),
            "grid_eval": grid_eval_full,
            "estimate": np.asarray(estimate, dtype=float),
            "plug_in": np.asarray(plug_in, dtype=float),
            "density_at_estimate": None if density_at_estimate is None else np.asarray(density_at_estimate, dtype=float),
            "log_normalizer": targeted_density.log_z,
            "integration_order": rule.order,
            "solver_used": solve_result.solver_used,
            "solver_status": solve_result.status,
            "score_residual": score_residual,
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
        )
