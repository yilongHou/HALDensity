import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from typing import Optional, Union
from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.solvers.cvxpy import (
    DEFAULT_SOLVER_THREADS,
    build_solver_sequence,
    solve_cvxpy_problem,
    thread_options,
    tight_ladder,
)
from haldensity.core.univariate.base_estimator import BaseEstimator
from haldensity.core.univariate.objective import (
    OBJECTIVE_NORMALIZERS,
    STRUCTURED_ORDERS,
    candidate_knots,
    normalizer_terms,
    resolve_max_knots,
    structured_problem,
)
from haldensity.core.univariate.support import FullDataHALProblem, certified_solve


SOLVER_TOLERANCES = ("tight", "default")


class CVXPYEstimator(BaseEstimator):
    """
    Density estimator using CVXPY for convex optimization.
    
    This class implements the original NPMLE approach using cumulative
    indicator basis functions with CVXPY for solving the convex optimization problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood.
    """

    # How objective_normalizer="knot_aware" is solved for basis_order 0 and 1: "structured" (sparse
    # reparametrization in the cell values, haldensity.core.univariate.objective.structured_problem)
    # or "dense" (the n x m basis; kept as a reference for verification). Same optimum either way.
    # If no solver reaches status "optimal" on the structured form, the fit falls back to the dense form;
    # knot_aware_formulation_used_ records which one produced the fit.
    knot_aware_formulation = "structured"
    
    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 3.0,
        n_grid_points: int = 200,
        basis_order: int = 0,
        solver: str = "MOSEK",
        log_dir: Optional[str] = None,
        log_frequency: int = 10,
        use_secondary_solver: bool = True,
        solver_waterfall: list[str] = ["MOSEK", "CLARABEL", "ECOS", "SCS"],
        normalizer: str = "gauss_legendre",
        objective_normalizer: str = "midpoint",
        max_knots: Union[int, str, None] = "auto",
        solver_tolerance: str = "tight",
        solver_threads: int = DEFAULT_SOLVER_THREADS,
    ):
        """
        Initialize CVXPY estimator.
        
        Args:
            tol: tol for pruning small theta coefficients
            norm_constraint: One-norm constraint for theta[1:]
            n_grid_points: Number of grid points for density evaluation
            solver: CVXPY solver to use (default: "MOSEK")
            normalizer: how the fitted density is normalized when evaluated (see BaseEstimator)
            objective_normalizer: the normalizer inside the fitting objective
                (haldensity.core.univariate.objective). "midpoint" (default): midpoint rule on the
                n_grid_points - 1 equal cells. "knot_aware": the candidate knots are breakpoints of
                the cells, so no gap between knots is unseen; exact for basis_order = 0 (one term
                per cell between consecutive knots, reusing the data term's values).
            max_knots: cap K on the number of candidate knots, placed at K data quantiles
                (observations, de-duplicated). "auto" (default): K = 1,000 for basis_order >= 1 and
                max(1,000, ceil(n^(2/3))) for basis_order = 0, so only samples above 1,000 are
                affected (haldensity.core.univariate.objective.default_max_knots); None: every
                unique data value (the behaviour before 0.2.0); an integer: that K.
            solver_tolerance: "tight" (default; for the final fit at the chosen bound): the solvers
                from `solver` and `solver_waterfall` with tight tolerances (MOSEK 1e-12; CLARABEL
                1e-12 then 1e-10; ECOS 1e-12 then 1e-10; SCS 1e-9 capped at 600 s), each solution
                pruned at `tol` and checked against the optimality conditions, the first certified
                one kept; if none is, the best attempt from any solver (fewest coordinates failing
                the conditions), with a warning and certified_ = False. The result is the point estimate and its
                basis the delta-method SE basis; certified_, kkt_ and solve_attempts_ record the
                check. "default": the solvers' own tolerances, first acceptable status wins, no
                check (v0.1.0; OptunaHyperparameterTuner uses it in the CV folds).
            solver_threads: threads per solver call (default 1: MOSEK, CLARABEL and the BLAS/OpenMP
                pools run single-threaded, so parallel runs on shared servers do not take other
                people's cores). Raise it when one fit runs alone on a dedicated machine.
        """
        if solver_tolerance not in SOLVER_TOLERANCES:
            raise ValueError(f"solver_tolerance must be one of {SOLVER_TOLERANCES}, got {solver_tolerance!r}")
        if objective_normalizer not in OBJECTIVE_NORMALIZERS:
            raise ValueError(f"objective_normalizer must be one of {OBJECTIVE_NORMALIZERS}, got {objective_normalizer!r}")
        resolve_max_knots(max_knots, 1, basis_order)       # validates; the cap itself is set at fit time
        super().__init__(
            normalizer=normalizer,
            tol=tol,
            basis_order=basis_order,
            log_dir=log_dir or "./local/logs/cvxpy.log",
            log_frequency=log_frequency,
        )
        self.norm_constraint = norm_constraint
        self.n_grid_points = n_grid_points
        self.solver = solver
        self.objective_normalizer = objective_normalizer
        self.max_knots = max_knots if max_knots is None or isinstance(max_knots, str) else int(max_knots)
        self.solver_tolerance = solver_tolerance
        thread_options("MOSEK", solver_threads)            # validates
        self.solver_threads = int(solver_threads)
        
        # Internal, non-standard attributes for inspection
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self.knot_aware_formulation_used_: Optional[str] = None
        self.max_knots_used_: Optional[int] = None
        self.certified_: Optional[bool] = None
        self.kkt_: Optional[dict] = None
        self.solve_attempts_: tuple = ()
        self.support_ = None
        self.solver_option_: Optional[str] = None
        self.use_secondary_solver = use_secondary_solver
        self.solver_waterfall = solver_waterfall
        self.solver_used_: Optional[str] = None
        self.problem_status_: Optional[str] = None
        
    def fit(self, data: pd.DataFrame) -> 'CVXPYEstimator':
        """
        Fit the CVXPY density estimator.
        
        Args:
            data: DataFrame with column 'W1' containing the observations
            
        Returns:
            Self for method chaining
        """
        # Candidate knots: every unique data value, or max_knots data quantiles
        observed = data['W1'].dropna().to_numpy(dtype=float)
        grid_points_hal = candidate_knots(observed, self.max_knots, basis_order=self.basis_order)
        self.max_knots_used_ = resolve_max_knots(self.max_knots, int(np.isfinite(observed).sum()), self.basis_order)
        self._grid_points_hal = grid_points_hal  # Store the full set of candidate knots
        self._train_x = data['W1'].to_numpy(dtype=float)
        
        grid_eval = np.linspace(0, 1, self.n_grid_points)
        self.certified_, self.kkt_, self.solve_attempts_, self.support_ = None, None, (), None
        self.solver_option_ = None
        if self.solver_tolerance == "tight":
            theta_value = self._tight_solve(data, grid_points_hal)
        else:
            theta_value = self._default_solve(data, grid_points_hal)

        # Check if optimization was successful
        theta_raw = theta_value()
        if theta_raw is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")
         # Store raw optimized value for inspection
        self.optimized_theta_raw = np.asarray(theta_raw, dtype=float).copy()

        # --- SET STANDARDIZED ATTRIBUTES ---
        # Create and set the official, pruned theta_hat
        self.theta_hat = self.optimized_theta_raw.copy()
        # Prune small coefficients (excluding polynomial terms)
        self.theta_hat[self.basis_order + 1:] = np.where(
            np.abs(self.theta_hat[self.basis_order + 1:]) > self.tol, 
            self.theta_hat[self.basis_order + 1:], 
            0
        )
        
        # Determine the selected grid points based on non-zero coefficients in theta_hat
        knot_coeffs_start_index = self.basis_order + 1
        non_zero_knot_indices = np.where(self.theta_hat[knot_coeffs_start_index:] != 0)[0]
        
        if non_zero_knot_indices.size > 0:
            self.grid_points_hal_selected = grid_points_hal[non_zero_knot_indices]
        else:
            self.grid_points_hal_selected = np.array([])
        # --- End Standardization ---
            
        grid_eval = np.sort(np.unique(np.concatenate((grid_eval, self.grid_points_hal_selected))))
        self.grid_midpoints = (grid_eval[:-1] + grid_eval[1:]) / 2
        self.delta_j = grid_eval[1:] - grid_eval[:-1]
        # Public evaluation grid for density queries
        self.grid_points = np.linspace(0, 1, self.n_grid_points)
        
        self.is_fitted = True

        # Store the fitted theta as a dictionary for inspection
        assert len(self.basis_names) == len(self.theta_hat), "Basis names count does not match theta_hat length"
        self.fitted_theta_dict = {name: float(value) for name, value in zip(self.basis_names, self.theta_hat)}

        return self
    
    def _default_solve(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """solver_tolerance="default": the solvers' own tolerances, first acceptable status wins (v0.1.0)."""
        structured = (
            self.objective_normalizer == "knot_aware"
            and self.basis_order in STRUCTURED_ORDERS
            and self.knot_aware_formulation == "structured"
        )
        solve_options = dict(
            primary_solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            solver_waterfall=self.solver_waterfall,
            solver_threads=self.solver_threads,
        )
        solve_result = None
        if structured:
            # Sparse form of the same problem: no n x m basis is built
            self.basis_names = create_basis_functions(
                pd.DataFrame({'W1': [0.5]}), grid_points_hal, order=self.basis_order
            )[1]
            sparse = structured_problem(self._train_x, grid_points_hal, self.basis_order, self.n_grid_points,
                                        self.norm_constraint)
            try:
                # only a fully converged solve; otherwise the dense form below, as before this formulation
                solve_result = solve_cvxpy_problem(problem=sparse.problem, acceptable_statuses=("optimal",),
                                                   **solve_options)
            except RuntimeError:
                solve_result = None
        if solve_result is not None:
            self.knot_aware_formulation_used_ = "structured"
            self.objective_value_ = solve_result.objective_value * sparse.objective_scale
            dual = sparse.l1_constraint.dual_value
            self.lambda_val_lag = None if dual is None else float(dual) * sparse.dual_scale
            theta_value = sparse.theta
        else:
            self.knot_aware_formulation_used_ = "dense" if self.objective_normalizer == "knot_aware" else None
            problem, l1_constraint, theta_value = self._dense_problem(data, grid_points_hal)
            solve_result = solve_cvxpy_problem(problem=problem, **solve_options)
            self.objective_value_ = solve_result.objective_value
            self.lambda_val_lag = l1_constraint.dual_value
        self.solver_used_ = solve_result.solver_used
        self.problem_status_ = solve_result.status
        return theta_value


    def _tight_solve(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """solver_tolerance="tight": one certified solve (haldensity.core.univariate.support.certified_solve)."""
        problem = FullDataHALProblem(data, self.basis_order, self.n_grid_points, self.objective_normalizer,
                                     self.max_knots, structured=self.knot_aware_formulation == "structured")
        if not np.array_equal(problem.knots, grid_points_hal):
            raise RuntimeError("internal error: the certified solve built different knots")
        solvers = build_solver_sequence(self.solver, use_secondary_solver=self.use_secondary_solver,
                                        solver_waterfall=self.solver_waterfall)
        fit = certified_solve(problem, float(self.norm_constraint), tight_ladder(solvers), prune=self.tol,
                              solver_threads=self.solver_threads)
        self.basis_names = create_basis_functions(
            pd.DataFrame({'W1': [0.5]}), grid_points_hal, order=self.basis_order
        )[1]
        self.knot_aware_formulation_used_ = fit.formulation if self.objective_normalizer == "knot_aware" else None
        self.objective_value_ = fit.objective_value
        self.lambda_val_lag = fit.dual_value
        self.solver_used_, self.solver_option_ = fit.solver_used, fit.option
        self.problem_status_ = fit.solver_status
        self.certified_, self.kkt_, self.solve_attempts_, self.support_ = fit.certified, fit.kkt, fit.attempts, fit
        if not fit.certified:
            warnings.warn(
                f"The fit at norm_constraint={self.norm_constraint} could not be certified by the optimality "
                f"check; the best attempt ({fit.solver_used} {fit.option}: {fit.kkt['n_failing']} coordinates fail "
                f"the conditions, max ratio {fit.kkt['max_ratio']:.4g}) is used and certified_ is False.",
                UserWarning,
                stacklevel=3,
            )
        return lambda: fit.theta_raw

    def _dense_problem(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """The problem with the n x m data basis (midpoint objective, and knot-aware for k >= 2)."""
        n_samples = len(data)
        # Build basis for the data points
        basis_array, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        b_ik = basis_array  # shape: (n_samples, K)
        self.basis_names = basis_names

        # Cells of the objective's normalizer (midpoint grid, or knot-aware)
        terms = normalizer_terms(self._train_x, grid_points_hal, self.basis_order, self.n_grid_points,
                                 self.objective_normalizer)

        # K: number of basis functions
        if self.basis_order == 0:
            K = 1 + len(grid_points_hal)  # {1, I(x >= ξ₁), ..., I(x >= ξₘ)}
        else:
            K = (self.basis_order + 1) + len(grid_points_hal)  # {1, x, ..., x^k, (x-ξ₁)₊^k, ..., (x-ξₘ)₊^k}

        theta = cp.Variable(K)

        # Data log-density at the data points and at the evaluation grid
        # Since our basis includes the intercept, we use the full basis @ theta
        data_log_density = b_ik @ theta
        first_term = -cp.sum(data_log_density)
        if terms.basis is not None:
            log_density_grid = terms.basis @ theta
        else:
            # k = 0 knot-aware: the level on the cell starting at a knot is the data term's value there
            parts = [theta[0:1]] if terms.leading_intercept_cell else []
            if terms.data_rows.size:
                parts.append(data_log_density[terms.data_rows])
            log_density_grid = cp.hstack(parts) if len(parts) > 1 else parts[0]
        log_terms = terms.log_widths + log_density_grid
        log_Z = cp.log_sum_exp(log_terms)
        second_term = n_samples * log_Z

        loss = first_term + second_term
        constraints = [cp.norm1(theta[1:]) <= self.norm_constraint]
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)
        return problem, constraints[0], (lambda: None if theta.value is None else theta.value.copy())

    def objective_log_normalizer(self, theta: Optional[np.ndarray] = None) -> float:
        """log Z of the fitting objective's own normalizer at ``theta`` (default: the raw solution).

        Compare with ``log_normalizer()`` (the accurate one): a large gap means the objective did
        not see part of the fitted mass.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted first")
        theta = np.asarray(self.optimized_theta_raw if theta is None else theta, dtype=float)
        terms = normalizer_terms(self._train_x, self._grid_points_hal, self.basis_order, self.n_grid_points,
                                 self.objective_normalizer, build_basis=False)
        from scipy.special import logsumexp
        return float(logsumexp(terms.log_widths + terms.log_density(theta, self._grid_points_hal, self.basis_order)))

    def get_results(self) -> dict:
        """Return standardized common results plus CVX-specific fields."""
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before getting results. Call fit() first.")
        base = self._get_common_results()
        base.update({
            "lambda_val_lag": self.lambda_val_lag,
            "optimized_theta_raw": self.optimized_theta_raw,
            "norm_constraint": float(self.norm_constraint),
            "n_grid_points": int(self.n_grid_points),
            "objective_normalizer": self.objective_normalizer,
            "max_knots": self.max_knots,
            "max_knots_used": self.max_knots_used_,
            "n_candidate_knots": int(len(self._grid_points_hal)),
            "knot_aware_formulation_used": self.knot_aware_formulation_used_,
            "solver_tolerance": self.solver_tolerance,
            "solver_used": self.solver_used_,
            "solver_option": self.solver_option_,
            "problem_status": self.problem_status_,
            "certified": self.certified_,
            "kkt": self.kkt_,
            "support_refit": self.support_,
        })
        return base
