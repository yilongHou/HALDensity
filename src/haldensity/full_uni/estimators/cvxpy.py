import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional
from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.solvers.cvxpy import solve_cvxpy_problem
from haldensity.core.univariate.base_estimator import BaseEstimator
from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.core.univariate.objective import OBJECTIVE_NORMALIZERS, candidate_knots, normalizer_terms


class CVXPYEstimator(BaseEstimator):
    """
    Density estimator using CVXPY for convex optimization.
    
    This class implements the original NPMLE approach using cumulative
    indicator basis functions with CVXPY for solving the convex optimization problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood.
    """
    
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
        max_knots: Optional[int] = None,
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
                (observations, de-duplicated); None (default) uses every unique data value.
        """
        if objective_normalizer not in OBJECTIVE_NORMALIZERS:
            raise ValueError(f"objective_normalizer must be one of {OBJECTIVE_NORMALIZERS}, got {objective_normalizer!r}")
        if max_knots is not None and int(max_knots) < 1:
            raise ValueError("max_knots must be a positive integer or None")
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
        self.max_knots = None if max_knots is None else int(max_knots)
        
        # Internal, non-standard attributes for inspection
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
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
        n_samples = len(data)
        # Candidate knots: every unique data value, or max_knots data quantiles
        grid_points_hal = candidate_knots(data['W1'].dropna().to_numpy(dtype=float), self.max_knots)
        self._grid_points_hal = grid_points_hal  # Store the full set of candidate knots
        self._train_x = data['W1'].to_numpy(dtype=float)
        
        # Build basis for the data points
        basis_array, basis_names = create_basis_functions(data, grid_points_hal, order=self.basis_order)
        b_ik = basis_array  # shape: (n_samples, K)
        self.basis_names = basis_names
        
        # Cells of the objective's normalizer (midpoint grid, or knot-aware)
        grid_eval = np.linspace(0, 1, self.n_grid_points)
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
        
        solve_result = solve_cvxpy_problem(
            problem=problem,
            primary_solver=self.solver,
            use_secondary_solver=self.use_secondary_solver,
            solver_waterfall=self.solver_waterfall,
        )
        self.solver_used_ = solve_result.solver_used
        self.problem_status_ = solve_result.status
        self.objective_value_ = solve_result.objective_value

        self.lambda_val_lag = problem.constraints[0].dual_value
        
        # Check if optimization was successful
        if theta.value is None:
            raise RuntimeError("CVXPY optimization failed - theta.value is None")
         # Store raw optimized value for inspection
        self.optimized_theta_raw = theta.value.copy()
        
        # --- SET STANDARDIZED ATTRIBUTES ---
        # Create and set the official, pruned theta_hat
        self.theta_hat = theta.value.copy()
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
    
    def objective_log_normalizer(self, theta: Optional[np.ndarray] = None) -> float:
        """log Z of the fitting objective's own normalizer at ``theta`` (default: the raw solution).

        Compare with ``log_normalizer()`` (the accurate one): a large gap means the objective did
        not see part of the fitted mass.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted first")
        theta = np.asarray(self.optimized_theta_raw if theta is None else theta, dtype=float)
        terms = normalizer_terms(self._train_x, self._grid_points_hal, self.basis_order, self.n_grid_points,
                                 self.objective_normalizer)
        data_log = None
        if terms.basis is None:
            data_log = log_unnormalized_density(self._train_x, theta, self._grid_points_hal, self.basis_order)
        from scipy.special import logsumexp
        return float(logsumexp(terms.log_widths + terms.log_density(theta, data_log)))

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
            "n_candidate_knots": int(len(self._grid_points_hal)),
        })
        return base
