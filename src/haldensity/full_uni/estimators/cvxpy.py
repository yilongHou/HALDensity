import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from typing import Optional, Union
from haldensity.core.defaults import DEFAULT_BASIS_ORDER
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
    STRUCTURED_ORDERS,
    OBJECTIVE_NORMALIZERS,
    candidate_knots,
    dense_problem,
    normalizer_terms,
    resolve_max_knots,
    resolve_objective_normalizer,
    structured_problem,
)
from haldensity.core.univariate.imbalance import check_imbalance, imbalance_diagnostics
from haldensity.core.univariate.support import FullDataHALProblem, certified_solve


SOLVER_TOLERANCES = ("tight", "default")


class CVXPYEstimator(BaseEstimator):
    """
    Density estimator using CVXPY for convex optimization.
    
    This class implements the original NPMLE approach using cumulative
    indicator basis functions with CVXPY for solving the convex optimization problem:
    
    min_θ  −ℓ(θ)  subject to  ‖θ[1:]‖₁ ≤ norm_constraint
    
    where ℓ is the HAL-basis log-likelihood. The normalizer inside the objective is the package's choice,
    by the order (haldensity.core.objective_rules): the exact objective at basis_order 0, the midpoint
    rule with the knots added to the grid at 1, the old 199-cell midpoint rule at 2 and above;
    get_results()["objective_rule"] records it.
    """

    # Internal (tests, benchmarks): how the knots-added objective is solved for basis_order 0 and 1,
    # "structured" (the step-height rewrite, haldensity.core.univariate.objective.structured_problem) or
    # "dense" (the problem in theta; a reference for verification). Same optimum either way. If no solver
    # reaches status "optimal" on the structured form, the fit falls back to the dense form;
    # knot_aware_formulation_used_ records which one produced the fit.
    _knot_aware_formulation = "structured"
    
    def __init__(
        self,
        tol: float = 1e-4,
        norm_constraint: float = 3.0,
        n_grid_points: int = 200,
        basis_order: int = DEFAULT_BASIS_ORDER,
        solver: str = "MOSEK",
        log_dir: Optional[str] = None,
        log_frequency: int = 10,
        use_secondary_solver: bool = True,
        solver_waterfall: list[str] = ["MOSEK", "CLARABEL", "ECOS", "SCS"],
        normalizer: str = "gauss_legendre",
        max_knots: Union[int, str, None] = "auto",
        solver_tolerance: str = "tight",
        solver_threads: int = DEFAULT_SOLVER_THREADS,
        _objective_normalizer: Optional[str] = None,
    ):
        """
        Initialize CVXPY estimator.
        
        Args:
            tol: tol for pruning small theta coefficients
            norm_constraint: One-norm constraint for theta[1:]
            n_grid_points: Number of grid points for density evaluation
            solver: CVXPY solver to use (default: "MOSEK")
            normalizer: how the fitted density is normalized when evaluated (see BaseEstimator)
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
        if _objective_normalizer is not None and _objective_normalizer not in OBJECTIVE_NORMALIZERS:
            raise ValueError(f"_objective_normalizer must be None or one of {OBJECTIVE_NORMALIZERS}")
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
        # Internal hook (tests, benchmarks, experiment repos): force the objective's normalizer, "midpoint"
        # or "knot_aware"; None (always for users) lets the package choose by the order, see objective_rule_
        self._objective_normalizer = _objective_normalizer
        self.max_knots = max_knots if max_knots is None or isinstance(max_knots, str) else int(max_knots)
        self.solver_tolerance = solver_tolerance
        thread_options("MOSEK", solver_threads)            # validates
        self.solver_threads = int(solver_threads)
        
        # Internal, non-standard attributes for inspection
        self.optimized_theta_raw: Optional[np.ndarray] = None
        self.lambda_val_lag: Optional[float] = None
        self.knot_aware_formulation_used_: Optional[str] = None
        self.max_knots_used_: Optional[int] = None
        self.objective_rule_: Optional[str] = None
        self.objective_normalizer_used_: Optional[str] = None
        self.imbalance_: Optional[dict] = None
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
        self.objective_normalizer_used_, self.objective_rule_ = resolve_objective_normalizer(
            self._objective_normalizer or "auto", self.basis_order)
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
        # internal safeguard at k = 0 (haldensity.core.univariate.imbalance): recorded for every fit,
        # a warning asking for a report if the final fit shows the coefficient imbalance
        self.imbalance_ = None
        if self.basis_order == 0:
            observed_x = self._train_x[np.isfinite(self._train_x)]
            if self.solver_tolerance == "tight":
                self.imbalance_ = check_imbalance(self.theta_hat, grid_points_hal, observed_x, stacklevel=3)
            else:
                self.imbalance_ = imbalance_diagnostics(self.theta_hat, grid_points_hal, observed_x)
        
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
    
    @property
    def _objective(self) -> str:
        """The implemented objective normalizer for this order ("auto" resolved)."""
        return resolve_objective_normalizer(self._objective_normalizer or "auto", self.basis_order)[0]

    def _default_solve(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """solver_tolerance="default": the solvers' own tolerances, first acceptable status wins (v0.1.0)."""
        structured = (
            self._objective == "knot_aware"
            and self.basis_order in STRUCTURED_ORDERS
            and self._knot_aware_formulation == "structured"
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
            self.knot_aware_formulation_used_ = "dense" if self._objective == "knot_aware" else None
            problem, l1_constraint, theta_value = self._dense_problem(data, grid_points_hal)
            solve_result = solve_cvxpy_problem(problem=problem, **solve_options)
            self.objective_value_ = solve_result.objective_value
            self.lambda_val_lag = l1_constraint.dual_value
        self.solver_used_ = solve_result.solver_used
        self.problem_status_ = solve_result.status
        return theta_value


    def _tight_solve(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """solver_tolerance="tight": one certified solve (haldensity.core.univariate.support.certified_solve)."""
        problem = FullDataHALProblem(data, self.basis_order, self.n_grid_points, self._objective,
                                     self.max_knots, structured=self._knot_aware_formulation == "structured")
        if not np.array_equal(problem.knots, grid_points_hal):
            raise RuntimeError("internal error: the certified solve built different knots")
        solvers = build_solver_sequence(self.solver, use_secondary_solver=self.use_secondary_solver,
                                        solver_waterfall=self.solver_waterfall)
        fit = certified_solve(problem, float(self.norm_constraint), tight_ladder(solvers), prune=self.tol,
                              solver_threads=self.solver_threads)
        self.basis_names = create_basis_functions(
            pd.DataFrame({'W1': [0.5]}), grid_points_hal, order=self.basis_order
        )[1]
        self.knot_aware_formulation_used_ = fit.formulation if self._objective == "knot_aware" else None
        self.objective_value_ = fit.objective_value
        self.lambda_val_lag = fit.dual_value
        self.solver_used_, self.solver_option_ = fit.solver_used, fit.option
        self.problem_status_ = fit.solver_status
        self.certified_, self.kkt_, self.solve_attempts_, self.support_ = fit.certified, fit.kkt, fit.attempts, fit
        if not fit.certified:
            warnings.warn(
                f"Basis cleaning failed at norm_constraint={self.norm_constraint}: no solver's solution passed "
                f"the optimality check. The best attempt ({fit.solver_used} {fit.option}: {fit.kkt['n_failing']} "
                f"coordinates fail the conditions, max ratio {fit.kkt['max_ratio']:.4g}) is the point estimate "
                f"(certified_ is False); no SE or intervals will be given. Use a bootstrap or another solver.",
                UserWarning,
                stacklevel=3,
            )
        return lambda: fit.theta_raw

    def _dense_problem(self, data: pd.DataFrame, grid_points_hal: np.ndarray):
        """The problem in theta (haldensity.core.univariate.objective.dense_problem): the old midpoint
        rule, the knots-added grid for k >= 2, and the reference form. The data term is the basis
        column sums times theta, so no n x m data basis is built."""
        self.basis_names = create_basis_functions(
            pd.DataFrame({'W1': [0.5]}), grid_points_hal, order=self.basis_order
        )[1]
        dense = dense_problem(self._train_x, grid_points_hal, self.basis_order, self.n_grid_points,
                              self._objective, self.norm_constraint)
        return dense.problem, dense.l1_constraint, dense.theta

    def objective_log_normalizer(self, theta: Optional[np.ndarray] = None) -> float:
        """log Z of the fitting objective's own normalizer at ``theta`` (default: the raw solution).

        Compare with ``log_normalizer()`` (the accurate one): a large gap means the objective did
        not see part of the fitted mass.
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted first")
        theta = np.asarray(self.optimized_theta_raw if theta is None else theta, dtype=float)
        terms = normalizer_terms(self._train_x, self._grid_points_hal, self.basis_order, self.n_grid_points,
                                 self._objective, build_basis=False)
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
            "objective_normalizer": self._objective,       # the implemented normalizer (cleaning rebuilds it)
            "objective_rule": self.objective_rule_,
            "imbalance": self.imbalance_,
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
