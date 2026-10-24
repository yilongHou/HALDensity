"""
Optuna-based hyperparameter tuning for supported HAL density estimators.

Supported estimators only:
- CVXPYEstimator, FISTAEstimator, ProjectedGDEstimator, ProximalGDEstimator,
  ProximalAdaGradEstimator, ProximalNewtonEstimator, ProximalNewtonLBFGSEstimator,
  ProximalNewtonLBFGSFullEstimator

Metrics:
- sll: Average log-likelihood (maximize)
- bic: Bayesian Information Criterion (minimize)

This module contains no plotting and operates purely on the data passed in.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Optional, Tuple, Any
from datetime import datetime
from scipy.interpolate import interp1d
import warnings
from tqdm import tqdm

import optuna
from sklearn.model_selection import KFold

from haldensity.core.cv.conservative import SELECTION_RULES, bisection_search, conservative_threshold
from haldensity.core.defaults import TUNER_DEFAULTS

# Import estimators
from haldensity.full_uni.estimators.cvxpy import CVXPYEstimator
from haldensity.full_uni.estimators.fista import FISTAEstimator
from haldensity.full_uni.estimators.projected_gd import ProjectedGDEstimator
from haldensity.full_uni.estimators.proximal_gd import ProximalGDEstimator
from haldensity.full_uni.estimators.proximal_adagrad import ProximalAdaGradEstimator
from haldensity.full_uni.estimators.proximal_newton import ProximalNewtonEstimator
from haldensity.full_uni.estimators.proximal_newton_lbfgs import ProximalNewtonLBFGSEstimator
from haldensity.full_uni.estimators.proximal_newton_lbfgs_full import (
    ProximalNewtonLBFGSFullEstimator,
)

CV_SCORINGS = ("exact", "interpolated")

# Available estimators for tuning (supported set only)
ESTIMATORS = {
    "CVXPYEstimator": CVXPYEstimator,
    "FISTAEstimator": FISTAEstimator,
    "ProjectedGDEstimator": ProjectedGDEstimator,
    "ProximalGDEstimator": ProximalGDEstimator,
    "ProximalAdaGradEstimator": ProximalAdaGradEstimator,
    "ProximalNewtonEstimator": ProximalNewtonEstimator,
    "ProximalNewtonLBFGSEstimator": ProximalNewtonLBFGSEstimator,
    "ProximalNewtonLBFGSFullEstimator": ProximalNewtonLBFGSFullEstimator,
}

class OptunaHyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for density estimation methods.
    
    Uses cross-validation to evaluate hyperparameters on held-out validation sets
    to prevent overfitting and ensure robust parameter selection.
    """
    
    def __init__(
        self,
        estimator_name: str,
        data: pd.DataFrame,
        cv_folds: int = 3,
        metric: str = "sll",
        random_state: int = 42,
        max_iter: int = 50_000,
        tolerance: float = 1e-6,
        log_frequency: int = -1,  # No logging during CV
        n_grid_points: int = 200,
        silent: bool = True,
        show_progress: bool = False,
        param_overrides: Optional[dict[str, Any]] = None,
        estimator_kwargs: Optional[dict[str, Any]] = None,
        cv_scoring: str = "exact",
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        conservative_selection_rule: str = TUNER_DEFAULTS.conservative_selection_rule,
        conservative_k_percent: float = TUNER_DEFAULTS.conservative_k_percent,
        conservative_se_multiplier: float = TUNER_DEFAULTS.conservative_se_multiplier,
        conservative_bracket_factor: float = TUNER_DEFAULTS.conservative_bracket_factor,
        conservative_min_norm_constraint: float = TUNER_DEFAULTS.conservative_min_norm_constraint,
        conservative_step_pct: float = TUNER_DEFAULTS.conservative_step_pct,
        conservative_max_steps: int = TUNER_DEFAULTS.conservative_max_steps,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            estimator_name: Name of the estimator to tune
            data: Full dataset for cross-validation
            cv_folds: Number of cross-validation folds
            metric: Optimization metric ('sll' or 'bic')
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for iterative estimators
            tolerance: Convergence tolerance
            log_frequency: Logging frequency (-1 = no logging)
            n_grid_points: Number of grid points for density evaluation
            silent: Whether to suppress all print statements
            param_overrides: Optional overrides/limits for hyperparameters. Format per param:
                - categorical: {"choices": [..]} or a list, or a single fixed value
                - float/int: {"low": ..., "high": ..., "log": bool (optional), "step": ... (optional)}
                  or a tuple/list (low, high), or a single fixed value
            estimator_kwargs: Optional fixed (not tuned) keyword arguments passed to every
                estimator the tuner builds, in the CV folds and in fit_best_model(), so both
                use the same settings. By default CVXPYEstimator uses its own solver defaults
                (primary solver plus fallback waterfall). For the old ECOS-only behaviour pass
                {"solver": "ECOS", "use_secondary_solver": False}. Keys must not overlap the
                tuned parameters; fix those with param_overrides instead.
            cv_scoring: How a fitted fold estimator is scored on its validation points.
                "exact" (default): the fitted log density evaluated exactly at each point
                (estimator.get_log_density_at_points, normalized per the estimator's
                `normalizer`, by default the accurate Gauss-Legendre log Z).
                "interpolated": the pre-0.2.0 score, linear interpolation of get_density()
                on its 200-point grid with a 1e-10 floor. Fully old behaviour also needs
                estimator_kwargs={"normalizer": "midpoint"}.
            use_conservative_adjustment: After the CV search, move to the smallest bound
                (norm_constraint) whose mean CV log-likelihood is at least the threshold below the
                best (haldensity.core.cv.conservative; bracketing then bisection in log M), as the
                censored-data tuners do. On by default. Applies to estimators tuned by
                norm_constraint (CVXPYEstimator) with metric "sll"; otherwise it is skipped and
                `adjustment_note` says why. The CV-best parameters stay in `max_cv_params`. See
                docs/dev/cvll-multicollinearity for why a flat CV curve makes this necessary.
            conservative_selection_rule, conservative_k_percent, conservative_se_multiplier:
                threshold = best - k_percent * SD ("legacy_percent_sd", default) or
                best - se_multiplier * SD / sqrt(folds) ("one_se").
            conservative_bracket_factor, conservative_min_norm_constraint, conservative_step_pct,
            conservative_max_steps: search settings (shrink factor, floor, relative tolerance,
                maximum CV evaluations).
        """
        if estimator_name not in ESTIMATORS:
            raise ValueError(f"Estimator '{estimator_name}' not supported. "
                           f"Available: {list(ESTIMATORS.keys())}")
        
        if metric not in ["sll", "bic"]:
            raise ValueError(f"Metric '{metric}' not supported. Use 'sll' or 'bic'.")
        if cv_scoring not in CV_SCORINGS:
            raise ValueError(f"cv_scoring must be one of {CV_SCORINGS}, got {cv_scoring!r}")
        
        self.estimator_name = estimator_name
        self.estimator_class = ESTIMATORS[estimator_name]
        self.data = data
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.log_frequency = log_frequency
        self.n_grid_points = n_grid_points
        self.silent = silent
        self.show_progress = show_progress
        self.param_overrides: dict[str, Any] = param_overrides or {}
        self.estimator_kwargs: dict[str, Any] = dict(estimator_kwargs or {})
        self.cv_scoring = cv_scoring
        self.use_conservative_adjustment = bool(use_conservative_adjustment)
        self.conservative_selection_rule = str(conservative_selection_rule).strip().lower()
        if self.conservative_selection_rule not in SELECTION_RULES:
            raise ValueError(f"conservative_selection_rule must be one of {SELECTION_RULES}")
        self.conservative_k_percent = float(conservative_k_percent)
        self.conservative_se_multiplier = float(conservative_se_multiplier)
        self.conservative_bracket_factor = float(conservative_bracket_factor)
        if not 0.0 < self.conservative_bracket_factor < 1.0:
            raise ValueError("conservative_bracket_factor must be in (0, 1)")
        self.conservative_min_norm_constraint = float(conservative_min_norm_constraint)
        self.conservative_step_pct = float(conservative_step_pct)
        self.conservative_max_steps = int(conservative_max_steps)

        # Cross-validation setup
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        # Full keyword arguments of the best trial's estimator (tuned, fixed and estimator_kwargs)
        self.best_estimator_params: Optional[dict[str, Any]] = None
        self._trial_estimator_params: dict[int, dict[str, Any]] = {}
        self._trial_fold_scores: dict[int, list[float]] = {}
        self._fold_errors: list[str] = []
        # Conservative adjustment results
        self.max_cv_params: Optional[dict[str, Any]] = None
        self.conservative_params: Optional[dict[str, Any]] = None
        self.adjustment_results: list[dict[str, Any]] = []
        self.adjustment_note: Optional[str] = None

        if not silent:
            print(f"Initialized tuner for {estimator_name}")
            print(f"Dataset size: {len(data)} samples")
            print(f"Cross-validation: {cv_folds} folds")
            print(f"Optimization metric: {metric} ({'maximize' if metric == 'sll' else 'minimize'})")

    def _suggest_param(self, trial: optuna.Trial, name: str, kind: str, **default_kwargs: Any) -> Any:
        """
        Suggest a hyperparameter value honoring optional user overrides.

        Args:
            trial: Optuna trial
            name: Parameter name
            kind: One of {"categorical", "float", "int"}
            default_kwargs: Default args for Optuna suggest_* (e.g., low/high/log or choices)

        Returns:
            Suggested (or fixed) parameter value
        """
        override = self.param_overrides.get(name, None)

        # Helper to call appropriate suggestor
        def suggest_with(kwargs: dict[str, Any]) -> Any:
            if kind == "categorical":
                return trial.suggest_categorical(name, kwargs["choices"])
            if kind == "float":
                return trial.suggest_float(
                    name,
                    kwargs["low"],
                    kwargs["high"],
                    step=kwargs.get("step", None),
                    log=kwargs.get("log", False),
                )
            if kind == "int":
                return trial.suggest_int(
                    name,
                    kwargs["low"],
                    kwargs["high"],
                    step=kwargs.get("step", 1),
                    log=kwargs.get("log", False),
                )
            raise ValueError(f"Unknown kind '{kind}' for parameter '{name}'")

        # No override: use defaults
        if override is None:
            return suggest_with(default_kwargs)

        # Fixed value provided directly
        if not isinstance(override, (list, tuple, dict)):
            return override

        # List/tuple override
        if isinstance(override, (list, tuple)):
            if kind == "categorical":
                return suggest_with({"choices": list(override)})
            # Numeric: interpret as (low, high) if len>=2, else fixed
            if len(override) >= 2:
                low, high = override[0], override[1]
                kw = dict(default_kwargs)
                kw.update({"low": low, "high": high})
                return suggest_with(kw)
            if len(override) == 1:
                return override[0]
            # Empty list/tuple: fall back to defaults
            return suggest_with(default_kwargs)

        # Dict override
        if isinstance(override, dict):
            if "fixed" in override:
                return override["fixed"]
            # Merge default with provided keys
            kw = dict(default_kwargs)
            kw.update(override)
            # Normalize categorical API
            if kind == "categorical" and "choices" not in kw and "values" in kw:
                kw["choices"] = kw.pop("values")
            return suggest_with(kw)

        # Fallback to defaults
        return suggest_with(default_kwargs)
    
    def _compute_log_likelihood_from_density(self, estimator: Any, validation_points: np.ndarray) -> float:
        """
        Average validation log-likelihood of a fitted estimator (see ``cv_scoring``).

        Args:
            estimator: Fitted estimator
            validation_points: Points to evaluate log-likelihood on

        Returns:
            Average log-likelihood
        """
        if self.cv_scoring == "exact":
            try:
                log_density = np.asarray(estimator.get_log_density_at_points(validation_points), dtype=float)
                return float(np.mean(log_density))
            except Exception as e:
                if not self.silent:
                    print(f"Warning: Failed to compute log-likelihood: {e}")
                return -np.inf

        try:
            # Get density from the estimator
            grid_points, density_values = estimator.get_density()
            
            # Interpolate density at validation points
            # Create interpolator with bounds handling
            density_interp = interp1d(
                grid_points, 
                density_values, 
                kind="linear",
                bounds_error=False, 
                fill_value=(density_values[0], density_values[-1])
            )
            
            # Get interpolated density values
            interpolated_density = density_interp(validation_points)
            
            # Ensure densities are positive (avoid log of zero/negative)
            interpolated_density = np.maximum(interpolated_density, 1e-10)
            
            # Compute average log-likelihood
            log_likelihood = np.log(interpolated_density)
            avg_log_likelihood = np.mean(log_likelihood)
            
            return float(avg_log_likelihood)
            
        except Exception as e:
            if not self.silent:
                print(f"Warning: Failed to compute log-likelihood: {e}")
            return -np.inf
    
    def _compute_bic_from_density(self, estimator: Any, validation_data: pd.DataFrame) -> float:
        """
        Compute BIC using the density output from any estimator.
        
        Args:
            estimator: Fitted estimator
            validation_data: Validation dataset
            
        Returns:
            BIC value (lower is better)
        """
        try:
            # Get number of parameters from the estimator results
            results = estimator.get_results()
            
            # For HAL-based methods, count non-zero coefficients
            if 'theta_hat' in results and results['theta_hat'] is not None:
                # Count non-zero coefficients (excluding intercept)
                theta_hat = results['theta_hat']
                n_params = np.sum(np.abs(theta_hat[1:]) > 1e-8) + 1  # +1 for intercept
            elif 'n_selected_knots' in results:
                n_params = results['n_selected_knots'] + 1  # +1 for intercept
            else:
                # Fallback: estimate parameters from density complexity
                n_params = 10  # Conservative estimate
            
            # Compute log-likelihood
            validation_points = validation_data['W1'].values
            avg_log_likelihood = self._compute_log_likelihood_from_density(estimator, validation_points)
            
            # BIC = -2 * log_likelihood + k * log(n)
            n_samples = len(validation_data)
            sum_log_likelihood = avg_log_likelihood * n_samples
            bic = -2 * sum_log_likelihood + n_params * np.log(n_samples)
            
            return float(bic)
            
        except Exception as e:
            if not self.silent:
                print(f"Warning: Failed to compute BIC: {e}")
            return np.inf
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for the given estimator based on Optuna trial.
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            dictionary of hyperparameters for the estimator
        """
        # Estimator-specific parameter ranges (supported set only)
        if self.estimator_name == "CVXPYEstimator":
            params = {
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "norm_constraint": self._suggest_param(trial, "norm_constraint", "float", low=1e-3, high=1e6, log=True),
            }
            
        elif self.estimator_name == "FISTAEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1e4, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "L": self._suggest_param(trial, "L", "float", low=1e0, high=1e5, log=True),
            }
        elif self.estimator_name == "ProjectedGDEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1e4, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "learning_rate": self._suggest_param(trial, "learning_rate", "float", low=1e-4, high=1e-1, log=True),
            }
        elif self.estimator_name == "ProximalGDEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1e4, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "L": self._suggest_param(trial, "L", "float", low=1e1, high=1e5, log=True),
            }
        elif self.estimator_name == "ProximalAdaGradEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1e2, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "alpha": self._suggest_param(trial, "alpha", "float", low=1e-5, high=1e-1, log=True),
            }
            
        elif self.estimator_name == "ProximalNewtonEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1.0, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "cd_sweeps": self._suggest_param(trial, "cd_sweeps", "int", low=1, high=5),
                "line_search_c": self._suggest_param(trial, "line_search_c", "float", low=1e-6, high=1e-1, log=True),
                "max_line_search_steps": self._suggest_param(trial, "max_line_search_steps", "int", low=10, high=50),
                "line_search_beta": self._suggest_param(trial, "line_search_beta", "float", low=0.3, high=0.8),
                "non_desc_clip_alpha": self._suggest_param(trial, "non_desc_clip_alpha", "categorical", choices=[True, False]),
                "hessian_regularization": self._suggest_param(trial, "hessian_regularization", "float", low=1e-10, high=1e-6, log=True),
                "non_descent_step_size": self._suggest_param(trial, "non_descent_step_size", "float", low=0.01, high=0.5),
            }
        elif self.estimator_name == "ProximalNewtonLBFGSEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1.0, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "line_search_c": self._suggest_param(trial, "line_search_c", "float", low=1e-5, high=1e-1, log=True),
                "max_line_search_steps": self._suggest_param(trial, "max_line_search_steps", "int", low=10, high=50),
                "lbfgs_memory": self._suggest_param(trial, "lbfgs_memory", "int", low=3, high=15),
                "non_desc_clip_alpha": self._suggest_param(trial, "non_desc_clip_alpha", "categorical", choices=[True, False]),
                "lbfgs_gamma_clip_range": (
                    self._suggest_param(trial, "lbfgs_gamma_clip_min", "float", low=1e-5, high=1e-1, log=True),
                    self._suggest_param(trial, "lbfgs_gamma_clip_max", "float", low=1e1, high=1e5, log=True)
                ),
            }
        elif self.estimator_name == "ProximalNewtonLBFGSFullEstimator":
            params = {
                "n_iterations": self.max_iter,
                "lam": self._suggest_param(trial, "lam", "float", low=1e-4, high=1.0, log=True),
                "basis_order": self._suggest_param(trial, "basis_order", "categorical", choices=[0, 1, 2]),
                "line_search_c": self._suggest_param(trial, "line_search_c", "float", low=1e-5, high=1e-1, log=True),
                "max_line_search_steps": self._suggest_param(trial, "max_line_search_steps", "int", low=10, high=50),
                "lbfgs_memory": self._suggest_param(trial, "lbfgs_memory", "int", low=3, high=15),
                "non_desc_clip_alpha": self._suggest_param(trial, "non_desc_clip_alpha", "categorical", choices=[True, False]),
                "lbfgs_gamma_clip_range": (
                    self._suggest_param(trial, "lbfgs_gamma_clip_min", "float", low=1e-5, high=1e-1, log=True),
                    self._suggest_param(trial, "lbfgs_gamma_clip_max", "float", low=1e1, high=1e5, log=True)
                ),
            }
        else:
            raise ValueError(f"Hyperparameter ranges not defined for {self.estimator_name}")

        overlap = sorted(set(params) & set(self.estimator_kwargs))
        if overlap:
            raise ValueError(
                f"estimator_kwargs {overlap} overlap tuned parameters; fix them with param_overrides instead."
            )
        params.update(self.estimator_kwargs)

        return params

    def _evaluate_params_cv(self, params: dict[str, Any]) -> float:
        """
        Evaluate hyperparameters using cross-validation.
        
        Args:
            params: Hyperparameters to evaluate
            
        Returns:
            Average metric value across CV folds
        """
        cv_scores = self._fold_scores(params)
        if len(cv_scores) == 0:
            # If all folds failed, return worst possible score
            return -np.inf if self.metric == "sll" else np.inf
        return np.mean(cv_scores)

    def _fold_scores(self, params: dict[str, Any]) -> list[float]:
        """Finite validation scores of the folds that succeeded (failures are recorded)."""
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.data)):
            try:
                # Split data
                train_data = self.data.iloc[train_idx].reset_index(drop=True)
                val_data = self.data.iloc[val_idx].reset_index(drop=True)
                
                # Create and fit estimator. CVXPY's "Solution may be inaccurate" notice is noise in
                # the search (the status is recorded on the estimator); the final fit still shows it.
                estimator = self.estimator_class(**params)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
                    estimator.fit(train_data)
                
                # Compute metric on validation set using density-based approach
                if self.metric == "sll":
                    # Average Log-Likelihood (higher is better)
                    score = self._compute_log_likelihood_from_density(estimator, val_data['W1'].values)
                else:  # bic
                    # Bayesian Information Criterion (lower is better)
                    score = self._compute_bic_from_density(estimator, val_data)
                
                if np.isfinite(score):
                    cv_scores.append(score)
                else:
                    self._fold_errors.append(f"fold {fold_idx}: non-finite score {score}")
                    if not self.silent:
                        print(f"Warning: Non-finite score {score} in fold {fold_idx}")

            except Exception as e:
                self._fold_errors.append(f"fold {fold_idx}: {type(e).__name__}: {e}")
                if not self.silent:
                    print(f"Warning: Error in fold {fold_idx}: {e}")
                continue

        self._last_fold_scores = list(cv_scores)
        return cv_scores
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to optimize (Optuna always minimizes)
        """
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        self._trial_estimator_params[trial.number] = dict(params)

        # Evaluate using cross-validation
        metric_value = self._evaluate_params_cv(params)
        self._trial_fold_scores[trial.number] = list(getattr(self, "_last_fold_scores", []))
        
        # Optuna minimizes, so negate SLL (which we want to maximize)
        if self.metric == "sll":
            return -metric_value  # Convert maximization to minimization
        else:
            return metric_value  # BIC is already for minimization
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds (None for no limit)
            show_progress: Whether to show progress bar
            
        Returns:
            dictionary containing optimization results
        """
        if not self.silent:
            print(f"\nStarting hyperparameter optimization...")
            print(f"Estimator: {self.estimator_name}")
            print(f"Trials: {n_trials}")
            print(f"Metric: {self.metric}")
        
        self._fold_errors = []
        self._trial_estimator_params = {}
        self._trial_fold_scores = {}
        self.max_cv_params = self.conservative_params = self.adjustment_note = None
        self.adjustment_results = []

        # Keep Optuna's per-trial INFO lines quiet during the search; restore the caller's level after.
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            # Create Optuna study
            direction = "minimize"  # Always minimize (we negate SLL if needed)
            study_name = f"{self.estimator_name}_{self.metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
            self.study = optuna.create_study(
                direction=direction,
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
        
            # Run optimization with tqdm progress bar
            if self.show_progress and n_trials > 1:
                # Use tqdm progress bar for trials (only if more than 1 trial)
                with tqdm(total=n_trials, desc=f"Optimizing {self.estimator_name}", 
                         unit="trial", dynamic_ncols=True) as pbar:
                    def callback(study, trial):
                        pbar.update(1)
                        # Update progress bar description with current best value
                        if study.best_value is not None:
                            best_display = f"{-study.best_value:.4f}" if self.metric == "sll" else f"{study.best_value:.4f}"
                            pbar.set_postfix_str(f"Best {self.metric.upper()}: {best_display}")
                
                    self.study.optimize(
                        self._objective,
                        n_trials=n_trials,
                        timeout=timeout,
                        callbacks=[callback],
                        show_progress_bar=False  # Disable optuna's progress bar
                    )
            else:
                # Run without progress bar
                self.study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=False
                )
        finally:
            optuna.logging.set_verbosity(previous_verbosity)

        # A trial whose every fold failed scores +inf; if no trial is finite the
        # "best" parameters would just be the first trial's, so fail loudly instead.
        if not any(t.value is not None and np.isfinite(t.value) for t in self.study.trials):
            distinct_errors = list(dict.fromkeys(self._fold_errors))
            raise RuntimeError(
                f"All {len(self.study.trials)} CV trials failed: no fold produced a finite "
                f"{self.metric} score. First errors: {distinct_errors[:3]}"
            )

        # Extract results
        self.best_params = self.study.best_params.copy()
        self.best_estimator_params = dict(self._trial_estimator_params[self.study.best_trial.number])
        best_objective_value = self.study.best_value
        
        # Convert back to original metric scale
        if self.metric == "sll":
            self.best_metric_value = -best_objective_value  # Convert back from negated
        else:
            self.best_metric_value = best_objective_value

        self.max_cv_params = dict(self.best_estimator_params)
        if self.use_conservative_adjustment:
            self._apply_conservative_adjustment(self.study.best_trial.number)
        
        if not self.silent:
            print(f"\nOptimization completed!")
            print(f"Best {self.metric.upper()}: {self.best_metric_value:.6f}")
            print(f"Best parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        return {
            "best_params": self.best_params,
            "best_estimator_params": self.best_estimator_params,
            "best_metric_value": self.best_metric_value,
            "max_cv_params": self.max_cv_params,
            "conservative_params": self.conservative_params,
            "adjustment_results": self.adjustment_results,
            "n_trials": len(self.study.trials),
            "study": self.study
        }
    
    def _apply_conservative_adjustment(self, best_trial_number: int) -> None:
        """Move norm_constraint to the smallest value whose CV score clears the threshold."""
        if self.metric != "sll":
            self.adjustment_note = "skipped: the conservative adjustment is defined for metric='sll'"
            return
        if "norm_constraint" not in self.best_estimator_params:
            self.adjustment_note = f"skipped: {self.estimator_name} is not tuned by norm_constraint"
            return
        best = dict(self.best_estimator_params)
        best_scores = self._trial_fold_scores.get(best_trial_number, [])
        best_mean = float(np.mean(best_scores)) if best_scores else float("-inf")
        if not np.isfinite(best_mean):
            self.adjustment_note = "skipped: the best trial has no finite fold score"
            return
        best_sd = float(np.std(best_scores, ddof=1)) if len(best_scores) >= 2 else 0.0
        threshold, details = conservative_threshold(
            best_mean=best_mean, best_sd=best_sd, n_valid=len(best_scores),
            selection_rule=self.conservative_selection_rule, k_percent=self.conservative_k_percent,
            se_multiplier=self.conservative_se_multiplier,
        )
        cache: dict[float, tuple[float, float]] = {float(best["norm_constraint"]): (best_mean, best_sd)}

        def is_above(norm_constraint: float, step: int) -> bool:
            nc = float(max(norm_constraint, self.conservative_min_norm_constraint))
            if nc not in cache:
                scores = self._fold_scores({**best, "norm_constraint": nc})
                mean = float(np.mean(scores)) if scores else float("-inf")
                cache[nc] = (mean, float(np.std(scores, ddof=1)) if len(scores) >= 2 else 0.0)
            mean, sd = cache[nc]
            above = bool(mean >= threshold)
            self.adjustment_results.append(dict(step=int(step), norm_constraint=nc, cv_ll=mean, cv_sd=sd,
                                                above_threshold=above, threshold=float(threshold), **details))
            return above

        chosen = bisection_search(
            is_above, float(best["norm_constraint"]), bracket_factor=self.conservative_bracket_factor,
            min_norm_constraint=self.conservative_min_norm_constraint, rel_tol=self.conservative_step_pct,
            max_steps=self.conservative_max_steps,
        )
        self.conservative_params = {**best, "norm_constraint": float(chosen)}
        self.best_estimator_params = dict(self.conservative_params)
        if "norm_constraint" in self.best_params:
            self.best_params["norm_constraint"] = float(chosen)
        self.best_metric_value = cache.get(float(chosen), (self.best_metric_value,))[0]

    # Removed plotting functions: no visualization in this module
    
    def fit_best_model(self) -> Any:
        """
        Fit the estimator with the best found hyperparameters on the full dataset.
        
        Returns:
            Fitted estimator with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        if not self.silent:
            print(f"\nFitting best model on full dataset...")
            print(f"Parameters: {self.best_params}")
        
        # Create estimator with best params
        # Refit with exactly the settings the best trial was cross-validated with.
        # study.best_params alone misses fixed overrides (e.g. basis_order), max_iter
        # and composite arguments such as lbfgs_gamma_clip_range.
        if self.best_estimator_params is not None:
            params = dict(self.best_estimator_params)
        else:
            params = {**self.best_params, **self.estimator_kwargs}
        best_estimator = self.estimator_class(**params)
        
        # Fit on full data
        best_estimator.fit(self.data)
        
        if not self.silent:
            print("Best model fitted successfully!")
        return best_estimator
    
    # Removed evaluation visualization; users can compute metrics via estimator APIs
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to JSON file.
        
        Args:
            filepath: Path to save results JSON file
        """
        if self.study is None or self.best_params is None:
            raise ValueError("No optimization results to save. Run optimize() first.")
        
        # Prepare results dictionary
        results = {
            "estimator_name": self.estimator_name,
            "optimization_config": {
                "cv_folds": self.cv_folds,
                "metric": self.metric,
                "cv_scoring": self.cv_scoring,
                "n_trials": len(self.study.trials),
                "random_state": self.random_state
            },
            "best_hyperparameters": self.best_params,
            "best_estimator_params": self.best_estimator_params,
            "best_metric_value": self.best_metric_value,
            "data_size": len(self.data),
            "optimization_timestamp": datetime.now().isoformat()
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not self.silent:
            print(f"Results saved to {filepath}")

