"""Cross-validation over the L1 bound shared by data types.

Provides:
- TuningResult: NamedTuple for standardized tuner output
- BaseCensoredInitTuner: Abstract base for Stage 1 (Init) tuners, including the
  conservative adjustment (bisection search below the CV-selected bound)

The Stage 2 (EM) tuner bases live in ``haldensity.core.em.tuners``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from haldensity.core.defaults import EM_DEFAULTS, TUNER_DEFAULTS
from haldensity.core.cv.conservative import bisection_search, conservative_threshold, cv_standard_error


class TuningResult(NamedTuple):
    """Result from hyperparameter tuning.
    
    Attributes
    ----------
    estimator : Any
        The fitted estimator instance with best hyperparameters.
    best_params : Dict[str, Any]
        Best hyperparameters found during tuning.
    metadata : Dict[str, Any]
        Additional metadata including CV results, study info, adjustment details.
    """
    estimator: Any
    best_params: Dict[str, Any]
    metadata: Dict[str, Any]


# Shared with the full-data tuner; private aliases kept for existing importers.
_compute_cv_standard_error = cv_standard_error
_compute_conservative_threshold = conservative_threshold


class BaseCensoredInitTuner(ABC):
    """Abstract base class for Stage 1 (Init) hyperparameter tuners.
    
    Provides shared CV tuning logic with Optuna and conservative adjustment.
    Subclasses implement data-specific methods for estimator creation and scoring.
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data.
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.
    n_grid_points : int
        Number of grid points for density evaluation.
    param_overrides : dict | None
        Override default tunable parameter ranges.
    use_conservative_adjustment : bool
        Whether to apply conservative 1-SE adjustment.
    conservative_k_percent : float
        Threshold as percentage of SD for conservative adjustment.
    conservative_max_steps : int
        Maximum steps for conservative search.
    conservative_step_pct : float
        Step size as percentage of best value (linear) or relative tolerance (bisection).
    conservative_method : str
        Conservative search method: {"bisection", "linear"}.
    conservative_bracket_factor : float
        Geometric shrink factor (0, 1) used to bracket the threshold in bisection mode.
    conservative_min_norm_constraint : float
        Minimum norm_constraint considered during conservative search.
    silent : bool
        Whether to suppress progress output.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        cv_folds: int = TUNER_DEFAULTS.cv_folds,
        random_state: int = TUNER_DEFAULTS.random_state,
        n_grid_points: int = TUNER_DEFAULTS.n_grid_points,
        param_overrides: Optional[Dict[str, Any]] = None,
        use_conservative_adjustment: bool = TUNER_DEFAULTS.use_conservative_adjustment,
        conservative_k_percent: float = TUNER_DEFAULTS.conservative_k_percent,
        conservative_max_steps: int = TUNER_DEFAULTS.conservative_max_steps,
        conservative_step_pct: float = TUNER_DEFAULTS.conservative_step_pct,
        conservative_method: str = TUNER_DEFAULTS.conservative_method,
        conservative_selection_rule: str = TUNER_DEFAULTS.conservative_selection_rule,
        conservative_se_multiplier: float = TUNER_DEFAULTS.conservative_se_multiplier,
        conservative_bracket_factor: float = TUNER_DEFAULTS.conservative_bracket_factor,
        conservative_min_norm_constraint: float = TUNER_DEFAULTS.conservative_min_norm_constraint,
        silent: bool = True,
    ):
        self.data = data.reset_index(drop=True)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.n_grid_points = int(n_grid_points)
        self.param_overrides = param_overrides or {}
        self.silent = bool(silent)
        
        self.use_conservative_adjustment = bool(use_conservative_adjustment)
        self.conservative_k_percent = float(conservative_k_percent)
        self.conservative_max_steps = int(conservative_max_steps)
        self.conservative_step_pct = float(conservative_step_pct)
        self.conservative_method = str(conservative_method)
        self.conservative_selection_rule = str(conservative_selection_rule).strip().lower()
        self.conservative_se_multiplier = float(conservative_se_multiplier)
        self.conservative_bracket_factor = float(conservative_bracket_factor)
        self.conservative_min_norm_constraint = float(conservative_min_norm_constraint)
        
        if self.conservative_method not in ("bisection", "linear"):
            raise ValueError("conservative_method must be one of {'bisection', 'linear'}")
        if self.conservative_selection_rule not in ("legacy_percent_sd", "one_se"):
            raise ValueError(
                "conservative_selection_rule must be one of {'legacy_percent_sd', 'one_se'}"
            )
        if self.conservative_se_multiplier < 0.0:
            raise ValueError("conservative_se_multiplier must be >= 0")
        if not (0.0 < self.conservative_bracket_factor < 1.0):
            raise ValueError("conservative_bracket_factor must be in (0, 1)")
        if not (0.0 < self.conservative_min_norm_constraint):
            raise ValueError("conservative_min_norm_constraint must be > 0")
        
        self.kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Results storage
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        self.optuna_params: Optional[Dict[str, Any]] = None
        self.conservative_params: Optional[Dict[str, Any]] = None
        self.adjustment_results: Optional[list] = None
        
        # Fixed defaults (non-tunable) - typed explicitly for mypy
        self._defaults: Dict[str, Any] = {
            "tol": float(EM_DEFAULTS.tol),
            "solver": str(TUNER_DEFAULTS.solver),
            "use_secondary_solver": bool(TUNER_DEFAULTS.use_secondary_solver),
        }
    
    @abstractmethod
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest tunable parameters for the trial."""
        pass
    
    @abstractmethod
    def _evaluate_fold(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        params: Dict[str, Any]
    ) -> float:
        """Evaluate parameters on a single CV fold. Returns log-likelihood."""
        pass
    
    @abstractmethod
    def _fit_final_estimator(self, params: Dict[str, Any]) -> Any:
        """Fit final estimator on full data with given parameters."""
        pass
    
    def _evaluate_folds(self, params: Dict[str, Any]) -> list[float]:
        """Evaluate parameters across all CV folds."""
        scores = []
        for train_idx, val_idx in self.kfold.split(self.data):
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            val_df = self.data.iloc[val_idx].reset_index(drop=True)
            try:
                score = self._evaluate_fold(train_df, val_df, params)
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))
        return scores
    
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters via CV. Returns negated mean for Optuna."""
        scores = self._evaluate_folds(params)
        mean_score = float(np.mean(scores))
        return -mean_score if np.isfinite(mean_score) else float("inf")
    
    def _compute_cv_score_with_sd(
        self, 
        norm_constraint: float, 
        basis_order: int
    ) -> Tuple[float, float, list[float]]:
        """Compute CV score and SD for given parameters."""
        params = {"norm_constraint": norm_constraint, "basis_order": basis_order}
        fold_scores = self._evaluate_folds(params)
        valid_scores = [s for s in fold_scores if np.isfinite(s)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else float("-inf")
        sd_score = float(np.std(valid_scores, ddof=1)) if len(valid_scores) >= 2 else 0.0
        return mean_score, sd_score, fold_scores
    
    def _apply_conservative_adjustment(self) -> Dict[str, Any]:
        """Apply conservative threshold adjustment after Optuna CV.
        
        Goal (same across methods):
        find the smallest `norm_constraint` such that:
            CV_LL(norm_constraint) >= CV_LL(optuna_best) - k% * SD(optuna_best)
        
        Notes
        -----
        - This is a *post-processing* step that keeps `basis_order` fixed and
          only adjusts `norm_constraint`.
        - In practice CV_LL vs norm_constraint is not perfectly monotone due to
          numerical noise and solver failures; implementations keep track of
          the smallest value observed above threshold and fall back gracefully.
        """
        if self.optuna_params is None:
            raise ValueError("Optuna optimization must run first")
        
        optuna_nc = float(self.optuna_params["norm_constraint"])
        basis_order = int(self.optuna_params["basis_order"])
        
        optuna_cv_ll, optuna_cv_sd, fold_scores = self._compute_cv_score_with_sd(optuna_nc, basis_order)
        n_valid_scores = int(sum(np.isfinite(score) for score in fold_scores))
        if not np.isfinite(optuna_cv_ll):
            # If the "best" point is not evaluable, do not attempt adjustment.
            self.adjustment_results = [{
                "step": 0,
                "norm_constraint": float(optuna_nc),
                "cv_ll": float(optuna_cv_ll),
                "cv_sd": float(optuna_cv_sd),
                "above_threshold": True,
                "method": self.conservative_method,
                "selection_rule": self.conservative_selection_rule,
                "note": "optuna_cv_ll_nonfinite_no_adjustment",
            }]
            self.conservative_params = {"norm_constraint": float(optuna_nc), "basis_order": int(basis_order)}
            return self.conservative_params
        
        threshold, threshold_details = _compute_conservative_threshold(
            best_mean=float(optuna_cv_ll),
            best_sd=float(optuna_cv_sd),
            n_valid=n_valid_scores,
            selection_rule=self.conservative_selection_rule,
            k_percent=self.conservative_k_percent,
            se_multiplier=self.conservative_se_multiplier,
        )
        
        if self.conservative_method == "linear":
            return self._apply_conservative_adjustment_linear(
                optuna_nc=optuna_nc,
                basis_order=basis_order,
                optuna_cv_ll=float(optuna_cv_ll),
                optuna_cv_sd=float(optuna_cv_sd),
                threshold=threshold,
                threshold_details=threshold_details,
            )
        return self._apply_conservative_adjustment_bisection(
            optuna_nc=optuna_nc,
            basis_order=basis_order,
            optuna_cv_ll=float(optuna_cv_ll),
            optuna_cv_sd=float(optuna_cv_sd),
            threshold=threshold,
            threshold_details=threshold_details,
        )
    
    def _apply_conservative_adjustment_linear(
        self,
        optuna_nc: float,
        basis_order: int,
        optuna_cv_ll: float,
        optuna_cv_sd: float,
        threshold: float,
        threshold_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Legacy linear stepping: nc = optuna_nc - step * (pct * optuna_nc)."""
        step_size = self.conservative_step_pct * optuna_nc
        best_conservative_nc = optuna_nc
        
        self.adjustment_results = [{
            "step": 0,
            "norm_constraint": float(optuna_nc),
            "cv_ll": float(optuna_cv_ll),
            "cv_sd": float(optuna_cv_sd),
            "above_threshold": True,
            "method": "linear",
            "threshold": float(threshold),
            **threshold_details,
        }]
        
        for step in range(1, self.conservative_max_steps + 1):
            candidate_nc = optuna_nc - step * step_size
            if candidate_nc <= self.conservative_min_norm_constraint:
                break
            cand_cv_ll, cand_cv_sd, _ = self._compute_cv_score_with_sd(candidate_nc, basis_order)
            above_threshold = bool(cand_cv_ll >= threshold)
            self.adjustment_results.append({
                "step": int(step),
                "norm_constraint": float(candidate_nc),
                "cv_ll": float(cand_cv_ll),
                "cv_sd": float(cand_cv_sd),
                "above_threshold": above_threshold,
                "method": "linear",
                "threshold": float(threshold),
                **threshold_details,
            })
            if above_threshold:
                best_conservative_nc = candidate_nc
            else:
                break
        
        self.conservative_params = {
            "norm_constraint": float(best_conservative_nc),
            "basis_order": int(basis_order),
        }
        return self.conservative_params
    
    def _apply_conservative_adjustment_bisection(
        self,
        optuna_nc: float,
        basis_order: int,
        optuna_cv_ll: float,
        optuna_cv_sd: float,
        threshold: float,
        threshold_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bracket + log-bisection to find smallest nc with CV_LL >= threshold.
        
        Uses geometric bracketing to rapidly move away from extremely large
        Optuna-selected constraints (common with highly correlated bases).
        """
        # Cache within this adjustment call (avoid duplicate CV runs).
        cache: dict[float, tuple[float, float]] = {}
        
        def _key(nc: float) -> float:
            # Stable-ish key: round to mitigate float noise but preserve scale.
            return float(np.round(float(nc), 12))
        
        def eval_nc(nc: float, step: int) -> tuple[float, float, bool]:
            nc = float(max(nc, self.conservative_min_norm_constraint))
            k = _key(nc)
            if k in cache:
                mean_ll, sd_ll = cache[k]
            else:
                mean_ll, sd_ll, _ = self._compute_cv_score_with_sd(nc, basis_order)
                mean_ll, sd_ll = float(mean_ll), float(sd_ll)
                cache[k] = (mean_ll, sd_ll)
            
            above = bool(mean_ll >= threshold)
            self.adjustment_results.append({
                "step": int(step),
                "norm_constraint": float(nc),
                "cv_ll": float(mean_ll),
                "cv_sd": float(sd_ll),
                "above_threshold": above,
                "method": "bisection",
                "threshold": float(threshold),
                **threshold_details,
            })
            return mean_ll, sd_ll, above
        
        self.adjustment_results = []
        best_nc = bisection_search(
            lambda nc, step: eval_nc(nc, step=step)[2],
            float(optuna_nc),
            bracket_factor=self.conservative_bracket_factor,
            min_norm_constraint=self.conservative_min_norm_constraint,
            rel_tol=self.conservative_step_pct,
            max_steps=self.conservative_max_steps,
        )
        self.conservative_params = {"norm_constraint": float(best_nc), "basis_order": int(basis_order)}
        return self.conservative_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        return self._evaluate(params)
    
    def optimize(self, n_trials: int = 50) -> TuningResult:
        """Run Optuna optimization with optional conservative adjustment.
        
        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
            
        Returns
        -------
        TuningResult
            Tuple of (estimator, best_params, metadata).
        """
        from tqdm import tqdm
        
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        
        if not self.silent:
            progress = tqdm(total=n_trials, desc="Init Tuner CV", unit="trial")
            best_metric = {"value": float("-inf")}
            
            def update_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                trial_val = trial.value if trial.value is not None else float("inf")
                metric = -float(trial_val) if np.isfinite(trial_val) else float("-inf")
                if metric > best_metric["value"]:
                    best_metric["value"] = metric
                progress.update(1)
                progress.set_postfix({
                    "LL": f"{metric:.2f}",
                    "best": f"{best_metric['value']:.2f}",
                })
            
            try:
                self.study.optimize(
                    self._objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                    callbacks=[update_progress],
                )
            finally:
                progress.close()
        else:
            self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)
        
        self.optuna_params = self.study.best_params.copy()
        self.best_metric_value = -self.study.best_value
        
        if self.use_conservative_adjustment:
            self._apply_conservative_adjustment()
            self.best_params = self.conservative_params
        else:
            self.conservative_params = None
            self.adjustment_results = None
            self.best_params = self.optuna_params
        
        # Fit final estimator - best_params is guaranteed non-None at this point
        assert self.best_params is not None
        estimator = self._fit_final_estimator(self.best_params)
        
        metadata = {
            "optuna_params": self.optuna_params,
            "conservative_params": self.conservative_params,
            "best_metric_value": self.best_metric_value,
            "conservative_selection_rule": self.conservative_selection_rule,
            "conservative_se_multiplier": self.conservative_se_multiplier,
            "study": self.study,
            "adjustment_results": self.adjustment_results,
        }
        
        return TuningResult(
            estimator=estimator,
            best_params=self.best_params,
            metadata=metadata,
        )
