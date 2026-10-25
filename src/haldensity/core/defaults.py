"""Default configuration models and result containers for censored data density estimation.

Contains:
- EMDefaults: Default parameters for EM algorithm
- TunerDefaults: Default parameters for hyperparameter tuners
- EMStageResult: Result container for EM stage outputs
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


# Package default basis order of the full-data univariate HAL-MLE (Yilong, 2026-10-24): fixed k = 2,
# CV over the order on request. At n = 320 the package's joint CV over {0, 1, 2} was never better than
# fixed k = 2 (KL of the joint CV relative to fixed k = 2: 0.96, 1.02, 1.16 and 1.34 on the jump,
# slope-kink, curvature-kink and smooth truths), and an order higher than the truth's costs little.
DEFAULT_BASIS_ORDER = 2

# Fitting grid of the full-data univariate HAL-MLE: n_grid_points - 1 equal cells of [0, 1] (public; whether
# the data are too concentrated for it is the user's call, Yilong 2026-10-24). Every fit records the
# missed-mass diagnostic, the accurate log Z minus the objective's log Z, and warns above
# MISSED_MASS_WARNING_NATS, suggesting about MISSED_MASS_CELLS_ACROSS_DATA cells across the data range.
DEFAULT_N_GRID_POINTS = 200
MISSED_MASS_WARNING_NATS = 0.01
MISSED_MASS_CELLS_ACROSS_DATA = 200


class EMDefaults(BaseModel):
    """Default parameters for EM algorithm.

    These defaults are shared across RightCensoredEMStage, EMIPCWEstimator, and tuners.
    """

    model_config = ConfigDict(frozen=True)

    tol: float = 1e-4
    m_imputations: int = 100
    max_em_iter: int = 10
    em_tol: float = 1e-1
    e_step_n_grid: int = 200
    use_sc_adjustment: bool = False
    m_step_solver: str = "MOSEK"
    init_solver: str = "MOSEK"


class TunerDefaults(BaseModel):
    """Default parameters for hyperparameter tuners.

    These defaults are shared across all Optuna-based tuners.
    """

    model_config = ConfigDict(frozen=True)

    cv_folds: int = 5
    random_state: int = 42
    n_grid_points: int = 200
    solver: str = "MOSEK"
    use_secondary_solver: bool = True

    # Conservative adjustment settings (to reduce CV oversmoothing)
    # Finds smallest norm_constraint where CV_LL >= max_CV_LL - k% * SD
    use_conservative_adjustment: bool = True
    conservative_k_percent: float = 0.05  # 5% of SD as threshold
    conservative_max_steps: int = 50  # Maximum search steps
    conservative_step_pct: float = 0.02  # 2% step size per step
    conservative_method: str = "bisection"  # {"bisection", "linear"}
    conservative_selection_rule: str = "legacy_percent_sd"  # {"legacy_percent_sd", "one_se"}
    conservative_se_multiplier: float = 1.0  # used when conservative_selection_rule == "one_se"
    conservative_bracket_factor: float = 0.5  # geometric shrink factor during bracketing
    conservative_min_norm_constraint: float = 1e-6  # floor to avoid invalid/zero constraints


# Default instances for convenience
EM_DEFAULTS = EMDefaults()
TUNER_DEFAULTS = TunerDefaults()


class EMStageResult:
    """Container for EM stage results.

    Used by both RightCensoredEMStage and IntervalCensoredEMStage.

    Attributes
    ----------
    final_estimator : Any
        The fitted estimator after EM refinement.
    theta_path : list[np.ndarray]
        History of theta values across EM iterations.
    em_iterations : int
        Number of EM iterations performed.
    em_converged : bool
        Whether EM converged before reaching max iterations.
    final_augmented_data : pd.DataFrame | None
        The final augmented (imputed) dataset from the last E-step.
    """

    def __init__(
        self,
        final_estimator,
        theta_path: list[np.ndarray],
        em_iterations: int,
        em_converged: bool,
        final_augmented_data: Optional[pd.DataFrame] = None,
    ):
        self.final_estimator = final_estimator
        self.theta_path = theta_path
        self.em_iterations = em_iterations
        self.em_converged = em_converged
        self.final_augmented_data = final_augmented_data

    def __repr__(self) -> str:
        return (
            f"EMStageResult(em_iterations={self.em_iterations}, "
            f"em_converged={self.em_converged}, "
            f"theta_path_len={len(self.theta_path)})"
        )


# Backward compatibility alias (will be removed in future)
RightCensoredEMStageResult = EMStageResult
