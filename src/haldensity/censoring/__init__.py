"""Deprecated import path. Use haldensity.rc and haldensity.ic; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.rc and haldensity.ic")

from haldensity.core.defaults import (  # noqa: E402
    EMStageResult,
)
from haldensity.ic import (  # noqa: E402
    IntervalCensoredCVOversmoothEMTuner,
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
    IntervalCensoredEMTuner,
    IntervalCensoredFISTAEstimator,
    IntervalCensoredFISTATuner,
    IntervalCensoredInitEstimator,
    IntervalCensoredInitTuner,
    IntervalCensoredJointTuner,
    IntervalCensoredProjectedGDEstimator,
    IntervalCensoredProjectedGDTuner,
    incomplete_loglik_interval,
)
from haldensity.rc import (  # noqa: E402
    KaplanMeier,
    RightCensoredCVOversmoothEMTuner,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
    RightCensoredEMTuner,
    RightCensoredInitEstimator,
    RightCensoredInitTuner,
    RightCensoredJointTuner,
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedFISTATuner,
    RightCensoredObservedFPGDEstimator,
    RightCensoredObservedFPGDTuner,
    RightCensoredObservedL1MLE,
    RightCensoredObservedL1Tuner,
    RightCensoredObservedProjectedGDEstimator,
    build_right_censored_knot_grid,
    compute_ipcw_weights,
    incomplete_loglik,
    ipcw_loglik,
    mi_complete_loglik,
    right_censored_observed_loglik_and_gradient,
)
from haldensity.core.cv import (  # noqa: E402
    TuningResult,
)
from haldensity.core import (  # noqa: E402
    WeightedHALMLEEstimator,
    kl_divergence,
)
from haldensity.datasets import (  # noqa: E402
    interval_censor,
    interval_censor_current_status_uniform,
    interval_censor_inspection_uniform,
)

__all__ = [
    "EMStageResult",
    "IntervalCensoredCVOversmoothEMTuner",
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    "IntervalCensoredEMTuner",
    "IntervalCensoredFISTAEstimator",
    "IntervalCensoredFISTATuner",
    "IntervalCensoredInitEstimator",
    "IntervalCensoredInitTuner",
    "IntervalCensoredJointTuner",
    "IntervalCensoredProjectedGDEstimator",
    "IntervalCensoredProjectedGDTuner",
    "KaplanMeier",
    "RightCensoredCVOversmoothEMTuner",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    "RightCensoredEMTuner",
    "RightCensoredInitEstimator",
    "RightCensoredInitTuner",
    "RightCensoredJointTuner",
    "RightCensoredObservedFISTAEstimator",
    "RightCensoredObservedFISTATuner",
    "RightCensoredObservedFPGDEstimator",
    "RightCensoredObservedFPGDTuner",
    "RightCensoredObservedL1MLE",
    "RightCensoredObservedL1Tuner",
    "RightCensoredObservedProjectedGDEstimator",
    "TuningResult",
    "WeightedHALMLEEstimator",
    "build_right_censored_knot_grid",
    "compute_ipcw_weights",
    "incomplete_loglik",
    "incomplete_loglik_interval",
    "interval_censor",
    "interval_censor_current_status_uniform",
    "interval_censor_inspection_uniform",
    "ipcw_loglik",
    "kl_divergence",
    "mi_complete_loglik",
    "right_censored_observed_loglik_and_gradient",
]


def __getattr__(name):
    if name in ("interval", "right", "tuners", "utils"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
