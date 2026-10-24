"""Deprecated import path. Use haldensity.rc; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.rc")

from haldensity.rc import (  # noqa: E402
    KaplanMeier,
    RightCensoredEMEstimator,
    RightCensoredEMStage,
    RightCensoredHALFit,
    RightCensoredInitEstimator,
    RightCensoredObservedFISTAEstimator,
    RightCensoredObservedFPGDEstimator,
    RightCensoredObservedL1MLE,
    RightCensoredObservedProjectedGDEstimator,
    build_right_censored_knot_grid,
    compute_ipcw_weights,
    incomplete_loglik,
    ipcw_loglik,
    mi_complete_loglik,
    right_censored_observed_loglik_and_gradient,
)

__all__ = [
    "KaplanMeier",
    "RightCensoredEMEstimator",
    "RightCensoredEMStage",
    "RightCensoredHALFit",
    "RightCensoredInitEstimator",
    "RightCensoredObservedFISTAEstimator",
    "RightCensoredObservedFPGDEstimator",
    "RightCensoredObservedL1MLE",
    "RightCensoredObservedProjectedGDEstimator",
    "build_right_censored_knot_grid",
    "compute_ipcw_weights",
    "incomplete_loglik",
    "ipcw_loglik",
    "mi_complete_loglik",
    "right_censored_observed_loglik_and_gradient",
]
