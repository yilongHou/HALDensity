"""Deprecated import path. Use haldensity.ic; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.ic")

from haldensity.ic import (  # noqa: E402
    IntervalCensoredEMEstimator,
    IntervalCensoredEMStage,
    IntervalCensoredFISTAEstimator,
    IntervalCensoredInitEstimator,
    IntervalCensoredProjectedGDEstimator,
    incomplete_loglik_interval,
)
from haldensity.datasets import (  # noqa: E402
    interval_censor,
    interval_censor_current_status_uniform,
    interval_censor_inspection_uniform,
)

__all__ = [
    "IntervalCensoredEMEstimator",
    "IntervalCensoredEMStage",
    "IntervalCensoredFISTAEstimator",
    "IntervalCensoredInitEstimator",
    "IntervalCensoredProjectedGDEstimator",
    "incomplete_loglik_interval",
    "interval_censor",
    "interval_censor_current_status_uniform",
    "interval_censor_inspection_uniform",
]
