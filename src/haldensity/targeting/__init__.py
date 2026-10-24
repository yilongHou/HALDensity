"""Deprecated import path. Use haldensity.full_uni.targeting and haldensity.rc.targeting; this module is removed in HALDensity 0.3.0."""

from haldensity._compat import warn_moved

warn_moved(__name__, "haldensity.full_uni.targeting and haldensity.rc.targeting")

from haldensity.full_uni.targeting import (  # noqa: E402
    BaseTargetLearner,
    CDFTargetLearner,
    MedianTargetLearner,
    MomentsTargetLearner,
    SurvivalTargetLearner,
    cdf_estimand_variance,
    cdf_targeting_M_step,
    median_estimand_variance,
    median_targeting_M_step,
    moments_estimand_variance,
    moments_targeting_M_step,
    survival_estimand_variance,
    survival_targeting_M_step,
)
from haldensity.rc.targeting import (  # noqa: E402
    RightCensoredDensityL2TargetLearner,
    RightCensoredEntropyTargetLearner,
    RightCensoredMeanTargetLearner,
    RightCensoredRMSTTargetLearner,
    RightCensoredSurvivalTargetLearner,
    right_censored_density_l2_estimand_variance,
    right_censored_density_l2_targeting_M_step,
    right_censored_entropy_estimand_variance,
    right_censored_entropy_targeting_M_step,
    right_censored_mean_estimand_variance,
    right_censored_mean_targeting_M_step,
    right_censored_rmst_estimand_variance,
    right_censored_rmst_targeting_M_step,
    right_censored_survival_estimand_variance,
    right_censored_survival_targeting_M_step,
    right_censored_survival_targeting_M_step_v2,
)
from haldensity.rc import (  # noqa: E402
    RightCensoredSurvivalAIPWLearner,
    right_censored_survival_aipw,
    right_censored_survival_aipw_estimand_variance,
)

__all__ = [
    "BaseTargetLearner",
    "CDFTargetLearner",
    "MedianTargetLearner",
    "MomentsTargetLearner",
    "RightCensoredDensityL2TargetLearner",
    "RightCensoredEntropyTargetLearner",
    "RightCensoredMeanTargetLearner",
    "RightCensoredRMSTTargetLearner",
    "RightCensoredSurvivalAIPWLearner",
    "RightCensoredSurvivalTargetLearner",
    "SurvivalTargetLearner",
    "cdf_estimand_variance",
    "cdf_targeting_M_step",
    "median_estimand_variance",
    "median_targeting_M_step",
    "moments_estimand_variance",
    "moments_targeting_M_step",
    "right_censored_density_l2_estimand_variance",
    "right_censored_density_l2_targeting_M_step",
    "right_censored_entropy_estimand_variance",
    "right_censored_entropy_targeting_M_step",
    "right_censored_mean_estimand_variance",
    "right_censored_mean_targeting_M_step",
    "right_censored_rmst_estimand_variance",
    "right_censored_rmst_targeting_M_step",
    "right_censored_survival_aipw",
    "right_censored_survival_aipw_estimand_variance",
    "right_censored_survival_estimand_variance",
    "right_censored_survival_targeting_M_step",
    "right_censored_survival_targeting_M_step_v2",
    "survival_estimand_variance",
    "survival_targeting_M_step",
]
