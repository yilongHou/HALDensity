"""Targeted learning for univariate full data: re-fit with added targeting basis functions."""

from haldensity.full_uni.targeting.base import BaseTargetLearner, TargetingResults, targeting_inputs
from haldensity.full_uni.targeting.cdf import (
    CDFTargetLearner,
    cdf_estimand_variance,
    cdf_targeting_M_step,
)
from haldensity.full_uni.targeting.density_functionals import (
    DensityL2TargetLearner,
    EntropyTargetLearner,
    density_l2_estimand_variance,
    density_l2_targeting_M_step,
    entropy_estimand_variance,
    entropy_targeting_M_step,
)
from haldensity.full_uni.targeting.median import (
    MedianTargetLearner,
    median_estimand_variance,
    median_targeting_M_step,
)
from haldensity.full_uni.targeting.moments import (
    MomentsTargetLearner,
    moments_estimand_variance,
    moments_targeting_M_step,
)
from haldensity.full_uni.targeting.survival import (
    SurvivalTargetLearner,
    survival_estimand_variance,
    survival_targeting_M_step,
)

__all__ = [
    "BaseTargetLearner",
    "TargetingResults",
    "targeting_inputs",
    "CDFTargetLearner",
    "DensityL2TargetLearner",
    "EntropyTargetLearner",
    "cdf_estimand_variance",
    "cdf_targeting_M_step",
    "density_l2_estimand_variance",
    "density_l2_targeting_M_step",
    "entropy_estimand_variance",
    "entropy_targeting_M_step",
    "MedianTargetLearner",
    "median_estimand_variance",
    "median_targeting_M_step",
    "MomentsTargetLearner",
    "moments_estimand_variance",
    "moments_targeting_M_step",
    "SurvivalTargetLearner",
    "survival_estimand_variance",
    "survival_targeting_M_step",
]
