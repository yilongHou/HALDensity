"""Pre-0.2.0 import paths still work, re-export the same objects, and warn (removed in 0.3.0)."""

import importlib
import sys

import pytest

import haldensity

OLD_PUBLIC_NAMES = {
    'haldensity.censoring': ['EMStageResult', 'IntervalCensoredCVOversmoothEMTuner', 'IntervalCensoredEMEstimator', 'IntervalCensoredEMStage', 'IntervalCensoredEMTuner', 'IntervalCensoredFISTAEstimator', 'IntervalCensoredFISTATuner', 'IntervalCensoredInitEstimator', 'IntervalCensoredInitTuner', 'IntervalCensoredJointTuner', 'IntervalCensoredProjectedGDEstimator', 'IntervalCensoredProjectedGDTuner', 'KaplanMeier', 'RightCensoredCVOversmoothEMTuner', 'RightCensoredEMEstimator', 'RightCensoredEMStage', 'RightCensoredEMTuner', 'RightCensoredInitEstimator', 'RightCensoredInitTuner', 'RightCensoredJointTuner', 'RightCensoredObservedFISTAEstimator', 'RightCensoredObservedFISTATuner', 'RightCensoredObservedFPGDEstimator', 'RightCensoredObservedFPGDTuner', 'RightCensoredObservedL1MLE', 'RightCensoredObservedL1Tuner', 'RightCensoredObservedProjectedGDEstimator', 'TuningResult', 'WeightedHALMLEEstimator', 'build_right_censored_knot_grid', 'compute_ipcw_weights', 'incomplete_loglik', 'incomplete_loglik_interval', 'interval', 'interval_censor', 'interval_censor_current_status_uniform', 'interval_censor_inspection_uniform', 'ipcw_loglik', 'kl_divergence', 'mi_complete_loglik', 'right', 'right_censored_observed_loglik_and_gradient', 'tuners', 'utils'],
    'haldensity.censoring.right': ['KaplanMeier', 'RightCensoredEMEstimator', 'RightCensoredEMStage', 'RightCensoredHALFit', 'RightCensoredInitEstimator', 'RightCensoredObservedFISTAEstimator', 'RightCensoredObservedFPGDEstimator', 'RightCensoredObservedL1MLE', 'RightCensoredObservedProjectedGDEstimator', 'build_right_censored_knot_grid', 'compute_ipcw_weights', 'incomplete_loglik', 'ipcw_loglik', 'mi_complete_loglik', 'right_censored_observed_loglik_and_gradient'],
    'haldensity.censoring.interval': ['IntervalCensoredEMEstimator', 'IntervalCensoredEMStage', 'IntervalCensoredFISTAEstimator', 'IntervalCensoredInitEstimator', 'IntervalCensoredProjectedGDEstimator', 'incomplete_loglik_interval', 'interval_censor', 'interval_censor_current_status_uniform', 'interval_censor_inspection_uniform'],
    'haldensity.censoring.tuners': ['CVOversmoothRecord', 'IntervalCensoredCVOversmoothEMTuner', 'IntervalCensoredEMTuner', 'IntervalCensoredFISTATuner', 'IntervalCensoredInitTuner', 'IntervalCensoredJointTuner', 'IntervalCensoredProjectedGDTuner', 'OverSmoothEMRecord', 'OverSmoothInitRecord', 'RightCensoredCVOversmoothEMTuner', 'RightCensoredEMTuner', 'RightCensoredInitTuner', 'RightCensoredJointTuner', 'RightCensoredObservedFISTATuner', 'RightCensoredObservedFPGDTuner', 'RightCensoredObservedL1Tuner', 'TuningResult'],
    'haldensity.censoring.utils': ['kl_divergence'],
    'haldensity.estimation': ['CVXPYEstimator', 'FISTAEstimator', 'ProjectedGDEstimator', 'ProximalAdaGradEstimator', 'ProximalGDEstimator', 'ProximalNewtonEstimator', 'ProximalNewtonLBFGSEstimator', 'ProximalNewtonLBFGSFullEstimator'],
    'haldensity.cross_validation': ['OptunaHyperparameterTuner'],
    'haldensity.targeting': ['BaseTargetLearner', 'CDFTargetLearner', 'MedianTargetLearner', 'MomentsTargetLearner', 'RightCensoredDensityL2TargetLearner', 'RightCensoredEntropyTargetLearner', 'RightCensoredMeanTargetLearner', 'RightCensoredRMSTTargetLearner', 'RightCensoredSurvivalAIPWLearner', 'RightCensoredSurvivalTargetLearner', 'SurvivalTargetLearner', 'cdf_estimand_variance', 'cdf_targeting_M_step', 'median_estimand_variance', 'median_targeting_M_step', 'moments_estimand_variance', 'moments_targeting_M_step', 'right_censored_density_l2_estimand_variance', 'right_censored_density_l2_targeting_M_step', 'right_censored_entropy_estimand_variance', 'right_censored_entropy_targeting_M_step', 'right_censored_mean_estimand_variance', 'right_censored_mean_targeting_M_step', 'right_censored_rmst_estimand_variance', 'right_censored_rmst_targeting_M_step', 'right_censored_survival_aipw', 'right_censored_survival_aipw_estimand_variance', 'right_censored_survival_estimand_variance', 'right_censored_survival_targeting_M_step', 'right_censored_survival_targeting_M_step_v2', 'survival_estimand_variance', 'survival_targeting_M_step'],
    'haldensity.multivariate': ['BivariateTruncatedNormal', 'MultvarHAL', 'build_exact_k0_normalizer_bank', 'build_midpoint_quadrature', 'build_sobol_normalizer_bank', 'coerce_multivariate_data', 'create_multivariate_basis', 'default_sobol_budget', 'exact_k0_cell_budget', 'expected_section_counts', 'get_basis_index', 'summarize_basis_catalog', 'summarize_design_matrix', 'theoretical_basis_count'],
    'haldensity.utils': ['BivariateTruncatedNormal', 'Sinusoidal', 'StepFunction', 'TruncatedGMM', 'TruncatedNormal', 'create_basis_functions', 'create_multivariate_basis'],
    'haldensity.utils.dgp': ['BivariateTruncatedNormal', 'Sinusoidal', 'StepFunction', 'TruncatedGMM', 'TruncatedNormal'],
    'haldensity.density_variance.density_variance': ['create_basis_functions', 'density_confidence_interval', 'estimate_covariance_beta'],
}


def _forget(prefix: str) -> None:
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


@pytest.mark.parametrize("old_path", sorted(OLD_PUBLIC_NAMES))
def test_old_path_reexports_public_names_with_a_deprecation_warning(old_path):
    _forget(".".join(old_path.split(".")[:2]))
    with pytest.warns(DeprecationWarning, match="removed in 0.3.0"):
        module = importlib.import_module(old_path)
        for name in OLD_PUBLIC_NAMES[old_path]:
            assert hasattr(module, name), f"{old_path}.{name}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_reexports_are_the_new_objects():
    from haldensity.censoring import RightCensoredInitEstimator as old_rc
    from haldensity.estimation import CVXPYEstimator as old_cvx
    from haldensity.targeting import right_censored_survival_aipw as old_aipw
    from haldensity.full_uni import CVXPYEstimator
    from haldensity.rc import RightCensoredInitEstimator, right_censored_survival_aipw

    assert old_cvx is CVXPYEstimator
    assert old_rc is RightCensoredInitEstimator
    assert old_aipw is right_censored_survival_aipw


def test_old_subpackages_reachable_as_attributes():
    with pytest.warns(DeprecationWarning):
        assert haldensity.censoring.right.KaplanMeier is haldensity.rc.KaplanMeier
