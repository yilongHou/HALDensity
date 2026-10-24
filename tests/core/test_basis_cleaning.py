import numpy as np
import pandas as pd
import pytest

from haldensity.core.univariate.support import (
    FullDataHALProblem,
    kkt_report,
    refit_support,
    refit_support_from_estimator,
)
from haldensity.core.variance import density_confidence_interval, estimate_covariance_beta
from haldensity.full_uni import CVXPYEstimator

POINTS = np.linspace(0.05, 0.95, 7)


@pytest.fixture(scope="module")
def data():
    return pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})


@pytest.fixture(scope="module")
def fit(data):
    # a loosely solved fit (v0.1.0 solver settings): the SE basis comes from a separate refit
    return CVXPYEstimator(norm_constraint=20.0, basis_order=1, log_frequency=-1, solver_tolerance="default").fit(data)


@pytest.fixture(scope="module")
def refit(fit, data):
    return refit_support_from_estimator(fit, data)


def test_refit_is_certified_and_no_larger_than_the_solver_basis(fit, refit):
    original = int(np.count_nonzero(np.asarray(fit.theta_hat)[2:]))
    assert refit.certified and refit.kkt["n_kept_violating"] == 0
    assert refit.theta.shape == np.asarray(fit.theta_hat).shape
    assert 0 < refit.n_selected <= original
    assert np.abs(refit.theta_raw[1:]).sum() <= refit.norm_constraint * (1 + 1e-6)


def test_kkt_report_flags_a_kept_coefficient_off_the_active_set(data, refit):
    problem = FullDataHALProblem(data, basis_order=1)
    theta = refit.theta_raw.copy()
    inactive = 2 + int(np.flatnonzero(np.abs(theta[2:]) <= 1e-8)[0])
    theta[inactive] = 1e-3                     # residue above the 1e-4 pruning cutoff
    report = kkt_report(problem, theta)
    assert report["n_kept_violating"] >= 1 and not report["certified"]


def test_cleaned_se_keeps_the_point_estimate(data, fit, refit):
    results = fit.get_results()
    original = density_confidence_interval(
        POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=None), 1, support_refit=None
    )
    cleaned = density_confidence_interval(
        POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=refit), 1, support_refit=refit
    )
    np.testing.assert_array_equal(cleaned["density"], original["density"])
    assert np.all(np.isfinite(cleaned["se"])) and np.all(cleaned["se"] > 0)
    assert cleaned.attrs["basis"] == "cleaned" and cleaned.attrs["basis_certified"]
    assert original.attrs["basis"] == "original"


def test_cleaning_is_the_default_and_can_be_switched_off(data, fit, refit):
    results = fit.get_results()
    default = density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1), 1)
    explicit = density_confidence_interval(
        POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=refit), 1, support_refit=refit
    )
    assert default.attrs["basis"] == "cleaned"
    np.testing.assert_allclose(default["se"], explicit["se"], rtol=1e-10)

    # a covariance on the original basis gets the original-basis SE, not a mismatched one
    off = density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=None), 1)
    assert off.attrs["basis"] == "original"


def test_fits_without_an_l1_bound_use_the_original_basis(data, fit):
    # e.g. a penalized estimator: no L1 bound and no certified solution carried with the results
    results = {k: v for k, v in fit.get_results().items() if k not in ("norm_constraint", "support_refit")}
    ci = density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1), 1)
    assert ci.attrs["basis"] == "original"


def test_mismatched_inputs_are_rejected(data, fit, refit):
    results = fit.get_results()
    with pytest.raises(ValueError, match="compute it with the same support_refit"):
        density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=None), 1,
                                    support_refit=refit)
    other = pd.DataFrame({"W1": np.random.default_rng(5).beta(2.0, 5.0, size=200)})
    with pytest.raises(ValueError, match="do not match the estimator's knots"):
        refit_support_from_estimator(fit, other)


def test_k0_refit_runs(data):
    refit = refit_support(data, basis_order=0, norm_constraint=5.0)
    assert refit.basis_order == 0 and refit.theta.size == 1 + np.unique(data["W1"]).size


def test_covariance_carries_its_basis(data, fit):
    results = fit.get_results()
    cleaned_cov = estimate_covariance_beta(data, results, 1)
    original_cov = estimate_covariance_beta(data, results, 1, support_refit=None)
    assert cleaned_cov.support_refit is not None and original_cov.support_refit is None
    # even if both bases had the same size, the pairing follows the covariance, not the size
    assert density_confidence_interval(POINTS, results, original_cov, 1).attrs["basis"] == "original"
    assert density_confidence_interval(POINTS, results, cleaned_cov, 1).attrs["basis"] == "cleaned"
    assert density_confidence_interval(POINTS, results, np.asarray(original_cov), 1).attrs["basis"] == "original"


# ----- proposed repair of an uncertified final fit (repair_support; off by default) -----

def _with_residue(refit):
    theta = refit.theta_raw.copy()
    inactive = 2 + int(np.flatnonzero(np.abs(theta[2:]) <= 1e-8)[0])
    theta[inactive] = 1e-3                     # a kept coefficient the optimality conditions say is zero
    return theta, inactive


def test_repair_by_zeroing_certifies_a_small_residue(data, refit):
    from haldensity.core.univariate.support import repair_support
    problem = FullDataHALProblem(data, basis_order=1)
    theta, inactive = _with_residue(refit)
    fixed = repair_support(problem, theta, refit.norm_constraint, polish=False)
    assert fixed["applied"] and fixed["certified"] and fixed["theta_raw"][inactive] == 0.0
    assert fixed["removed"] == 1 and fixed["removed_l1"] == pytest.approx(1e-3)


def test_repair_by_polishing_recovers_the_certified_solution(data, refit):
    from haldensity.core.univariate.support import repair_support
    problem = FullDataHALProblem(data, basis_order=1)
    theta, inactive = _with_residue(refit)
    fixed = repair_support(problem, theta, refit.norm_constraint, polish=True)
    assert fixed["certified"] and fixed["rounds"] == 1 and fixed["theta_raw"][inactive] == 0.0
    np.testing.assert_allclose(problem.objective(fixed["theta_raw"]), problem.objective(refit.theta_raw), rtol=1e-8)


def test_zeroing_is_limited_to_small_violations_polishing_is_not(data, refit):
    from haldensity.core.univariate.support import repair_support
    problem = FullDataHALProblem(data, basis_order=1)
    theta = refit.theta_raw.copy()
    inactive = 2 + np.flatnonzero(np.abs(theta[2:]) <= 1e-8)[:5]
    theta[inactive] = 0.5                      # 2.5 of the L1 norm 20 sits on coefficients that should be zero
    zeroed = repair_support(problem, theta, refit.norm_constraint, polish=False)
    assert not zeroed["applied"] and not zeroed["certified"] and "L1 norm" in zeroed["reason"]
    # polishing re-solves exactly on the near-active support, so the size removed does not matter
    polished = repair_support(problem, theta, refit.norm_constraint, polish=True)
    assert polished["certified"] and polished["removed"] == 5 and polished["removed_share"] > 0.1


def test_certified_solve_repairs_only_when_asked(monkeypatch, data, refit):
    import haldensity.core.univariate.support as support
    from haldensity.core.univariate.support import certified_solve
    real = support.FullDataHALProblem.build
    residue, _ = _with_residue(refit)

    def with_residue(self, norm_constraint, dense=False):    # every solve returns a solution with a residue
        built = real(self, norm_constraint, dense=dense)
        built.theta = lambda: residue.copy()
        return built

    monkeypatch.setattr(support.FullDataHALProblem, "build", with_residue)
    tight = support.LADDER[1:2]                                # CLARABEL 1e-12
    problem = FullDataHALProblem(data, basis_order=1)
    assert not certified_solve(problem, refit.norm_constraint, tight).certified
    repaired = certified_solve(problem, refit.norm_constraint, tight, repair="zero")
    assert repaired.certified and repaired.option.endswith("repair zero") and repaired.attempts[-1]["removed"] == 1
    with pytest.raises(ValueError, match="repair"):
        certified_solve(problem, refit.norm_constraint, tight, repair="yes")
