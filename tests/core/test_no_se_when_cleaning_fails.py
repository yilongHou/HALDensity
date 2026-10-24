"""Decision 5 (Yilong, 2026-10-24): if basis cleaning fails, the point estimate is kept but no SE or interval."""

import warnings

import numpy as np
import pandas as pd
import pytest

import haldensity.core.univariate.support as support
from haldensity.core.variance import density_confidence_interval, estimate_covariance_beta
from haldensity.full_uni import CVXPYEstimator

POINTS = np.linspace(0.05, 0.95, 7)
FREE = dict(solver="CLARABEL", solver_waterfall=["CLARABEL", "ECOS"], log_frequency=-1)


@pytest.fixture
def data():
    return pd.DataFrame({"W1": np.random.default_rng(4).beta(2.0, 5.0, size=200)})


@pytest.fixture
def cleaning_fails(monkeypatch):
    real = support.kkt_report
    monkeypatch.setattr(support, "kkt_report",
                        lambda *a, **kw: {**real(*a, **kw), "certified": False, "n_failing": 1})


@pytest.mark.parametrize("tolerance", ["tight", "default"])     # the fit's own cleaning, or a separate refit
def test_point_estimate_but_no_se(data, cleaning_fails, tolerance):
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, solver_tolerance=tolerance, **FREE).fit(data)
    if tolerance == "tight":
        assert any("Basis cleaning failed" in str(w.message) for w in fit_warnings) and fit.certified_ is False
    results = fit.get_results()
    with pytest.warns(UserWarning, match="bootstrap"):
        cov = estimate_covariance_beta(data, results, 1)
    assert np.all(np.isnan(cov))
    with pytest.warns(UserWarning, match="Basis cleaning failed"):
        ci = density_confidence_interval(POINTS, results, cov, 1)
    assert ci["density"].notna().all() and ci[["se", "lower", "upper"]].isna().all().all()
    assert ci.attrs["basis_certified"] is False
    # the point estimate is the fit's, and the original-basis SE is still there on request
    original = density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1, support_refit=None),
                                           1, support_refit=None)
    np.testing.assert_array_equal(ci["density"], original["density"])
    assert original["se"].notna().all()


def test_cleaned_fits_still_get_an_se(data):
    fit = CVXPYEstimator(norm_constraint=20.0, basis_order=1, **FREE).fit(data)
    results = fit.get_results()
    ci = density_confidence_interval(POINTS, results, estimate_covariance_beta(data, results, 1), 1)
    assert fit.certified_ and ci["se"].notna().all() and ci.attrs["basis_certified"]
