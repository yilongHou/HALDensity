import numpy as np
import pandas as pd

from haldensity.density_variance.density_variance import (
    density_confidence_interval,
    estimate_covariance_beta,
)
from haldensity.estimation import CVXPYEstimator


def test_delta_method_confidence_interval_runs_end_to_end():
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})
    estimator = CVXPYEstimator(
        norm_constraint=5.0, basis_order=1, log_frequency=-1, solver="ECOS", use_secondary_solver=False
    ).fit(data)
    results = estimator.get_results()

    cov = estimate_covariance_beta(data, results, basis_order=1)
    ci = density_confidence_interval(np.linspace(0.05, 0.95, 5), results, cov, basis_order=1)

    assert cov.shape[0] == cov.shape[1]
    assert np.all(np.isfinite(ci[["density", "se", "lower", "upper"]].to_numpy()))
    assert np.all(ci["se"] >= 0)
    assert np.all((ci["lower"] <= ci["density"]) & (ci["density"] <= ci["upper"]))
