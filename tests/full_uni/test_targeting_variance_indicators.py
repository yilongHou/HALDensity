"""The influence curves of S(t) = P(X > t) and F(t) = P(X <= t) at a tie x = t (review T2) and a list of points (T4)."""
import numpy as np
import pandas as pd

from haldensity.full_uni.targeting import cdf_estimand_variance, survival_estimand_variance

X = np.array([0.1, 0.3, 0.3, 0.5, 0.7])     # two observations exactly at t = 0.3


def test_survival_variance_uses_x_greater_than_t():
    fit = {"estimate": np.array([0.6])}
    variance = survival_estimand_variance(targeted_fit=fit, uncensored_data=pd.DataFrame({"W1": X}), targeting_points=[0.3])
    expected = np.var((X > 0.3).astype(float) - 0.6, ddof=1) / X.size
    np.testing.assert_allclose(variance, [expected], rtol=1e-14)


def test_cdf_variance_uses_x_at_most_t():
    fit = {"estimate": np.array([0.4])}
    variance = cdf_estimand_variance(targeted_fit=fit, uncensored_data=pd.DataFrame({"W1": X}), targeting_points=[0.3])
    expected = np.var((X <= 0.3).astype(float) - 0.4, ddof=1) / X.size
    np.testing.assert_allclose(variance, [expected], rtol=1e-14)
