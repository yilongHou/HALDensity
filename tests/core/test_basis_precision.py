"""Every basis and design matrix is built in float64 (float32 before 0.2.0)."""
import numpy as np
import pandas as pd
import pytest

from haldensity.core.basis import create_basis_functions, create_multivariate_basis
from haldensity.full_uni.targeting import cdf_basis, survival_basis


@pytest.mark.parametrize("k", [0, 1, 2])
def test_univariate_basis_is_float64_and_exact(k):
    x = np.random.default_rng(0).uniform(size=50)
    knots = np.sort(x)[::5]
    basis = create_basis_functions(pd.DataFrame({"W1": x}), knots, order=k)[0]
    assert basis.dtype == np.float64
    # a knot column equals its float64 formula exactly (float32 rounding was about 1e-7 relative)
    expected = (x >= knots[3]).astype(float) if k == 0 else np.maximum(x - knots[3], 0.0) ** k
    np.testing.assert_array_equal(basis[:, (1 if k == 0 else k + 1) + 3], expected)


@pytest.mark.parametrize("k", [0, 1])
def test_multivariate_basis_is_float64(k):
    data = np.random.default_rng(1).uniform(size=(12, 2))
    basis = create_multivariate_basis(data, k=k)[0]
    assert basis.dtype == np.float64


@pytest.mark.parametrize("module", [survival_basis, cdf_basis])
def test_targeting_bases_are_float64(module):
    data = pd.DataFrame({"W1": np.linspace(0.05, 0.95, 10)})
    assert module.create_targeting_basis_functions(data, [0.3, 0.6]).dtype == np.float64
