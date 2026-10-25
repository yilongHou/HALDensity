"""The dense (theta) problem holds no n x p data basis: its data term is -sum_i phi(x_i)' theta from prefix sums."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.core.univariate.objective import data_basis_sums, dense_problem, normalizer_terms

CASES = [(0, "midpoint"), (1, "midpoint"), (2, "midpoint"), (3, "midpoint"), (0, "knot_aware"), (1, "knot_aware"),
         (2, "knot_aware")]


def _x(n=400, seed=0):
    return np.random.default_rng(seed).beta(2.0, 5.0, size=n)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_data_sums_are_the_basis_column_sums(k):
    x = _x()
    knots = np.unique(x)
    basis = create_basis_functions(pd.DataFrame({"W1": x}), knots, order=k)[0]
    # the basis is float64 (float32 before 0.2.0, which differed by about 1e-7 relative)
    assert basis.dtype == np.float64
    np.testing.assert_allclose(data_basis_sums(x, knots, k), basis.sum(axis=0), rtol=1e-12, atol=1e-12)
    exact = np.array([np.sum(np.maximum(x - xi, 0.0) ** k) if k else np.sum(x >= xi) for xi in knots[:20]])
    np.testing.assert_allclose(data_basis_sums(x, knots, k)[1 + k:21 + k], exact, rtol=1e-13)


@pytest.mark.parametrize("k,normalizer", CASES)
def test_objective_is_exact_at_random_theta(k, normalizer):
    x = _x()
    knots = np.unique(x)
    dense = dense_problem(x, knots, k, 200, normalizer, 10.0)
    theta = np.random.default_rng(k).normal(scale=0.3, size=(1 if k == 0 else k + 1) + knots.size)
    variable = dense.problem.variables()[0]
    variable.value = theta
    terms = normalizer_terms(x, knots, k, 200, normalizer, build_basis=False)
    reference = (-log_unnormalized_density(x, theta, knots, k).sum()
                 + x.size * logsumexp(terms.log_widths + log_unnormalized_density(terms.points, theta, knots, k)))
    assert dense.problem.objective.value == pytest.approx(reference, rel=1e-12)


def test_no_matrix_has_a_row_per_observation():
    x = _x(n=3000)
    knots = np.unique(x)
    dense = dense_problem(x, knots, 2, 200, "midpoint", 10.0)
    rows = [c.shape[0] for c in dense.problem.constants() if len(c.shape) == 2]
    assert rows and max(rows) == 199                       # the 199 midpoint cells only
