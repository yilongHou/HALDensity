import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.univariate import BaseEstimator
from haldensity.core.univariate.normalizer import log_normalizer, log_unnormalized_density
from haldensity.full_uni import CVXPYEstimator


def _quad_log_z(theta, knots, k, active):
    edges = np.unique(np.concatenate(([0.0], active, [1.0])))
    f = lambda u: float(np.exp(log_unnormalized_density(np.array([u]), theta, knots, k)[0]))
    return float(np.log(sum(quad(f, a, b, epsabs=1e-14, epsrel=1e-13, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))))


def test_unnormalized_density_matches_the_basis():
    knots = np.array([0.2, 0.5, 0.7])
    x = np.linspace(0.0, 1.0, 41)
    for k in (0, 1, 2):
        theta = np.random.default_rng(k).normal(size=(1 if k == 0 else k + 1) + knots.size)
        basis, _ = create_basis_functions(pd.DataFrame({"W1": x}), knots, order=k)
        np.testing.assert_allclose(log_unnormalized_density(x, theta, knots, k), basis.astype(float) @ theta, atol=1e-6)


def test_k0_normalizer_is_the_exact_piecewise_constant_integral():
    knots = np.array([0.3, 0.30001, 0.8])          # a narrow cell between two close knots
    theta = np.array([0.0, 12.0, -12.0, 1.0])       # a tall spike on [0.3, 0.30001)
    widths = np.array([0.3, 0.00001, 0.49999, 0.2])
    levels = np.array([0.0, 12.0, 0.0, 1.0])
    expected = np.log(np.sum(widths * np.exp(levels)))
    result = log_normalizer(theta, knots, 0)
    assert result.converged and result.n_cells == 4
    assert result.log_z == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("k", [1, 2])
def test_gauss_legendre_matches_adaptive_quadrature(k):
    rng = np.random.default_rng(10 + k)
    knots = np.sort(rng.uniform(0.0, 1.0, size=15))
    theta = rng.normal(scale=3.0, size=k + 1 + knots.size)
    theta[k + 1 + 3] = 0.0                           # an inactive knot is not a breakpoint
    result = log_normalizer(theta, knots, k)
    assert result.converged
    assert result.log_z == pytest.approx(_quad_log_z(theta, knots, k, knots[theta[k + 1:] != 0]), abs=1e-10)


def test_estimator_density_integrates_to_one_and_midpoint_option_keeps_old_values():
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=200)})
    fit = CVXPYEstimator(norm_constraint=50.0, basis_order=1, log_frequency=-1, solver="ECOS",
                         use_secondary_solver=False).fit(data)
    assert fit.normalizer == "gauss_legendre"
    theta, knots = np.asarray(fit.theta_hat, float), np.asarray(fit._grid_points_hal, float)
    active = knots[theta[2:] != 0]
    assert fit.log_normalizer().log_z == pytest.approx(_quad_log_z(theta, knots, 1, active), abs=1e-10)

    points = np.linspace(0.01, 0.99, 9)
    np.testing.assert_allclose(fit.get_log_density_at_points(points), np.log(fit.get_density_at_points(points)))

    fit.normalizer = "midpoint"
    old = BaseEstimator.calculate_density_at_points(
        points=points, theta_hat=fit.theta_hat, basis_grid_points=fit._grid_points_hal, basis_order=1,
        norm_grid_midpoints=fit.grid_midpoints, norm_delta_j=fit.delta_j,
    )
    np.testing.assert_array_equal(fit.get_density_at_points(points), old)


def test_unknown_normalizer_is_rejected():
    with pytest.raises(ValueError, match="normalizer must be one of"):
        CVXPYEstimator(normalizer="trapezoid")
