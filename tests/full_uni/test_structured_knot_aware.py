"""The structured (sparse) knot-aware formulation solves the same problem as the dense one.

Tolerances, with the largest differences seen over these cases (n = 200 and 500, k = 0 and 1, a
reasonable and a 100x too large M, a knot cap; CLARABEL as in CI, and MOSEK) and, in brackets, over a
wider check up to n = 2,000 with K = 60 and 250 (not run here: the dense fits take minutes):
- objective (evaluated identically in float64 at both solutions): relative 1e-6; seen 4.9e-7 [3.3e-6];
  the structured solution was never worse by more than 1.6e-9, the larger gaps are the dense solve's;
- normalized log-density on the objective's cells that carry >= 1e-8 of the mass: 5e-3; seen 1.7e-3
  [3.4e-3] (a cell with ~1e-11 of the mass is not identified by the objective and is excluded);
- total variation between the two cell-mass distributions: 1e-3; seen 4.6e-5 [8.5e-5];
- support where it is well identified (coefficients >= 5% of the largest): Jaccard >= 0.9; seen 0.96
  [0.67 for k = 1, n = 2,000: the dense solution splits a coefficient over data values < 1e-3 apart].
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from haldensity.core.basis.univariate import create_basis_functions
from haldensity.core.univariate.normalizer import log_unnormalized_density
from haldensity.core.univariate.objective import normalizer_terms, objective_gradient
from haldensity.full_uni import CVXPYEstimator

CASES = [
    (0, 200, 10.0, None), (0, 200, 1000.0, None), (0, 500, 10.0, None), (0, 500, 10.0, 60),
    (1, 200, 50.0, None), (1, 200, 5000.0, None), (1, 500, 50.0, None), (1, 500, 50.0, 60),
]


def _fit(data, k, M, K, formulation, solver_tolerance="default"):
    fit = CVXPYEstimator(norm_constraint=M, basis_order=k, log_frequency=-1, _objective_normalizer="knot_aware",
                         max_knots=K, solver_tolerance=solver_tolerance)
    fit._knot_aware_formulation = formulation
    return fit.fit(data)


def _objective(fit, theta, terms):
    x, knots, k = fit._train_x, fit._grid_points_hal, fit.basis_order
    cells = terms.log_density(theta, knots, k)
    return -log_unnormalized_density(x, theta, knots, k).sum() + x.size * logsumexp(terms.log_widths + cells), cells


@pytest.mark.parametrize("k,n,M,K", CASES)
def test_structured_and_dense_knot_aware_give_the_same_solution(k, n, M, K):
    data = pd.DataFrame({"W1": np.random.default_rng(n + k).beta(2.0, 5.0, size=n)})
    # both through the tight final fit, so solver tolerance does not blur the comparison of the forms
    structured = _fit(data, k, M, K, "structured", solver_tolerance="tight")
    dense = _fit(data, k, M, K, "dense", solver_tolerance="tight")
    assert structured.knot_aware_formulation_used_ == "structured"
    assert dense.knot_aware_formulation_used_ == "dense"
    np.testing.assert_array_equal(structured._grid_points_hal, dense._grid_points_hal)
    terms = normalizer_terms(dense._train_x, dense._grid_points_hal, k, 200, "knot_aware", build_basis=False)

    obj_s, cells_s = _objective(structured, structured.optimized_theta_raw, terms)
    obj_d, cells_d = _objective(dense, dense.optimized_theta_raw, terms)
    assert abs(obj_s - obj_d) <= 1e-6 * max(1.0, abs(obj_d))

    log_s = cells_s - logsumexp(terms.log_widths + cells_s)
    log_d = cells_d - logsumexp(terms.log_widths + cells_d)
    mass_s, mass_d = np.exp(terms.log_widths + log_s), np.exp(terms.log_widths + log_d)
    carried = mass_d >= 1e-8
    assert np.max(np.abs(log_s - log_d)[carried]) <= 5e-3
    assert 0.5 * np.abs(mass_s - mass_d).sum() <= 1e-3

    first = 1 if k == 0 else 2
    coef_s, coef_d = structured.optimized_theta_raw[first:], dense.optimized_theta_raw[first:]
    big = max(np.abs(coef_s).max(), np.abs(coef_d).max())
    support_s = set(np.flatnonzero(np.abs(coef_s) >= 0.05 * big))
    support_d = set(np.flatnonzero(np.abs(coef_d) >= 0.05 * big))
    assert len(support_s & support_d) >= 0.9 * len(support_s | support_d)


def test_structured_failure_falls_back_to_the_dense_form(monkeypatch):
    import haldensity.full_uni.estimators.cvxpy as module

    real = module.solve_cvxpy_problem

    def fail_structured(problem, **kwargs):
        if kwargs.get("acceptable_statuses") == ("optimal",):
            raise RuntimeError("no solver reached optimal")
        return real(problem, **kwargs)

    monkeypatch.setattr(module, "solve_cvxpy_problem", fail_structured)
    data = pd.DataFrame({"W1": np.random.default_rng(1).beta(2.0, 5.0, size=150)})
    fit = _fit(data, 1, 50.0, None, "structured")
    assert fit.knot_aware_formulation_used_ == "dense"
    assert fit.get_results()["knot_aware_formulation_used"] == "dense"


def test_tight_fit_falls_back_to_the_dense_form(monkeypatch):
    import haldensity.core.univariate.support as support

    real = support.FullDataHALProblem.build

    def no_structured_solution(self, norm_constraint, dense=False):
        built = real(self, norm_constraint, dense=dense)
        if built.formulation == "structured":
            built.theta = lambda: None
        return built

    monkeypatch.setattr(support.FullDataHALProblem, "build", no_structured_solution)
    data = pd.DataFrame({"W1": np.random.default_rng(1).beta(2.0, 5.0, size=150)})
    fit = _fit(data, 1, 20.0, None, "structured", solver_tolerance="tight")
    assert fit.knot_aware_formulation_used_ == "dense" and fit.certified_
    assert {a["formulation"] for a in fit.solve_attempts_} == {"structured", "dense"}


@pytest.mark.parametrize("k,normalizer", [(0, "knot_aware"), (1, "knot_aware"), (1, "midpoint"), (2, "knot_aware")])
def test_tail_sum_gradient_matches_the_dense_formula(k, normalizer):
    rng = np.random.default_rng(3)
    x = rng.beta(2.0, 5.0, size=150)
    knots = np.unique(x)
    theta = rng.normal(scale=0.3, size=(1 if k == 0 else k + 1) + knots.size)
    terms = normalizer_terms(x, knots, k, 200, normalizer, build_basis=False)
    data_basis = create_basis_functions(pd.DataFrame({"W1": x}), knots, order=k)[0].astype(np.float64)
    cell_basis = create_basis_functions(pd.DataFrame({"W1": terms.points}), knots, order=k)[0].astype(np.float64)
    z = terms.log_widths + log_unnormalized_density(terms.points, theta, knots, k)
    dense = -data_basis.sum(axis=0) + x.size * (np.exp(z - logsumexp(z)) @ cell_basis)
    np.testing.assert_allclose(objective_gradient(theta, x, knots, k, terms), dense, rtol=1e-4, atol=1e-6 * x.size)
