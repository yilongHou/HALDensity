# HALDensity: Highly Adaptive Lasso Density Estimation (HAL-MLE/TMLE)

HALDensity provides density estimation with the Highly Adaptive Lasso (HAL) for full and censored data and optional targeting (TMLE) for functionals such as survival, moments, and the median. It includes multiple optimization backends (first-order, second-order, and conic optimization via CVXPY), a unified base estimator, cross-validation with Optuna, and tools for uncertainty quantification.

Features

- HAL basis with polynomial and truncated power terms (order 0 or k ≥ 1)
- Multiple solvers: CVX (conic), Proximal Newton (and LBFGS variants), FISTA, Projected/Proximal GD
- Standardized estimator interface with a lightweight `BaseEstimator`
- Pydantic `CommonEstimatorResults` schema for uniform results
- Cross-validation and hyperparameter tuning via Optuna
- Pointwise delta-method intervals for the density (`haldensity.core.variance`), with a standard error that
  does not depend on the data's units
- Targeted learning (TMLE) for survival, mean, median, entropy and density L2
- **Censored data support**:
  - Right-censored density estimation via IPCW initialization and the observed-data L1-MLE with censoring-aware tuning; four point estimators with one call each (`haldensity.rc.fit_right_censored`) and targeting (TMLE) for survival, mean, RMST, density L2 and entropy
  - Interval-censored density estimation via midpoint initialization and direct optimization of the observed-data likelihood (FISTA, projected gradient); current-status data through `haldensity.ic.current_status`
- Multivariate full-data HAL (`haldensity.full_multi`, experimental)
- Sample data and simple simulators (`haldensity.datasets`, including the galaxy velocities)

Install
This repo uses Python 3.11+. With pip:

```bash
pip install -e .             # the package
pip install -e ".[dev]"      # plus test and lint tools
pip install -e ".[examples]" # plus matplotlib and ipykernel, to run the tutorials
pip install -e ".[turnbull]" # plus lifelines, for knot_strategy="turnbull" with interval-censored data
```

Or install directly from GitHub:

```bash
pip install "git+https://github.com/yilongHou/HALDensity.git"
```

Parallel runs and shared servers

Every solver call is single-threaded by default: MOSEK (`MSK_IPAR_NUM_THREADS`), CLARABEL (`max_threads`)
and, for the duration of the call, the BLAS/OpenMP pools of numpy and scipy (through `threadpoolctl`);
SCS and ECOS are single-threaded anyway. Without this, one CLARABEL solve used 109 cores of a
256-core server and ran 4 times slower than on one thread. Run parallel work as separate single-threaded
processes. Raise the count only when one fit runs alone on a dedicated machine:
`CVXPYEstimator(solver_threads=8)` (or `estimator_kwargs={"solver_threads": 8}` in the tuner). numpy and
scipy calls outside the solvers keep numpy's own setting, so for many parallel workers also export
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before starting Python.

Solvers

The final fit tries MOSEK, CLARABEL, ECOS and SCS, in that order, and keeps the first solution that
passes the optimality check. MOSEK is optional and needs a license (`pip install mosek`); without it,
the fit starts at CLARABEL. `solver_used_` on the fitted estimator says which solver produced a fit.
Fits that pass the optimality check agree across solvers (log-densities within about 0.004 in our
cross-check).

Quickstart

```python
import numpy as np
import pandas as pd
from haldensity.full_uni import CVXPYEstimator

# Simulated data on [0, 1]
rng = np.random.default_rng(0)
data = pd.DataFrame({"W1": np.clip(rng.normal(0.07, 0.02, size=2000), 0, 1)})

# Fit CVX-based estimator
est = CVXPYEstimator(basis_order=0, norm_constraint=20)
est.fit(data)

# Evaluate density on a fine grid
grid = np.linspace(0, 1, 1000)
density = est.get_density_at_points(grid)

# Results (standardized)
results = est.get_results()
# Or a Pydantic model:
model = est.get_common_results_model()
```

Cross-Validation (Optuna)

```python
from haldensity.full_uni import OptunaHyperparameterTuner
tuner = OptunaHyperparameterTuner(
    'CVXPYEstimator', data, cv_folds=5, metric='sll', silent=True,
    param_overrides={"basis_order": [0, 2]}
)
res = tuner.optimize(n_trials=50)
best_est = tuner.fit_best_model()
```

Data in a small part of [0, 1]

The estimators work on [0, 1]; rescale the data by bounds fixed in advance (not the sample's minimum and
maximum). At order 2 (the default) the objective is normalized on a fitting grid of `n_grid_points - 1`
equal cells (default 200). If the data occupy only a small part [a, b] of [0, 1], few cells cover them and
the objective can miss part of the fitted mass. Raise the grid to about `200 / (b - a)`, so that about 200
cells cover the data, and give the tuner the same grid, so that every CV fold and the final fit use it:

```python
est = CVXPYEstimator(norm_constraint=..., n_grid_points=1000)
tuner = OptunaHyperparameterTuner("CVXPYEstimator", data, estimator_kwargs={"n_grid_points": 1000})
```

Every fit records the missed mass, `get_results()["log_z_gap"]` (the accurate log normalizer minus the
objective's, in nats), and warns above 0.01 nats with a suggested `n_grid_points`. Example: the galaxy
velocities divided by 3e5 occupy 8% of [0, 1]. At k = 2 and the CV choice M = 596,793, the default grid's
objective sees 93.8% of the fitted mass and the main peak is 80.7; with 1,000 points it sees all of it and
the peak is 65.2 (the same with 2,000), and a 5-fold CV trial takes 0.84 s instead of 0.20 s.

Targeted Learning (TMLE)

The targeting step starts from a fitted full-data estimator. `targeting_inputs` takes the fit (or its
`get_results()`) and returns the M-step's inputs: the intercept, the powers x, ..., x^k and the nonzero knot
coefficients, with their knots. Build each learner with the fit's `basis_order`. The M-step integrates
exactly (cells with edges at the knots and the targeting points), so when the targeting coefficients stay
inside their L1 bound the score equations hold at the targeting points themselves: the TMLE of S(t) is the
empirical survival at t, the TMLE of the mean is the sample mean. Each result holds `estimate` (the TMLE),
`plug_in` (the initial fit's value) and `score_residual` (the largest score-equation error).

**Cross-fitted by default.** Every targeting step, full-data and right-censored, is cross-fitted unless
`cross_fit="none"` (or `False`) is passed: each fold's fit is trained on the other folds, one targeting
coefficient serves all folds, and each observation's influence curve is evaluated under its own fold's targeted
fit, which gives the SE. `cross_fit` names where the fold fits come from:

- `"tuner"` (default, or `True`): the tuner's own fold fits at the parameters it chose, on its own splits.
  This is *cross-fitted with tuning on all folds*: the bound M (and, for right-censored data, Stage 1's order
  and knots) was chosen with every fold, including the one a fit is later scored on.
  `OptunaHyperparameterTuner.fit_best_model()` and `fit_right_censored(method="cvl1mle")` attach them to the
  estimator (`fold_fits_`), and `targeting_inputs(est)` passes them on. An estimator fitted at a fixed bound has
  none, and then `"tuner"` falls back to `"refit"`; the result's `cross_fit` says which ran.
- `"refit"`: the estimator's own settings refitted on each training fold.
- `"nested"`: the whole tuning repeated inside each training fold, so no fold fit has seen its validation fold
  (about five times the cost of the initial fit).
- `"none"` (or `False`): the full-sample step.

For survival, the CDF and the mean the cross-fitted estimate equals the empirical value, as the full-sample one
does; only the SE changes. For entropy and density L2, targeting removes the first-order term but not the
second-order remainder of the fit, which was still visible at n <= 1,600 in our simulations (`estimate`
versus `plug_in_cross_fitted`).

```python
import numpy as np
from haldensity.full_uni.targeting import (
    CDFTargetLearner, SurvivalTargetLearner, MomentsTargetLearner, MedianTargetLearner,
    cdf_estimand_variance, survival_estimand_variance, moments_estimand_variance, median_estimand_variance,
    targeting_inputs,
)

inputs = targeting_inputs(est)            # {"old_theta": ..., "grid_points_hal_selected": ...}
points = np.array([0.05, 0.07, 0.09])     # targeting points inside the data
```

**1) CDF targeting (F(t) at chosen points)**

```python
learner = CDFTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, targeting_points=points)
tmle, plug_in = targeted_fit["estimate"], targeted_fit["plug_in"]
se = np.sqrt(cdf_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=points))
```

**2) Survival targeting (S(t) = 1 - F(t) at chosen points)**

```python
learner = SurvivalTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, targeting_points=points)
tmle = targeted_fit["estimate"]
se = np.sqrt(survival_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=points))
```

**3) Moments targeting (E[X^r], e.g. the mean)**

```python
r = 1  # 1 for the mean, 2 for the second moment, etc.
learner = MomentsTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, x_moment=r)
tmle = targeted_fit["estimate"][0]
se = np.sqrt(moments_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, x_moment=r))[0]
```

**4) Median targeting**

```python
learner = MedianTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs)
tmle = targeted_fit["estimate"][0]
se = np.sqrt(median_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data))[0]
```

Notes

- For CDF/Survival, every interval cut by the targeting points (below the first point, between two
  consecutive points, above the last) should hold an observation. Where one does not, the empirical value asks
  the targeted density for no mass there, which it only approaches as its coefficients grow; once they reach
  the L1 bound the score equations hold only approximately (`score_residual`) and the learner warns. Raise
  `norm_constraint` (100 in the examples above; the learners' default is 20) or move the points.
- For Moments, give `x_moment` (integer power).
- For Median, no extra arguments: the learner takes one targeting step at the initial fit's median m, so
  F(m) = P_n(X < m) holds at m; the targeted median need not be an empirical median.

Examples

Tutorials per data type, in `examples/tutorials/`. They use the public API and run in CI; to run them, install
`pip install "haldensity[examples]"` (adds matplotlib and ipykernel):

- `full_uni/01_galaxies.ipynb`: the galaxy velocities (Roeder 1990): a larger fitting grid for data that
  occupy a small part of [0, 1], cross-validation, pointwise 95% intervals
- `full_uni/02_simulated_density.ipynb`: the HAL-MLE on simulated data, with the L1 bound chosen by
  cross-validation, compared with the truth; the fit's diagnostics
- `full_uni/03_targeting.ipynb`: targeted estimates (TMLE) of survival, the mean and the median, with
  influence-curve standard errors
- `full_multi/01_bivariate.ipynb`: the multivariate estimator (experimental) on a bivariate sample
- `rc/01_point_estimators.ipynb`: right-censored data: the four HAL point estimators and the fitted
  distribution
- `rc/02_tmle.ipynb`: right-censored data: targeted estimates (HAL-TMLE) of survival, the mean and the
  restricted mean survival time with influence-curve standard errors; density L2 and entropy

The notebooks directly in `examples/` predate 0.2.0 and use its old import paths:

- `examples/example_interval_censored_density_estimator.ipynb`: interval-censored density estimation workflow

Repository Structure

```
src/haldensity/
  core/         # shared by data types: basis (univariate and general d), univariate HAL base
                # (BaseEstimator, weighted MLE, HALFit), solvers, CV, variance, metrics
  full_uni/     # univariate full data: estimators (one per solver), Optuna tuner, targeting
  full_multi/   # multivariate full data (normalizer by Sobol QMC or exact k = 0 cells)
  rc/           # right-censored: IPCW, observed-data L1-MLE, tuners, pipelines, AIPW, targeting/
  ic/           # interval-censored: estimators, tuners; current_status/
  datasets/     # sample data (galaxies) and simple simulators
docs/           # censored-data overview; dev/ (layout, decisions, experiment write-ups)
examples/       # tutorials/ per data type (run in CI); older notebooks
```

See `CONTRIBUTING.md` for the development workflow and `CHANGELOG.md` for changes.

License
MIT
