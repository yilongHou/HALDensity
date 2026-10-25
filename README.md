# HALDensity: Highly Adaptive Lasso Density Estimation (HAL-MLE/TMLE)

HALDensity provides density estimation with the Highly Adaptive Lasso (HAL) for full and censored data and optional targeting (TMLE) for functionals such as survival, moments, and the median. It includes multiple optimization backends (first-order, second-order, and conic optimization via CVXPY), a unified base estimator, cross-validation with Optuna, and tools for uncertainty quantification.

Features

- HAL basis with polynomial and truncated power terms (order 0 or k ≥ 1)
- Multiple solvers: CVX (conic), Proximal Newton (and LBFGS variants), FISTA, Projected/Proximal GD
- Standardized estimator interface with a lightweight `BaseEstimator`
- Pydantic `CommonEstimatorResults` schema for uniform results
- Cross-validation and hyperparameter tuning via Optuna
- Delta-method style density bands; bootstrap example included
- Targeted learning (TMLE) for survival, mean, and median
- **Censored data support**:
  - Right-censored density estimation via IPCW initialization and the observed-data L1-MLE with censoring-aware tuning; four point estimators with one call each (`haldensity.rc.fit_right_censored`) and targeting (TMLE) for survival, mean, RMST, density L2 and entropy
  - Interval-censored density estimation via midpoint initialization and direct optimization of the observed-data likelihood (FISTA, projected gradient); current-status data through `haldensity.ic.current_status`
- Multivariate full-data HAL (`haldensity.full_multi`, experimental)
- Sample data and simple simulators (`haldensity.datasets`, including the galaxy velocities)

Install
This repo uses Python 3.11+. With uv:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .             # the package
pip install -e ".[dev]"      # plus test and lint tools
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

# 1) CDF targeting (F(t) at chosen points)

```python
learner = CDFTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, targeting_points=points)
tmle, plug_in = targeted_fit["estimate"], targeted_fit["plug_in"]
se = np.sqrt(cdf_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=points))
```

# 2) Survival targeting (S(t) = 1 - F(t) at chosen points)

```python
learner = SurvivalTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, targeting_points=points)
tmle = targeted_fit["estimate"]
se = np.sqrt(survival_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=points))
```

# 3) Moments targeting (E[X^r], e.g. the mean)

```python
r = 1  # 1 for the mean, 2 for the second moment, etc.
learner = MomentsTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs, x_moment=r)
tmle = targeted_fit["estimate"][0]
se = np.sqrt(moments_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, x_moment=r))[0]
```

# 4) Median targeting

```python
learner = MedianTargetLearner(norm_constraint=100, basis_order=est.basis_order)
targeted_fit = learner.run_m_step(uncensored_augmented=data, **inputs)
tmle = targeted_fit["estimate"][0]
se = np.sqrt(median_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data))[0]
```

Notes

- For CDF/Survival, give `targeting_points` inside the data: beyond the data the empirical value is 0 or 1,
  which the targeted density can only approach as its coefficients reach the L1 bound.
- For Moments, give `x_moment` (integer power).
- For Median, no extra arguments: the learner takes one targeting step at the initial fit's median m, so
  F(m) = P_n(X < m) holds at m; the targeted median need not be an empirical median.
- The example notebook `examples/example_full_data_target_learning.ipynb` still uses the pre-0.2.0 API.

Examples

The examples are being reorganized into small tutorials per data type that run with the public API;
until then some notebooks below still use pre-0.2.0 import paths.

- `examples/example_full_data_density_estimator.ipynb`: quick demo of estimators on fully observed data
- `examples/example_full_data_target_learning.ipynb`: targeting examples (TMLE for CDF, survival, moments, median)
- `examples/example_full_data_case_study.ipynb`: real-data galaxy velocity case study
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
examples/       # notebooks
```

See `CONTRIBUTING.md` for the development workflow and `CHANGELOG.md` for changes.

License
MIT
