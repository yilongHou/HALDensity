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
  - Right-censored density estimation via IPCW initialization, EM + MI, the observed-data L1-MLE and censoring-aware tuning; four point estimators with one call each (`haldensity.rc.fit_right_censored`) and targeting for survival, mean, RMST, density L2 and entropy
  - Interval-censored density estimation via midpoint initialization and EM refinement; current-status data through `haldensity.ic.current_status`
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

Quickstart

```python
import numpy as np
import pandas as pd
from haldensity.full_uni import CVXPYEstimator

# Simulated data on [0, 1]
rng = np.random.default_rng(0)
data = pd.DataFrame({"W1": np.clip(rng.normal(0.07, 0.02, size=2000), 0, 1)})

# Fit CVX-based estimator
est = CVXPYEstimator(basis_order=0, norm_constraint=20, n_grid_points=200)
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

Targeted Learning (TMLE)

```python
# Common prerequisites (after fitting an estimator)
import numpy as np
from haldensity.full_uni.targeting import (
    CDFTargetLearner, SurvivalTargetLearner, MomentsTargetLearner, MedianTargetLearner,
    cdf_estimand_variance, survival_estimand_variance, moments_estimand_variance, median_estimand_variance,
)

results = est.get_results()
old_theta = np.array(results["theta_hat"])                     # intercept + HAL coefs
hal_grid = results["grid_points_hal_selected"]                 # selected HAL grid points
```

# 1) CDF targeting (estimate F(x) at chosen points)

```python
import numpy as np

cdf_points = np.linspace(0.02, 0.98, 10)
learner = CDFTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    targeting_points=cdf_points,   # required
)
# Variance (SE^2) at the same points
var = cdf_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=cdf_points)
se = np.sqrt(var)
```

# 2) Survival targeting (estimate S(x)=1-F(x) at chosen points)

```python
surv_points = np.linspace(0.02, 0.98, 10)
learner = SurvivalTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    targeting_points=surv_points,  # required
)
var = survival_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, targeting_points=surv_points)
se = np.sqrt(var)
```

# 3) Moments targeting (estimate E[X^k], e.g., k=2)

```python
k = 2  # 1 for mean, 2 for second moment, etc.
learner = MomentsTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    x_moment=k,                    # required
)
var = moments_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data, x_moment=k)
se = np.sqrt(var)[0]  # single scalar
```

# 4) Median targeting (estimate the median)

```python
learner = MedianTargetLearner(norm_constraint=100, basis_order=0)
targeted_fit = learner.run_m_step(
    uncensored_augmented=data,
    grid_points_hal_selected=hal_grid,
    old_theta=old_theta,
    # no extra kwargs required
)
var = median_estimand_variance(targeted_fit=targeted_fit, uncensored_data=data)
se = np.sqrt(var)[0]  # single scalar
```

Notes

- The example notebook `examples/example_full_data_target_learning.ipynb` demonstrates all four targets. Set `TARGET_TYPE` to one of "CDF", "Survival", "Moments", or "Median" to switch modes.
- For CDF/Survival, provide `targeting_points` where the functional is targeted and where standard errors are reported.
- For Moments, provide `x_moment` (integer power).
- For Median, no additional arguments are needed beyond `old_theta` and the selected HAL grid.

Examples

The examples are being reorganized into small tutorials per data type that run with the public API;
until then some notebooks below still use pre-0.2.0 import paths.

- `examples/example_full_data_density_estimator.ipynb`: quick demo of estimators on fully observed data
- `examples/example_full_data_target_learning.ipynb`: targeting examples (TMLE for CDF, survival, moments, median)
- `examples/example_full_data_case_study.ipynb`: real-data galaxy velocity case study
- `examples/example_right_censored_density_estimator.ipynb`: right-censored density estimation workflow (IPCW + EM)
- `examples/example_interval_censored_density_estimator.ipynb`: interval-censored density estimation workflow

Repository Structure

```
src/haldensity/
  core/         # shared by data types: basis (univariate and general d), univariate HAL base
                # (BaseEstimator, weighted MLE, HALFit), solvers, CV, EM, variance, metrics
  full_uni/     # univariate full data: estimators (one per solver), Optuna tuner, targeting
  full_multi/   # multivariate full data (normalizer by Sobol QMC or exact k = 0 cells)
  rc/           # right-censored: IPCW/EM, observed-data L1-MLE, tuners, pipelines, AIPW, targeting/
  ic/           # interval-censored: estimators, tuners; current_status/
  datasets/     # sample data (galaxies) and simple simulators
docs/           # censored-data workflow notes; dev/ (layout, experiment write-ups)
examples/       # notebooks
```

See `CONTRIBUTING.md` for the development workflow and `CHANGELOG.md` for changes.

License
MIT
