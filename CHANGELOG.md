# Changelog

## Unreleased (0.2.0)

Relative to `v0.1.0` (`fd4e22a`, origin/main on 2026-10-23).

### Layout

The package is reorganized into one subpackage per data type plus a shared core:

| Before (0.1.0) | After |
| --- | --- |
| `haldensity.estimation` (8 estimators, `BaseEstimator`) | `haldensity.full_uni` (estimators); `haldensity.core.univariate` (`BaseEstimator`) |
| `haldensity.cross_validation.optuna_hyperparam_selector` | `haldensity.full_uni` (`OptunaHyperparameterTuner`) |
| `haldensity.targeting` (full-data targets) | `haldensity.full_uni.targeting` |
| `haldensity.targeting` (right-censored targets) | `haldensity.rc.targeting` |
| `haldensity.censoring.right`, `censoring.tuners` (right) | `haldensity.rc` |
| `haldensity.censoring.interval`, `censoring.tuners` (interval) | `haldensity.ic` |
| `haldensity.censoring._base_mle`, `_defaults`, tuner bases | `haldensity.core.univariate`, `core.defaults`, `core.cv`, `core.em` |
| `haldensity.multivariate` | `haldensity.full_multi` (estimator); `haldensity.core.basis` (basis) |
| `haldensity.utils` (basis, solver helper, density functionals) | `haldensity.core.basis`, `core.solvers`, `core.univariate` |
| `haldensity.utils.dgp`, `censoring.interval.simulate` | `haldensity.datasets` |
| `haldensity.density_variance.density_variance` | `haldensity.core.variance` |

Class and function names are unchanged. Tests mirror the layout under `tests/<subpackage>/`.

### Added

- `haldensity.rc.fit_right_censored(method=...)` and `fit_ipcw_hal`, `fit_l1mle_m1`,
  `fit_cvl1mle` (default method), `fit_undersmoothed`: the four right-censored point estimators
  with the same inputs and outputs (`RightCensoredPointEstimate`).
- `haldensity.datasets.load_galaxies()` (Roeder 1990 galaxy velocities, km/s) and
  `make_right_censored()`.
- `haldensity.ic.current_status.current_status_to_interval()`.
- `haldensity.core.univariate.HALFit`, the generic frozen fit; `RightCensoredHALFit` is its
  subclass.
- `OptunaHyperparameterTuner(estimator_kwargs=...)` and `best_estimator_params`.
- CI: the test suite on Python 3.11 for pushes and pull requests.
- From the consolidated development branch: right-censored observed-data L1-MLE, its CV tuner
  and fit object; right-censored targeting (survival, mean, density L2, RMST, entropy) and EIF
  helpers; AIPW; interval-censored FISTA and projected-GD estimators with knot strategies;
  CV-oversmooth EM tuners; the solver fallback helper; the multivariate estimator and
  tensor-product basis.

### Changed (behaviour)

- Default CVXPY solver: MOSEK, then CLARABEL, ECOS, SCS (was ECOS only); a solve is accepted
  only with status optimal or optimal_inaccurate. Restore ECOS-only with
  `solver="ECOS", use_secondary_solver=False`.
- Full-data CV uses the estimator's solver fallback, as the final fit does (it forced a single
  solver, so without MOSEK every trial failed); if no trial gives a finite score it raises
  instead of returning an arbitrary bound. ECOS-only: `estimator_kwargs={"solver": "ECOS",
  "use_secondary_solver": False}`.
- `fit_best_model()` refits with the best trial's full settings (fixed overrides such as
  `basis_order` were dropped before, and `ProximalNewtonLBFGS*` could not be refit).
- Conservative bound search in the censored-data init tuners: bisection (was a linear 2% step).
- Order-0 fits of the weighted MLE normalize on at least 2,000 grid points (was 200).
- `WeightedHALMLEEstimator.get_fit()` returns a `HALFit`; `RightCensoredInitEstimator.get_fit()`
  still returns a `RightCensoredHALFit` with the same state dictionary.
- `MultvarHAL` warns when fitted on one-dimensional data.
- `lifelines` is an optional extra: `pip install "haldensity[turnbull]"` (only
  `knot_strategy="turnbull"` uses it).

The old behaviour of these fixes is available at `v0.1.0`.

### Fixed

- `np.trapz` (removed in NumPy 2.4) replaced in the variance module; delta-method bands crashed
  on current NumPy.

### Removed

- `KFold_CV_LambdaSelector`, `CVSelectorMetric` (unused).
- Right-censored comparison helpers (truth objects, error metrics, M1–M4 experiment runner,
  two simulators) and the plug-in pipelines `fit_right_censored_cv_ipcw_l1mle_plugin` /
  `..._undersmoothed_plugin`; moved to the HAL_TMLE_RC project repository and replaced by
  `haldensity.rc.fit_right_censored`.
- `interval_censor_width_uniform` (legacy, informative censoring).
- `extract_term_type_metrics` (experiment reporting; moved to the multivar-HAL repository).
- Stray files (a matplotlib font cache, a 2 MB bootstrap result, scratch notebooks) and
  `examples/MCMC-HAL/`, summarized in `docs/dev/experiments/sobol-normalizer-1d.md`.
