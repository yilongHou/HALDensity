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

Old import paths keep working through 0.2.x as thin re-exports of the public names (156 names
from `haldensity.censoring[.right|.interval|.tuners|.utils]`, `haldensity.estimation`,
`haldensity.cross_validation`, `haldensity.targeting`, `haldensity.multivariate`,
`haldensity.utils[.dgp]` and `haldensity.density_variance.density_variance`); importing them
emits a `DeprecationWarning`, and they are removed in 0.3.0. Deeper module paths (e.g.
`haldensity.censoring.right.observed_mle`) and private names are not re-exported.

### Added

- `docs/dev/cvll-multicollinearity/`: Yilong's implementation note on multicollinearity in the
  0-order basis, the flat CV log-likelihood it causes, and the conservative adjustment (LaTeX
  source, figures, and a Markdown rendering with the 0.2.0 k = 0 diagnosis); linked from
  `OptunaHyperparameterTuner` and `haldensity.core.cv.conservative`.
- `CVXPYEstimator(objective_normalizer="knot_aware")` and `CVXPYEstimator(max_knots=K)`
  (`haldensity.core.univariate.objective`). Knot-aware: the candidate knots are breakpoints of the
  objective's normalizer cells; exact for k = 0 (one term per cell between consecutive knots,
  reusing the data term's values, so no extra matrix), a knot-inserted midpoint grid for k >= 1.
  `max_knots`: candidate knots at K data quantiles (observations, de-duplicated) instead of every
  unique data value. Both default to the previous behaviour (`"midpoint"`, `None`), which is
  reproduced bit for bit. Why: with the midpoint objective, k = 0, M = 50 (five simulated datasets),
  31-100% of the mass sat in one cell 0.0042-0.0049 wide between two close knots holding 5-10
  data points but no grid midpoint, so the objective gained likelihood without paying for the mass
  (training log-likelihood as low as -16.9 when evaluated exactly); with the exact k = 0 objective
  the spikes disappear and the 5-fold CV log-likelihood declines gradually past its peak instead of
  collapsing. Defaults will be set after a benchmark of time, memory and accuracy at large n. New
  diagnostics: `objective_log_normalizer()`, `objective_value_`; `get_results()` records the
  objective normalizer, `max_knots` and the number of candidate knots; basis cleaning rebuilds the
  same problem.
- Knot-aware objective for k = 0 and 1 solved in a structured (sparse) form
  (`haldensity.core.univariate.objective.structured_problem`), shared by the fit, CV and basis
  cleaning: k = 0 as a total-variation problem over the cell levels, k = 1 over the values at the
  knots and one slope per knot interval; O(n + cells) nonzeros instead of the n x m basis. Same
  optimum as the dense form (n = 200-2,000, k = 0, 1, M reasonable and 100x, K = 60, 250: objective
  within 3.3e-6 relative, the structured solution never worse by more than 1.6e-9; log-density on
  mass-carrying cells within 3.4e-3; cell masses within 8.5e-5 in total variation) and 1-1,900
  times faster. Scaled for the solvers (objective / n; k = 1 slopes / max(1, M)). If no solver
  reaches status "optimal" on it, the fit uses the dense form; `knot_aware_formulation_used_` and
  `get_results()["knot_aware_formulation_used"]` record which. The dense form stays available as
  `CVXPYEstimator.knot_aware_formulation = "dense"`. Defaults unchanged.
- Basis cleaning before the delta-method SE, on by default (`haldensity.core.univariate.support`):
  the full-data problem is re-solved at the fit's bound with tight tolerances (CLARABEL
  1e-12/1e-10, ECOS 1e-12/1e-10, SCS 1e-9), pruned at 1e-4 and certified by the optimality
  conditions, and `estimate_covariance_beta` / `density_confidence_interval` compute the SE on that
  basis (`support_refit="auto"`; the refit is computed once per fit, and the covariance carries its
  basis so the CI uses the same one). The point estimate stays the original fit; an uncertified
  refit is kept, warned about and flagged (`result.attrs["basis_certified"]`). `support_refit=None`
  uses the original basis; fits without an L1 bound (penalty-tuned estimators) always do.
  `CVXPYEstimator.get_results()` now includes `norm_constraint` and `n_grid_points`. With MOSEK, 100 replicates per setting (truncated normal and
  Gaussian mixture, n = 400 and 1,280, k = 1 and 2): basis 1,121-1,257 -> 5-33 functions at
  n = 1,280; mean SE / Monte Carlo sd 1.11-2.76 -> 0.80-1.25; coverage of the truth 85-100%
  (median 100%) -> 83-99% (median 94%).
- `OptunaHyperparameterTuner` uses 5 CV folds by default (`TUNER_DEFAULTS.cv_folds`, as the
  censored-data tuners) instead of 3, so default full-data CV results change; `cv_folds=3` gives
  the old split.
- The final fit is one tight, certified solve (`CVXPYEstimator(solver_tolerance="tight")`, the
  new default): MOSEK with interior-point tolerances 1e-12 (relative gap, primal and dual
  feasibility), then CLARABEL 1e-12/1e-10, ECOS 1e-12/1e-10, SCS 1e-9 (`haldensity.core.solvers.
  TIGHT_LADDER`, following `solver` / `solver_waterfall`); each solution is pruned at `tol` (1e-4)
  and checked against the optimality conditions, the first certified one is kept, else the least
  violating with a warning. That solution is the point estimate and its basis the SE basis
  (`estimate_covariance_beta` uses it directly, no separate refit). `certified_`, `kkt_`,
  `solve_attempts_`, `solver_option_` and `get_results()` record solver, status and the check.
  **Point estimates change slightly from v0.1.0**: on the 18 fixed fits of the restructure check
  (CLARABEL, no MOSEK) theta by at most 3.5e-4 and the density by at most 1.8e-4; on 32 fits (n =
  200 and 1,000, k = 0-2, midpoint and knot-aware, two bounds each) by at most 2.7% relative (MOSEK)
  or 8.4% (free solvers), apart from k = 0 midpoint cells the objective does not see, which it leaves
  unidentified. All 32 were certified; MOSEK alone certified 24 and stopped short of 1e-12 (status
  "optimal_inaccurate", MOSEK's near-optimal stop made visible) in 19. Without MOSEK the final fit
  equals the previous fit + cleaning refit bit for bit. Cost: about the old fit + refit; at n =
  1,000, 0.6-0.8 s when MOSEK certifies, 2-6 s when CLARABEL or ECOS has to. CV folds keep the
  solvers' own tolerances (`OptunaHyperparameterTuner(cv_solver_tolerance="default")`); the old
  final fit is `solver_tolerance="default"` (bit for bit). The optimality check now also handles a
  bound that is not binding (lambda / n <= 1e-5: certified when max |gradient| / n <= 1e-5; otherwise
  it also requires ||theta[1:]||_1 >= (1 - 1e-3) M), and basis
  cleaning of `solver_tolerance="default"` fits starts with MOSEK 1e-12 as well.
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

- Full-data CV can apply the conservative adjustment (core, shared with the censored-data tuners
  through `haldensity.core.cv.conservative`): after the search, the smallest bound whose mean CV
  log-likelihood is within 5% of a fold SD of the best, found by bracketing and bisection in log M.
  Interim default `use_conservative_adjustment="auto"`: on when the CV-best basis order is 0 (flat
  CV curves from nearly collinear indicator bases), off for k >= 1, where exact CV scoring already
  avoids the huge-M choices (in 16 simulated k = 1 datasets forcing it on shrank M to 0.11-0.87x
  the CV-best and made L2 error worse in 11). `True` / `False` force it on / off; the CV-best
  stays in `max_cv_params`. Defaults to be revisited after the knot-aware normalizer benchmark.
- Full-data CV scores each fold by evaluating the fitted log density exactly at the validation
  points (with the accurate normalizer) instead of linearly interpolating `get_density()` on its
  200-point grid. In 16 simulated datasets (4 densities, n = 400 and 1,280, k = 1) the selected
  bound was unchanged in 11; in the 4 n = 1,280 cases where the old score picked a much larger M,
  held-out log-likelihood rose by 0.019-0.030 nats and L2 error fell by 0.33-0.56. Old
  behaviour: `cv_scoring="interpolated"` with `estimator_kwargs={"normalizer": "midpoint"}`.
- Full-data estimators evaluate and report the density with an accurate normalizer: log Z by
  cell-wise Gauss-Legendre with breakpoints at the knots with nonzero coefficients (32 nodes per
  cell, doubled until the change is below 1e-12; exact for k = 0), in float64. The midpoint rule
  on the fitting grid underestimated Z for spiky fits (0.175 nats at k = 1, M = 1e5, n = 1280).
  The fitting objective is unchanged. Old behaviour: `normalizer="midpoint"`. New:
  `estimator.log_normalizer()`, `estimator.get_log_density_at_points()`. Censored-data estimators
  keep the normalizer frozen on their own fitting grid.
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
