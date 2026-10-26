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
- The exact objective for k = 0 and the midpoint rule with the knots added to the grid for k >= 1
  (`haldensity.core.univariate.objective`; the candidate knots are breakpoints of the normalizer's
  cells: exact for k = 0, one term per cell between consecutive knots), and
  `CVXPYEstimator(max_knots=K)`: candidate knots at K data quantiles (observations, de-duplicated)
  instead of every unique data value. Which objective a fit uses is the package's choice (see "The
  objective's normalizer is chosen by order" below); the knot cap's default is below too. Why: with the midpoint objective, k = 0, M = 50 (five simulated datasets),
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
  `CVXPYEstimator._knot_aware_formulation = "dense"` (internal). Defaults unchanged.
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
- **Default order k = 2** (`haldensity.core.defaults.DEFAULT_BASIS_ORDER`; Yilong, 2026-10-24) for
  every full-data univariate estimator and the full-data targeting functions; the tuner fixes the
  order at 2 unless asked to cross-validate it (`param_overrides={"basis_order": [0, 1, 2]}`, the
  default before 0.2.0). **Default results change.** Why: at n = 320 the package's joint CV over
  {0, 1, 2} was never better than fixed k = 2 (KL of the joint CV relative to fixed k = 2: 0.96,
  1.02, 1.16 and 1.34 on the jump, slope-kink, curvature-kink and smooth truths), and an order
  higher than the truth's costs little. The censored-data and multivariate estimators keep their
  own defaults.
- **No SE when basis cleaning fails** (Yilong, 2026-10-24): if the final fit does not pass the
  optimality check (or a separate cleaning refit does not), `estimate_covariance_beta` returns an
  all-NaN covariance and `density_confidence_interval` the point estimate with NaN `se`, `lower` and
  `upper`, each with a warning that suggests a bootstrap or another solver; before, the SE was
  computed on that basis with a warning. `support_refit=None` still gives the original-basis SE.
- **Numerical options are the package's choice** (Yilong, 2026-10-24). Public options of
  `CVXPYEstimator`: `norm_constraint`, `basis_order`, `max_knots` (None = every knot), `n_grid_points`
  (the one exception, see the next item), `solver_threads`, `log_dir`, `log_frequency`; of
  `OptunaHyperparameterTuner`: `cv_folds`, `metric`, `random_state`,
  `max_iter` and `tolerance` (first-order estimators), `silent`, `show_progress`,
  `param_overrides`, `estimator_kwargs`, `use_conservative_adjustment` and its margin settings; of the
  variance functions: `support_refit` and `alpha`. Options that were public in 0.1.0 still work through
  0.2.x with a `DeprecationWarning` naming the internal replacement and are removed in 0.3.0:
  `CVXPYEstimator(tol, solver, use_secondary_solver, solver_waterfall)` -> `_tol`, `_solver`,
  `_use_secondary_solver`, `_solver_waterfall`; `estimate_covariance_beta(ridge_param)` -> `_ridge_param`
  (passing `ridge_param` keeps the 0.1.0 rule, the absolute ridge, with `_inverse="ridge"`).
  Options added during the restructure are internal directly:
  `_normalizer`, `_solver_tolerance`, `_objective_normalizer` (estimator), `_cv_scoring`,
  `_cv_solver_tolerance` and the conservative search's `_conservative_bracket_factor`,
  `_conservative_min_norm_constraint`, `_conservative_step_pct`, `_conservative_max_steps` (tuner).
  Internal names are for tests, benchmarks and experiment repositories. Arguments after `data`
  (tuner) and all estimator arguments are keyword-only, so a 0.1.0 positional call fails loudly
  instead of shifting values.
- **The fitting grid `n_grid_points` is public, with a missed-mass warning** (Yilong, 2026-10-24; full
  data, univariate): whether the data are too concentrated for the default grid is the user's call, the one
  exception to "numerical options are internal"; the default stays 200. The tuner gives every CV fold and
  the final fit the same grid through `estimator_kwargs={"n_grid_points": ...}`; its own 0.1.0 option
  `n_grid_points`, which never reached the estimators, is passed on there with a `DeprecationWarning`
  (removed in 0.3.0). Every fit records the missed-mass diagnostic `get_results()["log_z_gap"]`
  (`log_z_gap_`): the accurate log normalizer minus the objective's, in nats, at the point estimate. Above
  0.01 nats the fit warns and suggests `n_grid_points` of about 200 / (max x - min x), 200 cells across the
  data, naming both routes: the estimator's argument and, through the tuner, `estimator_kwargs` (silent
  inside CV folds; the final fit shows it). Galaxies (v / 3e5, 8% of [0, 1]; k = 2, MOSEK, the
  case study's M = 596,793): with 200 points the gap is 0.064 nats (the objective saw 93.8% of the fitted
  mass), the main peak 80.7 and the exact 5-fold CV score 3.1670; with 1,000 points the gap is 0, the peak
  65.2 and the score 3.1792; with 2,000 the gap is 0, the peak 65.2 and the score 3.1794. One 5-fold CV
  trial at that M takes 0.20, 0.84 and 1.74 s. Default results are unchanged; the old midpoint rule forced
  at k = 0 with M = 20 (internal hook) sees only 12.5% of its fitted mass and now warns.
- **Internal safeguard at k = 0** (Yilong, 2026-10-24; `haldensity.core.univariate.imbalance`): every
  k = 0 fit records coefficient-imbalance diagnostics in `get_results()["imbalance"]`, and the final
  fit warns if adjacent selected knots less than 1e-3 apart carry opposite-sign coefficients both
  above 10 in absolute value (flag A), or coefficients above 10 carry more than half of ||theta||_1
  (flag B). With the package's objective this pattern (Yilong's CVLL note) should not occur (0 of 800
  fits in session 69's check), so the warning reports a numerical problem inside the package and
  asks for a report; it is not a setting, and nothing is refitted or adjusted.
- **The objective's normalizer is chosen by order, by the package** (Yilong, 2026-10-24; not a user
  option): the exact objective at k = 0, the midpoint rule with the knots added to the grid at k = 1
  (both through the step-height rewrite), the old midpoint rule (199 fixed cells) at k >= 2. **Default
  results change at k = 0 and 1**; k >= 2 is unchanged. In a CV search over orders each trial uses
  its order's rule (`trial.user_attrs["objective_rule"]` records it), and basis cleaning and the SE
  rebuild the same problem. The rule is `haldensity.core.objective_rules.objective_rule(k, d)`, which
  returns an identifier that each data-type package implements: d = 1 as above (`haldensity.full_uni`);
  d >= 2 the multivariate estimator's existing rule (exact below the cell quota, the Sobol bank above,
  at k = 0; the Sobol bank at k >= 1), now recorded in `MultvarHAL.get_results()["objective_rule"]`.
  Fits record `objective_rule` and the implemented `objective_normalizer`. Internal hooks for tests,
  benchmarks and experiment repos only: `CVXPYEstimator(_objective_normalizer="midpoint" or
  "knot_aware")` and the class attribute `_knot_aware_formulation`. Evidence:
  `docs/dev/benchmarks/knot_aware_normalizer/` (the old midpoint rule spiked at k = 0; the other two
  never did and were as accurate or better); at k = 2 the knots-added grid cost 107-415 s per 5-fold
  CV trial at n = 1,280 against 3.4-4.3 s, with about the same CV score.
- **The old midpoint rule without the n x p data basis** (Yilong, 2026-10-24): the data term
  -sum_i f(x_i) is linear in theta, -s' theta with s the basis column sums, which are computed exactly
  from sorted data and prefix sums (O(n log n); `haldensity.core.univariate.objective.data_basis_sums`).
  The problem then holds only the normalizer's cells x p matrix (199 x p for the old midpoint rule),
  built in float64. One builder (`objective.dense_problem`) serves the fit, the CV folds, basis
  cleaning and the knots-added grid's dense form. The objective agrees with a float64 evaluation to
  1e-12 at random theta; the 18 fixed fits of the restructure check change only at the float32
  level of the old basis (densities by at most 1.3e-5, k = 1 coefficients by 9.3e-6; k = 0 fits are
  identical).
- Estimators that log no longer keep log files open: the file is opened only while a record is
  written, and each estimator's logger is private instead of being registered with
  `logging.getLogger` (which keeps every logger, and its open file, for the life of the process). In
  0.1.0 each fit with logging on held one open file, about 250 per CV task (Yilong, 2026-10-24,
  decision 9). Logs are still written to `log_dir` and started afresh for each estimator.
- **No log file unless `log_dir` is given** (Yilong, 2026-10-24; core and full data): `CVXPYEstimator` and the
  weighted MLE (the censored-data initial estimators) wrote `./local/logs/cvxpy.log` by default, in every CV
  fold; now they log only when `log_dir` (the file's path) is given, and `log_frequency` still defaults to 10
  for them. **If you relied on the default file**, pass `log_dir="./local/logs/cvxpy.log"` (through the tuner:
  `estimator_kwargs={"log_dir": ..., "log_frequency": ...}`). The tuner's 0.1.0 option `log_frequency` never
  reached the fold estimators; it now only warns (`DeprecationWarning`, removed in 0.3.0). The FISTA, proximal
  GD and projected GD estimators no longer compute logging statistics when logging is off, and
  `log_frequency=0` no longer raises `ZeroDivisionError` there. A CV run and its final fit in an empty
  directory leave it empty (tested).
- **Solver calls are single-threaded by default** (`solver_threads=1`; Yilong, 2026-10-24, for
  shared servers): per call through MOSEK's `MSK_IPAR_NUM_THREADS` and CLARABEL's `max_threads`,
  and the numpy/scipy BLAS/OpenMP pools limited with `threadpoolctl` during the call; SCS and ECOS
  are single-threaded. Applies to every estimator, since all solve through
  `haldensity.core.solvers.solve_cvxpy_problem` or the final fit's ladder. Measured on a 256-core
  server: an uncapped CLARABEL solve used 109 cores and was 4x slower than one thread; MOSEK used
  5.6 cores (1.2 s against 1.6 s single-threaded). `CVXPYEstimator(solver_threads=k)` raises it;
  `threadpoolctl` is now a direct dependency (it already came with scikit-learn). README: "Parallel
  runs and shared servers".
- **Knot cap on by default** (`CVXPYEstimator(max_knots="auto")`, Yilong 2026-10-24): at most
  K = 1,000 candidate knots for k >= 1 and K = max(1,000, ceil(n^(2/3))) for k = 0, at K data
  quantiles (`haldensity.core.univariate.objective.default_max_knots`). **Default results change for
  n > 1,000**; samples up to 1,000 keep every knot, bit for bit. `max_knots=None` keeps every
  unique data value (the old behaviour). The k = 0 rule is rough: with K knots a jump can only be
  placed to within about 1/K, costing about h^2 / K in squared L2, which must stay below k = 0's
  statistical error n^(-2/3). In the benchmark (`docs/dev/benchmarks/knot_aware_normalizer/`, the
  exact objective at k = 0), K = 1,000 cost 5-13% in L2 on a jump density at n = 2,500-20,000
  against all knots, and nothing on smooth or kinked densities. `get_results()` records
  `max_knots` and the cap used, `max_knots_used`.
- `OptunaHyperparameterTuner` uses 5 CV folds by default (`TUNER_DEFAULTS.cv_folds`, as the
  censored-data tuners) instead of 3, so default full-data CV results change; `cv_folds=3` gives
  the old split.
- The final fit is one tight, certified solve (the
  new default; `_solver_tolerance="tight"` internally): MOSEK with interior-point tolerances 1e-12 (relative gap, primal and dual
  feasibility), then CLARABEL 1e-12/1e-10, ECOS 1e-12/1e-10, SCS 1e-9 capped at 600 s
  (`haldensity.core.solvers.TIGHT_LADDER`, following `solver` / `solver_waterfall`); each solution
  is pruned at `tol` (1e-4) and checked against the optimality conditions, the first certified one
  is kept. If none is, the best attempt from any solver is kept (the fewest coordinates failing the
  conditions, `kkt_["n_failing"]`), with a warning and `certified_ = False` (Yilong, 2026-10-24;
  evidence in `docs/dev/benchmarks/final_fit_repair/`). That solution is the point estimate and its basis the SE basis
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
  solvers' own tolerances (internally `_cv_solver_tolerance="default"`); the old
  final fit is the internal `_solver_tolerance="default"` (bit for bit). The optimality check now also handles a
  bound that is not binding (||theta[1:]||_1 < (1 - 1e-3) M: certified when max |gradient| / n <=
  1e-5; at the bound either test certifies), and basis
  cleaning of `_solver_tolerance="default"` fits starts with MOSEK 1e-12 as well.
- `haldensity.rc.fit_right_censored(method=...)` and `fit_ipcw_hal`, `fit_l1mle_m1`,
  `fit_cvl1mle` (default method), `fit_undersmoothed`: the four right-censored point estimators
  with the same inputs and outputs (`RightCensoredPointEstimate`).
- `density_confidence_interval(scale=...)` (Yilong, 2026-10-24; core): `"density"` (default, unchanged) is
  density +- z SE with the lower end cut at 0, the paper's formula; `"log"` maps the interval for the log
  density back, density * exp(+- z SE / density), from the same SE (always positive). Both are compared in
  the SE study; which one is claimed is decided there. `result.attrs["scale"]` records the choice.
- `haldensity.datasets.load_galaxies()` (Roeder 1990 galaxy velocities, km/s) and
  `make_right_censored()`.
- `haldensity.ic.current_status.current_status_to_interval()`.
- `haldensity.core.univariate.HALFit`, the generic frozen fit; `RightCensoredHALFit` is its
  subclass.
- `OptunaHyperparameterTuner(estimator_kwargs=...)` and `best_estimator_params`.
- CI: the test suite on Python 3.11 for pushes and pull requests.
- Tutorials per data type in `examples/tutorials/`, using the public API only (Yilong, 2026-10-24):
  `full_uni/01_galaxies`, `full_uni/02_simulated_density`, `full_uni/03_targeting`, `full_multi/01_bivariate`,
  `rc/01_point_estimators` and `rc/02_tmle` (standard errors for survival, the mean and RMST; density L2 and
  entropy without, as their SEs were too small in a simulation, Yilong 2026-10-25). CI runs them with nbmake (`pytest --nbmake examples/tutorials`, without MOSEK;
  nbmake joins the `dev` extra). The saved outputs come from a run with MOSEK.
- From the consolidated development branch: right-censored observed-data L1-MLE, its CV tuner
  and fit object; right-censored targeting (survival, mean, density L2, RMST, entropy) and EIF
  helpers; AIPW; interval-censored FISTA and projected-GD estimators with knot strategies;
  CV-oversmooth EM tuners; the solver fallback helper; the multivariate estimator and
  tensor-product basis.

### Changed (behaviour)

- **Every basis and design matrix is built in float64** (Yilong, 2026-10-24; core):
  `create_basis_functions`, the multivariate tensor basis (`create_multivariate_basis`) and the full-data
  targeting bases were float32 since the package's first commit, for no recorded reason. Full-data fits and
  their optimality check already ran in float64 (the lean data term), so full-data point estimates are
  unchanged bit for bit; what moves is every quantity that used the float32 basis. Before/after on 14 fixed
  fits (MOSEK): the delta-method SE by at most 1.7e-6 relative on simulated data and 3.9e-5 on Galaxies
  (k = 2, M = 596,793, ridge 1e-6 or 1e-10); full-data targeting coefficients by at most 1.1e-4 relative
  (k = 2); right- and interval-censored k = 1 densities by at most 1e-6 relative (their coefficients by up to
  2.9e-4, along a poorly determined direction); the multivariate k = 1 density by 5e-4 relative (coefficients
  by up to 13%: 1,004 columns and 512 normalizer points do not determine them); every k = 0 fit is identical
  (its basis is 0/1). Memory: a design matrix on its own doubles (n = 10,000 with every knot: 400 to 800 MB;
  multivariate d = 2, k = 1, n = 2,000: 80 to 160 MB), but whole fits grow little, because CVXPY keeps its
  own float64 copy: peak memory of an IPCW right-censored fit at n = 3,200 466 vs 464 MB, interval-censored
  n = 1,600 338 vs 345 MB, multivariate d = 2 at n = 500 1,395 vs 1,348 MB and at n = 1,000 (k = 0, the
  largest case measured) 2,377 vs 2,695 MB (+13%); times unchanged (188 vs 196 s for the largest).
- **The delta-method band's centre and the SE use the accurate normalizer** (Yilong, 2026-10-24; core):
  `estimate_covariance_beta` and `density_confidence_interval` computed the normalizing constant and E[phi]
  (the centring of the scores) with a 200-point trapezoid on [0, 1]. They now use the rule of the accurate
  normalizer (Gauss-Legendre with breakpoints at the selected knots, exact for k = 0;
  `haldensity.core.univariate.normalizer.quadrature_rule`), so the band's centre is the density that
  `get_density_at_points` reports, to the last digit (before: up to 5.1% below it on Galaxies, whose data
  occupy 8% of [0, 1], and 1.2% off for a k = 0 fit of simulated data). Before/after on the same five fixed
  fits (MOSEK): on Galaxies (k = 2, M = 596,793) the centre rises by 5.4% and the SE by 4-12% (ridge 1e-6)
  or 4-15% (ridge 1e-10); for the k = 0 fit (n = 400) both rise by 1.2%; for the k = 1 and 2 fits of simulated
  data the centre moves by less than 6e-5 and the SE by less than 8e-4 relative. Fits are unchanged.
- **The delta-method SE inverts the information with the scaled floor** (Yilong, 2026-10-24, after SE round
  2; core): `estimate_covariance_beta` adds a ridge of 1e-5 times each column's own score variance (the
  information scaled to unit diagonal, plus 1e-5 I, scaled back) instead of the absolute ridge 1e-6. The rule
  has no units, so the SE no longer changes when the data's units do; the intercept and any column constant on
  the data are dropped (no spread there, e.g. the knot at min(x) at k = 0, whose centred score is not 0; review
  C1, 2026-10-25: it had been kept, adding about p(x)^2 / n to the variance; k = 0 SEs fall by 1-7% where that
  knot is selected). Evidence (round 2, 180 cells, rules fixed in advance): on the original data the floor
  matches the ridge's coverage (mean |coverage - 95%|, floor minus ridge, -0.026, bootstrap SE 0.0015); on
  copies of the same data with squeezed units the ridge loses more than 3 coverage points in 98 of 180 cells,
  while the floor's SE is unchanged to 1e-9. **SEs and intervals change**: before/after on 12 fixed fits (the
  same cleaned fit, M chosen by CV; SE new / old, median over the data's 5-95% quantiles): k = 0 0.97-1.00
  (below 1 only where the knot at min(x) is selected); on three simulated samples that fill [0, 1] (n = 400) 1.01-1.13 at k = 1 and 1.18-1.19 at
  k = 2 (1.68 for a fit with 349 knots); on Galaxies (8% of [0, 1]) 1.64 at k = 1 and 4.38 at k = 2, where
  the old ridge was too narrow (band width over the data range 0.43 times a bootstrap's at fixed M; the floor's
  is 1.61 times). The absolute ridge stays available internally (`_inverse="ridge"`, `_ridge_param`) as the
  reference the paper's experiments compute alongside.
- **Full-data targeting integrates exactly** (Yilong, 2026-10-24; full data, univariate): the M-step's
  normalizer, the plug-in and TMLE functionals and the median's EIC variance were sums over a 5e-4 grid, so the
  score equations held at the nearest grid cell, not at the targeting point (on Galaxies the targeted survival
  at t = 0.0667, where the density is about 73, missed the empirical survival by 0.012). The integrals now use
  cells with edges at the knots and at the points where the targeting basis jumps (the targeting points, or the
  initial fit's median), with Gauss-Legendre inside each cell, doubled until log Z changes by less than 1e-12
  (`haldensity.full_uni.targeting.exact`). After the solver, a few Newton steps on the targeting coefficients
  (while the L1 bound is not binding) bring the score equations to about 1e-12. `run_m_step` also returns the
  exact values: `estimate` (the TMLE), `plug_in` (the initial fit's value), `density_at_estimate` (median),
  `log_normalizer`, `integration_order`, `score_residual`, `solver_used` and `solver_status`; its
  `estimated_density` on the fine grid is normalized exactly, and the variance functions use the exact estimate
  (a results dict without one falls back to the grid sums with a warning). The initial median that the median
  learner targets is computed exactly (it was read off a 2,001-point grid). Galaxies (k = 2; bound 100; survival
  at the old notebook's ten points, the mean, the median): every score equation holds to 2e-13 at the case
  study's fit (200 grid points, M = 596,793) and to 6e-16 at 1,000 points (M = 653,813); a survival M-step
  takes 0.2-0.4 s. Before/after on two fixed fits of simulated data (n = 400, k = 0 and 2): the score-equation
  errors fall from up to 1.0e-3 (survival) and 4.8e-4 (mean) to below 1e-15, the TMLE values move by up to
  1.0e-3, the survival and mean variances are unchanged (sample variances of the data), and the median's
  variance moves by 0.7% (k = 2) and 33% (k = 0, whose density at the median is a step). The median learner
  remains one targeting step at the initial median m0: F(m0) = P_n(X < m0) now holds exactly, but the
  targeted median need not be an empirical median (Galaxies: P_n(X <= median) = 43/82 with 200 points, 39/82
  with 1,000). The right-censored targeting keeps its grids (RC project).
- Full-data CV can apply the conservative adjustment (core, shared with the censored-data tuners
  through `haldensity.core.cv.conservative`): after the search, the smallest bound whose mean CV
  log-likelihood is within 5% of a fold SD of the best, found by bracketing and bisection in log M.
  **Off by default** (`use_conservative_adjustment=False`; Yilong, 2026-10-24): `True` turns it on,
  `"auto"` only when the CV-best basis order is 0 (an interim default earlier in 0.2.0); the
  CV-best stays in `max_cv_params`. Why off: at k = 0 with the exact objective (four truths, n = 320
  and 1,280, 50 replicates each) the fit at the CV-best bound had KL 0.82-0.99 times the adjusted
  fit's (8 of 16 KL/L2 intervals below 1, none above), the adjustment doubled the smooth truth's
  boundary bias (0.207 vs 0.123 at x = 0.01), and the flat CV curve it guards against appeared in
  0 of 800 fits; at k >= 1 it oversmoothed (in 16 simulated k = 1 datasets it shrank M to
  0.11-0.87x the CV-best and made L2 error worse in 11). Default results change at k = 0 against
  the interim default.
- Full-data CV scores each fold by evaluating the fitted log density exactly at the validation
  points (with the accurate normalizer) instead of linearly interpolating `get_density()` on its
  200-point grid. In 16 simulated datasets (4 densities, n = 400 and 1,280, k = 1) the selected
  bound was unchanged in 11; in the 4 n = 1,280 cases where the old score picked a much larger M,
  held-out log-likelihood rose by 0.019-0.030 nats and L2 error fell by 0.33-0.56. Old
  behaviour (internal hooks): `_cv_scoring="interpolated"` with `estimator_kwargs={"_normalizer": "midpoint"}`.
- Full-data estimators evaluate and report the density with an accurate normalizer: log Z by
  cell-wise Gauss-Legendre with breakpoints at the knots with nonzero coefficients (32 nodes per
  cell, doubled until the change is below 1e-12; exact for k = 0), in float64. The midpoint rule
  on the fitting grid underestimated Z for spiky fits (0.175 nats at k = 1, M = 1e5, n = 1280).
  The fitting objective is unchanged. Old behaviour (internal hook): `_normalizer="midpoint"`. New:
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
- **Python 3.11 or later** (`requires-python = ">=3.11"`; Yilong, 2026-10-25): 0.1.0 declared 3.9, but CI tests
  3.11 only and the README already said 3.11+. The license metadata is the SPDX expression `license = "MIT"`
  with `license-files`, and the build needs setuptools 77 or later (the TOML-table form and the license
  classifier are deprecated and stop building after 2027-02-18).
- **matplotlib and ipykernel are an extra** (`pip install "haldensity[examples]"`; Yilong, 2026-10-25): they
  were dependencies of the package since 0.1.0, but no package code imports them; only the tutorials need them.
  CI installs `.[dev,examples]`; `examples/tutorials/README.md` gives the install line.
- **Right-censored data: the largest observed time counts as an event when it is censored** (the
  Kaplan–Meier convention; Yilong, 2026-10-24; rc). `KaplanMeier.fit` applies it to the data it is fitted
  on, so the censoring survival estimate stays positive at every observed time; before, it fell to 0 at a
  censored largest time. The IPCW-HAL fit, the observed-data estimators, the EM estimators, the five
  targeting steps and AIPW apply it to their data, so the IPCW weights, the likelihoods and the influence
  curves count that observation as an event. The right-censored tuners split their folds on the data as
  observed and apply it to each training fold and to the final fit only; held-out losses see the
  validation data as observed (review R2, 2026-10-25: applying it to the sample before the split had let
  a validation fold score a censored largest time as an event; on the rc/01 tutorial's sample Stage-1 CV
  then picked M1 = 26.6 instead of 78.6).
  New `haldensity.rc.last_time_as_event(data)` applies it (use it before calling `compute_ipcw_weights`
  directly). Why: when the largest time was censored (about 10% of samples in the tutorials' design,
  Beta(2.2, 2.5) event times with Uniform[0, 1.2] censoring, at n = 200 to 1,000), the influence curves of
  the mean, RMST, entropy and density L2 blew up (down to −2,595 for the mean on the rc/02 tutorial's
  sample), and their default standard errors either used those values (density L2: 2.76 for an estimate of
  1.18) or left out every censored observation (mean, RMST, entropy). **Results change** for such samples,
  and slightly for others through CV training folds whose largest time is censored.
- **Right-censored targeting reports the SE over all observations** (Yilong, 2026-10-25; rc). The
  positivity-filtered SE is removed: the default of the mean, RMST and entropy targets, it left out the
  observations near or beyond the point where the censoring-survival estimate fell below its floor (every
  censored observation when the largest time was censored), and it divided by the number kept. Its summary
  columns (`clip_active_eic_*`, `positivity_eic_*`, `estimand_variance_clip_active_filtered*`,
  `standard_error_clip_active_filtered*`) and stored masks go with it. The floor on the censoring-survival
  estimate inside the influence curves, max(1/(sqrt(n) log n), clip), stays as a positivity safeguard; the new
  column `gbar_floor_active` says whether it was active. In the convention check (3,600 replicates of the
  tutorials' design, n = 200 to 1,000) neither the floor nor the filter triggered once the convention was in.
  Results where nothing was left out are unchanged (the one-step variances to rounding, 3e-16).

The old behaviour of these fixes is available at `v0.1.0`.

### Fixed

- Full-data targeting crashed for an order-0 fit with no selected knots (review T3; older code):
  `create_basis_functions` with no columns now returns an (n, 0) matrix instead of failing in `np.column_stack`.
- Full-data targeting variances (review T2 and T4; older code): the influence curve of S(t) = P(X > t) used
  I(x >= t) and that of F(t) = P(X <= t) used I(x < t), so an observation exactly at t counted on the wrong
  side; and both functions failed with a TypeError when `targeting_points` was a Python list. They now use
  I(x > t) and I(x <= t) directly and accept any sequence of points.
- **Right-censored influence curves: the censoring term is the jump of 1/G** (review R1, Yilong 2026-10-25;
  rc; older code). Each censoring jump u contributed E[phi(T*) | T* > u] dG(u) / G(u)^2; the discrete IPCW
  mapping needs dG(u) / (G(u-) G(u)), the jump of 1/G at u. The two differ most at the end of follow-up, where
  risk sets are small (by a factor of 2 with 2 at risk). Fixed in one helper used by all five targets, survival
  v1 and v2 and AIPW. Check: under a discrete censoring distribution, E[D | T* = t] = phi(t) - psi holds to
  1e-13 (the old form was off by up to 4.3 in the same test). Effect on three fixed samples (n = 200 and 400,
  the initial fit fixed): the SE of the mean +7% to +16%, the entropy +1% to +13%, density L2 -2% to -4%,
  survival at 0.8 and AIPW +1% to +7% (unchanged at 0.2-0.6), RMST(0.8) 0% to +1%; the targeted estimates
  move by at most 0.14 SE.
- `np.trapz` (removed in NumPy 2.4) replaced in the variance module; delta-method bands crashed
  on current NumPy.
- The README's targeting example failed at order k >= 1 with a shape error: `get_results()["theta_hat"]`
  holds a coefficient for every candidate knot, zeros included, while the M-step expects the intercept, the
  powers and the nonzero knot coefficients with their knots (Yilong, 2026-10-24; full data). New
  `haldensity.full_uni.targeting.targeting_inputs(fit)` builds those inputs from a fitted estimator or its
  results; `run_m_step` raises an error that names it when `old_theta` does not match the knots and the order;
  the README's targeting section uses it (targeting points inside the data, the exact `estimate` and
  `plug_in`), and a test runs the README's snippets at k = 0, 1 and 2. The median learner's docstring states
  that it takes one targeting step at the initial median, as in the paper.

### Removed

- `KFold_CV_LambdaSelector`, `CVSelectorMetric` (unused).
- Research notebooks moved to their project repositories (Yilong, 2026-10-24; copies checked against the
  blob hashes, each with a README_COPY.md there): `examples/multivar-HAL/` to multivar-HAL,
  `examples/AIPW-HAL-MLE/` to HAL_TMLE_RC, `examples/demo_conservative_adjustment_search.ipynb` to
  uni_EM_HAL_RC. They were never in `v0.1.0`; they stay in this repository's git history.
- `examples/example_right_censored_density_estimator.ipynb` (in `v0.1.0`) moved to HAL_TMLE_RC: EM stays in
  the API but is no longer presented in the README, the docs or the tutorials (Yilong, 2026-10-24: for
  interval-censored and current-status data, optimizing the observed-data likelihood directly works better;
  EM may return with stochastic EM). `docs/censored-data.md` is rewritten for the 0.2.0 layout around the
  recommended routes: the four right-censored point estimators and targeting, and for interval-censored data
  the midpoint initial fit followed by direct optimization (FISTA, projected gradient).
- Example notebooks replaced by the tutorials (still in git history): `example_full_data_density_estimator.ipynb`
  and `example_full_data_target_learning.ipynb` (both in `v0.1.0`), `example_multivariate_full_data_api.ipynb`
  and `example_right_censored_fit_object.ipynb`; `example_full_data_case_study.ipynb` (in `v0.1.0`; its copy
  with the bootstrap is in the paper repository HAL-MLE-DE-AOS, `from_haldensity/examples/`) and
  `examples/plotting.py` (in `v0.1.0`; each tutorial draws its own plots).
- `examples/HAL-TMLE-RC/` (not in `v0.1.0`): its right-censored targeting demos are replaced by
  `rc/02_tmle`; the whole folder, with the research notebooks and notes, is in HAL_TMLE_RC
  (`from_haldensity/examples/HAL-TMLE-RC/`, copied from 6001abe, every file identical).
- Right-censored comparison helpers (truth objects, error metrics, M1–M4 experiment runner,
  two simulators) and the plug-in pipelines `fit_right_censored_cv_ipcw_l1mle_plugin` /
  `..._undersmoothed_plugin`; moved to the HAL_TMLE_RC project repository and replaced by
  `haldensity.rc.fit_right_censored`.
- `interval_censor_width_uniform` (legacy, informative censoring).
- `extract_term_type_metrics` (experiment reporting; moved to the multivar-HAL repository).
- Stray files (a matplotlib font cache, a 2 MB bootstrap result, scratch notebooks) and
  `examples/MCMC-HAL/`, summarized in `docs/dev/experiments/sobol-normalizer-1d.md`.
