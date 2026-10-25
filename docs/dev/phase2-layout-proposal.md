# HALDensity layout proposal (Phase 2)

> **Status (2026-10-23).** Approved by Yilong with these changes, and carried out in Phase 3:
> the univariate base lives in `core/univariate/`; there is **no** `core/targeting/` (each data
> type keeps its own targeting; the shared SE/Wald helper was skipped because the full-data and
> right-censored code do not compute it the same way); the EM and tuner defaults stay one module
> (`core/defaults.py`) to avoid an import cycle; the two plug-in pipelines became four
> right-censored point estimators in `rc/pipelines.py`; generators stay one module per class in
> `datasets/`; `KFold_CV_LambdaSelector` is dropped; the current-status and interval-censored
> notebooks stay in place for now; `examples/MCMC-HAL/` is replaced by
> `docs/dev/experiments/sobol-normalizer-1d.md`. Kept below as the design record.

Proposal only. No file has moved. Base: branch `consolidate` at `43e93fd` (origin/main + 28 commits, 83/83 tests).
Written 2026-10-23 for Yilong's review, from his layout decisions of the same day.

## Labels used below

- **core**: code used by more than one data type. Lives in `haldensity/core/` and never imports a data-type package.
- **full_uni / full_multi / rc / ic**: code for one data type: univariate full data, multivariate full data, right-censored data, interval-censored data (current status nested in `ic/current_status/`). A data-type package imports only `core` and `datasets`, never another data-type package. Today `censoring/_base_mle.py` imports `censoring/right/fit.py`; that is the kind of import this rule forbids.
- **datasets**: sample data and simple simulators for tests and tutorials (`haldensity/datasets/`).
- **out**: leaves the package for a project repo. The lead copies the files; Phase 3 deletes them from the package.
- Kind of change:
  - **move**: the file moves unchanged except its import lines.
  - **split**: parts of one file go to several places, with the code unchanged.
  - **extract**: new shared code is factored out of existing code, with a before/after check that results are unchanged.
  - **new**: new code.

## 1. Target layout

```
haldensity/
  __init__.py        version; subpackages loaded on first use; installs the old-path aliases (section 5)
  _compat.py         old import path -> new module table, DeprecationWarning
  core/
    basis/           univariate.py (create_basis_functions), multivariate.py (general d, tensor products)
    base_estimator.py   univariate HAL estimator base (data column W1 on [0, 1], density evaluation, normalizer)
    weighted_mle.py     WeightedHALMLEEstimator: L1-constrained weighted HAL-MLE (full data = unit weights; IPCW; EM M-step)
    fit.py              HALFit: frozen fit object (theta, knots, normalizer), to_state/from_state       [extract]
    solvers/cvxpy.py    solver waterfall helper
    functionals.py      survival / CDF / median / moments of a univariate density on a grid
    cv/              tuner_base.py (shared Optuna CV over the L1 bound), conservative.py (bisection search) [extract],
                     defaults.py (TunerDefaults), lambda_selector.py (unused; see open question 8)
    em/              defaults.py (EMDefaults, EMStageResult), tuners.py (EM and CV-oversmooth tuner bases)
    targeting/       fluctuation.py (tilt + epsilon root-finding), loop.py (iterate until mean EIC is small),
                     inference.py (EIC variance, Wald interval)                                        [extract]
    variance.py      delta-method variance and CIs (+ basis cleaning, Phase 3 fix 3)
    metrics.py       kl_divergence
  full_uni/
    estimators/      cvxpy, fista, projected_gd, proximal_gd, proximal_adagrad, proximal_newton, proximal_newton_lbfgs(_full)
    tuner.py         OptunaHyperparameterTuner (full-data CV); will call core/cv/conservative.py
    normalizer.py    exact / Gauss-Legendre log Z for evaluation (Phase 3 fix 2), or in core (deviation 1)
    targeting/       base.py (BaseTargetLearner), cdf.py, median.py, moments.py, survival.py (+ the two indicator-basis files)
  full_multi/
    estimator.py     MultvarHAL (Sobol/QMC normalizer); warns when d = 1 and points to full_uni       [new warning]
    targeting/       (empty until a multivariate target exists)
  rc/
    km.py, weights.py, estimators.py (IPCW init + EM), observed_mle.py, fit.py (RightCensoredHALFit(HALFit)),
    metrics.py, tuners.py, _tuner_utils.py, aipw.py, pipelines.py (open question 1)
    targeting/       _common.py, eic.py, survival.py, mean.py, density_l2.py, rmst.py, entropy.py
  ic/
    estimators.py, metrics.py, tuners.py, _tuner_utils.py
    current_status/  adapter (C, Delta) -> (L, R); README noting CS theory differs from IC              [new]
  datasets/
    galaxies.py + data/galaxies.csv   load_galaxies()                                                  [new]
    densities.py     TruncatedNormal, TruncatedGMM, Sinusoidal, StepFunction, BivariateTruncatedNormal
    censoring.py     make_right_censored, make_interval_censored, make_current_status
```

## 2. Where I depart from the decided layout, and why

1. **Univariate normalizer and univariate HAL base go in core, not full_uni.** This covers `BaseEstimator`, `WeightedHALMLEEstimator`, the density functionals, and the accurate log Z of fix 2. `rc` and `ic` are univariate too and use all of them: `WeightedHALMLEEstimator` is the IPCW fit and the EM M-step for both. Under the import rule they cannot live in `full_uni`. `full_uni` would re-export them. The Sobol normalizer stays in `full_multi`.
2. **The 8 full-data solver estimators go to `full_uni/estimators/`**: `CVXPYEstimator`, FISTA, projected and proximal gradient, AdaGrad, and the three Newton variants. Each hard-codes the full-data likelihood. Generic solver code in core is today only the CVXPY waterfall. The first-order solvers are re-implemented three times: here, in `rc/observed_mle.py`, and in `ic/estimators.py` (FISTA and projected GD). Lifting one generic copy into core is a later refactor, not part of Phase 3.
3. **`BaseTargetLearner` goes to `full_uni/targeting/`, not core.** Its M-step refits the full-data likelihood, and only the four full-data targets use it. The RC targets do not use it: they carry their own machinery inside `right_censored_survival/learner.py`. The generic core targeting would instead be extracted from that file:
   - from `_find_score_bracket`, `_solve_epsilon_t0`, `_tilted_density_and_survival`: the tilt and the root-finding for epsilon;
   - from `_iterate_tmle_t0`, `_iterative_tolerance_from_sigma`: the iterate-until-small loop;
   - from `_estimate_eic_variance*`, `_build_wald_interval`: the inference.

   The RC log-likelihood, score and EIF stay in `rc`.
4. **EM machinery shared by `rc` and `ic` goes to `core/em/`**: `EMDefaults`, `EMStageResult`, `BaseCensoredEMTuner`, `BaseCVOversmoothEMTuner`, and the oversmooth records. Both data types use it, and they may not import each other.
5. **The two CV-IPCW-L1MLE plug-in pipelines in `comparison.py`** may stay in `rc` (open question 1).

## 3. File-by-file move list

Paths are under `src/haldensity/`. Line counts are at `43e93fd`. The 20 small `__init__.py` files below `estimation/` (11) and `targeting/<target>/` (9) are not listed; their re-exports go to the new `__init__` files.

| Current file | Lines | New place | Label | Kind |
| --- | ---: | --- | --- | --- |
| `__init__.py` | 10 | same | core | new: lazy subpackages, compat |
| `censoring/__init__.py` | 161 | old-path shim | — | compat |
| `censoring/README.md` | — | `docs/rc.md`, `docs/ic.md` | — | split |
| `censoring/_base_mle.py` | 381 | `core/weighted_mle.py` | core | move; `get_fit()` returns core `HALFit`, RC subclasses override → backwards import gone |
| `censoring/_defaults.py` | 109 | `core/em/defaults.py` (EMDefaults, EMStageResult) + `core/cv/defaults.py` (TunerDefaults) | core | split |
| `censoring/interval/estimators.py` | 1575 | `ic/estimators.py` | ic | move |
| `censoring/interval/metrics.py` | 94 | `ic/metrics.py` | ic | move |
| `censoring/interval/simulate.py` | 158 | `datasets/censoring.py` (inspection-uniform, current-status-uniform, dispatcher); `interval_censor_width_uniform` (legacy, informative) → out | datasets / out | split |
| `censoring/right/comparison.py` | 917 | out → HAL_TMLE_RC (DGPs, truth objects, error metrics, M1–M4 runner); the 2 plug-in pipelines → `rc/pipelines.py` without `truth=` (open question 1) | out / rc | split |
| `censoring/right/estimators.py` | 739 | `rc/estimators.py` | rc | move |
| `censoring/right/fit.py` | 311 | generic part → `core/fit.py` (`HALFit`); `RightCensoredHALFit(HALFit)` + `last_failure_from_frame` → `rc/fit.py` | core / rc | extract (state format unchanged; old states load) |
| `censoring/right/km.py` | 138 | `rc/km.py` | rc | move |
| `censoring/right/metrics.py` | 192 | `rc/metrics.py` | rc | move |
| `censoring/right/observed_mle.py` | 1256 | `rc/observed_mle.py` | rc | move |
| `censoring/right/weights.py` | 64 | `rc/weights.py` | rc | move |
| `censoring/tuners/_base.py` | 1205 | `TuningResult`, threshold helpers, `BaseCensoredInitTuner` → `core/cv/tuner_base.py`; records + EM tuner bases → `core/em/tuners.py`; conservative search → `core/cv/conservative.py` | core | split, then extract the search |
| `censoring/tuners/_utils.py` | 190 | `create_s_c_predict`, `prepare_ipcw_data`, `fit_rc_init_estimator` → `rc/_tuner_utils.py`; `fit_ic_init_estimator` → `ic/_tuner_utils.py` | rc / ic | split |
| `censoring/tuners/right_tuners.py` | 1224 | `rc/tuners.py` | rc | move |
| `censoring/tuners/interval_tuners.py` | 835 | `ic/tuners.py` | ic | move |
| `censoring/utils/common_metrics.py` | 66 | `core/metrics.py` | core | move |
| `cross_validation/optuna_hyperparam_selector.py` | 647 | `full_uni/tuner.py` | full_uni | move; then uses `core/cv/conservative.py` (Yilong: the conservative adjustment is core, full-data CV uses it too) |
| `cross_validation/lambda_selector.py` | 233 | `core/cv/lambda_selector.py` or drop | core | move (open question 8) |
| `density_variance/density_variance.py` | 160 | `core/variance.py` | core | move (fix 3 adds basis cleaning here) |
| `estimation/base_estimator.py` | 375 | `core/base_estimator.py` | core | move |
| `estimation/conic_optimization_method/cvxpy/estimator.py` | 171 | `full_uni/estimators/cvxpy.py` | full_uni | move |
| `estimation/first_order_method/fista/estimator.py` | 231 | `full_uni/estimators/fista.py` | full_uni | move |
| `estimation/first_order_method/projected_gradient_descent/estimator.py` | 255 | `full_uni/estimators/projected_gd.py` | full_uni | move |
| `estimation/first_order_method/proximal_gradient_descent/estimator.py` | 212 | `full_uni/estimators/proximal_gd.py` | full_uni | move |
| `estimation/second_order_method/proximal_adagrad/estimator.py` | 403 | `full_uni/estimators/proximal_adagrad.py` | full_uni | move |
| `estimation/second_order_method/proximal_newton/estimator.py` | 354 | `full_uni/estimators/proximal_newton.py` | full_uni | move |
| `estimation/second_order_method/proximal_newton_lbfgs/estimator.py` | 535 | `full_uni/estimators/proximal_newton_lbfgs.py` | full_uni | move |
| `estimation/second_order_method/proximal_newton_lbfgs_full/estimator.py` | 836 | `full_uni/estimators/proximal_newton_lbfgs_full.py` | full_uni | move |
| `multivariate/estimator.py` | 520 | `full_multi/estimator.py` | full_multi | move; + d = 1 warning (new) |
| `targeting/base_target_learner.py` | 302 | `full_uni/targeting/base.py` | full_uni | move |
| `targeting/cdf/learner.py`, `targeting_basis.py` | 244 + 40 | `full_uni/targeting/cdf.py`, `cdf_basis.py` | full_uni | move |
| `targeting/median/learner.py` | 269 | `full_uni/targeting/median.py` | full_uni | move |
| `targeting/moments/learner.py` | 222 | `full_uni/targeting/moments.py` | full_uni | move |
| `targeting/survival/learner.py`, `targeting_basis.py` | 243 + 40 | `full_uni/targeting/survival.py`, `survival_basis.py` | full_uni | move |
| `targeting/_right_censored_eic.py` | 567 | `rc/targeting/eic.py` | rc | move |
| `targeting/right_censored_survival/learner.py` | 2102 | RC machinery (fit/grid/censoring cache, Gbar floor, RC loglik/score/EIF) → `rc/targeting/_common.py`; survival target → `rc/targeting/survival.py`; AIPW (`PointwiseAIPWResult`, `_target_aipw_t0`, `_package_pointwise_aipw_fit`, `RightCensoredSurvivalAIPWLearner`, `right_censored_survival_aipw*`) → `rc/aipw.py`; generic pieces → `core/targeting/` | rc / core | split, then extract |
| `targeting/right_censored_mean/learner.py` | 1181 | `rc/targeting/mean.py` | rc | move |
| `targeting/right_censored_density_l2/learner.py` | 1238 | `rc/targeting/density_l2.py` | rc | move |
| `targeting/right_censored_rmst/learner.py` | 1253 | `rc/targeting/rmst.py` | rc | move |
| `targeting/right_censored_entropy/learner.py` | 1143 | `rc/targeting/entropy.py` | rc | move |
| `utils/basis.py` | 63 | `core/basis/univariate.py` | core | move |
| `utils/multivariate_basis.py` | 402 | `core/basis/multivariate.py` | core | move (at d = 1 it gives the same columns as the univariate basis) |
| `utils/cvxpy_solver.py` | 113 | `core/solvers/cvxpy.py` | core | move |
| `utils/density_computations.py` | 175 | `core/functionals.py` | core | move |
| `utils/dgp/*.py` (5 files) | 759 | `datasets/densities.py` | datasets | move |
| `targeting/__init__.py`, `utils/__init__.py`, `estimation/__init__.py`, `cross_validation/__init__.py`, `multivariate/__init__.py`, `censoring/{right,interval,tuners,utils}/__init__.py` | — | old-path shims | — | compat |

New files:
- `core/fit.py`
- `core/cv/conservative.py`
- `core/targeting/*`
- `ic/current_status/*`
- `datasets/galaxies.py` + `datasets/data/galaxies.csv`, from `examples/case_study/galaxies.csv`: 82 velocities in km/s; Roeder (1990), JASA 85, 617–624; also `MASS::galaxies`. Check the CSV against the MASS copy and its help page before shipping.
- `datasets/censoring.py`
- `_compat.py`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `docs/` (including Yilong's multicollinearity note)

**Tests** mirror the layout: `tests/core`, `tests/full_uni`, `tests/full_multi`, `tests/rc`, `tests/ic`, `tests/datasets`.
- The five RC targeting tests each carry an inline copy of the same Beta(2.2, 2.5) / Unif[0, 1.2] simulator. Those copies clip event times to [0.001, 0.999]; `simulate_beta_uniform_right_censored` does not clip. They switch to `datasets.make_right_censored`, which makes the same random-number calls and has an optional event clip, so both kinds of caller get identical draws.
- `test_right_censored_initial_comparison.py` goes out with `comparison.py`.
- `test_right_censored_cv_ipcw_l1mle_survival_api.py` stays if the pipelines stay (open question 1).

## 4. Generators, AIPW and comparison code

| Today | Proposal |
| --- | --- |
| `utils/dgp/` (TruncatedNormal, TruncatedGMM, Sinusoidal, StepFunction, BivariateTruncatedNormal) | `datasets/densities.py`. Simple densities with a known true density, used by 4 test files. |
| `censoring/interval/simulate.py` | `datasets/censoring.py`: `make_interval_censored` (inspection-uniform, used by 2 test files) and `make_current_status`. The legacy informative `interval_censor_width_uniform` goes out. |
| `comparison.py: simulate_beta_uniform_right_censored` | Out with `comparison.py`. `datasets.make_right_censored` replaces it for tests and tutorials, with the same defaults (Beta(2.2, 2.5) events, Unif[0, 1.2] censoring). |
| `comparison.py: simulate_truncnorm_uniform_right_censored` | Out with `comparison.py`. Only the comparison test uses it. |
| AIPW (in `right_censored_survival/learner.py`) | Kept: `rc/aipw.py`. The docstring states that AIPW is not a plug-in, so its estimate can leave the parameter's range (e.g. a survival probability outside [0, 1]), and that this is why targeting is preferred. |
| Rest of `comparison.py` (truth objects, error-vs-truth metrics, M1–M4 runner) | Out → HAL_TMLE_RC (the lead copies it). |

## 5. Public API before and after

**Before.**
- `import haldensity` loads `censoring` and `multivariate` eagerly.
- Users import from `haldensity.estimation`, `haldensity.cross_validation`, `haldensity.censoring[.right|.interval|.tuners|.utils]`, `haldensity.targeting[.<target>]`, `haldensity.multivariate`, `haldensity.utils[.dgp]` and `haldensity.density_variance.density_variance`.
- Tests and project code also import deep module paths and private functions, for example `haldensity.targeting.right_censored_survival.learner._solve_epsilon_t0` and `haldensity.censoring.tuners.right_tuners` (monkeypatched in tests).

**After.** Each subpackage's `__init__` exports its public names under the same class and function names as today. Phase 3 renames nothing.
- `haldensity.core`
- `haldensity.full_uni` (estimators, `OptunaHyperparameterTuner`, full-data targets)
- `haldensity.full_multi`
- `haldensity.rc` (+ `.targeting`)
- `haldensity.ic` (+ `.current_status`)
- `haldensity.datasets`

**Old paths keep working through 0.2.x**, with a `DeprecationWarning` naming the new path, and are removed in 0.3.0.
- **Aggregate packages** (`haldensity.censoring`, `.targeting`, `.utils`, `.estimation`, `.cross_validation`, `.multivariate`): small real shim modules that re-export from the new places.
- **Leaf modules** (e.g. `haldensity.censoring.right.observed_mle`): an alias table in `_compat.py`, installed as an import hook. The old name resolves to the *same module object* as the new one, so private names, monkeypatching and unpickling old objects all keep working.
- One test imports every old path and checks it resolves and warns.
- Downstream repos pin old commits and are unaffected until they upgrade. Checked 2026-10-23:
  - Pins on tagged commits: HAL_TMLE_RC (`ad2a5e7`), CS-HAL-MLE (`bbd3021`), multivar-HAL (`d58d14b`), HALDensity-Censored-Data-Experiments (`cae05ac`).
  - HAL_TMLE_RC_2 pins `c7ef168`, which is **untagged**. The archive tag Yilong already decided on is required before deleting `HAL_TMLE_RC_2_package`.
  - EM-HAL-MLE-uni pins the branch `right-censored-data-oversmooth`, which no longer exists on GitHub. That pin is already broken.

## 6. Examples triage (file by file)

Every row needs Yilong's decision.
- **T**: rewrite as a curated tutorial in `examples/tutorials/<data type>/`. Public API only, no absolute paths, small outputs, runs in a clean environment, executed in CI.
- **P**: move to a project repo (the lead copies; the package copy is deleted).
- **D**: delete (it stays in git history; only the five top-level notebooks and `plotting.py` were in `v0.1.0`).

| File | Size | Proposal |
| --- | ---: | --- |
| `example_full_data_case_study.ipynb` | 468K | T → `full_uni/01_galaxies.ipynb`: `load_galaxies()`, CV fit, delta-method bands, no bootstrap. Original with bootstrap → P HAL-MLE-DE-AOS. |
| `example_full_data_density_estimator.ipynb` | 232K | T → `full_uni/02_simulated_density.ipynb` |
| `example_full_data_target_learning.ipynb` | 264K | T → `full_uni/03_targeting.ipynb` |
| `example_multivariate_full_data_api.ipynb` | 820K | T → `full_multi/01_bivariate.ipynb` (outputs shrunk) |
| `example_right_censored_density_estimator.ipynb` | 148K | T → `rc/01_ipcw_em_density.ipynb` |
| `example_right_censored_fit_object.ipynb` | 12K | T → merged into `rc/02_observed_l1_and_fit_object.ipynb` |
| (copied out) `observed-l1-api-warm-start-demo` | 157K | T source for `rc/02` |
| (copied out) `cv-ipcw-l1mle-survival-api-demo` | 235K | T source for `rc/03_survival_plugin_and_targeting.ipynb` (needs the pipelines, open question 1) |
| `HAL-TMLE-RC/hal-tmle-rc-mean-demo.ipynb` | 364K | P HAL_TMLE_RC; T source for the targeting half of `rc/03` |
| `example_interval_censored_density_estimator.ipynb` | 148K | T → `ic/01_density.ipynb` |
| `HAL-TMLE-RC/` other 7 notebooks, `rc_target_learning_demo_utils.py`, 3 plan/notes `.md`, `README.md` | ~2.9M | P HAL_TMLE_RC. `hal-tmle-rc-density-l2-demo.ipynb` has an **uncommitted edit** in Yilong's main checkout; he should save it first. |
| `AIPW-HAL-MLE/aipw-hal-ipcw.ipynb` | 352K | P HAL_TMLE_RC |
| `current_status_data_examples/` (2 notebooks) | 2.4M | P CS-HAL-MLE or current_status_HAL-MLE (open question 5) |
| `example_ic_FISTA.ipynb`, `example_ic_initial_variant.ipynb` | 1.7M | P, IC experiments repo (open question 5) |
| `ic_FISTA.md` | 4K | D (pasted chat answer) |
| `multivar-HAL/` (4 notebooks + `multivar-basis.md`) | 2.6M | P multivar-HAL |
| `MCMC-HAL/` (`plan.md`, `univariate_sobol_normalizer_demo.ipynb`) | 444K | P multivar-HAL (Sobol normalizer study; open question 5) |
| `demo_conservative_adjustment_search.ipynb` | 92K | P uni_EM_HAL_RC, next to the multicollinearity/CVLL resources. Its `conservative_method="linear"` call already raises `TypeError` at the branch tip. |
| `plotting.py` | 8K | Keep as `examples/tutorials/_plotting.py` only if the tutorials need it; else D. |
| `case_study/galaxies.csv` | 4K | → package data (`datasets/data/`) |

Tutorials run in CI through `pytest --nbmake examples/tutorials` (adds `nbmake` to the `dev` extra), each under about a minute.

## 7. Development workflow (to go in CONTRIBUTING.md)

- **"Does it work?" checks** are package tests with tiny data (seconds), run by CI on every push.
- **Performance comparisons** (simulations, other methods, paper figures) live in project repos, never in the package.
- **During development** a project repo uses the local checkout in editable mode: `haldensity = { path = "../HALDensity", editable = true }`. That line is not committed.
- **Final results** pin a tagged release (`v0.2.0`, ...). Snapshot tags are never moved or deleted.
- **Before every change**, label it core or data-type-specific (section labels). Behaviour changes go in `CHANGELOG.md` with a before/after number.

## 8. Proposed Phase 3 order (each step its own commit, tests passing)

1. Pure moves and splits, plus the compat layer: no logic change. Check: the same tests pass, and a fixed set of fits gives bit-identical results before and after.
2. Untangle `core/weighted_mle.py` from `rc/fit.py` (core `HALFit`).
3. `datasets/`: `load_galaxies`, generators; tests switched; `comparison.py` and its test removed (after the lead's copy).
4. Extractions: the conservative search into `core/cv/conservative.py`, then used by full-data CV (Yilong's decision; full-data numbers change, before/after reported); generic targeting into `core/targeting/` (RC targets give identical numbers).
5. `full_multi` d = 1 warning; AIPW docstring; `ic/current_status` adapter and README.
6. Docs: CONTRIBUTING, CHANGELOG, Yilong's multicollinearity note (`multicolinear_cvll_write_up.tex` + figures) linked from the tuner docstring.
7. Checkpoint, then the three core fixes (exact CV scoring, accurate normalizer, basis cleaning before the SE) and the k = 0, M = 50 diagnosis with the exact k = 0 normalizer test.
8. Examples triage (after Yilong's rows above) and tutorials in CI.

## 9. Open questions for Yilong

1. **Plug-in pipelines.** `fit_right_censored_cv_ipcw_l1mle_plugin` and its undersmoothed variant are in `comparison.py`, which you decided moves out. They are HAL estimators, and the tutorial candidate `cv-ipcw-l1mle-survival-api-demo` uses them. Keep them as `rc/pipelines.py` (without the `truth=` argument), or move them out too?
2. **Normalizer and univariate base in core** (deviation 1). Agree?
3. **Generic targeting** extracted from the RC code into `core/targeting/`, with `BaseTargetLearner` staying in `full_uni` (deviation 3). Agree?
4. **Names**: `full_uni` / `full_multi` as given? (I see no better pair.) Also keep `MultvarHAL`, or add the alias `MultivariateHAL`?
5. **Destinations**: current-status notebooks to CS-HAL-MLE or current_status_HAL-MLE; IC notebooks to which repo; `MCMC-HAL/` to multivar-HAL?
6. **Galaxies scaling.** The case study divides velocities by c = 3×10⁵ km/s, so the data occupy [0.031, 0.114] of [0, 1]. The 199-cell fitting grid then has only about 17 cells over the data. Should `load_galaxies()` return km/s and leave the tutorial to rescale to a stated support (e.g. min–max with a margin)? This also matters for fix 2.
7. **Utilities.** Keep the experiment-reporting helpers in the multivariate basis (`extract_term_type_metrics`, catalog summaries) in core, or move them out?
8. **`KFold_CV_LambdaSelector`** (`cross_validation/lambda_selector.py`) is used nowhere in the package, tests or examples. Drop it in 0.2.0?
9. **Compat window**: old import paths warn in 0.2.x and are removed in 0.3.0. OK?
