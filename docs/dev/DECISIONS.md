# HALDensity: decisions on hyperparameters, solvers and rules

Status: reference list, 2026-10-24. Each item says what was decided, when, and why; "pending" items are open. Applies to HALDensity v0.2.0 (branch `consolidate`, not yet released) unless stated otherwise. **Update this file in the same commit as any change to a default or rule.** Paths are relative to the repository root unless marked as project notes; project notes live in the HAL-MLE-DE-AOS project repository, not in this package: `notes/haldensity-cv-scoring/`, `notes/hal-free-knot-argument/`, `notes/hal-order-choice/`, `notes/haldensity-restructure/spec_normalizer_knotcap.md`.

## 1. Cross-validation (choosing the L1 bound M)

| Setting | Decision | Why |
| --- | --- | --- |
| Search | Optuna TPE, 50 trials, M log-uniform on [1e-3, 1e6], 5 folds | HALDensity's defaults; confirmed 2026-10-23. 5 folds everywhere decided 2026-10-24: the full-data tuner's former default of 3 now follows the shared `TUNER_DEFAULTS.cv_folds = 5` (discrepancy resolved) |
| **CV score** | **Exact held-out log-likelihood** (`cv_scoring="exact"`, default): exact evaluation at the validation points with the accurate normalizer (§4). `"interpolated"` keeps the old score | The old score interpolated 200 grid values and used a midpoint normalizer; for spiky fits it was inflated by up to 0.1 nats and picked absurd M (e.g. 9.9e5) |
| A trial counts | Only if every fold fit qualifies; if no trial gets a finite score, raise an error (never return an arbitrary M) | A silent all-folds failure returned the first trial's M |
| CV solver | The estimator's own solver defaults with the fallback ladder (§3), **at the default tolerance** (`OptunaHyperparameterTuner(cv_solver_tolerance="default")`) | CV only compares fitted values; tight tolerance in 250 solves costs time for no benefit (decided 2026-10-24) |
| Solver statuses | `optimal_inaccurate` accepted and counted; L1 overshoot recorded, not gated | As in HALDensity |
| Degree k | **Package default: fixed k = 2** (decided 2026-10-24; `DEFAULT_BASIS_ORDER`); CV over the order on request (`param_overrides={"basis_order": [0, 1, 2]}`, the default before 0.2.0). Experiments: fixed when a study checks the theorem; chosen by CV jointly with M when HAL is compared with other estimators (decided 2026-10-23) | At n = 320 the package's joint CV over {0, 1, 2} was never better than fixed k = 2 (KL of the joint CV relative to fixed k = 2: 0.96, 1.02, 1.16, 1.34 on the jump, slope-kink, curvature-kink and smooth truths), and an order higher than the truth's costs little |
| **Conservative adjustment** | `"auto"` (default): **on only when the CV-best order is k = 0**; margin **5% of the CV log-likelihood SD** (Yilong's rule), bisection search. Off for k ≥ 1 | For k = 0 the CV curve can be flat (spikes, Yilong's CVLL note: [cvll-multicollinearity/](cvll-multicollinearity/README.md)). For k ≥ 1 it oversmoothed: L2 worse in 11 of 16 test datasets |
| One-SE rule | **Not used** | About 9× wider than the 5% margin; HAL's CV already picks M below the truth's norm |
| CV safeguard for k ≥ 1 | **None** (decided 2026-10-24): the paired-difference rule is not adopted | The k = 1 CV curves decline gently with no pathological fits; a continuous fit cannot hide spikes; plain CV at k = 1–3 beat both sieves in the experiments. Reopen only if the paper rerun's imbalance diagnostics flag a problem. The k = 0 conservative adjustment may also go (session 69's check pending) |

## 2. The fitting objective

| Setting | Decision | Why |
| --- | --- | --- |
| Basis and knots | Truncated powers of order k, knots at the data points (at most K of them, see the knot cap); L1 constraint on all coefficients except the intercept | HALDensity's definition |
| **Knot cap** `max_knots` | **Default `"auto"`: K = 1,000 for k ≥ 1; K = max(1,000, ⌈n^(2/3)⌉) for k = 0** (decided 2026-10-24), at K data quantiles; `None` = every data point. Changes default results only for n > 1,000 | Bounds memory and time at large n and keeps knots at least about 1/K apart. Rough rule for k = 0: a jump can only be placed to within about 1/K, costing about h²/K in squared L2, which must stay below n^(-2/3), so K ≳ n^(2/3) (acts only above n ≈ 31,600); k ≥ 1 needs far fewer knots (about n^(4/15) for k = 1). Benchmark: on a jump density at n = 20,000, K = 1,000 cost 5–13% in L2, K = 250 up to 43%, K = 100 up to 88%; nothing on smooth or kinked densities ([benchmarks/knot_aware_normalizer/](benchmarks/knot_aware_normalizer/README.md)) |
| **Objective normalizer** | **`"auto"` (default; decided 2026-10-24): chosen by order** — the exact objective at k = 0, the midpoint rule with the knots added to the grid at k = 1 (both through the step-height rewrite), the old midpoint rule (199 fixed cells) at k ≥ 2. Chosen per order inside one CV search. `haldensity.core.objective_rules.objective_rule(k, d)` returns the rule; each data-type package implements it (d ≥ 2: the multivariate estimator's rule, exact below the cell quota and the Sobol bank above at k = 0, the Sobol bank at k ≥ 1). Fits record the rule. `"midpoint"` = the old default | The old midpoint rule is blind to mass between two close knots: the k = 0 spike (objective log Z 0.19 vs exact 18.4). At k = 2 the knots-added grid cost 107–415 s per 5-fold CV trial at n = 1,280 against 3.4–4.3 s, with about the same CV score; Yilong: "we can tailor different rules for different k, and in future different d" |

## 3. Solvers and tolerances

| Setting | Decision |
| --- | --- |
| **Final fit at the chosen M** | **One solve: MOSEK at 1e-12** (relative gap, primal and dual feasibility). Fallback ladder with tight tolerances: **CLARABEL 1e-12 → CLARABEL 1e-10 → ECOS 1e-12 → ECOS 1e-10 → SCS 1e-9, the SCS step capped at 600 s** (cap decided 2026-10-24: without it SCS ran past 17 min at n = 20,000). Record solver, status and achieved accuracy (decided 2026-10-24). Code: `CVXPYEstimator(solver_tolerance="tight")` (default; `"default"` = v0.1.0), `haldensity.core.solvers.TIGHT_LADDER`, `haldensity.core.univariate.support.certified_solve` |
| CV trials | Default tolerances (MOSEK first, then the package's waterfall) |
| **Threads** | **Every solver call single-threaded by default** (decided 2026-10-24, so runs on shared servers do not take other people's cores): MOSEK `MSK_IPAR_NUM_THREADS`, CLARABEL `max_threads`, numpy/scipy BLAS pools limited during the call (`threadpoolctl`); `solver_threads` raises it |
| Pruning | Coefficients with \|θ\| ≤ 1e-4 set to zero (HALDensity's rule) |
| **KKT certification** of the final fit | λ = median \|∂L/∂θⱼ\| over kept coefficients with \|θⱼ\| ≥ 0.1·max\|θ\|; **certified** if no kept coefficient has \|∂L/∂θⱼ\| < 0.99λ and no coefficient has \|∂L/∂θⱼ\| > 1.01λ. If ‖θ[1:]‖₁ < (1 − 1e-3)M the bound is not binding and the fit is certified when max \|∂L/∂θⱼ\|/n ≤ 1e-5; at the bound, passing either test certifies (added 2026-10-24). If not certified, try the next solver in the ladder; if none certifies, keep the best attempt from any solver (the fewest coordinates failing the conditions), flagged and warned (decided 2026-10-24). No repair of uncertified solutions: setting small violators to zero and re-checking certified 0 of 18 in the study, and the re-solve-on-the-support variant never certified a problem the ladder could not ([benchmarks/final_fit_repair/](benchmarks/final_fit_repair/README.md)) |
| Point estimate | **The tight, pruned, certified final fit** (point estimates change slightly from v0.1.0; noted in the CHANGELOG) |
| Without MOSEK | The package must work: the free-solver ladder is the full path |

## 4. Normalizing constant (evaluation, reporting, CV scoring)

| Setting | Decision |
| --- | --- |
| Univariate, full data | **Accurate** (default): cell-wise Gauss–Legendre with breakpoints at the knots with nonzero coefficients, 32 nodes per cell, doubled until the change is below 1e-12; exact for k = 0. `normalizer="midpoint"` keeps the old behaviour |
| Multivariate | Exact below a quota; fixed scrambled Sobol bank above it (existing design). In 1-D, Sobol does not beat the midpoint rule ([experiments/sobol-normalizer-1d.md](experiments/sobol-normalizer-1d.md)) |
| **Censored-data estimators** | Keep their own grid normalizers for now. **Pending — part of the RC project handoff** (their targeting relies on grid sums) |
| Diagnostic (proposed) | Report the gap between the objective's log Z and the accurate one; warn above about 0.01 nats (signals a missed spike) |

## 5. Variance estimation and inference

| Setting | Decision |
| --- | --- |
| Delta-method SE | On the **certified basis of the final fit** (§3); on by default (`support_refit="auto"`; `None` switches it off). Claim strength: an empirical improvement in the simulations, not a guarantee; the selected basis is not unique (SE/sd 2.4–14 → 0.76–1.3 in the simulations). **If basis cleaning fails** (the final fit does not pass the optimality check): the point estimate is kept but **no SE and no intervals** (NaN), with a warning suggesting a bootstrap or another solver (decided 2026-10-24); `support_refit=None` still gives the original-basis SE on request |
| Pointwise intervals at kinks and boundaries | **Deferred to the paper-version experiment rerun (Yilong, 2026-10-24); report both variants there.** Evidence so far: undersmooth for inference, M ≈ 2× the CV choice (coverage at the k = 1 kink 46–58% → 78–89%; boundary 74–80% → ≥ 93%); CV for the point estimate. Never use the conservative adjustment when the goal is inference |
| Supports | Fixed in advance (e.g. [0, 1]); data rescaled by a known bound, not min–max of the sample. Galaxies: divide by c = 3×10⁵ km/s; raise `n_grid_points` if the data occupy a small part of [0, 1] |

## 6. Order of the HAL fit (theory and simulations, 2026-10-23)

| Rule | Status |
| --- | --- |
| Default to a moderately high order (k = 2); a higher order than the truth's costs little (for k₀ ≥ 1), a lower one costs more | Theory (project notes `notes/hal-order-choice/note.md`); Lemma A confirmed numerically; Phase B simulation (R = 100 test stage) running |
| If jumps are plausible (k₀ = 0), choose k by CV | Theory; any continuous fit fails in sup norm at a jump |
| Free smearing at the kink happens only when the fitted order equals the truth's and both are 0 or 1 | Theory + simulations (coverage 0.78 at k = 1, 0.94 at k = 2) |

## 7. Package rules

| Rule | Decision |
| --- | --- |
| Scope | Only our methods (HAL) and their implementation; comparison methods and experiments live in experiment repos that pin a tagged release |
| Layout | `core/` (shared by more than one data type; never imports a data-type package; univariate base in `core/univariate/`), `full_uni/`, `full_multi/`, `rc/`, `ic/` (with `ic/current_status/`), `datasets/` |
| Every change | Labelled core or data-type-specific before it is made |
| Targeting | Each data type keeps its own (full data adds basis functions and re-runs the optimization; RC solves an equation for ε); no shared framework |
| RC estimators | All on equal footing in `rc/pipelines.py`: IPCW-HAL, L1MLE at M₁, CVL1MLE (CV over [M₁, 2M₁]), undersmoothed (EIF-mean rule); default CVL1MLE (best in most but not all cases so far) |
| AIPW | Kept (Yilong's method); documented as not a plug-in |
| Names | `full_uni` / `full_multi`; `MultvarHAL` unchanged; `MultvarHAL` warns at d = 1 |
| Sample data | `load_galaxies()` returns raw km/s; simple simulators in `datasets/` |
| Warnings | No global warning filters; known noise suppressed locally inside CV folds; CI fails on deprecated calls |
| Compatibility | Old import paths re-export with a DeprecationWarning through 0.2.x, removed in 0.3.0 (kept light; can be dropped) |
| Removed | `KFold_CV_LambdaSelector`, the comparison module (copied to `HAL_TMLE_RC/from_haldensity/`), multivariate reporting helpers (copied to `multivar-HAL/from_haldensity/`), the MCMC-HAL notebook (conclusion kept in `docs/dev/experiments/`) |
| Versions | v0.1.0 tags the old `main`; branches are deleted only after archive tags on their tips |

## 8. Pending decisions

- Extending the step-height rewrite to k ≥ 2 (not needed for now: the old midpoint rule at k ≥ 2 has the lean data term).
- Whether the paper's theory-checking runs use every knot (experiment side).
- Accurate normalizer for the censored-data estimators (RC project handoff).
- Undersmoothing for pointwise intervals (c ≈ 2) vs the CV fit: decided at the paper-version rerun; Phase B's bias/sd at the kink for k = 2, 3 informs it.
- Current-status and interval-censored notebook destinations; examples triage and tutorials; release version, merge and branch clean-up.
