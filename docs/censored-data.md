# Censored-data density estimation

HAL density estimation when the event time is not always observed. All times live on [0, 1]; rescale by bounds
fixed in advance.

## Right-censored data (`haldensity.rc`)

The observed data are `T` (the smaller of the event time and the censoring time) and `Delta` (1 if the event
was observed). When the largest observed time is censored, it is treated as an event (the Kaplan–Meier
convention): every right-censored estimator applies it to its data (`last_time_as_event`), so the
estimate of the censoring survival stays positive at every observed time.

**Point estimators, one call each.** `fit_right_censored(data, targeting_points, method=...)` returns the
fitted density, its survival at the requested points and the frozen fits. All four methods share Stage 1, an
IPCW-weighted HAL-MLE whose L1 bound M1 is chosen by cross-validation on the IPCW log-likelihood:

| `method` | point estimator |
| --- | --- |
| `"ipcw_hal"` | the Stage-1 IPCW-HAL fit itself (bound M1) |
| `"l1mle_m1"` | the observed-data L1-MLE on the Stage-1 support at bound M1 |
| `"cvl1mle"` (default) | the observed-data L1-MLE with the bound chosen by cross-validation over [M1, 2 M1] |
| `"undersmoothed"` | the observed-data L1-MLE path over [M1, 2 M1], the bound chosen per time point by the TMLE rule |

The result's `fits` hold `RightCensoredHALFit` objects with `density`, `survival` and `rmst`.

**Targeting (TMLE), the method we propose for univariate right-censored data.** `haldensity.rc.targeting`
updates an initial fit so that the estimate of a chosen feature solves its efficient score equation, with an
influence-curve standard error: survival at time points, the mean, the restricted mean survival time, the
squared-density functional (density L2) and the entropy. Each target has a `..._targeting_M_step(initial_estimator,
observed_data, ...)` function and a matching `..._estimand_variance(...)`. The standard error uses every
observation's influence curve; `gbar_floor_active` in the summary says whether the positivity floor on the
censoring survival was active where that target's influence curve uses it.

The targeting is cross-fitted by default (`cross_fit="tuner"`; see the README's targeting section for the
values). Fold v's initial fit and its censoring Kaplan-Meier come from the other folds, with the Kaplan-Meier
convention applied to them, and the validation fold is scored as observed; one epsilon serves all folds, and
`mode` ("iterative", "one_step", "auto") works as before on the pooled score. With `method="cvl1mle"` the fold
fits are the L1 tuner's own fits at the chosen bound, cross-fitted with tuning on all folds: the bound, Stage
1's order and its knots were chosen with every fold. `cross_fit="refit"` repeats Stage 1 at the chosen order and
M1, then the bound, inside each training fold; `"nested"` repeats `fit_right_censored` with all its tuning there;
`"none"` is the full-sample step. The summary reports `cross_fit`, `n_folds`, `psi_init` (the full-sample
plug-in) and `psi_init_cross_fitted`. For entropy and density L2, targeting removes the first-order term, not
the fold fits' second-order remainder (a Kullback-Leibler divergence, minus a squared L2 distance), which was
still visible at n <= 1,600 in the CV-TMLE study (2026-10-26).

**Building blocks.** `KaplanMeier` and `compute_ipcw_weights` (the censoring weights),
`RightCensoredInitEstimator` (the IPCW-weighted HAL-MLE), `RightCensoredObservedL1MLE` (the observed-data
L1-MLE; FISTA and projected-gradient variants), and their tuners `RightCensoredInitTuner` and
`RightCensoredObservedL1Tuner`.

**AIPW.** `right_censored_survival_aipw` is an augmented-IPCW estimator of survival; it is not a plug-in of a
density fit.

The censored-data estimators normalize the density on their own fitting grid; the accurate normalizer of the
full-data estimators is not used here yet.

## Interval-censored data (`haldensity.ic`)

The observed data are intervals `(L, R]` that contain the event time.

**Initial fit.** `IntervalCensoredInitEstimator` fits the HAL-MLE to the interval midpoints;
`IntervalCensoredInitTuner` chooses its bound by cross-validation.

**Direct optimization of the observed-data likelihood.** `IntervalCensoredFISTAEstimator` and
`IntervalCensoredProjectedGDEstimator` maximize the interval likelihood itself, with the tuners
`IntervalCensoredFISTATuner` and `IntervalCensoredProjectedGDTuner`. First-order methods suit these problems
better than second-order ones.

**Current-status data** (one inspection time per subject, with an indicator of whether the event had happened):
`haldensity.ic.current_status.current_status_to_interval` turns them into intervals.

## Where things live

```
rc/            km.py, weights.py, estimators.py (IPCW initial fit), observed_mle.py, tuners.py,
               pipelines.py (fit_right_censored), fit.py (RightCensoredHALFit), aipw.py, targeting/
ic/            estimators.py, tuners.py, metrics.py, current_status/
core/          univariate/weighted_mle.py (the weighted HAL-MLE shared by both)
```
