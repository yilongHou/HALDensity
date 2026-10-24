# Sobol vs midpoint normalizer in one dimension

**Conclusion.** In one dimension, a fixed-bank scrambled Sobol normalizer does not beat the
midpoint rule at matched budgets: at every budget the midpoint fit is closer to the
high-resolution reference, and it does not depend on a random scramble. Sobol (quasi-Monte
Carlo) normalization is therefore for multivariate HAL (`haldensity.full_multi`), where dense
midpoint grids stop scaling; the univariate estimators keep deterministic grids.

## Question

The HAL-MLE normalizes `exp(f_theta)` by `Z(theta) = ∫_0^1 exp(f_theta(x)) dx`, which the fit
approximates by a finite rule. Is a fixed bank of B scrambled Sobol points a better approximation
than B midpoint cells for the same B?

## Design (as run)

- Data: one training sample (n = 100) and one held-out sample (5,000) from the package's
  `TruncatedNormal()` on [0, 1], data seed 0.
- Model: basis order k = 1, L1 bound M = 65, knots at the training points, ECOS with fallback.
  Only the normalizer approximation differs between fits.
- Normalizers, budget B in {32, 64, 128, 256, 512}:
  - midpoint: B equal cells on [0, 1];
  - Sobol: B scrambled Sobol points, `log Z ≈ logsumexp(Phi_bank theta) - log B` (still a
    convex program, since logsumexp of affine functions is convex);
  - reference: midpoint with 4,096 cells.
- Sobol is refit for 50 scramble seeds on the same data (bank sensitivity, not a Monte Carlo
  study over datasets).

The original plan (n = 500, Sobol banks of 256 to 4,096 points against the package estimator)
was replaced by this budget-matched design before running.

## Result

Two numbers per fit, both ideally 0:

- **held-out gap**: average held-out log-likelihood of the fit minus that of the 4,096-cell
  midpoint reference (nats per observation, absolute value);
- **log Z error**: |log Z under the fit's own rule − log Z on a 10,000-point reference grid|.

| B | held-out gap, midpoint | held-out gap, Sobol (mean of 50) | log Z error, midpoint | log Z error, Sobol (mean of 50) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0034 | 0.0235 | 0.0048 | 0.0344 |
| 64 | 0.0009 | 0.0075 | 0.0011 | 0.0095 |
| 128 | 0.00012 | 0.0019 | 0.00011 | 0.0019 |
| 256 | 0.00007 | 0.0005 | 0.000012 | 0.0005 |
| 512 | 0.00006 | 0.00014 | 0.000010 | 0.00013 |

All fits were valid (non-negative densities, normalization residual under their own rule at
machine precision, finite held-out log-likelihood). Sobol needs about four times the budget to
match midpoint, and its error varies across scrambles (at B = 32 the 10th to 90th percentile
held-out gap is 0.007 to 0.045).

## Source

Notebook `examples/MCMC-HAL/univariate_sobol_normalizer_demo.ipynb` and its `plan.md`, removed
from the package for 0.2.0. They remain in git at tag `snapshot/multivar-hal-d58d14b`
(not at `v0.1.0`, which predates them):

```
git show snapshot/multivar-hal-d58d14b:examples/MCMC-HAL/univariate_sobol_normalizer_demo.ipynb > sobol-1d.ipynb
```
