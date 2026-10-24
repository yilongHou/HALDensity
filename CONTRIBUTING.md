# Contributing to HALDensity

## What belongs in the package

- **Only our method (HAL) and its implementation.** Comparison methods, simulation studies and
  paper figures live in project (experiment) repositories, which depend on a tagged HALDensity
  release.
- **Label every change before making it:**
  - **core**: used by more than one data type; lives in `haldensity/core/`.
  - **data-type-specific**: belongs to one data type: `full_uni` (univariate full data),
    `full_multi` (multivariate full data), `rc` (right-censored), `ic` (interval-censored,
    with current status in `ic/current_status/`).
- **Import rule:** a data-type package imports only `core` and `datasets`, never another
  data-type package. `core` never imports a data-type package.
- **Behaviour changes** (anything that can move a number) say what changes, come with a
  before/after check on a few fits, and are recorded in `CHANGELOG.md`.

## Layout

| Package | Contents |
| --- | --- |
| `core/basis/` | HAL basis: univariate (any order k) and tensor products (general d) |
| `core/univariate/` | univariate HAL base shared by `full_uni`, `rc`, `ic`: `BaseEstimator`, `WeightedHALMLEEstimator`, `HALFit`, density functionals |
| `core/solvers/`, `core/cv/`, `core/em/` | solver fallback, CV over the L1 bound (incl. the conservative adjustment), EM tuner bases |
| `core/variance.py`, `core/metrics.py`, `core/defaults.py` | delta-method variance, metrics, shared defaults |
| `full_uni/` | full-data estimators (one per solver), the Optuna tuner, full-data targeting |
| `full_multi/` | `MultvarHAL` (normalizer by Sobol QMC or exact k = 0 cells) |
| `rc/` | IPCW and EM estimators, observed-data L1-MLE, fit object, tuners, pipelines, AIPW, `targeting/` |
| `ic/` | estimators, tuners, `current_status/` |
| `datasets/` | sample data (`load_galaxies`) and simple simulators used by tests and tutorials |

Targeting lives with its data type: full-data targeting refits with added basis functions,
right-censored targeting solves a score equation for epsilon, and the interval-censored and
multivariate cases are not worked out yet.

## Development workflow

- **"Does it work?" checks** are package tests on tiny data (seconds each) in
  `tests/<subpackage>/`. CI runs them on every push and pull request (Python 3.11):

  ```bash
  pip install -e ".[dev]"
  python -m pytest
  ```

  MOSEK is optional: without it the solver fallback uses CLARABEL, ECOS or SCS.
- **Performance comparisons** (simulations, competing methods, paper results) live in project
  repositories, never in the package.
- **During development**, a project repository uses the local checkout in editable mode, e.g.
  with uv:

  ```toml
  [tool.uv.sources]
  haldensity = { path = "../HALDensity", editable = true }
  ```

  Do not commit that line.
- **Final results** pin a tagged release (e.g. `v0.2.0`). Tags `v*`, `snapshot/*` and
  `archive/*` are pinned by project repositories and are never moved or deleted.

## Documentation

- `docs/`: user-facing notes (e.g. `docs/censored-data.md`).
- `docs/dev/`: design records and short write-ups of numerical studies that informed package
  decisions (`docs/dev/experiments/`), each pointing to the notebook at a tag.
