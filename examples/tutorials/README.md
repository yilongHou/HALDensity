# Tutorials

Small notebooks per data type that use the public API of `haldensity`; CI runs each of them from top to bottom.

Install the package with the tutorials' extra, which adds matplotlib and ipykernel:

```bash
pip install "haldensity[examples]"     # or, from a clone: pip install -e ".[examples]"
```

MOSEK is optional: without it the fits use the free solvers (CLARABEL, ECOS, SCS).

- `full_uni/`: univariate full data (the galaxy velocities, simulated data, targeting)
- `full_multi/`: the multivariate estimator (experimental)
- `rc/`: right-censored data
