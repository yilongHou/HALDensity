"""HAL density estimation: highly adaptive lasso MLE and targeted learning.

Subpackages
-----------
core        shared machinery (basis, univariate HAL base, solvers, CV, EM, variance)
full_uni    univariate full-data density
full_multi  multivariate full-data density
rc          right-censored data
ic          interval-censored data
datasets    sample data and simple simulators
"""

import importlib
from typing import Any

__version__ = "0.2.0"

_SUBPACKAGES = ("core", "full_uni", "full_multi", "rc", "ic", "datasets")
__all__ = list(_SUBPACKAGES)


# Pre-0.2.0 subpackages: thin re-exports that warn on import (removed in 0.3.0).
_DEPRECATED = ("censoring", "cross_validation", "density_variance", "estimation", "multivariate", "targeting", "utils")


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES or name in _DEPRECATED:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
