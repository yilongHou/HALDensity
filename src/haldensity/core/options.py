"""Options that were public in 0.1.0 and are internal since 0.2.0 (Yilong, 2026-10-24).

Numerical choices (pruning threshold, grids, the solver ladder, the ridge term) are the package's job,
not the user's. Their 0.1.0 names are still accepted through 0.2.x with a ``DeprecationWarning`` that
names the internal replacement (a leading-underscore argument, for tests, benchmarks and experiment
repositories); they will be removed in 0.3.0. Options added during the 0.2.0 restructure became
internal directly.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

REMOVED_IN = "0.3.0"


def take_deprecated(owner: str, given: dict[str, Any], renames: dict[str, Optional[str]],
                    stacklevel: int = 3) -> dict[str, Any]:
    """Pop the 0.1.0 names in ``renames`` from ``given`` (a ``**kwargs`` dict) with a warning each.

    Returns {internal name: value}; a name mapped to None had no effect and is dropped. Anything left in
    ``given`` is an unknown argument (TypeError).
    """
    internal: dict[str, Any] = {}
    for old in [name for name in given if name in renames]:
        value, new = given.pop(old), renames[old]
        if new is None:
            message = f"{owner}({old}=...) has no effect and will be removed in {REMOVED_IN}."
        else:
            message = (f"{owner}({old}=...) is set by the package since 0.2.0 and will be removed in {REMOVED_IN}; "
                       f"tests and benchmarks can still force it with {new}=....")
            internal[new] = value
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    if given:
        raise TypeError(f"{owner}() got unexpected keyword argument(s): {', '.join(sorted(given))}")
    return internal
