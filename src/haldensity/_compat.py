"""Support for import paths deprecated in 0.2.0 (removed in 0.3.0)."""

import warnings


def warn_moved(old: str, new: str) -> None:
    """Warn, at the importing line, that module ``old`` moved to ``new``."""
    warnings.warn(
        f"`{old}` is deprecated since HALDensity 0.2.0 and removed in 0.3.0; import from {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
