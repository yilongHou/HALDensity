"""Current-status data: each subject is inspected once.

At inspection time ``C`` one observes only whether the event has happened,
``Delta = 1{T <= C}``. This is interval-censored data with a single
inspection, so estimation reuses :mod:`haldensity.ic`: convert the data with
:func:`current_status_to_interval` and pass the intervals to the
interval-censored estimators and tuners.

The theory for current-status data differs from the general interval-censored
case and has not been worked out for HAL yet; no current-status targeting or
inference is provided here. See ``README.md`` in this folder.
"""

from haldensity.ic.current_status.adapter import current_status_to_interval

__all__ = ["current_status_to_interval"]
