# Current-status data

Each subject is inspected once, at time `C`, and we observe only whether the
event has happened: `Delta = 1{T <= C}`.

- **Estimation** reuses the interval-censored code in `haldensity.ic`. Convert
  the data with `current_status_to_interval` (event by the inspection gives
  `[0, C]`, otherwise `[C, 1]`) and use the interval-censored estimators and
  tuners.
- **Theory** for current-status data differs from the general interval-censored
  case and has not been worked out for HAL yet. No current-status targeting or
  inference is provided; they will live in this folder once derived.
- If current-status estimation later needs its own code, this folder can be
  promoted to a top-level subpackage.
