"""ProximalAdaGradEstimator: log_frequency only sets logging (review C2)."""
import numpy as np
import pandas as pd

from haldensity.full_uni import ProximalAdaGradEstimator


def test_log_frequency_zero_works_and_nonzero_values_keep_their_results():
    # log_frequency = 0 divided by zero: the objective check was paced by the logging frequency. It now has its own
    # interval, |log_frequency| (1 when 0), so -1 (the default) and 1 still check every iteration, as before.
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=150)})
    fits = {f: ProximalAdaGradEstimator(lam=3.0, n_iterations=300, basis_order=1, log_frequency=f).fit(data).theta_hat
            for f in (0, -1, 1)}
    np.testing.assert_array_equal(fits[0], fits[-1])
    np.testing.assert_array_equal(fits[1], fits[-1])
