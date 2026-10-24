import subprocess
import sys

import numpy as np
import optuna
import pandas as pd

from haldensity.full_uni import OptunaHyperparameterTuner


def test_importing_the_package_installs_no_catch_all_warning_filter():
    code = (
        "import warnings; before = list(warnings.filters); "
        "import haldensity.full_uni, haldensity.rc, haldensity.ic, haldensity.full_multi; "
        "new = [f for f in warnings.filters if f not in before]; "
        "assert not [f for f in new if f[0] == 'ignore' and f[1] is None], new"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_optimize_restores_the_callers_optuna_verbosity():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=60)})
        tuner = OptunaHyperparameterTuner(
            "CVXPYEstimator", data, cv_folds=2, param_overrides={"basis_order": 1, "norm_constraint": (1.0, 10.0)}
        )
        tuner.optimize(n_trials=1, show_progress=False)
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
    finally:
        optuna.logging.set_verbosity(previous)
