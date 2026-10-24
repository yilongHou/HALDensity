import numpy as np
import pandas as pd

from haldensity.core.defaults import TUNER_DEFAULTS
from haldensity.full_uni import OptunaHyperparameterTuner


def test_full_data_tuner_uses_the_shared_fold_default():
    data = pd.DataFrame({"W1": np.random.default_rng(0).beta(2.0, 5.0, size=50)})
    tuner = OptunaHyperparameterTuner("CVXPYEstimator", data)
    assert TUNER_DEFAULTS.cv_folds == 5
    assert tuner.cv_folds == TUNER_DEFAULTS.cv_folds and tuner.kfold.get_n_splits() == 5
