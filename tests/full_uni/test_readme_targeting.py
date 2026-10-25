"""The README's targeting snippets run as written at orders 0, 1 and 2 (before 0.2.0 they failed at k >= 1 with a
shape error: get_results()["theta_hat"] holds a coefficient for every candidate knot)."""
import pathlib
import re

import numpy as np
import pytest

from haldensity.full_uni import CVXPYEstimator
from haldensity.full_uni.targeting import SurvivalTargetLearner, targeting_inputs

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks(start: str, end: str) -> list[str]:
    text = README.read_text(encoding="utf-8")
    first = text.index(start)
    return re.findall(r"```python\n(.*?)```", text[first:text.index(end, first)], flags=re.S)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_readme_targeting_snippets_run_at_every_order(k, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # whatever the quickstart writes stays out of the repository
    namespace: dict = {}
    exec(_python_blocks("Quickstart", "Cross-Validation (Optuna)")[0], namespace)
    data = namespace["data"]
    x = data["W1"].to_numpy()
    namespace["est"] = CVXPYEstimator(basis_order=k, norm_constraint=20, log_frequency=-1).fit(data)
    prerequisites, cdf, survival, moments, median = _python_blocks("Targeted Learning (TMLE)", "\nNotes\n")
    exec(prerequisites, namespace)
    points = namespace["points"]
    for block, empirical in ((cdf, [np.mean(x <= t) for t in points]), (survival, [np.mean(x > t) for t in points]),
                             (moments, [x.mean()]), (median, None)):
        exec(block, namespace)
        fit = namespace["targeted_fit"]
        assert np.all(np.isfinite(namespace["se"])) and np.all(np.asarray(namespace["se"]) > 0)
        if empirical is not None:                     # the score equations hold at the targeting points
            np.testing.assert_allclose(fit["estimate"], empirical, atol=1e-8)


def test_full_theta_gives_a_clear_error():
    data = pd_data = __import__("pandas").DataFrame({"W1": np.random.default_rng(1).beta(2.0, 5.0, size=150)})
    est = CVXPYEstimator(basis_order=2, norm_constraint=10, log_frequency=-1).fit(pd_data)
    results = est.get_results()
    learner = SurvivalTargetLearner(norm_constraint=100, basis_order=2)
    with pytest.raises(ValueError, match="targeting_inputs"):
        learner.run_m_step(uncensored_augmented=data, grid_points_hal_selected=results["grid_points_hal_selected"],
                           old_theta=results["theta_hat"], targeting_points=[0.2])
    inputs = targeting_inputs(results)                # the results dict works as well as the estimator
    assert inputs["old_theta"].size == 3 + inputs["grid_points_hal_selected"].size
