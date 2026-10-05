import numpy as np
import pytest

pytest.importorskip("sklearn")
from sizeagent.baselines import MODELS  # noqa: E402


@pytest.mark.parametrize("name", list(MODELS))
def test_baseline_interface_and_learning(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    Y = np.stack([2 * X[:, 0] + X[:, 1] ** 2, np.sin(X[:, 2]) * 5 + 10], 1)
    m = MODELS[name]().fit(X[:120], Y[:120])
    mu, sd = m.predict(X[120:])
    assert mu.shape == sd.shape == (30, 2) and (sd >= 0).all()
    ss = ((Y[120:] - mu) ** 2).sum(0) / ((Y[120:] - Y[120:].mean(0)) ** 2).sum(0)
    assert (1 - ss).min() > 0.5, f"{name} should beat the mean predictor clearly"
