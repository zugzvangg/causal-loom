import numpy as np
import pytest

from cib_advanced_experiments.model import SynthControlModel


def test_fit_outputs_weights_and_loss_cpu():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3), dtype=np.float32)
    true_w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    y = X @ true_w

    model = SynthControlModel(epochs=10, lr=0.05)
    model.fit(X, y)

    weights = model.get_weights()
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= 0).all()

    loss = model.get_loss_value()
    assert np.isfinite(loss)


def test_model_raises_on_invalid_and_loss():
    with pytest.raises(AssertionError, match="Loss function should be from"):
        SynthControlModel(loss_func="huber")


def test_model_raises_when_access_before_fit():
    model = SynthControlModel()
    with pytest.raises(ValueError, match="Fit model first with .fit"):
        model.get_weights()
    with pytest.raises(ValueError, match="Fit model first with .fit"):
        model.get_loss_value()


