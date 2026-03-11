import numpy as np
import pytest
import torch

from cib_advanced_experiments.model import SynthControlModel


def test_fit_outputs_weights_and_loss_cpu():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3), dtype=np.float32)
    true_w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    y = X @ true_w

    model = SynthControlModel(epochs=10, lr=0.05, device="cpu")
    model.fit(X, y)

    weights = model.get_weights()
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= 0).all()

    loss = model.get_loss_value()
    assert np.isfinite(loss)


def test_model_raises_on_invalid_device_and_loss():
    with pytest.raises(AssertionError, match="Device should be from"):
        SynthControlModel(device="tpu")

    with pytest.raises(AssertionError, match="Loss function should be from"):
        SynthControlModel(loss_func="huber")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_cuda_device():
    rng = np.random.default_rng(1)
    X = rng.random((10, 4), dtype=np.float32)
    y = X @ np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    model = SynthControlModel(epochs=5, lr=0.05, device="cuda")
    model.fit(X, y)

    weights = model.get_weights()
    assert weights.shape == (4,)
