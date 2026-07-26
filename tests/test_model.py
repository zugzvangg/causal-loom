import numpy as np
import pytest

from causal_loom.model import SynthControlModel


def test_fit_outputs_weights_and_loss_cpu():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3), dtype=np.float32)
    true_w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    y = X @ true_w

    model = SynthControlModel(max_iter=2_000, tolerance=1e-12)
    model.fit(X, y)

    weights = model.get_weights()
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= 0).all()

    loss = model.get_loss_value()
    assert np.isfinite(loss)
    assert loss < 1e-10
    assert np.allclose(weights, true_w, atol=1e-4)


def test_model_raises_on_invalid_and_loss():
    with pytest.raises(ValueError, match="loss_func should be one of"):
        SynthControlModel(loss_func="huber")


def test_deprecated_optimizer_aliases_remain_compatible():
    with pytest.warns(DeprecationWarning, match="'epochs' is deprecated"):
        model = SynthControlModel(epochs=10)
    with pytest.warns(DeprecationWarning, match="'lr' is deprecated"):
        model = SynthControlModel(lr=1e-6)

    assert model.max_iter == 2_000
    assert model.tolerance == 1e-6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter should be positive"),
        ({"tolerance": 0.0}, "tolerance should be strictly between"),
        ({"epochs": 10, "max_iter": 10}, "Pass only one"),
        ({"lr": 1e-6, "tolerance": 1e-6}, "Pass only one"),
    ],
)
def test_model_validates_optimizer_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SynthControlModel(**kwargs)


def test_model_raises_when_access_before_fit():
    model = SynthControlModel()
    with pytest.raises(ValueError, match="Fit model first with .fit"):
        model.get_weights()
    with pytest.raises(ValueError, match="Fit model first with .fit"):
        model.get_loss_value()
