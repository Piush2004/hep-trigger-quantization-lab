"""Gate 3: model construction and forward-pass tests."""
import torch

from hep_quant.models.factory import build_model
from hep_quant.models.mlp import MLPClassifier


def test_build_mlp_from_config():
    cfg = {"type": "mlp", "hidden_dims": [32, 16], "dropout": 0.1, "activation": "relu"}
    model = build_model(cfg, input_dim=10, n_classes=2)
    assert isinstance(model, MLPClassifier)
    assert model.hidden_dims == [32, 16]


def test_mlp_forward_pass_shape():
    model = MLPClassifier(input_dim=16, hidden_dims=[64, 32], n_classes=2)
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


def test_mlp_architecture_is_configurable():
    model_a = MLPClassifier(input_dim=10, hidden_dims=[128, 64, 32], n_classes=2)
    model_b = MLPClassifier(input_dim=10, hidden_dims=[16], n_classes=2)
    assert model_a.num_parameters() > model_b.num_parameters()


def test_mlp_unsupported_activation_raises():
    import pytest

    with pytest.raises(ValueError):
        MLPClassifier(input_dim=10, hidden_dims=[16], activation="not_an_activation")
