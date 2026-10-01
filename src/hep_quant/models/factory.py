"""Model construction from a config dict."""
from __future__ import annotations

from hep_quant.models.mlp import MLPClassifier

_MODEL_TYPES = {"mlp": MLPClassifier}


def build_model(model_cfg: dict, input_dim: int, n_classes: int):
    model_type = model_cfg.get("type", "mlp")
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"Unsupported model type '{model_type}', choose from {list(_MODEL_TYPES)}")

    if model_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            n_classes=n_classes,
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
        )
    raise AssertionError("unreachable")
