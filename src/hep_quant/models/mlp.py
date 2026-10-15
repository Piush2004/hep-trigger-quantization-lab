"""Configurable compact MLP classifier.

Architecture is driven entirely by `hidden_dims` in the config — never
hardcoded — so the same class serves the FP32 baseline, PTQ, and QAT
experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class MLPClassifier(nn.Module):
    # Keys Trainer.save_checkpoint/load_checkpoint read/write generically via
    # get_config()/from_config() -- unchanged from the original top-level checkpoint
    # keys, so every existing frozen checkpoint on disk stays loadable as-is.
    CONFIG_KEYS = ("input_dim", "hidden_dims", "n_classes")
    # Attribute names of the Sequential blocks hep_quant.quantization.ptq's
    # _insert_activation_fake_quant walks when inserting fake-quant activations.
    QUANT_BLOCKS = ("net",)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_classes: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}', choose from {list(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        return {"input_dim": self.input_dim, "hidden_dims": self.hidden_dims, "n_classes": self.n_classes}

    @property
    def hls4ml_input_shape(self) -> tuple[int, ...]:
        """The `input_shape` kwarg hls4ml.utils.config_from_pytorch_model expects --
        generalizes hep_quant.hardware.hls_csim's previous `(model.input_dim,)` hardcode."""
        return (self.input_dim,)

    @classmethod
    def from_config(cls, cfg: dict) -> "MLPClassifier":
        return cls(input_dim=cfg["input_dim"], hidden_dims=cfg["hidden_dims"], n_classes=cfg["n_classes"])
