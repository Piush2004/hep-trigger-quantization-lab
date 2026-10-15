"""Configurable compact CNN classifier for jet images (Phase 14: second architecture, image input).

Mirrors MLPClassifier's design philosophy: architecture driven by config, never hardcoded, so
the same class serves the FP32 baseline and any later quantized variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JetImageCNN(nn.Module):
    # Keys Trainer.save_checkpoint/load_checkpoint read/write generically via
    # get_config()/from_config() -- see MLPClassifier.CONFIG_KEYS for the same contract.
    CONFIG_KEYS = ("image_size", "in_channels", "conv_channels", "n_classes")
    # Two Sequential blocks (not one flat "net" like MLPClassifier) -- see
    # MLPClassifier.QUANT_BLOCKS for the same contract.
    QUANT_BLOCKS = ("conv", "classifier")

    def __init__(
        self,
        image_size: int,
        in_channels: int = 1,
        conv_channels: list[int] | None = None,
        n_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        conv_channels = list(conv_channels) if conv_channels is not None else [16, 32, 64]

        layers: list[nn.Module] = []
        prev_c = in_channels
        for c in conv_channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            prev_c = c
        self.conv = nn.Sequential(*layers)

        reduced = max(image_size // (2 ** len(conv_channels)), 1)
        self.flatten_dim = prev_c * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        return {
            "image_size": self.image_size, "in_channels": self.in_channels,
            "conv_channels": self.conv_channels, "n_classes": self.n_classes,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "JetImageCNN":
        return cls(
            image_size=cfg["image_size"], in_channels=cfg.get("in_channels", 1),
            conv_channels=cfg.get("conv_channels"), n_classes=cfg["n_classes"],
        )
