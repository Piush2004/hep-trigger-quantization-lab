"""Configurable compact CNN classifier for jet images (Phase 14: second architecture, image input).

Mirrors MLPClassifier's design philosophy: architecture driven by config, never hardcoded, so
the same class serves the FP32 baseline and any later quantized variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JetImageCNN(nn.Module):
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
