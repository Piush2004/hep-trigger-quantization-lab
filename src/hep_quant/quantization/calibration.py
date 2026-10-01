"""Activation-range calibration via forward hooks over a calibration subset.

This is standard PTQ calibration: run representative data through the model
once, record observed min/max per activation, then use those ranges to fix
the quantization scale/zero-point.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_OBSERVED_TYPES = (nn.ReLU, nn.Tanh, nn.LeakyReLU)


class ActivationRangeObserver:
    def __init__(self):
        self.ranges: dict[str, list[float]] = {}

    def _hook(self, name: str):
        def _fn(module, inputs, output):
            cur_min = output.min().item()
            cur_max = output.max().item()
            if name not in self.ranges:
                self.ranges[name] = [cur_min, cur_max]
            else:
                self.ranges[name][0] = min(self.ranges[name][0], cur_min)
                self.ranges[name][1] = max(self.ranges[name][1], cur_max)

        return _fn

    def register(self, model: nn.Module) -> list:
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, _OBSERVED_TYPES):
                handles.append(module.register_forward_hook(self._hook(name)))
        return handles


def calibrate_activation_ranges(
    model: nn.Module, X_calib: np.ndarray, device: str = "cpu", batch_size: int = 256
) -> dict[str, tuple[float, float]]:
    """Return {module_name: (min, max)} observed over `X_calib`."""
    model = model.to(device)
    model.eval()
    observer = ActivationRangeObserver()
    handles = observer.register(model)
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_calib).to(device)
        for i in range(0, len(X_tensor), batch_size):
            model(X_tensor[i : i + batch_size])
    for h in handles:
        h.remove()
    return {name: (lo, hi) for name, (lo, hi) in observer.ranges.items()}
