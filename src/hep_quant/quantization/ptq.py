"""Post-Training Quantization via uniform affine fake-quantization.

IMPORTANT — how precision is actually represented (see docs/quantization.md):
PyTorch's native `torch.ao.quantization` only ships real bit-packed integer
kernels for INT8 (via fbgemm/qnnpack), not INT6/INT4. To compare FP32 / INT8 /
INT6 / INT4 on equal footing, this module instead implements a uniform
affine "fake quantization" (quantize-then-dequantize, computed in float32)
for ALL precisions, including INT8. Weights and activations are each rounded
to the requested bit-width's grid and immediately dequantized back to float,
so forward passes run in ordinary float32 arithmetic that *simulates* the
numerical effect of N-bit quantization. This is a standard, technically valid
PTQ research methodology (the same mechanism underlies most QAT literature),
but it does NOT produce bit-packed storage or integer-only compute — model
size and "estimated" memory footprint are computed analytically, not from an
actually-packed artifact. We state this explicitly rather than imply a real
int8 hardware kernel was used.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from hep_quant.quantization.calibration import calibrate_activation_ranges

SUPPORTED_PRECISIONS: dict[str, int | None] = {"fp32": None, "int8": 8, "int6": 6, "int4": 4}
_ACTIVATION_TYPES = (nn.ReLU, nn.Tanh, nn.LeakyReLU)


def fake_quantize_tensor(x: torch.Tensor, bits: int, min_val: float, max_val: float) -> torch.Tensor:
    """Uniform affine quantize-then-dequantize of `x` to `bits` bits over [min_val, max_val]."""
    if max_val <= min_val:
        return x
    qmin, qmax = 0, 2**bits - 1
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale


class FakeQuantActivation(nn.Module):
    """Applies fixed-range fake quantization to its input (ranges fixed at calibration time)."""

    def __init__(self, bits: int, min_val: float, max_val: float):
        super().__init__()
        self.bits = bits
        self.register_buffer("min_val", torch.tensor(float(min_val)))
        self.register_buffer("max_val", torch.tensor(float(max_val)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize_tensor(x, self.bits, self.min_val.item(), self.max_val.item())

    def extra_repr(self) -> str:
        return f"bits={self.bits}, range=[{self.min_val.item():.3f}, {self.max_val.item():.3f}]"


def _quantize_weights_in_place(model: nn.Module, bits: int) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                min_val, max_val = w.min().item(), w.max().item()
                module.weight.data = fake_quantize_tensor(w, bits, min_val, max_val)


def _insert_activation_fake_quant(
    model: nn.Module, bits: int, activation_ranges: dict[str, tuple[float, float]]
) -> None:
    """Rebuild `model.net` inserting a FakeQuantActivation after each activation module."""
    new_layers = []
    for name, module in model.net.named_children():
        new_layers.append(module)
        if isinstance(module, _ACTIVATION_TYPES):
            key = f"net.{name}"
            lo, hi = activation_ranges.get(key, (-6.0, 6.0))
            new_layers.append(FakeQuantActivation(bits, lo, hi))
    model.net = nn.Sequential(*new_layers)


def quantize_model_ptq(
    model: nn.Module, precision: str, X_calib: np.ndarray, device: str = "cpu"
) -> nn.Module:
    """Return a new model with weights + activations fake-quantized to `precision`.

    `precision` must be one of SUPPORTED_PRECISIONS ("fp32" returns an
    unmodified copy). Activation ranges are calibrated on `X_calib`.
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision '{precision}', choose from {list(SUPPORTED_PRECISIONS)}")

    quantized = copy.deepcopy(model)
    if precision == "fp32":
        return quantized

    bits = SUPPORTED_PRECISIONS[precision]
    _quantize_weights_in_place(quantized, bits)
    activation_ranges = calibrate_activation_ranges(quantized, X_calib, device=device)
    _insert_activation_fake_quant(quantized, bits, activation_ranges)
    quantized.precision = precision
    return quantized
