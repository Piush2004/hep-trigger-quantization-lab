"""Quantization-Aware Training (QAT).

Uses the same uniform affine fake-quantization as ptq.py, but applied DURING
training with a straight-through estimator (STE) so gradients can flow
through the (non-differentiable) round() operation. Weight ranges are
recomputed each forward pass from the live weight tensor; activation ranges
are tracked with an exponential moving average (EMA) observer updated during
training, matching the standard QAT recipe (e.g. PyTorch's FakeQuantize/
Observer design), reimplemented here because PyTorch's built-in quantized
kernels do not support INT6/INT4.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from hep_quant.quantization.ptq import SUPPORTED_PRECISIONS, fake_quantize_tensor

_ACTIVATION_TYPES = (nn.ReLU, nn.Tanh, nn.LeakyReLU)


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, min_val: float, max_val: float) -> torch.Tensor:
        return fake_quantize_tensor(x, bits, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None


def ste_fake_quantize(x: torch.Tensor, bits: int, min_val: float, max_val: float) -> torch.Tensor:
    return _STEQuantize.apply(x, bits, min_val, max_val)


class QATLinear(nn.Module):
    """nn.Linear with fake-quantized weights (range taken fresh from the live weight each forward)."""

    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_val, max_val = self.weight.min().item(), self.weight.max().item()
        q_weight = ste_fake_quantize(self.weight, self.bits, min_val, max_val)
        return nn.functional.linear(x, q_weight, self.bias)


class QATActivation(nn.Module):
    """Wraps an activation module; fake-quantizes its output using an EMA-tracked range."""

    def __init__(self, activation: nn.Module, bits: int, momentum: float = 0.1):
        super().__init__()
        self.activation = activation
        self.bits = bits
        self.momentum = momentum
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        if self.training:
            batch_min, batch_max = out.min().detach(), out.max().detach()
            if not bool(self.initialized):
                self.running_min.copy_(batch_min)
                self.running_max.copy_(batch_max)
                self.initialized.fill_(True)
            else:
                self.running_min.mul_(1 - self.momentum).add_(self.momentum * batch_min)
                self.running_max.mul_(1 - self.momentum).add_(self.momentum * batch_max)
        return ste_fake_quantize(out, self.bits, self.running_min.item(), self.running_max.item())


def prepare_qat_model(model: nn.Module, precision: str) -> nn.Module:
    """Return a copy of `model` with Linear/activation layers replaced by QAT-aware equivalents.

    Weights are initialized from the trained FP32 model (QAT fine-tunes from
    an FP32 checkpoint, per the standard QAT recipe), not trained from scratch.
    """
    if precision not in SUPPORTED_PRECISIONS or precision == "fp32":
        raise ValueError(f"QAT requires a quantized precision, got '{precision}'")
    bits = SUPPORTED_PRECISIONS[precision]

    qat_model = copy.deepcopy(model)
    new_layers = []
    for module in qat_model.net:
        if isinstance(module, nn.Linear):
            new_layers.append(QATLinear(module, bits))
        elif isinstance(module, _ACTIVATION_TYPES):
            new_layers.append(QATActivation(module, bits))
        else:
            new_layers.append(module)
    qat_model.net = nn.Sequential(*new_layers)
    qat_model.precision = precision
    qat_model.qat = True
    return qat_model
