"""Controlled PTQ variants for sensitivity studies (calibration, component, per-layer, mixed precision).

`build_ptq_variant` follows exactly the procedure of `hep_quant.quantization.ptq.quantize_model_ptq`
(quantize selected Linear weights in place -> calibrate activation ranges on that weight-quantized
model -> insert fixed-range fake-quant after selected activations), but lets each Linear weight tensor
and each hidden activation take its own bit-width (or stay FP32 with `None`). With every unit set to
the same bits it must reproduce `quantize_model_ptq` bit-for-bit (tested).

Units for an MLP with hidden layers h1,h2,h3:  L0..L3 = Linear weight tensors, A0..A2 = hidden ReLU outputs.
Not quantized in any variant: inputs, final logits, biases (kept FP32), accumulators.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from hep_quant.quantization.calibration import calibrate_activation_ranges
from hep_quant.quantization.ptq import FakeQuantActivation, fake_quantize_tensor

_ACT = (nn.ReLU, nn.Tanh, nn.LeakyReLU)


def unit_names(model: nn.Module) -> tuple[list[str], list[str]]:
    linears = [n for n, m in model.net.named_children() if isinstance(m, nn.Linear)]
    acts = [n for n, m in model.net.named_children() if isinstance(m, _ACT)]
    return linears, acts


def build_ptq_variant(
    fp32_model: nn.Module,
    X_calib: np.ndarray,
    weight_bits: dict[int, int | None],
    act_bits: dict[int, int | None],
    device: str = "cpu",
) -> nn.Module:
    """Return a copy of `fp32_model` with per-unit PTQ. Missing/None entries stay FP32."""
    model = copy.deepcopy(fp32_model).to(device).eval()
    linears, acts = unit_names(model)
    with torch.no_grad():
        for i, name in enumerate(linears):
            bits = weight_bits.get(i)
            if bits is not None:
                lin = model.net.get_submodule(name)
                w = lin.weight.data
                lin.weight.data = fake_quantize_tensor(w, bits, w.min().item(), w.max().item())
    ranges = calibrate_activation_ranges(model, X_calib, device=device)
    layers = []
    act_idx = 0
    for name, module in model.net.named_children():
        layers.append(module)
        if isinstance(module, _ACT):
            bits = act_bits.get(act_idx)
            if bits is not None:
                lo, hi = ranges[f"net.{name}"]
                layers.append(FakeQuantActivation(bits, lo, hi))
            act_idx += 1
    model.net = nn.Sequential(*layers)
    return model.eval()


def uniform_units(model: nn.Module, bits: int | None) -> tuple[dict[int, int | None], dict[int, int | None]]:
    linears, acts = unit_names(model)
    return {i: bits for i in range(len(linears))}, {i: bits for i in range(len(acts))}


def theoretical_weight_storage_bytes(model: nn.Module, weight_bits: dict[int, int | None]) -> float:
    """Analytical storage: each Linear weight tensor at its bit-width (FP32 if None), biases at 4 bytes."""
    linears, _ = unit_names(model)
    total = 0.0
    for i, name in enumerate(linears):
        lin = model.net.get_submodule(name)
        total += lin.weight.numel() * (weight_bits.get(i) or 32) / 8 + lin.bias.numel() * 4
    return total
