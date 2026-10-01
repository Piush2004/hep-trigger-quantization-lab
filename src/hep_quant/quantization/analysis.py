"""Model size / parameter-count analysis for FP32 vs quantized models.

Sizes for INT8/INT6/INT4 are ANALYTICAL ESTIMATES: this project's PTQ/QAT
implementation stores fake-quantized weights as float32 (see ptq.py docstring),
so we compute what the model WOULD occupy if weights were bit-packed at the
target width, rather than measuring an actually-packed artifact on disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from hep_quant.quantization.ptq import SUPPORTED_PRECISIONS

BYTES_PER_FP32_PARAM = 4


@dataclass
class ModelSizeReport:
    precision: str
    total_params: int
    weight_params: int
    bias_params: int
    estimated_size_bytes: float
    estimated_size_mb: float
    is_estimated: bool


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return (total_params, weight_params, bias_params) across all nn.Linear layers."""
    total = sum(p.numel() for p in model.parameters())
    weight_params = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    bias_params = sum(
        m.bias.numel() for m in model.modules() if isinstance(m, nn.Linear) and m.bias is not None
    )
    return total, weight_params, bias_params


def estimate_model_size(model: nn.Module, precision: str) -> ModelSizeReport:
    """Estimate model size in bytes for the given precision.

    FP32 size is a direct, exact measurement (4 bytes/param, all params at
    full precision). INT8/INT6/INT4 sizes are estimates: weights at the
    target bit-width (packed), biases kept at FP32 (consistent with this
    project's PTQ/QAT methodology, which does not quantize biases).
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision '{precision}', choose from {list(SUPPORTED_PRECISIONS)}")

    total, weight_params, bias_params = count_parameters(model)

    if precision == "fp32":
        size_bytes = total * BYTES_PER_FP32_PARAM
        is_estimated = False
    else:
        bits = SUPPORTED_PRECISIONS[precision]
        weight_bytes = weight_params * bits / 8
        bias_bytes = bias_params * BYTES_PER_FP32_PARAM
        size_bytes = weight_bytes + bias_bytes
        is_estimated = True

    return ModelSizeReport(
        precision=precision,
        total_params=total,
        weight_params=weight_params,
        bias_params=bias_params,
        estimated_size_bytes=size_bytes,
        estimated_size_mb=size_bytes / (1024 * 1024),
        is_estimated=is_estimated,
    )
