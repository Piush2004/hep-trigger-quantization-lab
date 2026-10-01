"""Software inference latency benchmarking (CPU/GPU wall-clock, NOT hardware/FPGA latency)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LatencyReport:
    device: str
    batch_size: int
    n_repeats: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    mean_ms_per_sample: float
    source: str = "measured_software"  # vs "estimated_hls" used by hardware backends


@torch.no_grad()
def benchmark_inference_latency(
    model: nn.Module,
    input_dim: int,
    device: str = "cpu",
    batch_size: int = 1,
    n_warmup: int = 10,
    n_repeats: int = 100,
) -> LatencyReport:
    """Measure wall-clock software inference latency for a single batch, repeated `n_repeats` times.

    This is a software-side CPU/GPU measurement using Python wall-clock timing
    around the forward pass. It is explicitly NOT a hardware/FPGA latency
    measurement — see hardware/resource_estimator.py for the HLS-estimated path.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(batch_size, input_dim, device=device)

    for _ in range(n_warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    durations_ms = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations_ms = np.array(durations_ms)
    return LatencyReport(
        device=device,
        batch_size=batch_size,
        n_repeats=n_repeats,
        mean_ms=float(durations_ms.mean()),
        std_ms=float(durations_ms.std()),
        p50_ms=float(np.percentile(durations_ms, 50)),
        p95_ms=float(np.percentile(durations_ms, 95)),
        mean_ms_per_sample=float(durations_ms.mean() / batch_size),
    )
