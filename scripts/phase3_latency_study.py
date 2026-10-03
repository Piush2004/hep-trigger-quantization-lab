#!/usr/bin/env python
"""Controlled software-latency study of the seed-42 FROZEN checkpoints (FP32 + 6 quantized variants).

Devices are measured and reported SEPARATELY (never compared as equivalent):
  cpu   : torch threads = 1, time.perf_counter around the forward pass
  cuda  : torch.cuda.Event timing around the forward pass, synchronize after every repetition
Batch sizes: 1 (single-jet latency) and 4096 (throughput). Warm-up then N repetitions; mean/median/std/p5/p95/p99.
These are SOFTWARE latencies of simulated (fake) quantization; they say nothing about FPGA latency. Run on an otherwise idle machine.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model

FROZEN = Path("experiments/results/frozen_phase2")
TAGS = {"FP32": ("none", "fp32"), "INT8_PTQ": ("ptq", "int8"), "INT6_PTQ": ("ptq", "int6"), "INT4_PTQ": ("ptq", "int4"),
        "INT8_QAT": ("qat", "int8"), "INT6_QAT": ("qat", "int6"), "INT4_QAT": ("qat", "int4")}


def stats(ms: np.ndarray) -> dict:
    return {"mean_ms": ms.mean(), "median_ms": np.median(ms), "std_ms": ms.std(ddof=1), "p5_ms": np.percentile(ms, 5),
            "p95_ms": np.percentile(ms, 95), "p99_ms": np.percentile(ms, 99)}


@torch.no_grad()
def time_cpu(model, x, warmup, reps) -> np.ndarray:
    for _ in range(warmup):
        model(x)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        model(x)
        out.append((time.perf_counter() - t) * 1e3)
    return np.array(out)


@torch.no_grad()
def time_cuda(model, x, warmup, reps) -> np.ndarray:
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        model(x)
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return np.array(out)


def main() -> int:
    torch.set_num_threads(1)
    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        for tag, (method, prec) in TAGS.items():
            model = load_checkpoint_model(FROZEN / f"EXP_S42_JET_{tag}__best_model.pt", method, prec).to(dev).eval()
            for batch, warm, reps in [(1, 100, 1000), (4096, 20, 100)]:
                x = torch.randn(batch, 16, device=dev)
                ms = (time_cuda if dev == "cuda" else time_cpu)(model, x, warm, reps)
                s = stats(ms)
                rows.append({"device": dev, "config": tag, "batch": batch, "reps": reps, **s, "per_sample_us_median": s["median_ms"] * 1e3 / batch,
                             "throughput_samples_per_s_median": batch / (s["median_ms"] / 1e3)})
    df = pd.DataFrame(rows)
    Path("reports/tables").mkdir(exist_ok=True)
    df.to_csv("reports/tables/phase3_latency_study.csv", index=False)
    ov = {}
    for dev in devices:
        b1 = df[(df.device == dev) & (df.batch == 1)].set_index("config")
        ov[dev] = {c: float(b1.loc[c, "median_ms"] / b1.loc["FP32", "median_ms"]) for c in b1.index}
    Path("reports/phase3_latency_study.json").write_text(json.dumps({"torch": torch.__version__, "devices": devices,
        "cpu_threads": 1, "median_ratio_to_fp32_batch1": ov, "protocol": "warmup 100/20, reps 1000/100, cpu perf_counter, cuda events + sync per rep"}, indent=2))
    fig, axes = plt.subplots(1, len(devices), figsize=(6.2 * len(devices), 4.6), squeeze=False)
    for ax, dev in zip(axes[0], devices):
        g = df[(df.device == dev) & (df.batch == 1)]
        ax.bar(g["config"], g["median_ms"], yerr=[g["median_ms"] - g["p5_ms"], g["p95_ms"] - g["median_ms"]], capsize=3, color="tab:gray")
        ax.set_ylabel("Median latency per forward pass, batch 1 (ms; bars = p5–p95)")
        ax.set_title(f"{dev.upper()} — software latency of simulated quantization")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("hls4ml_lhc_jets_hlf | MLP 16-64-32-32-5 | seed-42 frozen checkpoints | devices measured separately; NOT FPGA latency", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("reports/figures/phase3_latency_study.png", dpi=150)
    pd.set_option("display.width", 200)
    print(df[df.batch == 1][["device", "config", "median_ms", "std_ms", "p95_ms"]].round(4).to_string(index=False))
    print(df[df.batch == 4096][["device", "config", "median_ms", "throughput_samples_per_s_median"]].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
