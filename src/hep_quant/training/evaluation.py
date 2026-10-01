"""Model evaluation: inference + metrics on a held-out split."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from hep_quant.training.metrics import ClassificationMetrics, compute_metrics


@dataclass
class EvalResult:
    metrics: ClassificationMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray  # binary: 1D positive-class prob; multiclass: full (n_samples, n_classes) matrix
    inference_time_s: float
    n_samples: int

    @property
    def mean_latency_ms_per_sample(self) -> float:
        return 1000.0 * self.inference_time_s / self.n_samples if self.n_samples else float("nan")


@torch.no_grad()
def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu", batch_size: int = 1024) -> EvalResult:
    """Run inference over `X` in batches and compute metrics against `y`.

    Timing is measured as wall-clock software inference time on `device`
    (CPU or GPU) — explicitly NOT a hardware/FPGA latency measurement.
    """
    model = model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X).to(device)
    all_probs = []

    start = time.perf_counter()
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i : i + batch_size]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    probs = torch.cat(all_probs, dim=0).numpy()
    y_pred = probs.argmax(axis=1)
    # Binary: metrics/plots want the positive-class probability (1D).
    # Multiclass: macro ROC-AUC/PR-AUC need the full per-class probability matrix.
    y_prob = probs[:, 1] if probs.shape[1] == 2 else probs

    metrics = compute_metrics(y, y_pred, y_prob)
    return EvalResult(
        metrics=metrics,
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob,
        inference_time_s=elapsed,
        n_samples=len(y),
    )
