"""Helpers for auditing recorded real-data experiments (rebuild model + exact test split from a record).

Everything here re-derives artifacts from what an experiment *recorded* (its stored
config and saved checkpoint); nothing is re-trained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hep_quant.data.hls4ml_jets import load_hls4ml_jets
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.ptq import quantize_model_ptq
from hep_quant.quantization.qat import prepare_qat_model


def test_split_for_record(record: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rebuild the held-out, preprocessed test split exactly as the experiment saw it."""
    cfg = record["config"]
    d, seed = cfg["dataset"], cfg.get("seed", 42)
    ds = load_hls4ml_jets(subset_size=d.get("subset_size"), seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    return pre.transform(sp.X_test), sp.y_test, ds.class_names


def load_checkpoint_model(checkpoint_path: str | Path, method: str, precision: str) -> torch.nn.Module:
    """Rebuild the (possibly fake-quantized) model structure and load the saved weights/buffers.

    PTQ/QAT structures are re-created from an untrained MLP of the recorded shape and then
    fully overwritten by the checkpoint's state dict (including calibrated activation ranges).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MLPClassifier(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"], n_classes=ckpt["n_classes"])
    if method == "ptq" and precision != "fp32":
        dummy = np.zeros((8, ckpt["input_dim"]), dtype=np.float32)
        model = quantize_model_ptq(model, precision, dummy)
    elif method == "qat" and precision != "fp32":
        model = prepare_qat_model(model, precision)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    outs = [torch.softmax(model(torch.from_numpy(X[i : i + batch_size])), dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(outs).numpy()


def macro_roc_curve(y_true: np.ndarray, probs: np.ndarray, grid: int = 400) -> tuple[np.ndarray, np.ndarray, float]:
    """Macro-average one-vs-rest ROC curve (mean TPR over classes on a common FPR grid) and its AUC."""
    from sklearn.metrics import roc_curve

    fpr_grid = np.linspace(0, 1, grid)
    tprs = []
    for c in range(probs.shape[1]):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), probs[:, c])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    return fpr_grid, mean_tpr, float(np.trapezoid(mean_tpr, fpr_grid))
