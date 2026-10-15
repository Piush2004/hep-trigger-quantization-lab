#!/usr/bin/env python
"""Phase 14 quantization: PTQ + QAT sweep on JetImageCNN, mirroring the original MLP study's
scope (Phase 2/3: FP32 baseline, INT8/INT6/INT4 PTQ, INT8/INT6/INT4 QAT = 7 configurations),
now that ptq.py and qat.py both support Conv2d / JetImageCNN's two-block structure.

Scope disclosure: the CNN's FP32 training cost (measured ~8-30 min/seed, see
reports/tables/phase14_cnn_timing.json and reports/phase14_cnn_jet_images.md) makes a 3-seed
PTQ/QAT sweep (7 configs x 3 seeds = 21 training-adjacent runs, each needing its own FP32 base
plus, for QAT, several fine-tune epochs) materially more expensive than the MLP study's
equivalent. This script runs the full 7-config sweep on a single seed (42) as a demonstration
that the pipeline works end-to-end on the CNN -- disclosed as single-seed, not silently
presented as equivalent statistical power to the project's usual multi-seed convention.

Outputs: reports/tables/phase14_cnn_quantization.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.data.jet_images import load_and_downsample_jet_images  # noqa: E402
from hep_quant.models.cnn import JetImageCNN  # noqa: E402
from hep_quant.quantization.ptq import quantize_model_ptq  # noqa: E402
from hep_quant.quantization.qat import prepare_qat_model  # noqa: E402
from hep_quant.training.metrics import macro_background_rejection_at_efficiency  # noqa: E402
from hep_quant.training.trainer import Trainer  # noqa: E402
from hep_quant.utils.seeds import set_seed  # noqa: E402

TRAIN_DIR = ROOT / "data" / "raw" / "hls4ml_jet_images_30p" / "train"
VAL_DIR = ROOT / "data" / "raw" / "hls4ml_jet_images_30p" / "val"
TAB = ROOT / "reports" / "tables"
CKPT = ROOT / "experiments" / "checkpoints"

TARGET_SIZE = 25
EFFICIENCY = 0.70
TRAIN_SUBSET = 100_000
SEED = 42
TRAINING_CFG = {"epochs": 30, "batch_size": 512, "learning_rate": 0.001, "weight_decay": 0.0001,
                 "early_stopping_patience": 6, "device": "cpu"}
PRECISIONS = ["int8", "int6", "int4"]


def predict_proba(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[i : i + batch_size]))
            outs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outs, axis=0)


def score(y_true, probs, label: str) -> dict:
    y_pred = probs.argmax(axis=1)
    rej = macro_background_rejection_at_efficiency(y_true, probs, efficiency=EFFICIENCY)
    return {
        "config": label,
        "roc_auc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        f"bkg_rejection_at_{int(EFFICIENCY * 100)}pct_eff_macro": rej["macro"],
    }


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"loading official train/ (subsampled to {TRAIN_SUBSET} jets, seed={SEED})...")
    train_ds = load_and_downsample_jet_images(str(TRAIN_DIR), target_size=TARGET_SIZE, max_jets=TRAIN_SUBSET, seed=SEED)
    from sklearn.model_selection import train_test_split

    idx_train, idx_val = train_test_split(
        np.arange(len(train_ds.y)), test_size=0.15, random_state=SEED, stratify=train_ds.y
    )
    X_train = train_ds.images[idx_train][:, None, :, :]
    X_val = train_ds.images[idx_val][:, None, :, :]
    y_train, y_val = train_ds.y[idx_train], train_ds.y[idx_val]
    X_train_calib = X_train[:2048]  # small calibration subset for PTQ activation-range observation

    print("loading official val/ (held-out test, full 270,000 jets, touched once)...")
    test_ds = load_and_downsample_jet_images(str(VAL_DIR), target_size=TARGET_SIZE)
    X_test = test_ds.images[:, None, :, :]
    y_test = test_ds.y

    print(f"training FP32 base model (seed {SEED})...")
    set_seed(SEED)
    fp32_model = JetImageCNN(image_size=TARGET_SIZE, in_channels=1, conv_channels=[16, 32, 64], n_classes=5)
    ckpt_dir = CKPT / f"EXP_S{SEED}_CNN_QUANT_FP32"
    t0 = time.perf_counter()
    Trainer(fp32_model, TRAINING_CFG, ckpt_dir).fit(X_train, y_train, X_val, y_val)
    print(f"  FP32 training took {time.perf_counter() - t0:.1f}s")

    probs = predict_proba(fp32_model, X_test)
    rows.append({"seed": SEED, **score(y_test, probs, "FP32")})

    for precision in PRECISIONS:
        print(f"PTQ {precision}...")
        ptq_model = quantize_model_ptq(fp32_model, precision, X_train_calib, device="cpu")
        probs = predict_proba(ptq_model, X_test)
        rows.append({"seed": SEED, **score(y_test, probs, f"{precision.upper()}_PTQ")})

        print(f"QAT {precision}...")
        qat_epochs = max(5, TRAINING_CFG["epochs"] // 4)
        qat_cfg = {**TRAINING_CFG, "epochs": qat_epochs, "learning_rate": TRAINING_CFG["learning_rate"] * 0.1}
        set_seed(SEED)
        qat_model = prepare_qat_model(fp32_model, precision)
        qat_ckpt_dir = CKPT / f"EXP_S{SEED}_CNN_QUANT_{precision.upper()}_QAT"
        t0 = time.perf_counter()
        Trainer(qat_model, qat_cfg, qat_ckpt_dir).fit(X_train, y_train, X_val, y_val)
        print(f"  QAT {precision} fine-tune took {time.perf_counter() - t0:.1f}s")
        probs = predict_proba(qat_model, X_test)
        rows.append({"seed": SEED, **score(y_test, probs, f"{precision.upper()}_QAT")})

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "phase14_cnn_quantization.csv", index=False)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
