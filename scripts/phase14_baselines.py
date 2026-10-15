#!/usr/bin/env python
"""Phase 14 baselines: hand-engineered-features+GBM and flattened-pixel+LogisticRegression,
scored against the pre-declared rule in reports/phase14_baseline_predeclaration.md, BEFORE any
CNN training (matching the Phase 13 "baseline before NN" discipline exactly).

Official train/ -> internal 85/15 train/val split (seed=42, stratified). Official val/ (full
270,000 jets) -> held-out test, touched once. Images downsampled 100x100 -> 25x25
(energy-conserving sum-pool). Train is subsampled to 100,000 jets (matching the existing
tabular-MLP project's own subset_size convention) for training-time reasons; test uses the
full official val split (confirmed memory-safe: ~1.9GB peak RSS, measured directly).

Outputs: reports/tables/phase14_baselines.csv
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.data.image_features import extract_image_features  # noqa: E402
from hep_quant.data.jet_images import load_and_downsample_jet_images  # noqa: E402
from hep_quant.training.metrics import macro_background_rejection_at_efficiency  # noqa: E402

TRAIN_DIR = ROOT / "data" / "raw" / "hls4ml_jet_images_30p" / "train"
VAL_DIR = ROOT / "data" / "raw" / "hls4ml_jet_images_30p" / "val"
TAB = ROOT / "reports" / "tables"
TARGET_SIZE = 25
EFFICIENCY = 0.70
TRAIN_SUBSET = 100_000
SEED = 42


def score(y_true, y_pred, y_prob, label: str) -> dict:
    rej = macro_background_rejection_at_efficiency(y_true, y_prob, efficiency=EFFICIENCY)
    return {
        "model": label,
        "roc_auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        f"bkg_rejection_at_{int(EFFICIENCY * 100)}pct_eff_macro": rej["macro"],
    }


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)

    print(f"loading official train/ (subsampled to {TRAIN_SUBSET} jets)...")
    train_ds = load_and_downsample_jet_images(str(TRAIN_DIR), target_size=TARGET_SIZE, max_jets=TRAIN_SUBSET, seed=SEED)
    print(f"  {len(train_ds.y)} jets loaded")

    idx_train, idx_val = train_test_split(
        range(len(train_ds.y)), test_size=0.15, random_state=SEED, stratify=train_ds.y
    )
    X_train_img, y_train = train_ds.images[idx_train], train_ds.y[idx_train]
    # idx_val / internal val carve-out not used by these baselines (no early stopping / model
    # selection needed for default-hyperparameter sklearn models) -- kept for parity with the
    # CNN training script's split, which does use it.

    print("loading official val/ (held-out test, full 270,000 jets, touched once)...")
    test_ds = load_and_downsample_jet_images(str(VAL_DIR), target_size=TARGET_SIZE)
    X_test_img, y_test = test_ds.images, test_ds.y
    print(f"  {len(y_test)} test jets")

    rows = []

    print("training baseline 1: hand-engineered image features + HistGradientBoostingClassifier...")
    Xtr_feat = extract_image_features(X_train_img)
    Xte_feat = extract_image_features(X_test_img)
    gbm = HistGradientBoostingClassifier(random_state=SEED)
    gbm.fit(Xtr_feat, y_train)
    probs = gbm.predict_proba(Xte_feat)
    rows.append(score(y_test, probs.argmax(1), probs, "ImageFeatures+GBM"))

    print("training baseline 2: flattened pixels + LogisticRegression...")
    Xtr_flat = X_train_img.reshape(len(y_train), -1)
    Xte_flat = X_test_img.reshape(len(y_test), -1)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(Xtr_flat, y_train)
    probs = lr.predict_proba(Xte_flat)
    rows.append(score(y_test, probs.argmax(1), probs, "FlattenedPixels+LogReg"))

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "phase14_baselines.csv", index=False)
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
