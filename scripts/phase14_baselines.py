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
# Same 3 seeds scripts/phase14_jet_image_cnn.py uses (measured-cost-justified 3-seed precedent,
# see reports/tables/phase14_cnn_timing.json) -- for a fair, paired seed-by-seed comparison
# rather than a single-point baseline vs. a multi-seed CNN.
SEEDS = [42, 123, 2026]


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

    print(f"loading official train/ (subsampled to {TRAIN_SUBSET} jets, fixed pool, seed=42)...")
    # Subsample pool fixed once at seed=42, matching scripts/phase14_jet_image_cnn.py exactly --
    # each seed below only varies the internal 85/15 split + model randomness, not which jets
    # are in the 100,000-jet pool, for a fair comparison against what the CNN script actually ran.
    train_ds = load_and_downsample_jet_images(str(TRAIN_DIR), target_size=TARGET_SIZE, max_jets=TRAIN_SUBSET, seed=42)
    print(f"  {len(train_ds.y)} jets loaded")

    print("loading official val/ (held-out test, full 270,000 jets, touched once)...")
    test_ds = load_and_downsample_jet_images(str(VAL_DIR), target_size=TARGET_SIZE)
    X_test_img, y_test = test_ds.images, test_ds.y
    Xte_feat = extract_image_features(X_test_img)
    Xte_flat = X_test_img.reshape(len(y_test), -1)
    print(f"  {len(y_test)} test jets")

    rows = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        idx_train, _idx_val = train_test_split(
            range(len(train_ds.y)), test_size=0.15, random_state=seed, stratify=train_ds.y
        )
        X_train_img, y_train = train_ds.images[idx_train], train_ds.y[idx_train]

        Xtr_feat = extract_image_features(X_train_img)
        gbm = HistGradientBoostingClassifier(random_state=seed)
        gbm.fit(Xtr_feat, y_train)
        probs = gbm.predict_proba(Xte_feat)
        rows.append({"seed": seed, **score(y_test, probs.argmax(1), probs, "ImageFeatures+GBM")})

        Xtr_flat = X_train_img.reshape(len(y_train), -1)
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(Xtr_flat, y_train)
        probs = lr.predict_proba(Xte_flat)
        rows.append({"seed": seed, **score(y_test, probs.argmax(1), probs, "FlattenedPixels+LogReg")})

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "phase14_baselines.csv", index=False)
    summary = df.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        f1_mean=("f1", "mean"),
        bkg_rejection_mean=(f"bkg_rejection_at_{int(EFFICIENCY * 100)}pct_eff_macro", "mean"),
    ).round(4)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
