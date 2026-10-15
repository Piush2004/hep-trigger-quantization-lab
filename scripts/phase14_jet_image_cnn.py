#!/usr/bin/env python
"""Phase 14 CNN: trains JetImageCNN on the hls4ml jet-image dataset, scored against the
pre-declared baselines in reports/phase14_baselines.csv (reports/phase14_baseline_predeclaration.md).

Protocol (measure-then-decide, matching Phase 12's timing-benchmark pattern): times one seed's
full training run first, then decides between the project's full 5-seed convention
([42,123,2026,7,99]) and its existing 3-seed precedent ([42,123,2026], the same subset
freeze_phase2.py uses) based on the measured per-run cost -- never silently under-seeds without
stating the measured cost and the resulting choice.

Official train/ -> internal 85/15 train/val split (seed-dependent, stratified). Official val/
(full 270,000 jets) -> held-out test, touched once after all seeds' validation scores are in.
Images downsampled 100x100 -> 25x25. Train subsampled to 100,000 jets (same subset size and
rationale as scripts/phase14_baselines.py, for an apples-to-apples comparison).

Outputs: reports/tables/phase14_cnn_timing.json, phase14_cnn_validation.csv, phase14_cnn_test.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.data.jet_images import load_and_downsample_jet_images  # noqa: E402
from hep_quant.models.cnn import JetImageCNN  # noqa: E402
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
FULL_SEEDS = [42, 123, 2026, 7, 99]
REDUCED_SEEDS = [42, 123, 2026]
TRAINING_CFG = {"epochs": 30, "batch_size": 512, "learning_rate": 0.001, "weight_decay": 0.0001,
                 "early_stopping_patience": 6, "device": "cpu"}
# CPU-only: torch.cuda.is_available() confirmed False in this environment -- CNN training runs
# on CPU regardless of preference, matching the project's documented CPU/CUDA-divergence
# discipline for anything meant to be comparable across runs.


def score(y_true, probs, label: str) -> dict:
    y_pred = probs.argmax(axis=1)
    rej = macro_background_rejection_at_efficiency(y_true, probs, efficiency=EFFICIENCY)
    return {
        "label": label,
        "roc_auc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        f"bkg_rejection_at_{int(EFFICIENCY * 100)}pct_eff_macro": rej["macro"],
    }


def predict_proba(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[i : i + batch_size]))
            outs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(outs, axis=0)


def run_one_seed(seed: int, train_images: np.ndarray, train_y: np.ndarray):
    idx_train, idx_val = train_test_split(
        np.arange(len(train_y)), test_size=0.15, random_state=seed, stratify=train_y
    )
    X_train = train_images[idx_train][:, None, :, :]
    X_val = train_images[idx_val][:, None, :, :]
    y_train, y_val = train_y[idx_train], train_y[idx_val]

    set_seed(seed)
    model = JetImageCNN(image_size=TARGET_SIZE, in_channels=1, conv_channels=[16, 32, 64], n_classes=5)
    ckpt_dir = CKPT / f"EXP_S{seed}_CNN_FP32"
    trainer = Trainer(model, TRAINING_CFG, ckpt_dir)

    t0 = time.perf_counter()
    history = trainer.fit(X_train, y_train, X_val, y_val)
    elapsed = time.perf_counter() - t0

    val_probs = predict_proba(model, X_val)
    val_metrics = score(y_val, val_probs, f"CNN_S{seed}_val")
    return model, history, elapsed, val_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-only", action="store_true", help="Run one seed, print timing, exit.")
    args = parser.parse_args()

    TAB.mkdir(parents=True, exist_ok=True)
    print(f"loading official train/ (subsampled to {TRAIN_SUBSET} jets)...")
    train_ds = load_and_downsample_jet_images(str(TRAIN_DIR), target_size=TARGET_SIZE, max_jets=TRAIN_SUBSET, seed=42)
    print(f"  {len(train_ds.y)} jets loaded")

    print("timing one seed (42) to decide the full seed count, measure don't guess...")
    _model, history, elapsed, val_metrics = run_one_seed(42, train_ds.images, train_ds.y)
    n_epochs_run = len(history.epochs)
    mean_epoch_s = float(np.mean(history.epoch_duration_s))
    timing = {
        "seed": 42, "elapsed_s": elapsed, "n_epochs_run": n_epochs_run,
        "mean_epoch_duration_s": mean_epoch_s, "cuda_available": torch.cuda.is_available(), "device": "cpu",
    }
    (TAB / "phase14_cnn_timing.json").write_text(json.dumps(timing, indent=2))
    print(f"  seed 42: {elapsed:.1f}s total, {n_epochs_run} epochs, {mean_epoch_s:.2f}s/epoch mean")
    if args.timing_only:
        return 0

    # Decide seed count from the measured cost -- full 5 seeds if a run is a few minutes,
    # the project's existing 3-seed precedent if tens of minutes (documented either way).
    full_cost_min = elapsed * len(FULL_SEEDS) / 60
    if elapsed <= 300:  # a single run <= 5 min -> full grid is a few-minutes-per-seed affair
        seeds = FULL_SEEDS
        seed_rationale = f"seed 42 took {elapsed:.1f}s; full 5-seed cost ~{full_cost_min:.1f} min -- using all 5 seeds"
    else:
        seeds = REDUCED_SEEDS
        seed_rationale = (f"seed 42 took {elapsed:.1f}s ({elapsed/60:.1f} min); full 5-seed cost would be "
                           f"~{full_cost_min:.1f} min -- using the project's existing 3-seed precedent instead")
    print(seed_rationale)

    val_rows = [val_metrics]
    models = {42: _model}
    for s in seeds:
        if s == 42:
            continue
        print(f"training seed {s}...")
        model, _h, _e, vm = run_one_seed(s, train_ds.images, train_ds.y)
        val_rows.append(vm)
        models[s] = model

    val_df = pd.DataFrame(val_rows)
    val_df["seed_rationale"] = seed_rationale
    val_df.to_csv(TAB / "phase14_cnn_validation.csv", index=False)
    print(val_df.round(4).to_string(index=False))

    print("loading official val/ (held-out test, full 270,000 jets, touched once)...")
    test_ds = load_and_downsample_jet_images(str(VAL_DIR), target_size=TARGET_SIZE)
    X_test = test_ds.images[:, None, :, :]
    y_test = test_ds.y

    test_rows = []
    for s in seeds:
        probs = predict_proba(models[s], X_test)
        test_rows.append({"seed": s, **score(y_test, probs, f"CNN_S{s}_test")})
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(TAB / "phase14_cnn_test.csv", index=False)
    print(test_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
