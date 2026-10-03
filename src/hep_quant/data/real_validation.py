"""Gate B: end-to-end validation report for the real HEP dataset.

Loads the full dataset and the configured subset, runs integrity checks, a
single-feature leakage screen, split-contamination checks, and writes
`real_data_validation.{json,md}`. All numbers come from the loaded data; the
report also records content fingerprints so two independent runs (after wiping
the download cache) can be compared for determinism.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from hep_quant.data.hls4ml_jets import EXPECTED_CLASSES, OPENML_NAME, OPENML_VERSION, load_hls4ml_jets
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import check_no_leakage, split_dataset
from hep_quant.data.validation import validate_dataset

LEAKAGE_FLAG_ACCURACY = 0.90  # a single shallow-tree feature this predictive would be suspicious


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _class_table(y: np.ndarray, names: list[str]) -> dict:
    counts = np.bincount(y, minlength=len(names))
    return {n: {"count": int(c), "fraction": float(c / len(y))} for n, c in zip(names, counts)}


def _single_feature_leakage(X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int) -> dict:
    n = min(len(X), 40000)
    half = n // 2
    out = {}
    for j, name in enumerate(feature_names):
        tree = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X[:half, j : j + 1], y[:half])
        out[name] = float(tree.score(X[half:n, j : j + 1], y[half:n]))
    return out


def validate_real_dataset(dataset_cfg: dict, seed: int = 42) -> dict:
    subset_size = dataset_cfg.get("subset_size")
    val_fraction = dataset_cfg.get("val_fraction", 0.15)
    test_fraction = dataset_cfg.get("test_fraction", 0.15)

    full = load_hls4ml_jets(subset_size=None, seed=seed)
    full_report = validate_dataset(full.X, full.y, raise_on_error=False)

    stds = full.X.std(axis=0)
    safe = np.where(stds > 0, stds, 1.0)
    zscores = np.abs((full.X - full.X.mean(axis=0)) / safe)
    feature_stats = {}
    for j, name in enumerate(full.feature_names):
        col = full.X[:, j]
        feature_stats[name] = {
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "std": float(col.std()),
            "n_abs_z_gt_10": int((zscores[:, j] > 10).sum()),
        }

    sub = load_hls4ml_jets(subset_size=subset_size, seed=seed)
    splits = split_dataset(sub.X, sub.y, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    overlap = check_no_leakage(splits)
    pre = Preprocessor.fit(splits.X_train)
    X_train_scaled = pre.transform(splits.X_train)

    report = {
        "dataset": OPENML_NAME,
        "openml_version": OPENML_VERSION,
        "seed": seed,
        "full_dataset": {
            "n_samples": full_report.n_samples,
            "n_features": full_report.n_features,
            "classes": EXPECTED_CLASSES,
            "class_distribution": _class_table(full.y, full.class_names),
            "n_nan": full_report.n_nan,
            "n_inf": full_report.n_inf,
            "n_duplicate_rows": full_report.n_duplicate_rows,
            "warnings": full_report.warnings,
            "fingerprint_sha256_X": _sha256(full.X),
            "fingerprint_sha256_y": _sha256(full.y),
            "feature_stats": feature_stats,
        },
        "subset": {
            "requested_subset_size": subset_size,
            "n_samples": int(len(sub.y)),
            "class_distribution": _class_table(sub.y, sub.class_names),
            "fingerprint_sha256_X": _sha256(sub.X),
        },
        "splits": {
            "train": int(len(splits.y_train)), "validation": int(len(splits.y_val)), "test": int(len(splits.y_test)),
            "train_class_distribution": _class_table(splits.y_train, sub.class_names),
            "test_class_distribution": _class_table(splits.y_test, sub.class_names),
            "cross_split_duplicate_rows": int(overlap),
            "fingerprint_sha256_train_X": _sha256(splits.X_train),
            "fingerprint_sha256_test_X": _sha256(splits.X_test),
            "preprocessed_train_mean_abs_max": float(np.abs(X_train_scaled.mean(axis=0)).max()),
            "preprocessed_train_std_range": [float(X_train_scaled.std(axis=0).min()), float(X_train_scaled.std(axis=0).max())],
        },
        "leakage_screen": {
            "method": f"depth-3 decision tree on each single feature, fit on first half / scored on second half of up to 40000 subset rows; flag if accuracy > {LEAKAGE_FLAG_ACCURACY}",
            "chance_accuracy": 1.0 / len(EXPECTED_CLASSES),
            "per_feature_accuracy": _single_feature_leakage(sub.X, sub.y, sub.feature_names, seed),
        },
    }
    accs = report["leakage_screen"]["per_feature_accuracy"]
    report["leakage_screen"]["flagged_features"] = [k for k, v in accs.items() if v > LEAKAGE_FLAG_ACCURACY]
    report["status"] = (
        "PASS"
        if (full_report.n_nan == 0 and full_report.n_inf == 0 and overlap == 0 and not report["leakage_screen"]["flagged_features"])
        else "FAIL"
    )
    return report


def to_markdown(report: dict) -> str:
    f, s, sp, lk = report["full_dataset"], report["subset"], report["splits"], report["leakage_screen"]
    lines = [
        "# Real-data validation",
        "",
        f"Dataset: `{report['dataset']}` (OpenML version {report['openml_version']}), seed {report['seed']}. **Status: {report['status']}**",
        "",
        "## Full dataset",
        f"- Samples: {f['n_samples']:,}; features: {f['n_features']}; classes: {', '.join(f['classes'])}",
        f"- NaN: {f['n_nan']}; Inf: {f['n_inf']}; duplicate feature rows: {f['n_duplicate_rows']}",
        f"- Fingerprint (sha256 of float32 X): `{f['fingerprint_sha256_X']}`",
        "",
        "| Class | Count | Fraction |", "|---|---:|---:|",
    ]
    lines += [f"| {k} | {v['count']:,} | {v['fraction']:.4f} |" for k, v in f["class_distribution"].items()]
    lines += ["", "| Feature | min | max | mean | std | #(|z|>10) |", "|---|---:|---:|---:|---:|---:|"]
    lines += [
        f"| {k} | {v['min']:.4g} | {v['max']:.4g} | {v['mean']:.4g} | {v['std']:.4g} | {v['n_abs_z_gt_10']} |"
        for k, v in f["feature_stats"].items()
    ]
    lines += [
        "", "## Configured subset and splits",
        f"- Requested subset size: {s['requested_subset_size']}; actual: {s['n_samples']:,} (stratified)",
        f"- Train / validation / test: {sp['train']:,} / {sp['validation']:,} / {sp['test']:,}",
        f"- Duplicate feature rows shared across splits: {sp['cross_split_duplicate_rows']}",
        f"- Train fingerprint `{sp['fingerprint_sha256_train_X']}`; test fingerprint `{sp['fingerprint_sha256_test_X']}`",
        f"- Preprocessed train: max |mean| = {sp['preprocessed_train_mean_abs_max']:.2e}, std range = {sp['preprocessed_train_std_range']}",
        "", "## Leakage screen",
        f"Method: {lk['method']}. Chance accuracy = {lk['chance_accuracy']:.2f}. Flagged features: {lk['flagged_features'] or 'none'}.",
        "", "| Feature | single-feature accuracy |", "|---|---:|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in lk["per_feature_accuracy"].items()]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path = "reports", stem: str = "real_data_validation") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jp.write_text(json.dumps(report, indent=2))
    mp.write_text(to_markdown(report))
    return jp, mp
