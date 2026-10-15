#!/usr/bin/env python
"""Phase 13: non-NN baselines, and background rejection at fixed signal efficiency.

Two things a reviewer would ask for that nothing earlier in this project reports:
  1. Does the MLP's complexity earn its keep over a simple baseline? Trains logistic
     regression and gradient-boosted trees (HistGradientBoostingClassifier) on the
     identical preprocessed splits/seeds the MLP uses, with the same metric suite.
  2. A HEP-standard operating-point metric: background rejection (1/FPR) at a fixed
     signal efficiency (TPR), here 70% -- chosen as a common trigger-relevant working
     point, not tuned to flatter any model. This is a 5-class jet-tagging task, not a
     binary signal-vs-background search, so it is computed one-vs-rest per class and
     macro-averaged (see hep_quant.training.metrics.macro_background_rejection_at_efficiency)
     -- the adaptation is reported explicitly, not hidden behind one averaged number.

Reuses the exact dataset/split/preprocessing path the frozen FP32 record was built with
(load_hls4ml_jets + split_dataset + Preprocessor, same config, same seeds), so every row in
the final comparison table is evaluated on identical splits. Existing MLP configurations
(FP32, uniform INT6 PTQ, and this project's own Phase 12 mixed-precision PTQ/QAT selections)
are loaded/rebuilt read-only (no retraining of the FP32 base; only the already-selected
mixed-precision PTQ/QAT variants are rebuilt, which is cheap -- PTQ is a forward pass, QAT
reuses the exact fine-tune recipe already validated in Phase 12).

Outputs: reports/tables/phase13_baselines_and_efficiency.csv
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "reports" / "tables"

from hep_quant.analysis.phase2 import load_checkpoint_model, predict_proba  # noqa: E402
from hep_quant.analysis.ptq_studies import build_ptq_variant  # noqa: E402
from hep_quant.analysis.qat_studies import build_qat_variant  # noqa: E402
from hep_quant.data.hls4ml_jets import load_hls4ml_jets  # noqa: E402
from hep_quant.data.preprocessing import Preprocessor  # noqa: E402
from hep_quant.data.splits import split_dataset  # noqa: E402
from hep_quant.training.metrics import macro_background_rejection_at_efficiency  # noqa: E402

SEEDS = [42, 123, 2026, 7, 99]
EFFICIENCY = 0.70
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
CKPT = ROOT / "experiments" / "checkpoints"


def _load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_gate", ROOT / "scripts" / "phase5_mixed_precision_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_phase12():
    spec = importlib.util.spec_from_file_location("phase12_gate", ROOT / "scripts" / "phase12_exhaustive_mixed_precision_pareto.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_full_splits(seed: int):
    """Same dataset/config/split/preprocessing path the frozen FP32 record used,
    returning everything (including y_train) rather than just what load_splits exposes."""
    frozen_json = FROZEN / f"EXP_S{seed}_JET_FP32.json"
    rec_path = frozen_json if frozen_json.exists() else RESULTS / f"EXP_S{seed}_JET_FP32.json"
    d = json.loads(rec_path.read_text())["config"]["dataset"]
    ds = load_hls4ml_jets(subset_size=d["subset_size"], seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    return {
        "X_train": pre.transform(sp.X_train), "y_train": sp.y_train,
        "X_val": pre.transform(sp.X_val), "y_val": sp.y_val,
        "X_test": pre.transform(sp.X_test), "y_test": sp.y_test,
        "training_cfg": rec_path and json.loads(rec_path.read_text())["config"]["training"],
    }


def score_row(label: str, seed: int, y_test, probs: np.ndarray, n_params: int | None,
              fit_time_s: float | None, predict_time_s: float) -> dict:
    y_pred = probs.argmax(axis=1)
    rej = macro_background_rejection_at_efficiency(y_test, probs, efficiency=EFFICIENCY)
    return {
        "model": label, "seed": seed,
        "roc_auc": roc_auc_score(y_test, probs, multi_class="ovr", average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        f"bkg_rejection_at_{int(EFFICIENCY*100)}pct_eff_macro": rej["macro"],
        "n_params_or_equivalent": n_params,
        "fit_time_s": fit_time_s,
        "predict_time_s_per_sample_us": predict_time_s / len(y_test) * 1e6,
    }


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    p5 = _load_phase5()
    p12 = _load_phase12()
    qat_sel = json.loads((TAB / "phase12_qat_selection.json").read_text())
    ptq_sel = json.loads((TAB / "phase12_selection.json").read_text())
    by_name = p12.layouts_by_name()
    qat_layout = by_name[qat_sel["selected_layout"]]
    ptq_layout = by_name[ptq_sel["selected_layout"]]

    rows = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        data = load_full_splits(seed)
        Xtr, ytr = data["X_train"], data["y_train"]
        Xte, yte = data["X_test"], data["y_test"]

        # --- Logistic regression baseline ---
        t0 = time.perf_counter()
        lr = LogisticRegression(max_iter=1000, random_state=seed)  # sklearn >=1.5 always fits multinomial for >2 classes
        lr.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        probs = lr.predict_proba(Xte)
        pred_t = time.perf_counter() - t0
        n_params = lr.coef_.size + lr.intercept_.size
        rows.append(score_row("LogisticRegression", seed, yte, probs, n_params, fit_t, pred_t))

        # --- Gradient-boosted trees baseline ---
        t0 = time.perf_counter()
        gbm = HistGradientBoostingClassifier(random_state=seed)
        gbm.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        probs = gbm.predict_proba(Xte)
        pred_t = time.perf_counter() - t0
        rows.append(score_row("HistGradientBoosting", seed, yte, probs, None, fit_t, pred_t))

        # --- FP32 MLP reference (frozen, read-only) ---
        model, _tr_val, _ = p5.load_splits(seed, include_test=False)
        t0 = time.perf_counter()
        probs = predict_proba(model, Xte)
        pred_t = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters())
        rows.append(score_row("MLP FP32 (reference)", seed, yte, probs, n_params, None, pred_t))

        # --- Uniform INT6 PTQ reference (frozen, read-only) ---
        frozen_json = FROZEN / f"EXP_S{seed}_JET_INT6_PTQ.json"
        ckpt = (FROZEN / f"EXP_S{seed}_JET_INT6_PTQ__best_model.pt") if frozen_json.exists() \
            else (CKPT / f"EXP_S{seed}_JET_INT6_PTQ" / "best_model.pt")
        int6_model = load_checkpoint_model(ckpt, "ptq", "int6")
        t0 = time.perf_counter()
        probs = predict_proba(int6_model, Xte)
        pred_t = time.perf_counter() - t0
        rows.append(score_row("MLP INT6 PTQ uniform (reference)", seed, yte, probs, n_params, None, pred_t))

        # --- Phase 12 PTQ-selected mixed layout (rebuilt, read-only base, no retraining) ---
        w, a = ptq_layout
        v = build_ptq_variant(model, Xtr[:20480], w, a)
        t0 = time.perf_counter()
        probs = predict_proba(v, Xte)
        pred_t = time.perf_counter() - t0
        rows.append(score_row(f"MLP mixed PTQ selected ({ptq_sel['selected_layout']})", seed, yte, probs, n_params, None, pred_t))

        # --- Phase 12 QAT-selected mixed layout (rebuilt via the validated fine-tune recipe) ---
        w, a = qat_layout
        t0 = time.perf_counter()
        v = build_qat_variant(model, w, a, Xtr, ytr, data["X_val"], data["y_val"],
                               data["training_cfg"], str(CKPT / "phase13_qat_tmp" / f"S{seed}"), device="cpu", seed=seed)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        probs = predict_proba(v, Xte)
        pred_t = time.perf_counter() - t0
        rows.append(score_row(f"MLP mixed QAT selected ({qat_sel['selected_layout']})", seed, yte, probs, n_params, fit_t, pred_t))

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "phase13_baselines_and_efficiency.csv", index=False)
    summary = df.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        f1_mean=("f1", "mean"),
        bkg_rejection_mean=(f"bkg_rejection_at_{int(EFFICIENCY*100)}pct_eff_macro", "mean"),
        predict_us_mean=("predict_time_s_per_sample_us", "mean"),
    ).round(4)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
