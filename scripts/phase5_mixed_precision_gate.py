#!/usr/bin/env python
"""Phase 5 gate: validation-selected mixed-precision PTQ, then ONE untouched test evaluation.

Replaces the test-selected exploratory result (Phase 3 S3 -> "L0+A0 INT8, rest INT4", 0.9278 ± 0.0037, n = 3),
which chose its layout using test-split ROC-AUC and is therefore BIASED.

Protocol (fixed before running; nothing here is tuned after seeing results):
  1. Candidate layouts are a pre-declared list (base INT4 + mixed layouts). No candidate is generated from data.
  2. Stage A (validation only): for each of the 5 seeds, build each candidate from the FP32 checkpoint with
     calibration on the first 20,480 TRAIN jets, and score macro ROC-AUC on the VALIDATION split.
     Stage A loads the split arrays together (the loader returns them in one call, so test arrays are
     materialised in memory) but never scores, selects on, or otherwise uses the test arrays.
  3. Selection rule (pre-declared): maximise mean validation ROC-AUC over the 5 seeds among MIXED candidates
     (at least one unit INT4 and at least one unit above INT4). Tie-break: fewer INT8 units, then name.
     The selection is written to disk BEFORE stage B starts.
  4. Stage B (test, once): evaluate only the selected layout on the TEST split, for the same 5 seeds.
     Reference rows (FP32, INT8/INT6/INT4 PTQ, INT4 QAT) are the recorded test metrics from the experiment
     JSON files; they were not used for selection.

Outputs: reports/tables/phase5_mixed_candidates_validation.csv, reports/tables/phase5_mixed_selection.json,
         reports/tables/phase5_mixed_final_test.csv, reports/tables/phase5_mixed_reference_test.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model, predict_proba  # noqa: E402
from hep_quant.analysis.ptq_studies import build_ptq_variant, theoretical_weight_storage_bytes, uniform_units, unit_names  # noqa: E402
from hep_quant.data.hls4ml_jets import load_hls4ml_jets  # noqa: E402
from hep_quant.data.preprocessing import Preprocessor  # noqa: E402
from hep_quant.data.splits import split_dataset  # noqa: E402

SEEDS = [42, 123, 2026, 7, 99]
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
CKPT = ROOT / "experiments" / "checkpoints"
TAB = ROOT / "reports" / "tables"

# Pre-declared mixed layouts. Units: L0..L3 = Linear weights, A0..A2 = hidden activations.
# Each entry: name -> (weight_bits_by_unit, act_bits_by_unit); unlisted units are INT4.
MIXED = {
    "L0_int8": ({0: 8}, {}),
    "L1_int8": ({1: 8}, {}),
    "L2_int8": ({2: 8}, {}),
    "L3_int8": ({3: 8}, {}),
    "A0_int8": ({}, {0: 8}),
    "A1_int8": ({}, {1: 8}),
    "A2_int8": ({}, {2: 8}),
    "L0A0_int8": ({0: 8}, {0: 8}),
    "L0L1A0_int8": ({0: 8, 1: 8}, {0: 8}),
}
BASE_BITS = 4


def load_splits(seed: int, include_test: bool):
    """Return the FP32 model, the train/val arrays, and the test arrays only when include_test is True.

    Stage A calls this with include_test=False, so the test split is never preprocessed or held in memory there.
    """
    frozen_json = FROZEN / f"EXP_S{seed}_JET_FP32.json"
    if frozen_json.exists():
        rec_path, ckpt = frozen_json, FROZEN / f"EXP_S{seed}_JET_FP32__best_model.pt"
    else:
        rec_path, ckpt = RESULTS / f"EXP_S{seed}_JET_FP32.json", CKPT / f"EXP_S{seed}_JET_FP32" / "best_model.pt"
    d = json.loads(rec_path.read_text())["config"]["dataset"]
    ds = load_hls4ml_jets(subset_size=d["subset_size"], seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    model = load_checkpoint_model(ckpt, "none", "fp32")
    tr_val = {
        "train": pre.transform(sp.X_train),
        "val": (pre.transform(sp.X_val), sp.y_val),
    }
    test = (pre.transform(sp.X_test), sp.y_test) if include_test else None
    return model, tr_val, test


def score(model, X, y) -> dict:
    p = predict_proba(model, X)
    return {
        "roc_auc": roc_auc_score(y, p, multi_class="ovr", average="macro"),
        "f1": f1_score(y, p.argmax(1), average="macro"),
        "accuracy": accuracy_score(y, p.argmax(1)),
    }


def layout_bits(model, weight_over, act_over):
    """Full per-unit bit maps: every unit INT4 unless overridden."""
    linears, acts = unit_names(model)
    w = {i: BASE_BITS for i in range(len(linears))}
    a = {i: BASE_BITS for i in range(len(acts))}
    w.update(weight_over)
    a.update(act_over)
    return w, a


def stage_a_validation() -> pd.DataFrame:
    """Validation-only scoring. The test split is not loaded here (include_test=False)."""
    rows = []
    for s in SEEDS:
        model, tr_and_val, _no_test = load_splits(s, include_test=False)
        Xtr = tr_and_val["train"]
        Xva, yva = tr_and_val["val"]
        rows.append({"seed": s, "candidate": "FP32", "int8_units": 0, "mixed": False,
                     **score(model, Xva, yva), "theoretical_weight_storage_bytes": theoretical_weight_storage_bytes(model, {})})
        for name, (wo, ao) in [("INT4_base", ({}, {}))] + [(k, v) for k, v in MIXED.items()]:
            w, a = layout_bits(model, wo, ao)
            v = build_ptq_variant(model, Xtr[:20480], w, a)
            n8 = sum(1 for b in list(w.values()) + list(a.values()) if b == 8)
            is_mixed = name != "INT4_base"
            rows.append({"seed": s, "candidate": name, "int8_units": n8, "mixed": is_mixed,
                         **score(v, Xva, yva), "theoretical_weight_storage_bytes": theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def select(val_df: pd.DataFrame) -> dict:
    mixed = val_df[val_df["mixed"]]
    summary = mixed.groupby("candidate").agg(mean_val_auc=("roc_auc", "mean"), int8_units=("int8_units", "first")).reset_index()
    summary = summary.sort_values(["mean_val_auc", "int8_units", "candidate"], ascending=[False, True, True])
    best = summary.iloc[0]
    return {
        "rule": "maximise mean validation macro ROC-AUC over 5 seeds among mixed layouts; tie-break fewer INT8 units, then name",
        "selected": best["candidate"],
        "selected_mean_val_auc": float(best["mean_val_auc"]),
        "ranking": summary.to_dict(orient="records"),
        "selection_inputs": "validation split only: stage_a_validation() loads with include_test=False and scores with score(model, Xva, yva)",
    }


def stage_b_test(selected: str) -> pd.DataFrame:
    """Test evaluation of the single selected layout. The selection is already on disk."""
    wo, ao = MIXED[selected]
    rows = []
    for s in SEEDS:
        model, _tr_val, (Xte, yte) = load_splits(s, include_test=True)
        w, a = layout_bits(model, wo, ao)
        v = build_ptq_variant(model, _tr_val["train"][:20480], w, a)
        rows.append({"seed": s, "candidate": selected, **score(v, Xte, yte),
                     "theoretical_weight_storage_bytes": theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def reference_rows() -> pd.DataFrame:
    """Recorded test metrics (historical experiment records). Not used for selection."""
    rows = []
    for s in SEEDS:
        for prec, method, label in [("FP32", "none", "FP32"), ("INT8", "ptq", "INT8 PTQ"), ("INT6", "ptq", "INT6 PTQ"),
                                    ("INT4", "ptq", "INT4 PTQ"), ("INT4", "qat", "INT4 QAT")]:
            name = f"EXP_S{s}_JET_{prec}_{method.upper()}" if prec != "FP32" else f"EXP_S{s}_JET_FP32"
            p = FROZEN / f"{name}.json"
            if not p.exists():
                p = RESULTS / f"{name}.json"
            rec = json.loads(p.read_text())
            t = rec["metrics"]["test"]
            rows.append({"seed": s, "candidate": label, "roc_auc": t["roc_auc"], "f1": t["f1"], "accuracy": t["accuracy"],
                         "source_file": str(p.relative_to(ROOT)).replace("\\", "/")})
    return pd.DataFrame(rows)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    device = {"cuda_available": torch.cuda.is_available(), "device": "cpu"}

    # ---- Stage A: validation only ----
    val_df = stage_a_validation()
    val_df.to_csv(TAB / "phase5_mixed_candidates_validation.csv", index=False)

    # ---- Selection (written before any test evaluation) ----
    sel = select(val_df)
    sel.update(device)
    (TAB / "phase5_mixed_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    selected = sel["selected"]
    print(f"selected on validation: {selected} (mean val ROC-AUC {sel['selected_mean_val_auc']:.4f})")

    # ---- Stage B: test, once, selected layout only ----
    final_df = stage_b_test(selected)
    final_df.to_csv(TAB / "phase5_mixed_final_test.csv", index=False)
    ref_df = reference_rows()
    ref_df.to_csv(TAB / "phase5_mixed_reference_test.csv", index=False)

    print(final_df[["seed", "roc_auc", "f1", "accuracy"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
