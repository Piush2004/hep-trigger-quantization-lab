#!/usr/bin/env python
"""Phase 12, QAT stage: fine-tune the PTQ Pareto-frontier layouts (not all 2187), then select
and test exactly once, mirroring the PTQ-side protocol in phase12_exhaustive_mixed_precision_pareto.py.

Reads reports/tables/phase12_pareto_frontier.csv (computed from the full confirm table by
find_pareto_frontier -- 23 layouts for the run this was built against) and fine-tunes each at all
5 seeds using build_qat_variant (hep_quant.analysis.qat_studies), with the same epoch/LR formula
hep_quant.experiments.runner uses for uniform QAT (qat_epochs = max(5, epochs // 4), qat_lr = lr * 0.1).

Protocol (fixed before running):
  1. Validation-only QAT scoring for every frontier layout, all 5 seeds (stage A).
  2. Selection (pre-declared, same rule as the PTQ stage): highest mean validation ROC-AUC among
     layouts within the dynamic INT6-uniform storage limit; tie-break smaller storage, then name.
     Written to disk before any test-set evaluation.
  3. Stage B: the single selected QAT layout, test split, once, all 5 seeds.

Outputs: reports/tables/phase12_qat_confirm_validation.csv, phase12_qat_selection.json,
         phase12_qat_final_test.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "reports" / "tables"

SEEDS = [42, 123, 2026, 7, 99]
TIE_EPS = 0.001


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


def _training_cfg(seed: int) -> dict:
    """Reuse exactly the training config the frozen FP32 checkpoint for this seed was built with."""
    frozen = ROOT / "experiments" / "results" / "frozen_phase2" / f"EXP_S{seed}_JET_FP32.json"
    rec_path = frozen if frozen.exists() else ROOT / "experiments" / "results" / f"EXP_S{seed}_JET_FP32.json"
    return json.loads(rec_path.read_text())["config"]["training"]


def stage_a_validation(p5, p12, layouts: list[tuple[str, dict, dict]], seeds: list[int] = SEEDS) -> pd.DataFrame:
    from hep_quant.analysis.qat_studies import build_qat_variant

    rows = []
    for s in seeds:
        model, tr_val, _ = p5.load_splits(s, include_test=False)
        Xtr = tr_val["train"]
        Xva, yva = tr_val["val"]
        y_train = _y_train_for_seed(p5, s)
        training_cfg = _training_cfg(s)
        for name, w, a in layouts:
            ckpt = ROOT / "experiments" / "checkpoints" / "phase12_qat_tmp" / f"S{s}_{name}"
            v = build_qat_variant(model, w, a, Xtr, y_train, Xva, yva, training_cfg, ckpt, device="cpu", seed=s)
            rows.append({"seed": s, "layout": name, **p5.score(v, Xva, yva),
                         "theoretical_weight_storage_bytes": p5.theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def _y_train_for_seed(p5, seed: int):
    """load_splits doesn't return y_train (PTQ never needs labels for calibration); rebuild the
    exact same split to get it, reusing the identical config/seed path load_splits already used."""
    import json as _json

    from hep_quant.data.hls4ml_jets import load_hls4ml_jets
    from hep_quant.data.splits import split_dataset

    frozen = ROOT / "experiments" / "results" / "frozen_phase2" / f"EXP_S{seed}_JET_FP32.json"
    rec_path = frozen if frozen.exists() else ROOT / "experiments" / "results" / f"EXP_S{seed}_JET_FP32.json"
    d = _json.loads(rec_path.read_text())["config"]["dataset"]
    ds = load_hls4ml_jets(subset_size=d["subset_size"], seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    return sp.y_train


def select(confirm_df: pd.DataFrame, limit: float) -> dict:
    summary = confirm_df.groupby("layout").agg(
        val_auc_mean=("roc_auc", "mean"), val_auc_sd=("roc_auc", "std"),
        storage=("theoretical_weight_storage_bytes", "first")).reset_index()
    base = {
        "selection_rule": ("among QAT-fine-tuned frontier layouts with theoretical_weight_storage_bytes "
                           "<= size_limit_bytes, select highest mean validation macro ROC-AUC over the 5 "
                           "seeds; ties within tie_eps broken by smaller storage, then layout name"),
        "size_limit_bytes": float(limit), "tie_eps": TIE_EPS, "candidate_count": int(len(summary)),
        "selection_inputs": "validation split only (QAT stage A, include_test=False)",
        "selection_used_test_set": False,
    }
    elig = summary[summary.storage <= limit]
    if elig.empty:
        base.update({"selected_layout": None, "status": "NO ELIGIBLE CONFIGURATION"})
        return base
    best = elig.val_auc_mean.max()
    tied = elig[elig.val_auc_mean >= best - TIE_EPS].sort_values(["storage", "layout"])
    chosen = tied.iloc[0]
    base.update({
        "selected_layout": chosen["layout"], "selected_storage_bytes": float(chosen["storage"]),
        "validation_mean": float(chosen["val_auc_mean"]),
        "validation_sd": float(chosen["val_auc_sd"]) if pd.notna(chosen["val_auc_sd"]) else None,
        "indistinguishable_from_best": tied["layout"].tolist(), "status": "SELECTED",
        "test_set_used_after_selection": None,
    })
    return base


def stage_b_test(p5, name: str, w: dict, a: dict, seeds: list[int] = SEEDS) -> pd.DataFrame:
    from hep_quant.analysis.qat_studies import build_qat_variant

    rows = []
    for s in seeds:
        model, tr_val, (Xte, yte) = p5.load_splits(s, include_test=True)
        Xva, yva = tr_val["val"]
        y_train = _y_train_for_seed(p5, s)
        training_cfg = _training_cfg(s)
        ckpt = ROOT / "experiments" / "checkpoints" / "phase12_qat_tmp" / f"test_S{s}_{name}"
        v = build_qat_variant(model, w, a, tr_val["train"], y_train, Xva, yva, training_cfg, ckpt, device="cpu", seed=s)
        rows.append({"seed": s, "layout": name, **p5.score(v, Xte, yte),
                     "theoretical_weight_storage_bytes": p5.theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def main() -> int:
    p5 = _load_phase5()
    p12 = _load_phase12()
    TAB.mkdir(parents=True, exist_ok=True)
    device = {"cuda_available": torch.cuda.is_available(), "device": "cpu"}

    frontier = pd.read_csv(TAB / "phase12_pareto_frontier.csv")
    by_name = p12.layouts_by_name()
    layouts = []
    for name in frontier["layout"]:
        w, a = by_name[name]
        layouts.append((name, w, a))
    print(f"QAT fine-tuning {len(layouts)} frontier layouts x {len(SEEDS)} seeds")

    model0, _tr_val, _ = p5.load_splits(42, include_test=False)
    limit = p12.size_limit_bytes(model0, p5)

    confirm = stage_a_validation(p5, p12, layouts, seeds=SEEDS)
    confirm.to_csv(TAB / "phase12_qat_confirm_validation.csv", index=False)

    sel = select(confirm, limit)
    sel.update(device)
    (TAB / "phase12_qat_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    if sel.get("selected_layout") is None:
        print(sel["status"])
        return 0
    name = sel["selected_layout"]
    w, a = by_name[name]
    print(f"QAT selected: {name} (mean val ROC-AUC {sel['validation_mean']:.4f}, "
          f"storage {sel['selected_storage_bytes']:.0f} B)")

    final = stage_b_test(p5, name, w, a, seeds=SEEDS)
    final.to_csv(TAB / "phase12_qat_final_test.csv", index=False)
    print(final[["seed", "roc_auc", "f1", "accuracy"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
