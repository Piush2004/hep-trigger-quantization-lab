#!/usr/bin/env python
"""Phase 5B: size-constrained, validation-selected mixed-precision gate.

Reuses the Phase 5 candidate space and stage functions from scripts/phase5_mixed_precision_gate.py
(unchanged), so the candidate set and validation scores are the same computation.

Pre-declared before any run (not tuned afterwards):
  SIZE_LIMIT_BYTES = 3724.0   theoretical weight storage of uniform INT6 PTQ
  TIE_EPS          = 0.001    mean validation ROC-AUC difference treated as indistinguishable;
                              within that band the SMALLER layout is preferred, then candidate name.
Selection rule:
  among mixed candidates with theoretical storage <= SIZE_LIMIT_BYTES, select the highest mean
  validation macro ROC-AUC over the five seeds (with the tie rule above). No test metric participates.

Order of operations (enforced in code):
  1. Stage A: validation scoring for every candidate (load with include_test=False).
  2. Selection: computed from the validation table only; the selection file is written.
  3. Stage B: the single selected layout is evaluated on the test split, once.
  4. Pareto and statistics are computed after stage B.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "reports" / "tables"

SIZE_LIMIT_BYTES = 3724.0
TIE_EPS = 0.001
SEEDS = [42, 123, 2026, 7, 99]


def _load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_gate", ROOT / "scripts" / "phase5_mixed_precision_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_table(p5, val_df: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate validation summary with eligibility. Validation columns only."""
    assert not any("test" in c for c in val_df.columns), "validation table must not contain test columns"
    mixed = val_df[val_df["mixed"]]
    rows = []
    for name, g in mixed.groupby("candidate"):
        wo, ao = p5.MIXED[name]
        storage = float(g["theoretical_weight_storage_bytes"].iloc[0])
        per_seed = {f"val_auc_seed{s}": float(g.loc[g.seed == s, "roc_auc"].iloc[0]) for s in SEEDS}
        rows.append({
            "candidate": name,
            "bit_widths": "; ".join([f"L{k}=INT8" for k in wo] + [f"A{k}=INT8" for k in ao]) + " (rest INT4)",
            "int8_units": int(g["int8_units"].iloc[0]),
            "theoretical_weight_storage_bytes": storage,
            **per_seed,
            "val_auc_mean": float(g["roc_auc"].mean()),
            "val_auc_sd": float(g["roc_auc"].std(ddof=1)),
            "eligible": storage <= SIZE_LIMIT_BYTES,
        })
    df = pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)
    df["selection_status"] = "not eligible"
    return df


def select(cand: pd.DataFrame) -> dict:
    elig = cand[cand["eligible"]].copy()
    base = {
        "selection_rule": ("among mixed candidates with theoretical_weight_storage_bytes <= size_limit_bytes, select highest "
                           "mean validation macro ROC-AUC over the five seeds; candidates within tie_eps of the best are "
                           "treated as indistinguishable and the smallest storage is preferred, then candidate name"),
        "size_limit_bytes": SIZE_LIMIT_BYTES,
        "tie_eps": TIE_EPS,
        "candidate_count": int(len(cand)),
        "eligible_candidate_count": int(len(elig)),
        "seeds": SEEDS,
        "selection_inputs": "validation split only (stage A, include_test=False)",
        "selection_used_test_set": False,
    }
    if elig.empty:
        base.update({"selected_layout": None, "status": "NO ELIGIBLE MIXED-PRECISION CONFIGURATION"})
        return base
    best = elig["val_auc_mean"].max()
    tied = elig[elig["val_auc_mean"] >= best - TIE_EPS].sort_values(["theoretical_weight_storage_bytes", "candidate"])
    chosen = tied.iloc[0]
    base.update({
        "selected_layout": chosen["candidate"],
        "selected_bit_widths": chosen["bit_widths"],
        "selected_storage_bytes": float(chosen["theoretical_weight_storage_bytes"]),
        "validation_mean": float(chosen["val_auc_mean"]),
        "validation_sd": float(chosen["val_auc_sd"]),
        "indistinguishable_from_best": tied["candidate"].tolist(),
        "status": "SELECTED",
        "test_set_used_after_selection": None,  # set to True only after stage B completes
    })
    return base


def main() -> int:
    p5 = _load_phase5()
    TAB.mkdir(parents=True, exist_ok=True)

    # ---- 1. stage A: validation only (same code path as Phase 5) ----
    val_df = p5.stage_a_validation()
    cand = candidate_table(p5, val_df)

    # ---- 2. selection, written before any test evaluation ----
    sel = select(cand)
    cand["selection_status"] = np.where(
        ~cand["eligible"], "not eligible (storage > limit)",
        np.where(cand["candidate"] == sel.get("selected_layout"), "SELECTED", "eligible, not selected"))
    cand.to_csv(TAB / "phase5b_mixed_candidates_validation.csv", index=False)
    (TAB / "phase5b_mixed_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    if sel.get("selected_layout") is None:
        print(sel["status"])
        return 0
    layout = sel["selected_layout"]
    print(f"selected on validation: {layout} (mean val AUC {sel['validation_mean']:.4f}, "
          f"storage {sel['selected_storage_bytes']:.0f} B)")

    # ---- 3. stage B: test, once, selected layout only ----
    final = p5.stage_b_test(layout)
    final.to_csv(TAB / "phase5b_mixed_final_test.csv", index=False)
    sel.update({
        "test_set_used_after_selection": True,
        "final_test_mean": float(final["roc_auc"].mean()),
        "final_test_sd": float(final["roc_auc"].std(ddof=1)),
        "final_test_per_seed": {int(r.seed): float(r.roc_auc) for r in final.itertuples()},
    })
    (TAB / "phase5b_mixed_selection.json").write_text(json.dumps(sel, indent=2, default=float))

    # ---- 4. comparison and Pareto (after test; reference rows are recorded test metrics) ----
    ref = p5.reference_rows()
    ref_by = ref.groupby("candidate").agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std")).reset_index()
    storage_of = {"FP32": 17556.0, "INT8 PTQ": 4788.0, "INT6 PTQ": 3724.0, "INT4 PTQ": 2660.0, "INT4 QAT": 2660.0}
    points = [{"configuration": k, "storage_bytes": v,
               "roc_auc_mean": float(ref_by.loc[ref_by.candidate == k, "roc_auc_mean"].iloc[0])} for k, v in storage_of.items()]
    points.append({"configuration": "Mixed L0L1A0 INT8 (Phase 5, validation-selected, not size-constrained)",
                   "storage_bytes": 4196.0, "roc_auc_mean": 0.9281})
    points.append({"configuration": f"Mixed {layout} (Phase 5B, size-constrained)",
                   "storage_bytes": sel["selected_storage_bytes"], "roc_auc_mean": float(final["roc_auc"].mean())})
    pts = pd.DataFrame(points)
    dominated = []
    for i, p in pts.iterrows():
        others = pts.drop(i)
        d = others[(others.storage_bytes <= p.storage_bytes) & (others.roc_auc_mean >= p.roc_auc_mean)
                   & ((others.storage_bytes < p.storage_bytes) | (others.roc_auc_mean > p.roc_auc_mean))]
        dominated.append(bool(len(d)))
    pts["dominated"] = dominated
    pts.to_csv(TAB / "phase5b_pareto.csv", index=False)

    # paired statistics, mixed (5B) minus each comparator, same seeds
    p = ref.pivot(index="seed", columns="candidate", values="roc_auc")
    f = final.set_index("seed")["roc_auc"]
    stat_rows = []
    for name in ["INT6 PTQ", "INT4 QAT", "INT4 PTQ", "FP32"]:
        d = f - p[name]
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        stat_rows.append({"comparator": name, "mean_diff": float(d.mean()), "ci_low": float(d.mean() - t * se),
                          "ci_high": float(d.mean() + t * se), "seeds_positive": int((d > 0).sum()), "n": n})
    pd.DataFrame(stat_rows).to_csv(TAB / "phase5b_paired_comparisons.csv", index=False)

    print(final[["seed", "roc_auc"]].round(4).to_string(index=False))
    print(pd.DataFrame(stat_rows).round(4).to_string(index=False))
    print(pts.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
