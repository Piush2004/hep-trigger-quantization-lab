#!/usr/bin/env python
"""Export the Phase 6 architecture results under the required name reports/tables/phase6_architecture_results.csv.

Source: reports/tables/phase6_run1_per_seed.csv (run 1). Run 2 matched it exactly (see phase6_compare_runs.py),
and this script asserts that run 2 matches before exporting, so the exported file cannot silently differ.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"

ARCH_NAME = {"A": "A_[64,32,32]", "B": "B_[128,64,64,32]"}


def main() -> int:
    a = pd.read_csv(TAB / "phase6_run1_per_seed.csv")
    b = pd.read_csv(TAB / "phase6_run2_per_seed.csv")
    m = a.merge(b, on=["arch", "seed", "precision", "method"], suffixes=("_r1", "_r2"))
    for metric in ["roc_auc", "pr_auc", "accuracy", "f1"]:
        if (m[f"{metric}_r1"] != m[f"{metric}_r2"]).any():
            print(f"run1 and run2 differ on {metric}; refusing to export")
            return 1
    out = pd.DataFrame({
        "architecture": a["arch"].map(ARCH_NAME),
        "precision": a["precision"],
        "method": a["method"],
        "seed": a["seed"],
        "roc_auc": a["roc_auc"],
        "f1": a["f1"],
        "accuracy": a["accuracy"],
        "pr_auc": a["pr_auc"],
        "parameter_count": a["params"],
        "theoretical_weight_storage_bytes": a["storage_bytes"],
        "device": "cpu",
        "source": "experiments/results_phase6_run1 (identical to run2)",
    })
    metrics = out[["roc_auc", "pr_auc", "accuracy", "f1"]]
    if not (metrics.notna().all().all() and pd.Series(metrics.values.ravel()).apply(lambda x: x == x and abs(x) != float("inf")).all()):
        print("NaN or Inf found in metrics; refusing to export")
        return 1
    out.to_csv(TAB / "phase6_architecture_results.csv", index=False)
    print(f"wrote {len(out)} rows to reports/tables/phase6_architecture_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
