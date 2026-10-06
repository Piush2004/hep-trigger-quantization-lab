#!/usr/bin/env python
"""Phase 6 TEST #1 vs TEST #2: compare per-seed metrics and summaries of the two independent runs.

Reports exact-equality counts and the maximum absolute difference per metric. Writes no results.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"


def main() -> int:
    a = pd.read_csv(TAB / "phase6_run1_per_seed.csv")
    b = pd.read_csv(TAB / "phase6_run2_per_seed.csv")
    m = a.merge(b, on=["arch", "seed", "precision", "method"], suffixes=("_r1", "_r2"))
    print(f"matched rows: {len(m)} (expected 50)")
    ok = True
    for metric in ["roc_auc", "pr_auc", "accuracy", "f1"]:
        d = (m[f"{metric}_r1"] - m[f"{metric}_r2"]).abs()
        exact = int((d == 0).sum())
        print(f"{metric:9s} max|diff| = {d.max():.3e}   exact-equal rows = {exact}/{len(m)}")
    d_auc = (m.roc_auc_r1 - m.roc_auc_r2).abs()
    if d_auc.max() > 1e-9:
        ok = False
    sa = pd.read_csv(TAB / "phase6_run1_summary.csv")
    sb = pd.read_csv(TAB / "phase6_run2_summary.csv")
    s = sa.merge(sb, on=["arch", "precision", "method"], suffixes=("_r1", "_r2"))
    dd = (s.roc_auc_mean_r1 - s.roc_auc_mean_r2).abs().max()
    print(f"summary roc_auc_mean max|diff| across 10 configurations = {dd:.3e}")
    print("RESULT:", "IDENTICAL (within 1e-9)" if ok and dd < 1e-9 else "DIFFERENT - investigate before proceeding")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
