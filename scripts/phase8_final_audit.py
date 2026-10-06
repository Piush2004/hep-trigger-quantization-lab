#!/usr/bin/env python
"""Phase 8: canonical final results table, built only from existing per-seed records (no new training).

Sources (read-only):
  primary architecture A, CUDA-era frozen seeds 42/123/2026 : experiments/results/frozen_phase2/EXP_S*_JET_*.json
  primary architecture A, seeds 7/99 (added in Phase 3)      : experiments/results/EXP_S{7,99}_JET_*.json
  mixed precision, validation-selected (Phase 5, not size-constrained)  : reports/tables/phase5_mixed_final_test.csv
  mixed precision, validation-selected under 3,724 B (Phase 5B)         : reports/tables/phase5b_mixed_final_test.csv
  architecture B and architecture A re-run on CPU (Phase 6)             : reports/tables/phase6_architecture_results.csv

Each row: n, mean, SD, min, max, 95% t-CI, absolute and percentage degradation vs the SAME architecture's FP32 on
the SAME seeds, theoretical weight storage. Rows with metrics not recorded for that source are left blank, not invented.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
SEEDS = [42, 123, 2026, 7, 99]
STORAGE_A = {"FP32": 17556, "INT8 PTQ": 4788, "INT6 PTQ": 3724, "INT4 PTQ": 2660, "INT4 QAT": 2660}


def primary_record(seed: int, tag: str) -> dict:
    name = f"EXP_S{seed}_JET_{tag}.json"
    path = FROZEN / name if (FROZEN / name).exists() else RESULTS / name
    rec = json.loads(path.read_text())
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "test": rec["metrics"]["test"]}


def row(config: str, arch: str, source: str, metrics: dict, storage, fp32: dict | None) -> dict:
    """metrics: name -> list of per-seed values (same seed order). fp32: same-architecture FP32 per-seed values."""
    auc = np.asarray(metrics["roc_auc"], dtype=float)
    n = len(auc)
    sd = auc.std(ddof=1) if n > 1 else np.nan
    t = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    out = {"architecture": arch, "configuration": config, "source": source, "n": n,
           "roc_auc_mean": auc.mean(), "roc_auc_sd": sd, "roc_auc_min": auc.min(), "roc_auc_max": auc.max(),
           "roc_auc_ci95_low": auc.mean() - t * sd / np.sqrt(n), "roc_auc_ci95_high": auc.mean() + t * sd / np.sqrt(n)}
    for k in ["f1", "accuracy", "pr_auc"]:
        out[f"{k}_mean"] = float(np.mean(metrics[k])) if k in metrics else np.nan
    if fp32 is not None:
        d = auc - np.asarray(fp32["roc_auc"], dtype=float)
        out["abs_degradation_vs_fp32"] = d.mean()
        out["pct_degradation_vs_fp32"] = 100 * d.mean() / np.mean(fp32["roc_auc"])
    else:
        out["abs_degradation_vs_fp32"] = np.nan
        out["pct_degradation_vs_fp32"] = np.nan
    out["theoretical_weight_storage_bytes"] = storage
    return out


def main() -> int:
    rows = []
    # ---- primary architecture A, frozen/CUDA-era records (5 seeds) ----
    prim = {tag: [primary_record(s, tag) for s in SEEDS] for tag in ["FP32", "INT8_PTQ", "INT6_PTQ", "INT4_PTQ", "INT4_QAT"]}
    fp32_auc = [r["test"]["roc_auc"] for r in prim["FP32"]]
    names = {"FP32": "FP32", "INT8_PTQ": "INT8 PTQ", "INT6_PTQ": "INT6 PTQ", "INT4_PTQ": "INT4 PTQ", "INT4_QAT": "INT4 QAT"}
    for tag, label in names.items():
        recs = prim[tag]
        m = {k: [r["test"][k] for r in recs] for k in ["roc_auc", "f1", "accuracy", "pr_auc"]}
        src = "experiments/results(/frozen_phase2) EXP_S*_JET_" + tag + " (CUDA-era records; seeds 42/123/2026 frozen)"
        rows.append(row(label, "A primary [64,32,32]", src, m, STORAGE_A[label], {"roc_auc": fp32_auc}))

    # ---- mixed precision (ROC-AUC only recorded in these tables) ----
    m5 = pd.read_csv(TAB / "phase5b_mixed_final_test.csv")
    rows.append(row("Mixed L0+A0 INT8 (size-constrained, Phase 5B)", "A primary [64,32,32]",
                    "reports/tables/phase5b_mixed_final_test.csv (ROC-AUC only recorded)",
                    {"roc_auc": m5.set_index("seed").loc[SEEDS, "roc_auc"].tolist()}, 3172, {"roc_auc": fp32_auc}))
    m5a = pd.read_csv(TAB / "phase5_mixed_final_test.csv")
    rows.append(row("Mixed L0+L1+A0 INT8 (validation-selected, Phase 5)", "A primary [64,32,32]",
                    "reports/tables/phase5_mixed_final_test.csv (ROC-AUC only recorded)",
                    {"roc_auc": m5a.set_index("seed").loc[SEEDS, "roc_auc"].tolist()}, 4196, {"roc_auc": fp32_auc}))

    # ---- Phase 6 architectures (CPU, same protocol) ----
    p6 = pd.read_csv(TAB / "phase6_architecture_results.csv")
    for arch_label, arch_key in [("A re-run on CPU [64,32,32]", "A_[64,32,32]"), ("B [128,64,64,32]", "B_[128,64,64,32]")]:
        sub = p6[p6.architecture == arch_key]
        f = sub[sub.precision == "FP32"].set_index("seed").loc[SEEDS]
        for prec, method, label in [("FP32", "none", "FP32"), ("INT8", "PTQ", "INT8 PTQ"), ("INT6", "PTQ", "INT6 PTQ"),
                                    ("INT4", "PTQ", "INT4 PTQ"), ("INT4", "QAT", "INT4 QAT")]:
            g = sub[(sub.precision == prec) & (sub.method == method)].set_index("seed").loc[SEEDS]
            m = {"roc_auc": g.roc_auc.tolist(), "f1": g.f1.tolist(), "accuracy": g.accuracy.tolist(), "pr_auc": g.pr_auc.tolist()}
            rows.append(row(label, arch_label, "reports/tables/phase6_architecture_results.csv (CPU, Phase 6)", m,
                            int(g.theoretical_weight_storage_bytes.iloc[0]), {"roc_auc": f.roc_auc.tolist()}))

    df = pd.DataFrame(rows)
    assert np.isfinite(df[["roc_auc_mean", "roc_auc_sd", "roc_auc_min", "roc_auc_max"]].fillna(0).values).all()
    df.to_csv(TAB / "final_results.csv", index=False)
    print(df[["architecture", "configuration", "n", "roc_auc_mean", "roc_auc_sd", "abs_degradation_vs_fp32",
              "theoretical_weight_storage_bytes"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
