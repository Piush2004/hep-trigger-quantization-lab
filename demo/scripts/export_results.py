#!/usr/bin/env python
"""Export the frozen research results to JSON for the demo layer. Read-only on the research files.

Reads: reports/tables/*.csv, reports/numerical_consistency.json, experiments/results(/frozen_phase2)/*.json
Writes: demo/data/*.json   (metadata, baseline, pareto, architecture, calibration, hls4ml, quantization)

Every row carries its device and evidence tier. Nothing is retrained and nothing is invented: values absent
from the sources are written as null. Re-run this script only if the research results change (they do not
on branch demo/web-release).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "reports" / "tables"
OUT = ROOT / "demo" / "data"
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
SEEDS = [42, 123, 2026, 7, 99]
RESEARCH_COMMIT = "970f44a"

MEASURED = "MEASURED"
SIMULATED = "SIMULATED"
ESTIMATED = "ESTIMATED"
NOT_AVAILABLE = "NOT AVAILABLE"


def _nan_to_none(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def clean(rows):
    return [{k: _nan_to_none(v) for k, v in r.items()} for r in rows]


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=float))


def primary_per_seed(tag: str):
    out = []
    for s in SEEDS:
        name = f"EXP_S{s}_JET_{tag}.json"
        p = FROZEN / name if (FROZEN / name).exists() else RESULTS / name
        t = json.loads(p.read_text())["metrics"]["test"]
        out.append({"seed": s, "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "f1": t["f1"], "accuracy": t["accuracy"]})
    return out


def main() -> int:
    final = pd.read_csv(TAB / "final_results.csv")
    p6 = pd.read_csv(TAB / "phase6_architecture_results.csv")
    pareto = pd.read_csv(TAB / "phase5b_pareto.csv")
    calib = pd.read_csv(TAB / "phase4_int4_D_calib_size.csv")
    numc = json.loads((ROOT / "reports" / "numerical_consistency.json").read_text())

    # ---- metadata ----
    write("metadata.json", {
        "research_commit": RESEARCH_COMMIT,
        "dataset": {"name": "OpenML hls4ml_lhc_jets_hlf v1", "doi": "10.5281/zenodo.3602260", "licence": "CC BY 4.0",
                    "jets": 830000, "features": 16, "classes": ["g", "q", "t", "W", "Z"],
                    "nature": "public simulated particle-level Monte Carlo; not detector data",
                    "fingerprint_sha256_X": "b2863e6c92578c774dc4b677f88a8f054906226197e2ade5a938d3f2531f096c"},
        "seeds": SEEDS,
        "split": "70/15/15 stratified, seed-controlled; 100,000-jet subset",
        "devices": {"primary_frozen": "CUDA-era records (seeds 42/123/2026 frozen; 7/99 as recorded)",
                    "architecture_rerun": "CPU (Phase 6, both architectures)",
                    "calibration": "CPU (Phase 4, architecture A)",
                    "hls4ml": "CPU, C-simulation"},
        "evidence_tiers": {
            "MEASURED": "PyTorch evaluation on held-out test split",
            "SIMULATED": "hls4ml C-simulation (fixed-point arithmetic, no hardware)",
            "ESTIMATED": "software hardware-resource heuristic (not validated by synthesis)",
            "NOT AVAILABLE": "HLS synthesis and physical FPGA measurement (no tool or device)"},
        "storage_definition": "theoretical weight storage in bytes (analytical); checkpoints are float32; not file size",
        "not_available": ["HLS synthesis (Vivado/Vitis/Quartus not installed)", "physical FPGA measurement",
                          "latency on hardware", "power or energy"],
    })

    # ---- baseline: canonical table, one row per configuration ----
    base_rows = []
    for _, r in final.iterrows():
        if "Mixed" in r.configuration:
            device = "FP32 checkpoint CUDA-era; PTQ layout evaluated on CPU"
        elif r.architecture.startswith("A primary"):
            device = "CUDA-era frozen"
        elif r.architecture.startswith("A re-run"):
            device = "CPU"
        else:
            device = "CPU"
        base_rows.append({
            "architecture": r.architecture, "configuration": r.configuration, "device": device, "n": int(r.n),
            "roc_auc_mean": r.roc_auc_mean, "roc_auc_sd": r.roc_auc_sd, "roc_auc_min": r.roc_auc_min,
            "roc_auc_max": r.roc_auc_max, "roc_auc_ci95_low": r.roc_auc_ci95_low, "roc_auc_ci95_high": r.roc_auc_ci95_high,
            "pr_auc_mean": r.pr_auc_mean, "f1_mean": r.f1_mean, "accuracy_mean": r.accuracy_mean,
            "delta_vs_fp32": r.abs_degradation_vs_fp32, "pct_vs_fp32": r.pct_degradation_vs_fp32,
            "storage_bytes": int(r.theoretical_weight_storage_bytes), "tier": MEASURED,
            "source": r.source,
        })
    write("baseline.json", {"rows": clean(base_rows)})

    # ---- quantization: primary per-seed values, so the UI can show seed variance ----
    quant = {}
    for label, tag in [("FP32", "FP32"), ("INT8 PTQ", "INT8_PTQ"), ("INT6 PTQ", "INT6_PTQ"),
                       ("INT8 QAT", "INT8_QAT"), ("INT6 QAT", "INT6_QAT"),
                       ("INT4 PTQ", "INT4_PTQ"), ("INT4 QAT", "INT4_QAT")]:
        quant[label] = primary_per_seed(tag)
    write("quantization.json", {"device": "CUDA-era frozen", "tier": MEASURED, "per_seed": quant})

    # INT8 QAT and INT6 QAT exist in the frozen primary records but are not in the canonical table.
    # Add them from the same per-seed records so the explorer is complete. Nothing is typed by hand.
    fp_seeds = {r["seed"]: r["roc_auc"] for r in quant["FP32"]}
    for label, tag, bytes_ in [("INT8 QAT", "INT8_QAT", 4788), ("INT6 QAT", "INT6_QAT", 3724)]:
        rec = quant[label]
        aucs = pd.Series({r["seed"]: r["roc_auc"] for r in rec})
        d = aucs - pd.Series(fp_seeds)
        n = len(aucs)
        sd = aucs.std(ddof=1)
        t = stats.t.ppf(0.975, n - 1)
        base_rows.append({
            "architecture": "A primary [64,32,32]", "configuration": label, "device": "CUDA-era frozen", "n": n,
            "roc_auc_mean": aucs.mean(), "roc_auc_sd": sd, "roc_auc_min": aucs.min(), "roc_auc_max": aucs.max(),
            "roc_auc_ci95_low": aucs.mean() - t * sd / np.sqrt(n), "roc_auc_ci95_high": aucs.mean() + t * sd / np.sqrt(n),
            "pr_auc_mean": float(np.mean([r["pr_auc"] for r in rec])), "f1_mean": float(np.mean([r["f1"] for r in rec])),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rec])),
            "delta_vs_fp32": d.mean(), "pct_vs_fp32": 100 * d.mean() / float(np.mean(list(fp_seeds.values()))),
            "storage_bytes": bytes_, "tier": MEASURED,
            "source": f"experiments/results/frozen_phase2 EXP_S*_JET_{tag} (per-seed records)",
        })
    write("baseline.json", {"rows": clean(base_rows)})

    # ---- pareto: points and dominance flags as recorded by phase5b_pareto.csv ----
    write("pareto.json", {
        "axes": {"x": "theoretical weight storage (bytes, analytical)", "y": "test macro ROC-AUC (mean)"},
        "note": "Dominance flags as recorded in reports/tables/phase5b_pareto.csv. The mixed point is validation-selected under the 3,724 B constraint.",
        "points": clean(pareto.to_dict(orient="records")),
    })

    # ---- architecture: both architectures, same protocol, CPU ----
    arch_rows = []
    for arch, g in p6.groupby("architecture"):
        fp = g[g.precision == "FP32"].set_index("seed")
        for (prec, method), gg in g.groupby(["precision", "method"], sort=False):
            gg = gg.set_index("seed").loc[SEEDS]
            d = gg["roc_auc"] - fp["roc_auc"]
            arch_rows.append({"architecture": arch, "precision": prec, "method": method, "device": "CPU", "n": 5,
                              "roc_auc_mean": gg.roc_auc.mean(), "roc_auc_sd": gg.roc_auc.std(ddof=1),
                              "roc_auc_min": gg.roc_auc.min(), "roc_auc_max": gg.roc_auc.max(),
                              "delta_vs_own_fp32_mean": d.mean(), "delta_vs_own_fp32_sd": d.std(ddof=1),
                              "parameter_count": int(gg.parameter_count.iloc[0]),
                              "storage_bytes": int(gg.theoretical_weight_storage_bytes.iloc[0]),
                              "per_seed_roc_auc": {int(s): float(v) for s, v in gg.roc_auc.items()},
                              "tier": MEASURED})
    write("architecture.json", {"note": "Same protocol and seeds for both architectures; re-run on CPU. The CUDA-era primary records are in baseline.json.",
                                "rows": clean(arch_rows)})

    # ---- calibration: Architecture A, CPU, Phase 4 ----
    crow = []
    for (bits, n), g in calib.groupby(["bits", "calib_size"]):
        crow.append({"precision": f"INT{bits}", "calib_samples": int(n), "n": int(len(g)),
                     "roc_auc_mean": g.roc_auc.mean(), "roc_auc_sd": g.roc_auc.std(ddof=1),
                     "act_range_hi_mean": g.A0_hi.mean(), "act_step_mean": g.A0_step.mean(),
                     "clip_rate_mean": g.A0_clip_rate.mean(), "argmax_disagree_vs_fp32": g.argmax_disagree_vs_fp32.mean(),
                     "device": "CPU", "architecture": "A [64,32,32]", "tier": MEASURED})
    write("calibration.json", {"note": "Min/max calibration. Model seed dominates the spread; calibration size acts through the activation range. Clip rate is ~0 under min/max.",
                               "rows": clean(crow)})

    # ---- hls4ml: PyTorch rows MEASURED, C-simulation rows SIMULATED, synthesis and hardware NOT AVAILABLE ----
    hls_rows = [{"stage": "PyTorch FP32", "tier": MEASURED, "accuracy": numc["fp32_pytorch"]["accuracy"],
                 "roc_auc": numc["fp32_pytorch"]["roc_auc"]}]
    for k, v in numc["pytorch_quantized"].items():
        hls_rows.append({"stage": f"PyTorch {k}", "tier": MEASURED, "accuracy": v["accuracy"], "roc_auc": v["roc_auc"],
                         "agreement_with_fp32": v["class_agreement_with_fp32"], "d_roc_auc_vs_fp32": v["d_roc_auc_vs_fp32"]})
    for r in numc["hls4ml_csim"]:
        hls_rows.append({"stage": "hls4ml C-simulation", "mode": r["mode"], "width_W": r["ap_fixed_total_bits"],
                         "int_bits": r["int_bits"], "tier": SIMULATED, "accuracy": r["accuracy"], "roc_auc": r["roc_auc"],
                         "agreement_with_fp32": r["class_agreement_with_fp32"], "d_roc_auc_vs_fp32": r["d_roc_auc_vs_fp32"],
                         "logit_mae": r["logit_mae"], "prob_mae": r["prob_mae"], "prob_max_abs_err": r["prob_max_abs_err"]})
    hls_rows.append({"stage": "HLS synthesis", "tier": NOT_AVAILABLE, "note": "HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT"})
    hls_rows.append({"stage": "Physical FPGA", "tier": NOT_AVAILABLE, "note": "no FPGA measurement exists"})
    write("hls4ml.json", {"rows": clean(hls_rows), "note": "SIMULATED values come from fixed-point C-simulation of the seed-42 model; they are not hardware results."})

    print("wrote demo/data/: metadata, baseline, quantization, pareto, architecture, calibration, hls4ml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
