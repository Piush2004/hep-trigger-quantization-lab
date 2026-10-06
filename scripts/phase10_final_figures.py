#!/usr/bin/env python
"""Phase 10: the final figure set 01–08. Built only from existing tables and per-seed records (read-only).

Output names are as requested by the final sprint. Older unnumbered/phase-prefixed figures are left in place.
Figures show means, SDs and per-seed points; no value is invented, and missing values are left out.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
SEEDS = [42, 123, 2026, 7, 99]
PREC = [("FP32", "none", "FP32"), ("INT8", "PTQ", "INT8 PTQ"), ("INT6", "PTQ", "INT6 PTQ"),
        ("INT4", "PTQ", "INT4 PTQ"), ("INT4", "QAT", "INT4 QAT")]


def primary_auc(tag: str) -> np.ndarray:
    out = []
    for s in SEEDS:
        name = f"EXP_S{s}_JET_{tag}.json"
        p = FROZEN / name if (FROZEN / name).exists() else RESULTS / name
        out.append(json.loads(p.read_text())["metrics"]["test"]["roc_auc"])
    return np.array(out)


def primary_table() -> dict:
    keys = {"FP32": "FP32", "INT8 PTQ": "INT8_PTQ", "INT6 PTQ": "INT6_PTQ", "INT4 PTQ": "INT4_PTQ", "INT4 QAT": "INT4_QAT"}
    return {label: primary_auc(tag) for label, tag in keys.items()}


def cpu_table(arch_key: str) -> dict:
    p6 = pd.read_csv(TAB / "phase6_architecture_results.csv")
    sub = p6[p6.architecture == arch_key]
    out = {}
    for prec, method, label in PREC:
        g = sub[(sub.precision == prec) & (sub.method == method)].set_index("seed").loc[SEEDS]
        out[label] = g.roc_auc.to_numpy()
    return out


def fig01_baseline() -> None:
    d = primary_table()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (label, v) in enumerate(d.items()):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, alpha=0.7, color="tab:blue")
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="_k", capsize=6, ms=16)
    ax.set_xticks(range(len(d)), list(d.keys()), fontsize=8)
    ax.set_ylabel("Test macro ROC-AUC")
    ax.set_title("Architecture A (primary, CUDA-era frozen records): ROC-AUC by precision\n5 seeds, mean ± SD; hls4ml_lhc_jets_hlf (simulated)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "01_baseline_quantization.png", dpi=150); plt.close(fig)


def fig02_int4_ptq_vs_qat() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    series = [("A primary (CUDA)", primary_table(), "tab:blue"),
              ("A re-run (CPU)", cpu_table("A_[64,32,32]"), "tab:cyan"),
              ("B (CPU)", cpu_table("B_[128,64,64,32]"), "tab:orange")]
    for k, (name, d, c) in enumerate(series):
        ptq, qat = d["INT4 PTQ"], d["INT4 QAT"]
        for s_i in range(len(ptq)):
            ax.plot([0 + k * 0.08, 1 + k * 0.08], [ptq[s_i], qat[s_i]], color=c, alpha=0.45, lw=1)
        ax.errorbar([k * 0.08, 1 + k * 0.08], [ptq.mean(), qat.mean()], yerr=[ptq.std(ddof=1), qat.std(ddof=1)],
                    color=c, marker="o", capsize=4, label=name)
    ax.set_xticks([0, 1], ["INT4 PTQ", "INT4 QAT"])
    ax.set_ylabel("Test macro ROC-AUC")
    ax.legend(fontsize=7)
    ax.set_title("INT4: PTQ → QAT per seed (lines) and mean ± SD", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "02_int4_ptq_vs_qat.png", dpi=150); plt.close(fig)


def fig03_degradation() -> None:
    sets = [("A primary (CUDA)", primary_table()), ("A re-run (CPU)", cpu_table("A_[64,32,32]")),
            ("B (CPU)", cpu_table("B_[128,64,64,32]"))]
    labels = ["INT8 PTQ", "INT6 PTQ", "INT4 PTQ", "INT4 QAT"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.27
    for k, (name, d) in enumerate(sets):
        fp = d["FP32"]
        dm, ds = [], []
        for lab in labels:
            diff = d[lab] - fp
            dm.append(diff.mean()); ds.append(diff.std(ddof=1))
        ax.bar(np.arange(len(labels)) + (k - 1) * width, dm, width, yerr=ds, capsize=3, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("ΔROC-AUC vs same-architecture FP32 (same seeds)")
    ax.legend(fontsize=7)
    ax.set_title("Quantization degradation relative to each architecture's own FP32", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "03_quantization_degradation.png", dpi=150); plt.close(fig)


def fig04_calibration() -> None:
    D = pd.read_csv(TAB / "phase4_int4_D_calib_size.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for bits, c in [(8, "tab:green"), (6, "tab:purple"), (4, "tab:red")]:
        g = D[D.bits == bits].groupby("calib_size").roc_auc
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std(ddof=1).values, color=c, marker="o", capsize=3, label=f"INT{bits} PTQ")
    ax.set_xscale("log")
    ax.set_xlabel("Calibration set size (training jets)")
    ax.set_ylabel("Test macro ROC-AUC (mean ± SD over 5 seeds × 5 draws)")
    ax.legend(fontsize=7)
    ax.set_title("Calibration sensitivity (Architecture A, CPU): INT4 falls with calibration size\nAll 5 seeds, 5 draws per size; model seed is the larger source of variation", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "04_calibration_sensitivity.png", dpi=150); plt.close(fig)


def fig05_architecture() -> None:
    A = cpu_table("A_[64,32,32]"); B = cpu_table("B_[128,64,64,32]")
    labels = [p[2] for p in PREC]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels))
    for k, (name, d, c) in enumerate([("A [64,32,32] CPU", A, "tab:blue"), ("B [128,64,64,32] CPU", B, "tab:orange")]):
        axes[0].errorbar(x + (k - 0.5) * 0.15, [d[l].mean() for l in labels], yerr=[d[l].std(ddof=1) for l in labels],
                         fmt="o-", color=c, capsize=3, label=name)
        dfp = [(d[l] - d["FP32"]).mean() for l in labels]
        dsd = [(d[l] - d["FP32"]).std(ddof=1) for l in labels]
        axes[1].errorbar(x + (k - 0.5) * 0.15, dfp, yerr=dsd, fmt="o-", color=c, capsize=3, label=name)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Test macro ROC-AUC (mean ± SD)")
    axes[1].set_ylabel("ΔROC-AUC vs own FP32")
    axes[1].axhline(0, color="k", lw=0.8)
    fig.suptitle("Architecture comparison (same protocol, same seeds, CPU). Parameters: A 4,389; B 16,837", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "05_architecture_comparison.png", dpi=150); plt.close(fig)


def fig06_pareto() -> None:
    F = pd.read_csv(TAB / "final_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {"A primary": ("tab:blue", "o"), "A re-run": ("tab:cyan", "^"), "B": ("tab:orange", "s"), "Mixed": ("tab:green", "D")}
    for _, r in F.iterrows():
        if r.architecture.startswith("A primary") and "Mixed" in r.configuration:
            key = "Mixed"
        elif r.architecture.startswith("A primary"):
            key = "A primary"
        elif r.architecture.startswith("A re-run"):
            key = "A re-run"
        else:
            key = "B"
        c, m = styles[key]
        ax.errorbar(r.theoretical_weight_storage_bytes, r.roc_auc_mean, yerr=r.roc_auc_sd, color=c, marker=m, capsize=3, ms=6)
        ax.annotate(r.configuration.split(" (")[0][:26], (r.theoretical_weight_storage_bytes, r.roc_auc_mean), fontsize=5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical weight storage (bytes, log; analytical, not file size)")
    ax.set_ylabel("Test macro ROC-AUC (mean ± SD, n = 5)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=k) for k, (c, m) in styles.items()]
    ax.legend(handles=handles, fontsize=7)
    ax.set_title("Accuracy–storage view (non-dominated points are interpreted in final_statistical_audit.md)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "06_accuracy_storage_pareto.png", dpi=150); plt.close(fig)


def fig07_hls() -> None:
    d = json.loads((ROOT / "reports" / "numerical_consistency.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fp = d["fp32_pytorch"]["accuracy"]
    modes = {}
    for r in d["hls4ml_csim"]:
        modes.setdefault(r["mode"], []).append((r["ap_fixed_total_bits"], r["accuracy"]))
    colours = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for (mode, pts), colour in zip(modes.items(), colours):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colour, label=f"hls4ml C-sim (SIMULATED): {mode}")
    ax.axhline(fp, color="k", ls="--", lw=1, label=f"PyTorch FP32 accuracy ({fp:.4f}; MEASURED)")
    ax.set_xlabel("ap_fixed total width W (integer bits 6)")
    ax.set_ylabel("Accuracy on 15,000 test jets")
    ax.legend(fontsize=5)
    ax.set_title("hls4ml fixed-point C-simulation vs PyTorch (SIMULATED, not synthesized or physical)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "07_hls_numerical_consistency.png", dpi=150); plt.close(fig)


def fig08_summary() -> None:
    F = pd.read_csv(TAB / "final_results.csv")
    prim = F[F.architecture.str.startswith("A primary") & ~F.configuration.str.startswith("Mixed")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(prim.configuration, prim.roc_auc_mean, yerr=prim.roc_auc_sd, capsize=3, color="tab:blue")
    axes[0].set_ylim(0.85, 0.94)
    axes[0].set_ylabel("Test macro ROC-AUC (mean ± SD, n = 5)")
    axes[0].set_title("Primary architecture, frozen records", fontsize=9)
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=7)
    axes[1].bar(prim.configuration, prim.theoretical_weight_storage_bytes, color="tab:gray")
    axes[1].set_ylabel("Theoretical weight storage (B)")
    axes[1].set_title("Analytical storage (not checkpoint size)", fontsize=9)
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.suptitle("Summary: INT8/INT6 near-lossless, INT4 PTQ degraded and variable, INT4 QAT recovers. Hardware: none measured.", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "08_final_summary.png", dpi=150); plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig01_baseline(); fig02_int4_ptq_vs_qat(); fig03_degradation(); fig04_calibration()
    fig05_architecture(); fig06_pareto(); fig07_hls(); fig08_summary()
    print("wrote 01–08 final figures to reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
