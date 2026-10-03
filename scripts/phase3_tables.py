#!/usr/bin/env python
"""Build the Phase-3 machine-readable tables and figures from experiment artifacts (nothing is typed by hand).

Seeds 42/123/2026 are read from the FROZEN snapshot (the primary n=3 record); seeds 7/99 and the FP32-fine-tune controls
come from experiments/results (new IDs). All pooled seeds must share one device; the script refuses to pool otherwise.
Statistics: sample SD (ddof=1); 95% t-interval; paired effect size d_z = mean(Δ)/SD(Δ). With n<=5 these are DESCRIPTIVE aids,
not significance tests; no p-values are produced.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

FROZEN = Path("experiments/results/frozen_phase2")
LIVE = Path("experiments/results")
TAB, FIG = Path("reports/tables"), Path("reports/figures")
FROZEN_SEEDS, EXTRA_SEEDS = [42, 123, 2026], [7, 99]
TAGS = ["FP32", "INT8_PTQ", "INT6_PTQ", "INT4_PTQ", "INT8_QAT", "INT6_QAT", "INT4_QAT"]
ORDER = {"fp32": 0, "int8": 1, "int6": 2, "int4": 3}
M = ["roc_auc", "f1", "accuracy", "pr_auc"]
BANNER = "hls4ml_lhc_jets_hlf (100k-jet subsets) | MLP 16-64-32-32-5 | simulated fake-quantization"


def load(path: Path, frozen: bool) -> dict | None:
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r.get("status") != "completed":
        return None
    t, hw, mi = r["metrics"]["test"], r["hardware"], r["model_info"]
    return {"experiment_id": r["experiment_id"], "seed": r["config"]["seed"], "precision": mi["precision"], "method": mi["method"],
            "frozen": frozen, **{k: t[k] for k in M}, "precision_score": t["precision"], "recall": t["recall"], "params": mi["n_parameters"],
            "theoretical_weight_storage_mb": r["metrics"]["model_size"]["estimated_size_mb"],
            "latency_ms_indicative": r["metrics"]["latency"]["mean_ms_per_sample"], "latency_device": r["metrics"]["latency"]["device"],
            "hw_tier": hw.get("measurement_tier"), "lut": hw.get("lut"), "dsp": hw.get("dsp"), "bram": hw.get("bram"), "ff": hw.get("ff"),
            "hw_source": hw.get("source", "SoftwareEstimateBackend" if hw.get("backend") == "software_estimate" else "N/A"),
            "hw_tool": hw.get("tool", "analytical heuristic (hep_quant)" if hw.get("backend") == "software_estimate" else "N/A"),
            "hw_tool_version": hw.get("tool_version", "N/A (field added after this run)"),
            "hw_configuration": json.dumps(hw.get("configuration", {"note": "field added after this run; heuristic parameters in docs/hardware.md"})),
            "git_commit": r["reproducibility"]["system"]["git_commit"], "git_dirty": r["reproducibility"]["system"].get("git_dirty", "N/A"),
            "environment_hash": r["reproducibility"]["system"].get("environment_hash", "N/A")}


def tci(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    half = float(st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return float(np.mean(x)), sd, half


def main() -> int:
    rows = []
    for s in FROZEN_SEEDS:
        for t in TAGS:
            rows.append(load(FROZEN / f"EXP_S{s}_JET_{t}.json", True))
    for s in EXTRA_SEEDS:
        for t in TAGS:
            rows.append(load(LIVE / f"EXP_S{s}_JET_{t}.json", False))
    for s in FROZEN_SEEDS + EXTRA_SEEDS:
        rows.append(load(LIVE / f"EXP_C{s}_JET_FP32_FT.json", False))
    df = pd.DataFrame([r for r in rows if r is not None])
    df["config"] = df["precision"].str.upper() + "_" + df["method"].map({"none": "BASE", "ptq": "PTQ", "qat": "QAT", "fp32_finetune": "FT_CONTROL"})
    devices = sorted(df["latency_device"].unique())
    if len(devices) != 1:
        raise SystemExit(f"refusing to pool results from several devices: {devices}")
    df.to_csv(TAB / "experiment_summary.csv", index=False)

    base = df[df.config == "FP32_BASE"].set_index("seed")
    core = df[df.config != "FP32_FT_CONTROL"]
    seed_sets = {"n3_frozen": FROZEN_SEEDS, "n5_extended": FROZEN_SEEDS + EXTRA_SEEDS}
    out_rows, cmp_rows = [], []
    for label, seeds in seed_sets.items():
        sub = core[core.seed.isin(seeds)]
        for cfg, g in sub.groupby("config", sort=False):
            if len(g) != len(seeds):
                continue
            r = {"seed_set": label, "config": cfg, "n_seeds": len(g), "seeds": ",".join(map(str, sorted(g.seed)))}
            for m in M:
                mean, sd, half = tci(g[m].to_numpy())
                r |= {f"{m}_mean": mean, f"{m}_sd": sd, f"{m}_ci95_halfwidth": half, f"{m}_min": g[m].min(), f"{m}_max": g[m].max()}
            out_rows.append(r)
            if cfg != "FP32_BASE":
                d = np.array([g.set_index("seed").loc[s, "roc_auc"] - base.loc[s, "roc_auc"] for s in seeds])
                dacc = np.array([g.set_index("seed").loc[s, "accuracy"] - base.loc[s, "accuracy"] for s in seeds])
                mean, sd, half = tci(d)
                cmp_rows.append({"seed_set": label, "config": cfg, "n_seeds": len(seeds), "d_roc_auc_mean": mean, "d_roc_auc_sd": sd, "d_roc_auc_ci95_halfwidth": half,
                                 "rel_roc_auc_pct_mean": float(np.mean(d / np.array([base.loc[s, "roc_auc"] for s in seeds])) * 100),
                                 "d_accuracy_mean": float(dacc.mean()), "d_accuracy_sd": float(dacc.std(ddof=1)),
                                 "paired_effect_size_dz": mean / sd if sd and sd > 0 else float("nan"),
                                 "theoretical_weight_storage_mb": g["theoretical_weight_storage_mb"].iloc[0]})
    pd.DataFrame(out_rows).to_csv(TAB / "seed_statistics.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(TAB / "quantization_comparison.csv", index=False)

    # QAT vs PTQ vs compute-matched control (n=5)
    ctrl = df[df.config == "FP32_FT_CONTROL"].set_index("seed")
    q_rows = []
    seeds = FROZEN_SEEDS + EXTRA_SEEDS
    for label, ss in seed_sets.items():
        for p in ["int8", "int6", "int4"]:
            g = {m: core[(core.precision == p) & (core.method == m) & core.seed.isin(ss)].set_index("seed") for m in ["ptq", "qat"]}
            if any(len(v) != len(ss) for v in g.values()):
                continue
            for metric in ["roc_auc", "f1", "accuracy"]:
                d = np.array([g["qat"].loc[s, metric] - g["ptq"].loc[s, metric] for s in ss])
                mean, sd, half = tci(d)
                q_rows.append({"seed_set": label, "comparison": f"QAT - PTQ at {p.upper()}", "metric": metric, "n_seeds": len(ss), "mean_diff": mean, "sd": sd,
                               "ci95_halfwidth": half, "paired_effect_size_dz": mean / sd if sd and sd > 0 else float("nan"), "seeds_qat_higher": int((d > 0).sum())})
            if all(s in ctrl.index for s in ss):
                for metric in ["roc_auc", "f1", "accuracy"]:
                    d = np.array([g["qat"].loc[s, metric] - ctrl.loc[s, metric] for s in ss])
                    mean, sd, half = tci(d)
                    q_rows.append({"seed_set": label, "comparison": f"QAT {p.upper()} - FP32 compute-matched control", "metric": metric, "n_seeds": len(ss), "mean_diff": mean, "sd": sd,
                                   "ci95_halfwidth": half, "paired_effect_size_dz": mean / sd if sd and sd > 0 else float("nan"), "seeds_qat_higher": int((d > 0).sum())})
        if all(s in ctrl.index for s in ss):
            for metric in ["roc_auc", "f1", "accuracy"]:
                d = np.array([ctrl.loc[s, metric] - base.loc[s, metric] for s in ss])
                mean, sd, half = tci(d)
                q_rows.append({"seed_set": label, "comparison": "FP32 fine-tune control - FP32 baseline (effect of extra epochs alone)", "metric": metric, "n_seeds": len(ss),
                               "mean_diff": mean, "sd": sd, "ci95_halfwidth": half, "paired_effect_size_dz": mean / sd if sd and sd > 0 else float("nan"), "seeds_qat_higher": int((d > 0).sum())})
    pd.DataFrame(q_rows).to_csv(TAB / "qat_vs_ptq.csv", index=False)

    # hardware summary (estimates) + csim rows (simulated)
    hw = core[core.seed == 42][["experiment_id", "precision", "method", "lut", "dsp", "bram", "ff", "hw_tier", "hw_source", "hw_tool", "hw_tool_version", "hw_configuration"]].copy()
    hw["quantity"] = "LUT/DSP/BRAM/FF (software heuristic)"
    csim = json.loads(Path("reports/hls_csim_accuracy.json").read_text())
    csim_rows = [{"experiment_id": csim["experiment"], "precision": f"ap_fixed{r['ap_fixed']}", "method": "hls4ml C-simulation (FP32 weights)", "lut": None, "dsp": None, "bram": None, "ff": None,
                  "hw_tier": "simulated", "hw_source": "hls4ml_csim", "hw_tool": "hls4ml", "hw_tool_version": "1.3.0",
                  "hw_configuration": json.dumps({"backend": "Vivado", "io_type": "io_parallel", "rounding": "AP_TRN(default)", "overflow": "AP_WRAP(default)", "accum": "same as default type", "csim_accuracy": r["accuracy"], "agreement_with_fp32": r["agreement_with_fp32_predictions"]}),
                  "quantity": "accuracy under fixed-point arithmetic (no latency/resources exist from C-simulation)"} for r in csim["rows"]]
    pd.concat([hw, pd.DataFrame(csim_rows)]).to_csv(TAB / "hardware_summary.csv", index=False)

    # ---- figures (n=5): strip + mean±CI; QAT/PTQ/control
    st5 = pd.DataFrame(out_rows)
    st5 = st5[st5.seed_set == "n5_extended"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs, labels = [], []
    order = [("FP32_BASE", "FP32"), ("INT8_PTQ", "INT8 PTQ"), ("INT8_QAT", "INT8 QAT"), ("INT6_PTQ", "INT6 PTQ"), ("INT6_QAT", "INT6 QAT"), ("INT4_PTQ", "INT4 PTQ"), ("INT4_QAT", "INT4 QAT")]
    for i, (cfg, lab) in enumerate(order):
        g = core[(core.config == cfg) & core.seed.isin(seeds)]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g["roc_auc"], s=26, color="tab:red" if "QAT" in cfg else ("tab:blue" if "PTQ" in cfg else "black"), alpha=0.8, zorder=3)
        r = st5[st5.config == cfg].iloc[0]
        ax.errorbar(i + 0.3, r["roc_auc_mean"], yerr=r["roc_auc_ci95_halfwidth"], fmt="D", color="gray", capsize=3, label="mean ± 95% t-CI (n=5)" if i == 0 else None)
        xs.append(i); labels.append(lab)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30); ax.set_ylabel("Macro one-vs-rest ROC-AUC (test)")
    ax.legend(fontsize=8); fig.suptitle(BANNER + "\nevery point = one seed (42/123/2026/7/99); grey diamond = mean ± 95% t-interval, n=5 (descriptive)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase3_auc_by_config_all_seeds.png", dpi=150); plt.close(fig)

    qq = pd.DataFrame(q_rows)
    qq5 = qq[(qq.seed_set == "n5_extended") & (qq.metric == "roc_auc")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labs = [c for c in qq5.comparison]
    ax.barh(range(len(qq5)), qq5["mean_diff"], xerr=qq5["ci95_halfwidth"], capsize=3, color="tab:gray")
    ax.set_yticks(range(len(qq5))); ax.set_yticklabels(labs, fontsize=7); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Paired ΔROC-AUC, mean ± 95% t-CI (n=5)"); fig.suptitle(BANNER + "\nQAT vs PTQ and vs the compute-matched FP32 fine-tune control", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase3_qat_vs_ptq_vs_control.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 60)
    print(st5[["config", "n_seeds", "roc_auc_mean", "roc_auc_sd", "roc_auc_ci95_halfwidth", "roc_auc_min", "roc_auc_max"]].round(4).to_string(index=False))
    print(pd.DataFrame(cmp_rows).query("seed_set=='n5_extended'")[["config", "d_roc_auc_mean", "d_roc_auc_sd", "d_roc_auc_ci95_halfwidth", "rel_roc_auc_pct_mean", "paired_effect_size_dz"]].round(4).to_string(index=False))
    print(qq[(qq.seed_set == "n5_extended") & (qq.metric == "roc_auc")][["comparison", "mean_diff", "sd", "ci95_halfwidth", "seeds_qat_higher"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
