#!/usr/bin/env python
"""Phase-2 results audit over the FROZEN snapshot (experiments/results/frozen_phase2).

Reads only the frozen records/checkpoints. Produces:
  reports/tables/phase2_*.{csv,md}           statistics, degradation, QAT-vs-PTQ, size audit, consistency
  reports/figures/phase2_fig{1..8}_*.png     the eight scientific figures
  reports/phase2_audit_data.json             machine-readable numbers used by the written report
Nothing here re-trains a model or touches the frozen files.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model, macro_roc_curve, predict_proba, test_split_for_record
from hep_quant.quantization.ptq import SUPPORTED_PRECISIONS, fake_quantize_tensor
from hep_quant.visualization.pareto import find_pareto_frontier

FROZEN = Path("experiments/results/frozen_phase2")
FIG, TAB = Path("reports/figures"), Path("reports/tables")
ORDER = {"fp32": 0, "int8": 1, "int6": 2, "int4": 3}
BANNER = "hls4ml_lhc_jets_hlf (100k-jet stratified subset, 5-class) | MLP 16-64-32-32-5 (4,389 params) | mean ± SD over seeds 42/123/2026"
METRICS = ["accuracy", "f1", "roc_auc", "pr_auc"]
COLORS = {"none": "black", "ptq": "tab:blue", "qat": "tab:red"}
MARKERS = {"none": "o", "ptq": "s", "qat": "^"}


def label(method: str) -> str:
    return {"none": "FP32 baseline", "ptq": "PTQ", "qat": "QAT"}[method]


def load_frozen() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text())
    rows = []
    for e in manifest["experiments"]:
        rec = json.loads((FROZEN / f"{e['experiment_id']}.json").read_text())
        t, m, hw = rec["metrics"]["test"], rec["metrics"], rec["hardware"]
        rows.append({
            "experiment_id": e["experiment_id"], "seed": e["seed"], "precision": e["precision"], "method": e["method"],
            **{k: t[k] for k in METRICS + ["recall"]}, "precision_score": t["precision"],
            "latency_ms": m["latency"]["mean_ms_per_sample"], "latency_p95_ms": m["latency"]["p95_ms"], "latency_device": m["latency"]["device"],
            "params": m["model_size"]["total_params"], "est_storage_mb": m["model_size"]["estimated_size_mb"],
            "size_is_estimated": m["model_size"]["is_estimated"],
            "lut": hw.get("lut"), "dsp": hw.get("dsp"), "bram": hw.get("bram"), "ff": hw.get("ff"), "hw_tier": hw.get("measurement_tier"),
            "epochs_run": e["epochs_run"], "train_seconds": e["train_seconds_sum_of_epochs"],
            "ckpt_bytes": (FROZEN / f"{e['experiment_id']}__best_model.pt").stat().st_size,
            "git_commit": e["git_commit_recorded"],
        })
    df = pd.DataFrame(rows)
    df["cfg"] = df["precision"].str.upper() + "_" + df["method"].map({"none": "BASE", "ptq": "PTQ", "qat": "QAT"})
    df["_o"] = df["precision"].map(ORDER)
    return df.sort_values(["seed", "_o", "method"]).drop(columns="_o").reset_index(drop=True), manifest


def configuration_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (p, m), g in df.groupby(["precision", "method"]):
        r = {"precision": p, "method": m, "n_seeds": len(g)}
        for k in METRICS + ["latency_ms"]:
            r |= {f"{k}_mean": g[k].mean(), f"{k}_std": g[k].std(ddof=1), f"{k}_min": g[k].min(), f"{k}_max": g[k].max()}
        r |= {"est_storage_mb": g["est_storage_mb"].iloc[0], "params": g["params"].iloc[0], "latency_device": ",".join(sorted(set(g["latency_device"]))),
              "lut": g["lut"].iloc[0], "hw_tier": g["hw_tier"].iloc[0]}
        rows.append(r)
    out = pd.DataFrame(rows)
    out["_o"] = out["precision"].map(ORDER)
    return out.sort_values(["_o", "method"]).drop(columns="_o").reset_index(drop=True)


def degradation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[df.precision == "fp32"].set_index("seed")
    rows = []
    for _, r in df[df.precision != "fp32"].iterrows():
        row = {"cfg": r["cfg"], "precision": r["precision"], "method": r["method"], "seed": r["seed"]}
        for k in METRICS:
            row[f"d_{k}"] = r[k] - base.loc[r["seed"], k]
            row[f"rel_{k}_pct"] = 100.0 * (r[k] - base.loc[r["seed"], k]) / base.loc[r["seed"], k]
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["precision", "method"]).agg(**{f"{c}_{s}": (c, s) for c in per_seed.columns if c.startswith(("d_", "rel_")) for s in ("mean", "std")}).reset_index()
    agg["_o"] = agg["precision"].map(ORDER)
    return per_seed, agg.sort_values(["_o", "method"]).drop(columns="_o").reset_index(drop=True)


def qat_vs_ptq(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for p in ["int8", "int6", "int4"]:
        for seed in sorted(df.seed.unique()):
            a = df[(df.precision == p) & (df.method == "qat") & (df.seed == seed)].iloc[0]
            b = df[(df.precision == p) & (df.method == "ptq") & (df.seed == seed)].iloc[0]
            rows.append({"precision": p, "seed": seed, **{f"d_{k}": a[k] - b[k] for k in ["roc_auc", "f1", "accuracy"]}})
    per_seed = pd.DataFrame(rows)
    agg = []
    for p, g in per_seed.groupby("precision", sort=False):
        r = {"precision": p}
        for k in ["roc_auc", "f1", "accuracy"]:
            r |= {f"d_{k}_mean": g[f"d_{k}"].mean(), f"d_{k}_std": g[f"d_{k}"].std(ddof=1), f"seeds_qat_better_{k}": int((g[f"d_{k}"] > 0).sum())}
        agg.append(r)
    return per_seed, pd.DataFrame(agg)


def consistency_and_size_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-evaluate every frozen checkpoint on its rebuilt test split; verify weight grids and storage."""
    cons, size = [], []
    cache = {}
    for _, r in df.iterrows():
        rec = json.loads((FROZEN / f"{r['experiment_id']}.json").read_text())
        if r["seed"] not in cache:
            cache[r["seed"]] = test_split_for_record(rec)
        X, y, _ = cache[r["seed"]]
        ckpt_path = FROZEN / f"{r['experiment_id']}__best_model.pt"
        model = load_checkpoint_model(ckpt_path, r["method"], r["precision"])
        probs = predict_proba(model, X)
        acc, auc = accuracy_score(y, probs.argmax(1)), roc_auc_score(y, probs, multi_class="ovr", average="macro")
        cons.append({"experiment_id": r["experiment_id"], "recorded_accuracy": r["accuracy"], "recomputed_accuracy": acc,
                     "abs_diff_accuracy": abs(acc - r["accuracy"]), "recorded_roc_auc": r["roc_auc"], "recomputed_roc_auc": auc,
                     "abs_diff_roc_auc": abs(auc - r["roc_auc"]), "n_test": len(y)})
        # --- storage audit ---
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        dtypes = sorted({str(v.dtype) for v in sd.values()})
        numel = sum(v.numel() for v in sd.values())
        bits = SUPPORTED_PRECISIONS[r["precision"]]
        lin_levels, eff_levels = [], []
        for k, v in sd.items():
            if k.endswith("weight") and v.ndim == 2:
                lin_levels.append(int(torch.unique(v).numel()))
                if r["method"] == "qat":  # latent FP32 weights; QAT quantizes at forward time over the live [min,max]
                    q = fake_quantize_tensor(v, bits, v.min().item(), v.max().item())
                    eff_levels.append(int(torch.unique(q).numel()))
        weight_params = sum(v.numel() for k, v in sd.items() if k.endswith("weight") and v.ndim == 2)
        bias_params = sum(v.numel() for k, v in sd.items() if k.endswith("bias"))
        theo = (weight_params * (bits if bits else 32) / 8) + bias_params * 4
        size.append({"experiment_id": r["experiment_id"], "precision": r["precision"], "method": r["method"], "params": r["params"],
                     "weight_params": weight_params, "bias_params": bias_params, "checkpoint_dtypes": ",".join(dtypes),
                     "checkpoint_tensor_elements": numel, "checkpoint_file_bytes": r["ckpt_bytes"],
                     "theoretical_weight_storage_bytes": theo,
                     "recorded_estimated_size_mb_x1048576": r["est_storage_mb"] * 1048576,
                     "unique_weight_levels_per_layer_as_stored": str(lin_levels),
                     "unique_weight_levels_per_layer_effective_forward": str(eff_levels) if eff_levels else "same as stored",
                     "max_allowed_levels": 2 ** bits if bits else "n/a (fp32)"})
    return pd.DataFrame(cons), pd.DataFrame(size)


def banner(ax_or_fig, extra: str = "") -> None:
    ax_or_fig.suptitle(BANNER + (f"\n{extra}" if extra else ""), fontsize=7)


def _errbar(ax, xs, g, key, **kw):
    ax.errorbar(xs, g[f"{key}_mean"], yerr=g[f"{key}_std"].fillna(0.0), capsize=3, **kw)


def figures(df, stats, deg_agg, csim, split_cache) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    # Fig 1: ROC curves (seed 42), FP32 vs quantized
    seed = 42
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    X, y, _ = split_cache[seed]
    def roc_for(tag_prec, tag_method):
        r = df[(df.seed == seed) & (df.precision == tag_prec) & (df.method == tag_method)].iloc[0]
        model = load_checkpoint_model(FROZEN / f"{r['experiment_id']}__best_model.pt", tag_method, tag_prec)
        return macro_roc_curve(y, predict_proba(model, X))
    fpr, tpr, auc = roc_for("fp32", "none")
    for ax, method in zip(axes, ["ptq", "qat"]):
        ax.plot(fpr, tpr, "k-", lw=2, label=f"FP32 (macro AUC {auc:.4f})")
        for prec, ls in [("int8", "--"), ("int6", "-."), ("int4", ":")]:
            f, t, a = roc_for(prec, method)
            ax.plot(f, t, ls, color=COLORS[method], label=f"{prec.upper()} {method.upper()} ({a:.4f})")
        ax.set_xlim(0, 0.4); ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("False positive rate (one-vs-rest, macro-avg)"); ax.set_title(f"FP32 vs {method.upper()} (seed 42, held-out test)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("True positive rate (macro-avg)")
    banner(fig, "ROC curves, seed 42 only (zoomed to FPR ≤ 0.4); legend: macro one-vs-rest AUC from this curve")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase2_fig1_roc_fp32_vs_quantized.png", dpi=150); plt.close(fig)

    # Fig 2: mean ROC-AUC ± SD by precision
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    base = stats[stats.precision == "fp32"].iloc[0]
    ax.axhspan(base["roc_auc_mean"] - base["roc_auc_std"], base["roc_auc_mean"] + base["roc_auc_std"], color="gray", alpha=0.25, label="FP32 mean ± SD")
    ax.axhline(base["roc_auc_mean"], color="black", lw=1)
    for method in ["ptq", "qat"]:
        g = stats[stats.method == method].sort_values("precision", key=lambda c: c.map(ORDER))
        _errbar(ax, g["precision"].str.upper(), g, "roc_auc", fmt=f"-{MARKERS[method]}", color=COLORS[method], label=f"{method.upper()} (simulated fake-quantization)")
    ax.set_xlabel("Precision"); ax.set_ylabel("Macro one-vs-rest ROC-AUC (test)"); ax.legend(fontsize=8)
    banner(fig, "error bars = sample SD (ddof=1), n=3 seeds"); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase2_fig2_auc_by_precision.png", dpi=150); plt.close(fig)

    # Fig 3: degradation vs precision
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axhline(0, color="black", lw=1)
    for method in ["ptq", "qat"]:
        g = deg_agg[deg_agg.method == method]
        ax.errorbar(g["precision"].str.upper(), g["d_roc_auc_mean"], yerr=g["d_roc_auc_std"].fillna(0.0), fmt=f"-{MARKERS[method]}", color=COLORS[method], capsize=3, label=method.upper())
    ax.set_xlabel("Precision"); ax.set_ylabel("ΔROC-AUC = AUC(quantized) − AUC(FP32, same seed)"); ax.legend(fontsize=8)
    banner(fig, "paired per-seed differences, mean ± SD (n=3)"); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase2_fig3_degradation_vs_precision.png", dpi=150); plt.close(fig)

    # Fig 4: PTQ vs QAT
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    w = 0.35
    precs = ["int8", "int6", "int4"]
    for ax, key, yl in zip(axes, ["roc_auc", "accuracy"], ["Macro ROC-AUC (test)", "Accuracy (test)"]):
        for i, method in enumerate(["ptq", "qat"]):
            g = stats[stats.method == method].set_index("precision").loc[precs]
            ax.bar(np.arange(3) + (i - 0.5) * w, g[f"{key}_mean"], w, yerr=g[f"{key}_std"], capsize=3, color=COLORS[method], label=method.upper())
        ax.axhline(base[f"{key}_mean"], color="black", ls="--", lw=1, label="FP32 mean")
        ax.set_xticks(range(3)); ax.set_xticklabels([p.upper() for p in precs]); ax.set_ylabel(yl)
        lo = min(stats[f"{key}_mean"]) - 0.02; ax.set_ylim(lo, max(stats[f"{key}_mean"]) + 0.01); ax.legend(fontsize=8)
    banner(fig, "PTQ vs QAT; y-axes truncated (not zero-based) to make small differences visible"); fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(FIG / "phase2_fig4_ptq_vs_qat.png", dpi=150); plt.close(fig)

    # Fig 5 + 6 + 8 helpers
    def scatter(ax, xcol, xerr=None, logx=False):
        for method in ["none", "ptq", "qat"]:
            g = stats[stats.method == method]
            ax.errorbar(g[xcol] if xerr is None else g[f"{xcol}_mean"], g["roc_auc_mean"], yerr=g["roc_auc_std"].fillna(0), xerr=None if xerr is None else g[f"{xcol}_std"].fillna(0),
                        fmt=MARKERS[method], color=COLORS[method], capsize=3, label=label(method))
            for _, r in g.iterrows():
                ax.annotate(r["precision"].upper(), (r[xcol] if xerr is None else r[f"{xcol}_mean"], r["roc_auc_mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if logx: ax.set_xscale("log")
        ax.set_ylabel("Macro ROC-AUC (mean ± SD)"); ax.legend(fontsize=8)
    fig, ax = plt.subplots(figsize=(6.6, 4.8)); scatter(ax, "est_storage_mb")
    ax.set_xlabel("Theoretical weight storage (MB; bit-packed equivalent; NOT file size)"); banner(fig, "storage is analytical: weights × bits/8 + FP32 biases")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase2_fig5_storage_vs_auc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.8)); scatter(ax, "latency_ms", xerr=True)
    ax.set_xlabel("Software latency, ms/sample (batch 1, CUDA, simulation overhead — NOT hardware latency)")
    banner(fig, "all points measured on the same device (cuda); fake-quant ops add overhead, so lower precision is SLOWER here")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "phase2_fig6_latency_vs_auc.png", dpi=150); plt.close(fig)

    # Fig 7: hardware (estimated) + C-sim (simulated) in separate panels
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    for method in ["none", "ptq", "qat"]:
        g = stats[stats.method == method]
        ax.errorbar(g["lut"], g["roc_auc_mean"], yerr=g["roc_auc_std"].fillna(0), fmt=MARKERS[method], mfc="none", color=COLORS[method], capsize=3, label=label(method))
        for _, r in g.iterrows():
            ax.annotate(r["precision"].upper(), (r["lut"], r["roc_auc_mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("LUT — SOFTWARE HEURISTIC ESTIMATE (tier: estimated; not synthesized)"); ax.set_ylabel("Macro ROC-AUC (mean ± SD)")
    ax.set_title("(a) ESTIMATED resources (hollow markers)", color="tab:orange"); ax.legend(fontsize=8)
    ax = axes[1]
    if csim is not None:
        rows = csim["rows"]
        ax.plot([r["total_bits"] for r in rows], [r["accuracy"] for r in rows], "D-", color="tab:green", label=f"hls4ml C-sim ap_fixed<W,{rows[0]['int_bits']}> accuracy")
        ax.axhline(csim["fp32_torch_accuracy_on_same_rows"], color="gray", ls="--", label="FP32 PyTorch accuracy (same rows)")
    ax.set_xlabel("Total fixed-point width W (bits)"); ax.set_ylabel("Accuracy (seed-42 model, 15k test jets, single run)")
    ax.set_title("(b) SIMULATED C-simulation (no latency/resources)", color="tab:green"); ax.legend(fontsize=8)
    banner(fig, "panels use different quantities and different measurement tiers; they are not comparable axes. Physical FPGA synthesis NOT performed.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(FIG / "phase2_fig7_hardware_vs_auc.png", dpi=150); plt.close(fig)

    # Fig 8: Pareto
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, xcol, xl, tier in [(axes[0], "est_storage_mb", "Theoretical weight storage (MB)", "analytical"), (axes[1], "lut", "LUT — software estimate (tier: estimated)", "estimated")]:
        s = stats.copy(); s["acc"] = s["roc_auc_mean"]
        mask = find_pareto_frontier(s, maximize=["acc"], minimize=[xcol])
        for method in ["none", "ptq", "qat"]:
            g = s[s.method == method]
            ax.errorbar(g[xcol], g["roc_auc_mean"], yerr=g["roc_auc_std"].fillna(0), fmt=MARKERS[method], color=COLORS[method], capsize=3, label=label(method), alpha=0.9)
            for _, r in g.iterrows():
                ax.annotate(r["precision"].upper(), (r[xcol], r["roc_auc_mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        f = s[mask.values].sort_values(xcol)
        ax.plot(f[xcol], f["roc_auc_mean"], "--", color="crimson", label="Pareto frontier (means)")
        ax.set_xlabel(xl); ax.set_ylabel("Macro ROC-AUC (mean ± SD)"); ax.set_title(f"Pareto on means; cost axis tier: {tier}"); ax.legend(fontsize=8)
        if xcol == "lut": ax.set_xscale("log")
    banner(fig, "frontier computed on seed-means only; points within one SD of each other are not statistically separable")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(FIG / "phase2_fig8_pareto.png", dpi=150); plt.close(fig)
    return {"pareto_storage_frontier": stats.assign(m=find_pareto_frontier(stats.assign(acc=stats["roc_auc_mean"]), ["acc"], ["est_storage_mb"]).values)
            .query("m")[["precision", "method"]].values.tolist()}


def md(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> int:
    df, manifest = load_frozen()
    TAB.mkdir(parents=True, exist_ok=True)
    stats = configuration_stats(df)
    per_seed_deg, deg_agg = degradation(df)
    per_seed_qp, qp_agg = qat_vs_ptq(df)
    cons, size = consistency_and_size_audit(df)
    csim_path = Path("reports/hls_csim_accuracy.json")
    csim = json.loads(csim_path.read_text()) if csim_path.exists() else None
    cache = {s: test_split_for_record(json.loads((FROZEN / f"EXP_S{s}_JET_FP32.json").read_text())) for s in [42]}
    extra = figures(df, stats, deg_agg, csim, cache)

    for name, t in [("stats", stats), ("degradation_per_seed", per_seed_deg), ("degradation_mean", deg_agg), ("qat_vs_ptq_per_seed", per_seed_qp),
                    ("qat_vs_ptq_mean", qp_agg), ("checkpoint_consistency", cons), ("model_size_audit", size), ("all_runs", df)]:
        t.to_csv(TAB / f"phase2_{name}.csv", index=False)
    (TAB / "phase2_tables.md").write_text(
        "## Per-configuration statistics (3 seeds)\n\n" + md(stats[["precision", "method", "n_seeds", "roc_auc_mean", "roc_auc_std", "roc_auc_min", "roc_auc_max", "f1_mean", "f1_std", "accuracy_mean", "accuracy_std", "pr_auc_mean", "pr_auc_std", "latency_ms_mean", "latency_ms_std", "est_storage_mb", "latency_device"]], ".5f")
        + "\n\n## Degradation vs FP32 (paired per seed; mean ± SD over seeds)\n\n" + md(deg_agg[["precision", "method", "d_roc_auc_mean", "d_roc_auc_std", "rel_roc_auc_pct_mean", "d_f1_mean", "d_f1_std", "rel_f1_pct_mean", "d_accuracy_mean", "d_accuracy_std", "rel_accuracy_pct_mean"]], ".5f")
        + "\n\n## Degradation per seed\n\n" + md(per_seed_deg[["cfg", "seed", "d_roc_auc", "rel_roc_auc_pct", "d_f1", "rel_f1_pct", "d_accuracy", "rel_accuracy_pct"]], ".5f")
        + "\n\n## QAT − PTQ (paired per seed; mean ± SD)\n\n" + md(qp_agg, ".5f")
        + "\n\n## QAT − PTQ per seed\n\n" + md(per_seed_qp, ".5f")
        + "\n\n## Checkpoint ↔ record consistency\n\n" + md(cons[["experiment_id", "recorded_accuracy", "recomputed_accuracy", "abs_diff_accuracy", "recorded_roc_auc", "recomputed_roc_auc", "abs_diff_roc_auc"]], ".6f")
        + "\n\n## Model-size audit\n\n" + md(size, ".1f") + "\n", encoding="utf-8")

    summary = {
        "n_experiments": int(len(df)), "n_metrics_finite": int(np.isfinite(df[METRICS]).all(axis=1).sum()),
        "max_abs_diff_accuracy_recompute": float(cons["abs_diff_accuracy"].max()), "max_abs_diff_roc_auc_recompute": float(cons["abs_diff_roc_auc"].max()),
        "latency_devices": sorted(df["latency_device"].unique().tolist()), "hw_tiers": sorted(df["hw_tier"].dropna().unique().tolist()),
        "stats": stats.to_dict("records"), "degradation_mean": deg_agg.to_dict("records"), "qat_vs_ptq_mean": qp_agg.to_dict("records"), **extra,
    }
    Path("reports/phase2_audit_data.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ["n_experiments", "n_metrics_finite", "max_abs_diff_accuracy_recompute", "max_abs_diff_roc_auc_recompute", "latency_devices", "hw_tiers", "pareto_storage_frontier"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
