#!/usr/bin/env python
"""Phase 6 analysis: tables and figures from one Phase 6 run-tag (run1 or run2). Read-only on results.

Storage (theoretical weight storage, bytes): quantized weights at their bit-width, biases kept at 4 bytes.
  Architecture A (4,256 weight, 133 bias):   FP32 17,556 | INT8 4,788 | INT6 3,724 | INT4 2,660
  Architecture B (16,544 weight, 293 bias):  FP32 67,348 | INT8 17,716 | INT6 13,580 | INT4 9,444
These are analytical, not checkpoint sizes (checkpoints are float32; no packed artifact exists).

Outputs (tag-specific, never overwriting earlier phases):
  reports/tables/phase6_<tag>_per_seed.csv
  reports/tables/phase6_<tag>_summary.csv
  reports/tables/phase6_<tag>_paired.csv
  reports/figures/phase6_<tag>_fig*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2026, 7, 99]
PRECS = [("FP32", "FP32", "none"), ("INT8_PTQ", "INT8", "PTQ"), ("INT6_PTQ", "INT6", "PTQ"),
         ("INT4_PTQ", "INT4", "PTQ"), ("INT4_QAT", "INT4", "QAT")]
STORAGE = {"A": {"FP32": 17556, "INT8": 4788, "INT6": 3724, "INT4": 2660},
           "B": {"FP32": 67348, "INT8": 17716, "INT6": 13580, "INT4": 9444}}
PARAMS = {"A": 4389, "B": 16837}
ARCH_LABEL = {"A": "Architecture A [64,32,32]", "B": "Architecture B [128,64,64,32]"}


def load(tag: str) -> pd.DataFrame:
    rd = ROOT / "experiments" / f"results_phase6_{tag}"
    rows = []
    for arch in ["A", "B"]:
        for seed in SEEDS:
            for key, prec, method in PRECS:
                rec = json.loads((rd / f"EXP_P6{arch}_S{seed}_JET_{key}.json").read_text())
                assert rec.get("status") == "completed", (arch, seed, key)
                t = rec["metrics"]["test"]
                rows.append({"arch": arch, "seed": seed, "precision": prec, "method": method,
                             "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "accuracy": t["accuracy"], "f1": t["f1"],
                             "params": PARAMS[arch], "storage_bytes": STORAGE[arch][prec]})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arch, prec, method), g in df.groupby(["arch", "precision", "method"], sort=False):
        fp = df[(df.arch == arch) & (df.precision == "FP32")].set_index("seed")["roc_auc"]
        d = g.set_index("seed")["roc_auc"] - fp
        n = len(g)
        sd = g["roc_auc"].std(ddof=1)
        t = stats.t.ppf(0.975, n - 1)
        out.append({"arch": arch, "precision": prec, "method": method, "n": n,
                    "roc_auc_mean": g["roc_auc"].mean(), "roc_auc_sd": sd,
                    "roc_auc_min": g["roc_auc"].min(), "roc_auc_max": g["roc_auc"].max(),
                    "roc_auc_ci95_low": g["roc_auc"].mean() - t * sd / np.sqrt(n),
                    "roc_auc_ci95_high": g["roc_auc"].mean() + t * sd / np.sqrt(n),
                    "pr_auc_mean": g["pr_auc"].mean(), "pr_auc_sd": g["pr_auc"].std(ddof=1),
                    "accuracy_mean": g["accuracy"].mean(), "f1_mean": g["f1"].mean(),
                    "delta_vs_fp32_mean": d.mean(), "delta_vs_fp32_sd": d.std(ddof=1) if n > 1 else np.nan,
                    "delta_vs_fp32_pct": 100 * d.mean() / fp.mean(),
                    "params": int(g["params"].iloc[0]), "storage_bytes": int(g["storage_bytes"].iloc[0])})
    return pd.DataFrame(out)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Within-architecture paired comparisons (same seeds), and cross-architecture differences of degradation."""
    rows = []
    piv = df.pivot_table(index=["arch", "seed"], columns=["precision", "method"], values="roc_auc")

    def pair(name, a, b):
        d = a - b
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        return {"comparison": name, "n": n, "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                "ci95_low": d.mean() - t * se, "ci95_high": d.mean() + t * se,
                "seeds_positive": int((d > 0).sum()), "seeds_negative": int((d < 0).sum())}

    for arch in ["A", "B"]:
        p = piv.loc[arch]
        rows.append(pair(f"{arch}: INT8 PTQ - FP32", p[("INT8", "PTQ")], p[("FP32", "none")]))
        rows.append(pair(f"{arch}: INT6 PTQ - FP32", p[("INT6", "PTQ")], p[("FP32", "none")]))
        rows.append(pair(f"{arch}: INT4 PTQ - FP32", p[("INT4", "PTQ")], p[("FP32", "none")]))
        rows.append(pair(f"{arch}: INT4 QAT - FP32", p[("INT4", "QAT")], p[("FP32", "none")]))
        rows.append(pair(f"{arch}: INT4 QAT - INT4 PTQ (recovery)", p[("INT4", "QAT")], p[("INT4", "PTQ")]))
    # Cross-architecture: is the degradation the same? paired by seed index (same seed numbers, different models)
    for name, key in [("INT8 PTQ", ("INT8", "PTQ")), ("INT6 PTQ", ("INT6", "PTQ")),
                      ("INT4 PTQ", ("INT4", "PTQ")), ("INT4 QAT", ("INT4", "QAT"))]:
        dA = piv.loc["A"][key] - piv.loc["A"][("FP32", "none")]
        dB = piv.loc["B"][key] - piv.loc["B"][("FP32", "none")]
        rows.append(pair(f"degradation B minus A: {name}", dB, dA))
    return pd.DataFrame(rows)


def figures(df: pd.DataFrame, summ: pd.DataFrame, tag: str) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = ["FP32", "INT8", "INT6", "INT4"]
    for k, arch in enumerate(["A", "B"], start=1):
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        g = df[df.arch == arch]
        labels = ["FP32", "INT8 PTQ", "INT6 PTQ", "INT4 PTQ", "INT4 QAT"]
        keys = [("FP32", "none"), ("INT8", "PTQ"), ("INT6", "PTQ"), ("INT4", "PTQ"), ("INT4", "QAT")]
        for i, (prec, method) in enumerate(keys):
            v = g[(g.precision == prec) & (g.method == method)]["roc_auc"].values
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v, s=22, alpha=0.7)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="_k", capsize=6, ms=16)
        ax.set_xticks(range(5), labels)
        ax.set_ylabel("Test macro ROC-AUC")
        ax.set_title(f"{ARCH_LABEL[arch]}: ROC-AUC by precision (5 seeds, mean ± SD)", fontsize=9)
        fig.suptitle(f"hls4ml_lhc_jets_hlf | CPU | Phase 6 {tag}", fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"phase6_{tag}_fig{k}_roc_by_precision_arch{arch}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arch, c in [("A", "tab:blue"), ("B", "tab:orange")]:
        s = summ[(summ.arch == arch) & (summ.method != "none") | ((summ.arch == arch) & (summ.precision == "FP32"))]
        s = s.set_index(["precision", "method"])
        xs = ["INT8", "INT6", "INT4"]
        for method, ls in [("PTQ", "-"), ("QAT", "--")]:
            ys, es = [], []
            for x in xs:
                if (x, method) in s.index:
                    ys.append(-s.loc[(x, method), "delta_vs_fp32_mean"])
                    es.append(s.loc[(x, method), "delta_vs_fp32_sd"])
            if ys:
                ax.errorbar(range(len(ys)), ys, yerr=es, fmt="o" + ls, color=c, capsize=4, label=f"{ARCH_LABEL[arch]} {method}")
    ax.set_xticks(range(3), ["INT8", "INT6", "INT4"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ROC-AUC loss vs own FP32 (mean ± SD)")
    ax.legend(fontsize=7)
    ax.set_title("Quantization degradation relative to each architecture's own FP32", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"phase6_{tag}_fig3_degradation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for arch, c in [("A", "tab:blue"), ("B", "tab:orange")]:
        for i, method in enumerate(["PTQ", "QAT"]):
            v = df[(df.arch == arch) & (df.precision == "INT4") & (df.method == method)]["roc_auc"].values
            ax.scatter(np.full(len(v), i) + (0.1 if arch == "B" else -0.1) + np.random.default_rng(1).uniform(-0.03, 0.03, len(v)),
                       v, color=c, s=22, alpha=0.7, label=ARCH_LABEL[arch] if i == 0 else None)
            ax.errorbar(i + (0.1 if arch == "B" else -0.1), v.mean(), yerr=v.std(ddof=1), fmt="_k", capsize=6, ms=14)
    ax.set_xticks([0, 1], ["INT4 PTQ", "INT4 QAT"])
    ax.set_ylabel("Test macro ROC-AUC")
    ax.legend(fontsize=7)
    ax.set_title("INT4: PTQ vs QAT, both architectures (5 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"phase6_{tag}_fig4_int4_ptq_vs_qat.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for arch, c, mk in [("A", "tab:blue", "o"), ("B", "tab:orange", "s")]:
        s = summ[summ.arch == arch]
        ax.scatter(s.storage_bytes, s.roc_auc_mean, color=c, marker=mk, s=40, label=ARCH_LABEL[arch])
        for _, r in s.iterrows():
            ax.annotate(f"{r.precision} {r.method if r.method != 'none' else ''}".strip(), (r.storage_bytes, r.roc_auc_mean),
                        fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical weight storage (bytes, log scale)")
    ax.set_ylabel("Test macro ROC-AUC (mean, 5 seeds)")
    ax.legend(fontsize=7)
    ax.set_title("Storage vs ROC-AUC, both architectures (Pareto view; analytical storage)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"phase6_{tag}_fig5_storage_vs_auc.png", dpi=150)
    plt.close(fig)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
    df = load(tag)
    summ = summarise(df)
    pr = paired(df)
    TAB = ROOT / "reports" / "tables"
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"phase6_{tag}_per_seed.csv", index=False)
    summ.to_csv(TAB / f"phase6_{tag}_summary.csv", index=False)
    pr.to_csv(TAB / f"phase6_{tag}_paired.csv", index=False)
    figures(df, summ, tag)
    print(summ[["arch", "precision", "method", "roc_auc_mean", "roc_auc_sd", "delta_vs_fp32_mean", "storage_bytes"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
