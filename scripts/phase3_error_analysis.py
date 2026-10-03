#!/usr/bin/env python
"""Per-class robustness, probability calibration and FP32-vs-quantized error analysis on the FROZEN checkpoints.

Outputs: reports/tables/phase3_per_class*.csv, phase3_calibration*.csv, phase3_transitions.csv,
phase3_flip_by_feature.csv, reports/figures/phase3_per_class_int4.png, reports/phase3_error_analysis.json
No explanations are inferred from the numbers; the report states observations only.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model, predict_proba, test_split_for_record
from hep_quant.data.hls4ml_jets import EXPECTED_CLASSES, EXPECTED_FEATURES

FROZEN = Path("experiments/results/frozen_phase2")
TAB, FIG = Path("reports/tables"), Path("reports/figures")
SEEDS = [42, 123, 2026]
TAGS = {"FP32": ("none", "fp32"), "INT8_PTQ": ("ptq", "int8"), "INT6_PTQ": ("ptq", "int6"), "INT4_PTQ": ("ptq", "int4"),
        "INT8_QAT": ("qat", "int8"), "INT6_QAT": ("qat", "int6"), "INT4_QAT": ("qat", "int4")}


def ece_top_label(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1).mean())


def main() -> int:
    P, Y, X = {}, {}, {}
    for s in SEEDS:
        rec0 = json.loads((FROZEN / f"EXP_S{s}_JET_FP32.json").read_text())
        X[s], Y[s], _ = test_split_for_record(rec0)
        for tag, (method, prec) in TAGS.items():
            m = load_checkpoint_model(FROZEN / f"EXP_S{s}_JET_{tag}__best_model.pt", method, prec)
            P[(s, tag)] = predict_proba(m, X[s])

    pc, cal = [], []
    for (s, tag), p in P.items():
        y = Y[s]
        pr, rc, f1, _ = precision_recall_fscore_support(y, p.argmax(1), labels=range(5), zero_division=0)
        for c, name in enumerate(EXPECTED_CLASSES):
            pc.append({"seed": s, "config": tag, "class": name, "precision": pr[c], "recall": rc[c], "f1": f1[c],
                       "roc_auc_ovr": roc_auc_score((y == c).astype(int), p[:, c])})
        cal.append({"seed": s, "config": tag, "brier": brier(p, y), "ece_top_label_15bins": ece_top_label(p, y),
                    "mean_confidence": float(p.max(1).mean()), "accuracy": float((p.argmax(1) == y).mean())})
    pc, cal = pd.DataFrame(pc), pd.DataFrame(cal)
    pc.to_csv(TAB / "phase3_per_class.csv", index=False); cal.to_csv(TAB / "phase3_calibration.csv", index=False)

    base_pc = pc[pc.config == "FP32"].set_index(["seed", "class"])
    d = pc[pc.config != "FP32"].copy()
    for k in ["recall", "precision", "roc_auc_ovr"]:
        d[f"d_{k}"] = [r[k] - base_pc.loc[(r.seed, r["class"]), k] for _, r in d.iterrows()]
    dd = d.groupby(["config", "class"]).agg(**{f"{k}_{a}": (k, a) for k in ["d_recall", "d_precision", "d_roc_auc_ovr"] for a in ("mean", "std")}).reset_index()
    dd.to_csv(TAB / "phase3_per_class_delta.csv", index=False)

    base_cal = cal[cal.config == "FP32"].set_index("seed")
    cal["d_brier"] = [r.brier - base_cal.loc[r.seed, "brier"] for _, r in cal.iterrows()]
    cal["d_ece"] = [r.ece_top_label_15bins - base_cal.loc[r.seed, "ece_top_label_15bins"] for _, r in cal.iterrows()]
    cal_mean = cal.groupby("config")[["brier", "ece_top_label_15bins", "mean_confidence", "d_brier", "d_ece"]].agg(["mean", "std"])
    cal_mean.columns = ["_".join(c) for c in cal_mean.columns]
    cal_mean.reset_index().to_csv(TAB / "phase3_calibration_mean.csv", index=False)

    # 4-way agreement with FP32 and where flips happen
    trans, flips = [], []
    for s in SEEDS:
        y, pf = Y[s], P[(s, "FP32")].argmax(1)
        for tag in ["INT8_PTQ", "INT6_PTQ", "INT4_PTQ", "INT8_QAT", "INT6_QAT", "INT4_QAT"]:
            pq = P[(s, tag)].argmax(1)
            cf, cq = pf == y, pq == y
            trans.append({"seed": s, "config": tag, "n": len(y),
                          "fp32_correct__quant_correct": int((cf & cq).sum()), "fp32_correct__quant_wrong": int((cf & ~cq).sum()),
                          "fp32_wrong__quant_correct": int((~cf & cq).sum()), "fp32_wrong__quant_wrong": int((~cf & ~cq).sum()),
                          "prediction_agreement": float((pf == pq).mean())})
            if tag in ("INT4_PTQ", "INT4_QAT"):
                flip = cf & ~cq
                for j, f in enumerate(EXPECTED_FEATURES):
                    q = np.quantile(X[s][:, j], [0.25, 0.5, 0.75])
                    b = np.digitize(X[s][:, j], q)
                    rates = [float(flip[(b == k) & cf].mean()) if ((b == k) & cf).any() else float("nan") for k in range(4)]
                    flips.append({"seed": s, "config": tag, "feature": f, **{f"flip_rate_q{k + 1}": r for k, r in enumerate(rates)}})
    trans = pd.DataFrame(trans); trans.to_csv(TAB / "phase3_transitions.csv", index=False)
    fl = pd.DataFrame(flips)
    fl_mean = fl.groupby(["config", "feature"])[[f"flip_rate_q{k}" for k in range(1, 5)]].mean().reset_index()
    fl_mean["max_over_min"] = fl_mean[[f"flip_rate_q{k}" for k in range(1, 5)]].max(axis=1) / fl_mean[[f"flip_rate_q{k}" for k in range(1, 5)]].min(axis=1)
    fl_mean.to_csv(TAB / "phase3_flip_by_feature.csv", index=False)

    # per-class figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, k, yl in zip(axes, ["d_roc_auc_ovr", "d_recall"], ["ΔAUC (one-vs-rest) vs FP32", "Δrecall vs FP32"]):
        for i, (tag, c) in enumerate([("INT4_PTQ", "tab:blue"), ("INT4_QAT", "tab:red")]):
            g = dd[dd.config == tag].set_index("class").loc[EXPECTED_CLASSES]
            ax.bar(np.arange(5) + (i - 0.5) * 0.38, g[f"{k}_mean"], 0.38, yerr=g[f"{k}_std"], capsize=3, color=c, label=tag.replace("_", " "))
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(["g (gluon)", "q (quark)", "t (top)", "w (W)", "z (Z)"])
        ax.set_ylabel(yl); ax.legend(fontsize=8)
    fig.suptitle("hls4ml_lhc_jets_hlf | MLP 16-64-32-32-5 | INT4 simulated fake-quantization | mean ± SD over seeds 42/123/2026 (n=3)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "phase3_per_class_int4.png", dpi=150); plt.close(fig)

    summary = {"transitions_mean": trans.groupby("config")[["fp32_correct__quant_correct", "fp32_correct__quant_wrong", "fp32_wrong__quant_correct", "fp32_wrong__quant_wrong", "prediction_agreement"]].mean().round(4).to_dict("index"),
               "calibration_mean": cal_mean.round(5).to_dict("index")}
    Path("reports/phase3_error_analysis.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(dd[dd.config.isin(["INT4_PTQ", "INT4_QAT"])].round(4).to_string())
    print(cal_mean[["brier_mean", "ece_top_label_15bins_mean", "d_ece_mean", "d_ece_std", "d_brier_mean"]].round(4))
    print(pd.DataFrame(summary["transitions_mean"]).T.round(4))
    print(fl_mean[fl_mean.config == "INT4_PTQ"].sort_values("max_over_min", ascending=False).head(5).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
