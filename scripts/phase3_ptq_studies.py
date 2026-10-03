#!/usr/bin/env python
"""Phase-3 PTQ sensitivity studies on the FROZEN FP32 checkpoints (no training, nothing in the frozen dir is modified).

S0  sanity: uniform INT-N variant rebuilt from the frozen FP32 checkpoint vs the recorded PTQ experiment
S1  calibration-set size / draw sensitivity
S2  weights-only vs activations-only vs both
S3  per-unit sensitivity at INT4: each unit alone, and everything-except-one-unit
S4  (optional --mixed) mixed-precision strategies derived from S3
Outputs: reports/tables/phase3_ptq_*.csv, reports/figures/phase3_ptq_*.png, reports/phase3_ptq_studies.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model, predict_proba
from hep_quant.analysis.ptq_studies import build_ptq_variant, theoretical_weight_storage_bytes, uniform_units, unit_names
from hep_quant.data.hls4ml_jets import load_hls4ml_jets
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset

FROZEN = Path("experiments/results/frozen_phase2")
SEEDS = [42, 123, 2026]
TAB, FIG = Path("reports/tables"), Path("reports/figures")
BANNER = "hls4ml_lhc_jets_hlf | MLP 16-64-32-32-5 | PTQ applied to the frozen FP32 checkpoints of seeds 42/123/2026 | simulated fake-quantization"


def metrics(model, X, y) -> dict:
    p = predict_proba(model, X)
    return {"accuracy": accuracy_score(y, p.argmax(1)), "f1": f1_score(y, p.argmax(1), average="macro"),
            "roc_auc": roc_auc_score(y, p, multi_class="ovr", average="macro")}


def load_seed(seed: int):
    rec = json.loads((FROZEN / f"EXP_S{seed}_JET_FP32.json").read_text())
    cfg, d = rec["config"], rec["config"]["dataset"]
    ds = load_hls4ml_jets(subset_size=d["subset_size"], seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    model = load_checkpoint_model(FROZEN / f"EXP_S{seed}_JET_FP32__best_model.pt", "none", "fp32")
    return model, pre.transform(sp.X_train), pre.transform(sp.X_test), sp.y_test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", default=None, help="JSON list of strategies, e.g. '{\"name\":\"x\",\"weights\":{\"0\":8,\"1\":4},\"acts\":{}}' (S4 only)")
    args = ap.parse_args()
    data = {s: load_seed(s) for s in SEEDS}
    TAB.mkdir(parents=True, exist_ok=True)
    out = {}

    if args.mixed is None:
        # ---- S0 ----
        rows = []
        for s in SEEDS:
            model, Xtr, Xte, yte = data[s]
            base = metrics(model, Xte, yte)
            rows.append({"seed": s, "bits": 32, **base})
            for bits in [8, 6, 4]:
                wb, ab = uniform_units(model, bits)
                m = metrics(build_ptq_variant(model, Xtr[:20480], wb, ab), Xte, yte)
                rec = json.loads((FROZEN / f"EXP_S{s}_JET_INT{bits}_PTQ.json").read_text())["metrics"]["test"]
                rows.append({"seed": s, "bits": bits, **m, "recorded_accuracy": rec["accuracy"], "abs_diff_vs_record": abs(m["accuracy"] - rec["accuracy"])})
        s0 = pd.DataFrame(rows); s0.to_csv(TAB / "phase3_ptq_s0_sanity.csv", index=False)
        out["s0_max_abs_diff_vs_record"] = float(s0["abs_diff_vs_record"].max())

        # ---- S1 calibration ----
        rows = []
        sizes = [32, 128, 512, 2048, 8192, 20480, 70000]
        for s in SEEDS:
            model, Xtr, Xte, yte = data[s]
            n_train = len(Xtr)
            for bits in [8, 6, 4]:
                wb, ab = uniform_units(model, bits)
                for n in sizes:
                    n = min(n, n_train)
                    draws = [None] if n >= n_train else range(5)
                    for dr in draws:
                        idx = np.arange(n) if dr is None else np.random.default_rng(1000 * s + dr).choice(n_train, n, replace=False)
                        m = metrics(build_ptq_variant(model, Xtr[idx], wb, ab), Xte, yte)
                        rows.append({"seed": s, "bits": bits, "calib_size": n, "draw": -1 if dr is None else dr, **m})
        s1 = pd.DataFrame(rows); s1.to_csv(TAB / "phase3_ptq_s1_calibration.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
        for ax, bits in zip(axes, [8, 6, 4]):
            for s, c in zip(SEEDS, ["tab:blue", "tab:orange", "tab:green"]):
                g = s1[(s1.seed == s) & (s1.bits == bits)].groupby("calib_size")["roc_auc"].agg(["median", "min", "max"]).reset_index()
                ax.plot(g["calib_size"], g["median"], "o-", color=c, label=f"seed {s} (median of draws)")
                ax.fill_between(g["calib_size"], g["min"], g["max"], color=c, alpha=0.2)
                base = s0[(s0.seed == s) & (s0.bits == 32)]["roc_auc"].iloc[0]
                ax.axhline(base, color=c, ls=":", lw=1)
            ax.set_xscale("log"); ax.set_xlabel("Calibration set size (training jets)"); ax.set_title(f"INT{bits} PTQ"); ax.legend(fontsize=7)
        axes[0].set_ylabel("Macro ROC-AUC (test)")
        fig.suptitle(BANNER + "\nband = min–max over 5 random calibration draws; dotted = that seed's FP32", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "phase3_ptq_calibration_sensitivity.png", dpi=150); plt.close(fig)

        # ---- S2 components ----
        rows = []
        for s in SEEDS:
            model, Xtr, Xte, yte = data[s]
            base = metrics(model, Xte, yte)["roc_auc"]
            for bits in [8, 6, 4]:
                wb, ab = uniform_units(model, bits)
                for name, w, a in [("weights only", wb, {}), ("activations only", {}, ab), ("both", wb, ab)]:
                    m = metrics(build_ptq_variant(model, Xtr[:20480], w, a), Xte, yte)
                    rows.append({"seed": s, "bits": bits, "variant": name, **m, "d_roc_auc": m["roc_auc"] - base})
        s2 = pd.DataFrame(rows); s2.to_csv(TAB / "phase3_ptq_s2_components.csv", index=False)

        # ---- S3 per-unit at INT4 ----
        rows = []
        for s in SEEDS:
            model, Xtr, Xte, yte = data[s]
            lin, act = unit_names(model)
            base = metrics(model, Xte, yte)["roc_auc"]
            units = [("L", i) for i in range(len(lin))] + [("A", i) for i in range(len(act))]
            allw, alla = uniform_units(model, 4)
            for kind, i in units:
                w, a = ({i: 4}, {}) if kind == "L" else ({}, {i: 4})
                m = metrics(build_ptq_variant(model, Xtr[:20480], w, a), Xte, yte)
                rows.append({"seed": s, "mode": "only this unit at INT4", "unit": f"{kind}{i}", **m, "d_roc_auc": m["roc_auc"] - base})
                w2, a2 = dict(allw), dict(alla)
                (w2 if kind == "L" else a2)[i] = None
                m = metrics(build_ptq_variant(model, Xtr[:20480], w2, a2), Xte, yte)
                rows.append({"seed": s, "mode": "all INT4 except this unit (FP32)", "unit": f"{kind}{i}", **m, "d_roc_auc": m["roc_auc"] - base})
        s3 = pd.DataFrame(rows); s3.to_csv(TAB / "phase3_ptq_s3_units.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for ax, mode in zip(axes, ["only this unit at INT4", "all INT4 except this unit (FP32)"]):
            g = s3[s3["mode"] == mode]
            units = sorted(g["unit"].unique())
            for k, s in enumerate(SEEDS):
                gg = g[g.seed == s].set_index("unit").loc[units]
                ax.bar(np.arange(len(units)) + (k - 1) * 0.27, gg["d_roc_auc"], 0.27, label=f"seed {s}")
            ax.set_xticks(range(len(units))); ax.set_xticklabels(units); ax.axhline(0, color="k", lw=0.8)
            ax.set_title(mode); ax.set_xlabel("unit (L = Linear weights, A = hidden activation)"); ax.legend(fontsize=7)
        axes[0].set_ylabel("ΔROC-AUC vs FP32 (same seed)")
        fig.suptitle(BANNER, fontsize=8); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "phase3_ptq_unit_sensitivity_int4.png", dpi=150); plt.close(fig)

        out["s1_int4_auc_by_calib_size_median"] = s1[s1.bits == 4].groupby("calib_size")["roc_auc"].median().to_dict()
    else:
        strategies = json.loads(args.mixed)
        rows = []
        for s in SEEDS:
            model, Xtr, Xte, yte = data[s]
            base = metrics(model, Xte, yte)["roc_auc"]
            for st in strategies:
                w = {int(k): v for k, v in st["weights"].items()}
                a = {int(k): v for k, v in st["acts"].items()}
                m = metrics(build_ptq_variant(model, Xtr[:20480], w, a), Xte, yte)
                rows.append({"seed": s, "strategy": st["name"], "weights": json.dumps(st["weights"]), "acts": json.dumps(st["acts"]),
                             "theoretical_weight_storage_bytes": theoretical_weight_storage_bytes(model, w), **m, "d_roc_auc": m["roc_auc"] - base})
        pd.DataFrame(rows).to_csv(TAB / "phase3_ptq_s4_mixed.csv", index=False)

    if args.mixed is None:
        Path("reports/phase3_ptq_studies.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
