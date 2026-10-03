#!/usr/bin/env python
"""Fixed-point accuracy sweep of the trained real-data FP32 MLP via hls4ml C-simulation.

Loads the FP32 checkpoint of a real-data experiment, rebuilds the exact held-out
test split from the experiment's own config (same adapter, subset, seed), and
evaluates hls4ml's bit-accurate ap_fixed<W,I> simulation for several widths.
Output: reports/hls_csim_accuracy.{json,md} and a figure. Tier: simulated.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.config import load_config
from hep_quant.data.hls4ml_jets import load_hls4ml_jets
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.hardware.hls_csim import csim_accuracy_sweep
from hep_quant.models.mlp import MLPClassifier
from hep_quant.training.trainer import Trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jet_fp32.yaml")
    ap.add_argument("--experiment", default="EXP_S42_JET_FP32", help="checkpoint dir name under experiments/checkpoints")
    ap.add_argument("--seed", type=int, default=None, help="must match the seed the experiment was trained with")
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18])
    ap.add_argument("--int-bits", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=15000)
    ap.add_argument("--out", default="reports/hls_csim_accuracy")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    dcfg = cfg["dataset"]
    ds = load_hls4ml_jets(subset_size=dcfg.get("subset_size"), seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=dcfg["val_fraction"], test_fraction=dcfg["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    X_test, y_test = pre.transform(sp.X_test)[: args.n_eval], sp.y_test[: args.n_eval]

    model = Trainer.load_checkpoint(f"experiments/checkpoints/{args.experiment}/best_model.pt", MLPClassifier, device="cpu")
    with torch.no_grad():
        ref_pred = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()
    fp32_acc = float((ref_pred == y_test).mean())

    rows = csim_accuracy_sweep(model, X_test, y_test, args.widths, int_bits=args.int_bits, reference_pred=ref_pred)
    result = {
        "experiment": args.experiment,
        "config": args.config,
        "seed": seed,
        "n_eval": int(len(y_test)),
        "fp32_torch_accuracy_on_same_rows": fp32_acc,
        "note": "hls4ml C-simulation (bit-accurate ap_fixed on host CPU). Tier: simulated. No latency/resource data: physical FPGA synthesis not performed.",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    md = [
        "# hls4ml C-simulation accuracy vs fixed-point width",
        "",
        f"Model: `{args.experiment}` (FP32 weights), {len(y_test):,} held-out test jets, measurement tier **simulated**. "
        f"FP32 (PyTorch) accuracy on the same rows: {fp32_acc:.4f}. Physical FPGA synthesis not performed; no latency/resources.",
        "",
        "| ap_fixed<W,I> | frac bits | accuracy | agreement with FP32 predictions |",
        "|---|---:|---:|---:|",
    ]
    md += [f"| {r['ap_fixed']} | {r['frac_bits']} | {r['accuracy']:.4f} | {r['agreement_with_fp32_predictions']:.4f} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([r["total_bits"] for r in rows], [r["accuracy"] for r in rows], "o-", label=f"hls4ml C-sim ap_fixed<W,{args.int_bits}>")
    ax.axhline(fp32_acc, color="gray", ls="--", label="FP32 (PyTorch)")
    ax.set_xlabel("Total fixed-point width W (bits), integer bits fixed")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs fixed-point width (C-simulation, simulated tier)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    (out.parent / "figures").mkdir(exist_ok=True)
    fig.savefig(out.parent / "figures" / "hls_csim_accuracy_vs_width.png", dpi=150)
    print(json.dumps({"fp32": fp32_acc, "rows": [(r["ap_fixed"], round(r["accuracy"], 4)) for r in rows]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
