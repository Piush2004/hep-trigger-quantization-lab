#!/usr/bin/env python
"""End-to-end numerical agreement: FP32 PyTorch -> quantized PyTorch -> hls4ml fixed-point C-simulation (seed-42 models).

Reference = FP32 PyTorch logits on the 15,000 held-out test jets of EXP_S42. Every candidate is compared on:
class-prediction agreement, accuracy, logit abs error (mean/max), softmax-probability abs error (mean/max),
macro one-vs-rest ROC-AUC and its difference to FP32. hls4ml rows are measurement tier **simulated**
(C-simulation); PyTorch rows are measured. Writes reports/numerical_consistency.{md,json} and a figure.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.analysis.phase2 import load_checkpoint_model, test_split_for_record
from hep_quant.hardware.hls_csim import csim_predict

FROZEN = Path("experiments/results/frozen_phase2")
WIDTHS = [8, 10, 12, 14, 16]
MODES = {
    "default (TRN, WRAP, accum = W bits)": {},
    "RND + SAT, accum = W bits": {"rounding": "AP_RND", "saturation": "AP_SAT"},
    "TRN, WRAP, wide accum ap_fixed<28,12>": {"accum_bits": (28, 12)},
    "RND + SAT, wide accum ap_fixed<28,12>": {"rounding": "AP_RND", "saturation": "AP_SAT", "accum_bits": (28, 12)},
}


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def compare(ref_logits, logits, y, ref_auc) -> dict:
    p_ref, p = softmax(ref_logits), softmax(logits)
    finite = bool(np.isfinite(logits).all())
    auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro")) if finite else float("nan")
    return {"accuracy": float((logits.argmax(1) == y).mean()), "class_agreement_with_fp32": float((logits.argmax(1) == ref_logits.argmax(1)).mean()),
            "logit_mae": float(np.abs(logits - ref_logits).mean()), "logit_max_abs_err": float(np.abs(logits - ref_logits).max()),
            "prob_mae": float(np.abs(p - p_ref).mean()), "prob_max_abs_err": float(np.abs(p - p_ref).max()),
            "roc_auc": auc, "d_roc_auc_vs_fp32": auc - ref_auc, "finite": finite}


def main() -> int:
    rec = json.loads((FROZEN / "EXP_S42_JET_FP32.json").read_text())
    X, y, _ = test_split_for_record(rec)
    X, y = X[:15000], y[:15000]
    fp32 = load_checkpoint_model(FROZEN / "EXP_S42_JET_FP32__best_model.pt", "none", "fp32")
    with torch.no_grad():
        ref = fp32(torch.from_numpy(X)).numpy()
    ref_auc = float(roc_auc_score(y, softmax(ref), multi_class="ovr", average="macro"))
    out = {"n_eval": int(len(y)), "fp32_pytorch": {"accuracy": float((ref.argmax(1) == y).mean()), "roc_auc": ref_auc}, "pytorch_quantized": {}, "hls4ml_csim": []}

    pt_logits = {}
    for tag, (method, prec) in {"INT8_PTQ": ("ptq", "int8"), "INT6_PTQ": ("ptq", "int6"), "INT4_PTQ": ("ptq", "int4"),
                                "INT8_QAT": ("qat", "int8"), "INT6_QAT": ("qat", "int6"), "INT4_QAT": ("qat", "int4")}.items():
        m = load_checkpoint_model(FROZEN / f"EXP_S42_JET_{tag}__best_model.pt", method, prec)
        with torch.no_grad():
            pt_logits[tag] = m(torch.from_numpy(X)).numpy()
        out["pytorch_quantized"][tag] = compare(ref, pt_logits[tag], y, ref_auc)

    hls_logits = {}
    for mode, kw in MODES.items():
        for w in WIDTHS:
            lg = csim_predict(fp32, X, total_bits=w, int_bits=6, output_dir="experiments/hls4ml_csim", **kw)
            hls_logits[(mode, w)] = lg
            out["hls4ml_csim"].append({"mode": mode, "ap_fixed_total_bits": w, "int_bits": 6, "measurement_tier": "simulated", **compare(ref, lg, y, ref_auc)})
    # cross-implementation agreement: quantized PyTorch INT8 PTQ vs hls4ml 16-bit (default mode)
    out["cross_check_int8_ptq_pytorch_vs_hls4ml_w16_default_class_agreement"] = float((pt_logits["INT8_PTQ"].argmax(1) == hls_logits[(list(MODES)[0], 16)].argmax(1)).mean())
    Path("reports/numerical_consistency.json").write_text(json.dumps(out, indent=2))

    md = ["# Numerical consistency: FP32 PyTorch ↔ quantized PyTorch ↔ hls4ml fixed point", "",
          f"Model family: MLP 16-64-32-32-5, seed-42 models, {len(y):,} held-out test jets (hls4ml_lhc_jets_hlf). Reference: FP32 PyTorch "
          f"(accuracy {out['fp32_pytorch']['accuracy']:.4f}, macro ROC-AUC {ref_auc:.4f}). Logit/probability errors are against the FP32 reference on the same jets.", "",
          "## Quantized PyTorch (fake-quantization; measured)", "",
          "| model | class agreement | accuracy | logit MAE | logit max abs | prob MAE | prob max abs | ROC-AUC | ΔROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in out["pytorch_quantized"].items():
        md.append(f"| {t} | {r['class_agreement_with_fp32']:.4f} | {r['accuracy']:.4f} | {r['logit_mae']:.4f} | {r['logit_max_abs_err']:.3f} | {r['prob_mae']:.4f} | {r['prob_max_abs_err']:.3f} | {r['roc_auc']:.4f} | {r['d_roc_auc_vs_fp32']:+.4f} |")
    md += ["", "## hls4ml C-simulation of the FP32 weights, ap_fixed<W,6> (simulated; physical FPGA synthesis not performed)", "",
           "Input, weights, biases, layer outputs and result share ap_fixed<W,6> (plus the modes below). hls4ml 1.3.0, backend Vivado, io_parallel, Strategy Latency, ReuseFactor 1.", "",
           "| mode | W | class agreement | accuracy | logit MAE | logit max abs | prob MAE | ROC-AUC | ΔROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out["hls4ml_csim"]:
        md.append(f"| {r['mode']} | {r['ap_fixed_total_bits']} | {r['class_agreement_with_fp32']:.4f} | {r['accuracy']:.4f} | {r['logit_mae']:.3f} | {r['logit_max_abs_err']:.2f} | {r['prob_mae']:.4f} | {r['roc_auc']:.4f} | {r['d_roc_auc_vs_fp32']:+.4f} |")
    md += ["", f"Cross-implementation check: class agreement between quantized-PyTorch INT8 PTQ and hls4ml `ap_fixed<16,6>` (default mode): "
           f"{out['cross_check_int8_ptq_pytorch_vs_hls4ml_w16_default_class_agreement']:.4f} (both are compared to FP32 above; they are different quantization schemes)."]
    Path("reports/numerical_consistency.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for mode in MODES:
        rows = [r for r in out["hls4ml_csim"] if r["mode"] == mode]
        ax.plot([r["ap_fixed_total_bits"] for r in rows], [r["class_agreement_with_fp32"] for r in rows], "o-", label=mode)
    ax.set_xlabel("Total fixed-point width W (bits), integer bits = 6"); ax.set_ylabel("Class-prediction agreement with FP32 PyTorch")
    ax.set_title("hls4ml C-simulation (tier: simulated) — fixed-point mode ablation", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("hls4ml_lhc_jets_hlf | MLP 16-64-32-32-5 | seed-42 FP32 weights | 15k test jets | no FPGA synthesis", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("reports/figures/phase3_hls_fixed_point_modes.png", dpi=150)
    print(json.dumps({"cross": out["cross_check_int8_ptq_pytorch_vs_hls4ml_w16_default_class_agreement"],
                      "hls": [(r["mode"][:12], r["ap_fixed_total_bits"], round(r["class_agreement_with_fp32"], 3)) for r in out["hls4ml_csim"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
