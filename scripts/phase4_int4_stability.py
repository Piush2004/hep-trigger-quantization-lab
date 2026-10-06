#!/usr/bin/env python
"""Phase 4 / Phase 2: causal diagnosis of INT4 PTQ instability on the FROZEN FP32 checkpoints.

No training. Frozen artifacts are read only. Five FP32 checkpoints (seeds 42, 123, 2026 frozen; 7, 99 added).

Experiments (deterministic: every random draw is seeded):
  A  fixed model (seed 42), 10 different calibration subsets of size 1000
  B  per model seed, the SAME calibration procedure (5 draws of 1000) -> model-vs-calibration variance split
  C  fixed model + fixed calibration: repeated inference and repeated variant build -> determinism check
  D  calibration-set size 100/500/1000/5000/10000, 5 draws each, all 5 seeds, INT8/INT6/INT4 PTQ

Mechanism diagnostics recorded per variant (computed on TEST inputs, labels untouched by calibration):
  - calibrated activation range [lo, hi] per hidden ReLU
  - quantization step (hi-lo)/(2^b-1)
  - clipping rate: fraction of test pre-quantizer activations outside [lo, hi]
  - output disagreement with the FP32 model (argmax disagreement, mean |dp|)
  - weight-tensor step per layer (weights use their own min/max; they do NOT depend on calibration)

Note: the fake-quantizer zero-point is NOT rounded to an integer (see ptq.fake_quantize_tensor). This is
recorded as a limitation; it is not changed here because changing it would change the frozen methodology.

Outputs: reports/tables/phase4_int4_*.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hep_quant.analysis.phase2 import predict_proba  # noqa: E402
from hep_quant.analysis.ptq_studies import build_ptq_variant, uniform_units, unit_names  # noqa: E402
from hep_quant.quantization.ptq import FakeQuantActivation  # noqa: E402
from hep_quant.data.hls4ml_jets import load_hls4ml_jets  # noqa: E402
from hep_quant.data.preprocessing import Preprocessor  # noqa: E402
from hep_quant.data.splits import split_dataset  # noqa: E402
from hep_quant.analysis.phase2 import load_checkpoint_model  # noqa: E402

SEEDS = [42, 123, 2026, 7, 99]
TAB = ROOT / "reports" / "tables"
BITS = [8, 6, 4]
FROZEN = ROOT / "experiments" / "results" / "frozen_phase2"
RESULTS = ROOT / "experiments" / "results"
CKPT = ROOT / "experiments" / "checkpoints"


def load_seed_any(seed: int):
    """Same procedure as phase3_ptq_studies.load_seed, but seeds 7/99 live outside the frozen directory.

    Read-only: nothing under experiments/ is written.
    """
    frozen_json = FROZEN / f"EXP_S{seed}_JET_FP32.json"
    if frozen_json.exists():
        rec_path, ckpt = frozen_json, FROZEN / f"EXP_S{seed}_JET_FP32__best_model.pt"
    else:
        rec_path = RESULTS / f"EXP_S{seed}_JET_FP32.json"
        ckpt = CKPT / f"EXP_S{seed}_JET_FP32" / "best_model.pt"
    rec = json.loads(rec_path.read_text())
    d = rec["config"]["dataset"]
    ds = load_hls4ml_jets(subset_size=d["subset_size"], seed=seed)
    sp = split_dataset(ds.X, ds.y, val_fraction=d["val_fraction"], test_fraction=d["test_fraction"], seed=seed)
    pre = Preprocessor.fit(sp.X_train)
    model = load_checkpoint_model(ckpt, "none", "fp32")
    return model, pre.transform(sp.X_train), pre.transform(sp.X_test), sp.y_test


def metrics(model, X, y):
    p = predict_proba(model, X)
    return p, {
        "accuracy": accuracy_score(y, p.argmax(1)),
        "f1": f1_score(y, p.argmax(1), average="macro"),
        "roc_auc": roc_auc_score(y, p, multi_class="ovr", average="macro"),
    }


def draw_indices(n_train: int, n: int, seed: int, draw: int) -> np.ndarray:
    return np.random.default_rng(10_000 * seed + 100 * n + draw).choice(n_train, n, replace=False)


def diagnostics(variant, X_test, p_fp32, p_var, bits) -> dict:
    """Clip rate and step per hidden activation; weight step per layer; output disagreement."""
    clip = {}
    hooks = []
    fq = [m for m in variant.modules() if isinstance(m, FakeQuantActivation)]
    for k, m in enumerate(fq):
        def pre(mod, inp, k=k):
            x = inp[0]
            clip[k] = float(((x < mod.min_val) | (x > mod.max_val)).float().mean())
        hooks.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        variant(torch.as_tensor(X_test, dtype=torch.float32))
    for h in hooks:
        h.remove()
    row = {}
    for k, m in enumerate(fq):
        row[f"A{k}_lo"] = float(m.min_val)
        row[f"A{k}_hi"] = float(m.max_val)
        row[f"A{k}_step"] = float((m.max_val - m.min_val) / (2**bits - 1))
        row[f"A{k}_clip_rate"] = clip[k]
    row["argmax_disagree_vs_fp32"] = float((p_var.argmax(1) != p_fp32.argmax(1)).mean())
    row["mean_abs_dp_vs_fp32"] = float(np.abs(p_var - p_fp32).mean())
    return row


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    data = {s: load_seed_any(s) for s in SEEDS}
    rows_a, rows_b, rows_c, rows_d, rows_w = [], [], [], [], []

    # Weight step per layer for each model/precision: depends only on the weights (calibration-free).
    for s in SEEDS:
        model, Xtr, Xte, yte = data[s]
        linears, _ = unit_names(model)
        for bits in BITS:
            for i, name in enumerate(linears):
                w = model.net.get_submodule(name).weight.data
                lo, hi = float(w.min()), float(w.max())
                rows_w.append({"seed": s, "bits": bits, "layer": f"L{i}", "w_min": lo, "w_max": hi,
                               "w_step": (hi - lo) / (2**bits - 1)})

    def run_variant(s, bits, X_calib, draw_tag):
        model, Xtr, Xte, yte = data[s]
        p_fp32, _ = metrics(model, Xte, yte)
        wb, ab = uniform_units(model, bits)
        v = build_ptq_variant(model, X_calib, wb, ab)
        p_v, m = metrics(v, Xte, yte)
        return v, p_fp32, p_v, m, diagnostics(v, Xte, p_fp32, p_v, bits)

    # ---- A: fixed model (seed 42), 10 calibration subsets of 1000 ----
    s = 42
    model, Xtr, Xte, yte = data[s]
    for draw in range(10):
        idx = draw_indices(len(Xtr), 1000, s, draw)
        for bits in BITS:
            _, _, _, m, d = run_variant(s, bits, Xtr[idx], draw)
            rows_a.append({"seed": s, "bits": bits, "draw": draw, "calib_size": 1000, **m, **d})

    # ---- B: every seed, same calibration procedure (5 draws of 1000) ----
    for s in SEEDS:
        model, Xtr, Xte, yte = data[s]
        for draw in range(5):
            idx = draw_indices(len(Xtr), 1000, s, draw)
            for bits in BITS:
                _, _, _, m, d = run_variant(s, bits, Xtr[idx], draw)
                rows_b.append({"seed": s, "bits": bits, "draw": draw, "calib_size": 1000, **m, **d})

    # ---- C: determinism, fixed model + fixed calibration set ----
    s = 42
    model, Xtr, Xte, yte = data[s]
    idx = draw_indices(len(Xtr), 1000, s, 0)
    digests = []
    for rep in range(5):
        v, p_fp32, p_v, m, _ = run_variant(s, 4, Xtr[idx], 0)
        digests.append(hashlib.sha256(np.ascontiguousarray(p_v).tobytes()).hexdigest())
        rows_c.append({"seed": s, "bits": 4, "build_repeat": rep, **m, "prob_sha256": digests[-1][:16]})
    # repeated inference on the SAME built variant
    for rep in range(3):
        with torch.no_grad():
            pr = v(torch.as_tensor(Xte, dtype=torch.float32)).numpy()
        rows_c.append({"seed": s, "bits": 4, "build_repeat": f"infer{rep}", "prob_sha256":
                       hashlib.sha256(np.ascontiguousarray(pr).tobytes()).hexdigest()[:16]})
    c_identical = len(set(digests)) == 1

    # ---- D: calibration-size sweep ----
    for s in SEEDS:
        model, Xtr, Xte, yte = data[s]
        for n in [100, 500, 1000, 5000, 10000]:
            for draw in range(5):
                idx = draw_indices(len(Xtr), n, s, draw)
                for bits in BITS:
                    _, _, _, m, d = run_variant(s, bits, Xtr[idx], draw)
                    rows_d.append({"seed": s, "bits": bits, "draw": draw, "calib_size": n, **m, **d})

    pd.DataFrame(rows_w).to_csv(TAB / "phase4_int4_weight_steps.csv", index=False)
    pd.DataFrame(rows_a).to_csv(TAB / "phase4_int4_A_fixed_model_calib_subsets.csv", index=False)
    pd.DataFrame(rows_b).to_csv(TAB / "phase4_int4_B_model_seeds.csv", index=False)
    pd.DataFrame(rows_c).to_csv(TAB / "phase4_int4_C_determinism.csv", index=False)
    pd.DataFrame(rows_d).to_csv(TAB / "phase4_int4_D_calib_size.csv", index=False)

    summary = {
        "C_build_and_inference_bitwise_identical": c_identical,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "seeds": SEEDS,
    }
    (ROOT / "reports" / "phase4_int4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
