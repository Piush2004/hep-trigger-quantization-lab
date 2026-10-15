#!/usr/bin/env python
"""Phase 11, application stage: run the trained surrogate (and the existing heuristic) over
Phase 6's real frozen configurations.

Uses Phase 6 (both CPU, avoiding the documented CPU/CUDA INT4 divergence -- see
docs/REPRODUCIBILITY.md section 6) rather than Phase 2's CUDA-era checkpoints. Phase 6 produced
5 configs per architecture (FP32, INT8 PTQ, INT6 PTQ, INT4 PTQ, INT4 QAT -- no INT6/INT8 QAT),
x 2 architectures (A: hidden=[64,32,32], B: hidden=[128,64,64,32]) = 10 configurations, read-only
via load_checkpoint_model, no retraining. Run 1 only (Run 2 is a reported-identical duplicate,
see reports/phase6_architecture_generalization.md section 13).

Outputs: reports/tables/phase11_surrogate_hardware.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "reports" / "tables"

from hep_quant.analysis.phase2 import load_checkpoint_model  # noqa: E402
from hep_quant.hardware.resource_estimator import SoftwareEstimateBackend, SurrogateEstimateBackend  # noqa: E402

SEEDS = [42, 123, 2026, 7, 99]
ARCHITECTURES = {"A": "P6A", "B": "P6B"}
TAGS = [("FP32", "none", "fp32"), ("INT8_PTQ", "ptq", "int8"), ("INT6_PTQ", "ptq", "int6"),
        ("INT4_PTQ", "ptq", "int4"), ("INT4_QAT", "qat", "int4")]
RESULTS_DIR = ROOT / "experiments" / "results_phase6_run1"
CKPT_DIR = ROOT / "experiments" / "checkpoints_phase6_run1"


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    heuristic = SoftwareEstimateBackend()
    surrogate = SurrogateEstimateBackend()
    print(f"surrogate backend available: {surrogate.is_available()}")

    rows = []
    for arch_label, arch_prefix in ARCHITECTURES.items():
        for seed in SEEDS:
            for tag, method, precision in TAGS:
                name = f"EXP_{arch_prefix}_S{seed}_JET_{tag}"
                rec_path = RESULTS_DIR / f"{name}.json"
                ckpt_path = CKPT_DIR / name / "best_model.pt"
                if not rec_path.exists() or not ckpt_path.exists():
                    print(f"  skip (missing): {name}")
                    continue
                model = load_checkpoint_model(ckpt_path, method, precision)
                h = heuristic.analyze(model, precision, {"reuse_factor": 1})
                s = surrogate.analyze(model, precision, {"reuse_factor": 1})
                rows.append({
                    "architecture": arch_label, "seed": seed, "precision": tag, "method": method,
                    "heuristic_lut": h.lut, "heuristic_dsp": h.dsp, "heuristic_bram": h.bram, "heuristic_ff": h.ff,
                    "surrogate_lut": s.lut, "surrogate_dsp": s.dsp, "surrogate_bram": s.bram, "surrogate_ff": s.ff,
                    "surrogate_latency_ns": s.latency_estimate_ns, "surrogate_ii": s.initiation_interval,
                    "surrogate_measurement_tier": s.measurement_tier,
                })

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "phase11_surrogate_hardware.csv", index=False)
    print(f"wrote {len(df)} rows to {TAB / 'phase11_surrogate_hardware.csv'}")
    print(df.groupby(["architecture", "precision"])[
        ["heuristic_lut", "surrogate_lut", "heuristic_dsp", "surrogate_dsp"]
    ].mean().round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
