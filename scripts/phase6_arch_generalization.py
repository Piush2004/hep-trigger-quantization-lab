#!/usr/bin/env python
"""Phase 6: architecture-generalization experiment (2 architectures x 5 configurations x 5 seeds, CPU).

Architecture A = primary MLP, hidden [64, 32, 32]   (re-run on CPU as a device-matched control)
Architecture B = deeper/wider MLP, hidden [128, 64, 64, 32]

Only `model.hidden_dims` differs between A and B. Everything else (dataset, split, training budget,
QAT recipe, calibration) comes from the existing configs, unchanged. See reports/phase6_architecture_selection.md.

Usage:  python scripts/phase6_arch_generalization.py --run-tag run1
        python scripts/phase6_arch_generalization.py --run-tag run2
Each --run-tag writes to its own folders, so the two independent runs cannot resume from each other.
Resumable within one run-tag: completed experiments are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.experiments.runner import run_experiment  # noqa: E402
from hep_quant.utils.logging import get_logger  # noqa: E402

logger = get_logger("phase6")

SEEDS = [42, 123, 2026, 7, 99]
CONFIGS = {
    "FP32": "configs/jet_fp32.yaml",
    "INT8_PTQ": "configs/jet_int8.yaml",
    "INT6_PTQ": "configs/jet_int6.yaml",
    "INT4_PTQ": "configs/jet_int4.yaml",
    "INT4_QAT": "configs/jet_int4_qat.yaml",
}
ARCHS = {
    "A": [],                                   # primary config, hidden [64, 32, 32]
    "B": ["model.hidden_dims=[128,64,64,32]"],  # deeper/wider, everything else identical
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True, choices=["run1", "run2"])
    args = ap.parse_args()
    results_dir = ROOT / "experiments" / f"results_phase6_{args.run_tag}"
    ckpt_dir = ROOT / "experiments" / f"checkpoints_phase6_{args.run_tag}"

    for arch, arch_overrides in ARCHS.items():
        for seed in SEEDS:
            for tag, cfg in CONFIGS.items():
                exp_id = f"EXP_P6{arch}_S{seed}_JET_{tag}"
                done = results_dir / f"{exp_id}.json"
                if done.exists() and json.loads(done.read_text()).get("status") == "completed":
                    logger.info(f"skip {exp_id}")
                    continue
                t0 = time.time()
                overrides = arch_overrides + [f"seed={seed}", f"experiment.id={exp_id}", "hardware.backend=software_estimate"]
                run_experiment(str(ROOT / cfg), overrides=overrides,
                               checkpoint_dir=str(ckpt_dir), results_dir=str(results_dir))
                logger.info(f"done {exp_id} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
