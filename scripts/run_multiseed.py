#!/usr/bin/env python
"""Run the real-dataset experiment matrix across several seeds (resumable).

Each (seed, config) gets experiment id EXP_S<seed>_JET_<TAG>. The seed controls
the stratified subsample, the train/val/test split, weight init and shuffling,
so the reported std includes data-sampling variance, not only init variance.
Hardware analysis uses the software-estimate backend (labeled `estimated`)
because hls4ml cannot convert fake-quantized models; see docs/hardware.md.

Usage: python scripts/run_multiseed.py [--seeds 42 123 2026]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.experiments.runner import run_experiment
from hep_quant.utils.logging import get_logger

logger = get_logger("multiseed")

CONFIGS = {
    "FP32": "configs/jet_fp32.yaml",
    "INT8_PTQ": "configs/jet_int8.yaml",
    "INT6_PTQ": "configs/jet_int6.yaml",
    "INT4_PTQ": "configs/jet_int4.yaml",
    "INT8_QAT": "configs/jet_int8_qat.yaml",
    "INT6_QAT": "configs/jet_int6_qat.yaml",
    "INT4_QAT": "configs/jet_int4_qat.yaml",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    ap.add_argument("--results-dir", default="experiments/results")
    args = ap.parse_args()

    failures = []
    for seed in args.seeds:
        for tag, cfg in CONFIGS.items():
            exp_id = f"EXP_S{seed}_JET_{tag}"
            path = Path(args.results_dir) / f"{exp_id}.json"
            if path.exists() and json.loads(path.read_text()).get("status") == "completed":
                logger.info(f"skip {exp_id} (already completed)")
                continue
            t0 = time.time()
            try:
                run_experiment(
                    cfg,
                    overrides=[f"seed={seed}", f"experiment.id={exp_id}", "hardware.backend=software_estimate"],
                    results_dir=args.results_dir,
                )
                logger.info(f"done {exp_id} in {time.time() - t0:.0f}s")
            except Exception as exc:  # keep going; failures are recorded in the registry and reported at the end
                logger.error(f"{exp_id} failed: {exc}")
                failures.append(exp_id)
    if failures:
        logger.error(f"failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
