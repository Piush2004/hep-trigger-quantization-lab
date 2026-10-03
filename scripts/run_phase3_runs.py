#!/usr/bin/env python
"""Phase-3 additional runs (new IDs; the frozen Phase-2 snapshot is never touched).

  --extra-seeds 7 99   full 7-configuration matrix  -> EXP_S<seed>_JET_<TAG>   (extends n=3 to n=5)
  --control-seeds ...  compute-matched FP32 fine-tune control -> EXP_C<seed>_JET_FP32_FT
Resumable: completed experiments are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.experiments.runner import run_experiment
from hep_quant.utils.logging import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multiseed import CONFIGS  # noqa: E402

logger = get_logger("phase3_runs")
FROZEN_DIR = Path("experiments/results/frozen_phase2")


def run_one(cfg: str, exp_id: str, seed: int) -> None:
    path = Path("experiments/results") / f"{exp_id}.json"
    if (FROZEN_DIR / f"{exp_id}.json").exists():
        raise RuntimeError(f"{exp_id} is a frozen Phase-2 experiment; refusing to re-run")
    if path.exists() and json.loads(path.read_text()).get("status") == "completed":
        logger.info(f"skip {exp_id}")
        return
    t0 = time.time()
    run_experiment(cfg, overrides=[f"seed={seed}", f"experiment.id={exp_id}", "hardware.backend=software_estimate"])
    logger.info(f"done {exp_id} in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--control-seeds", type=int, nargs="*", default=[])
    args = ap.parse_args()
    for seed in args.control_seeds:
        run_one("configs/jet_fp32_finetune_control.yaml", f"EXP_C{seed}_JET_FP32_FT", seed)
    for seed in args.extra_seeds:
        for tag, cfg in CONFIGS.items():
            run_one(cfg, f"EXP_S{seed}_JET_{tag}", seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
