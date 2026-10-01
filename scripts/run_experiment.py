#!/usr/bin/env python
"""Run every experiment config in sequence and generate the aggregate report.

Usage: python scripts/run_experiment.py [configs/fp32.yaml configs/int8.yaml ...]
Defaults to the full matrix (FP32 baseline + PTQ/QAT at INT8/INT6/INT4) if no
configs are given.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.experiments.runner import run_experiment
from hep_quant.reporting import generate_report
from hep_quant.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIGS = [
    "configs/fp32.yaml",
    "configs/int8.yaml",
    "configs/int6.yaml",
    "configs/int4.yaml",
    "configs/int8_qat.yaml",
    "configs/int6_qat.yaml",
    "configs/int4_qat.yaml",
]


def main() -> int:
    configs = sys.argv[1:] or DEFAULT_CONFIGS
    failures = []
    for config in configs:
        logger.info(f"Running {config}")
        try:
            run_experiment(config)
        except Exception as exc:
            logger.error(f"{config} failed: {exc}")
            failures.append((config, str(exc)))

    generate_report()

    if failures:
        logger.error(f"{len(failures)} experiment(s) failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
