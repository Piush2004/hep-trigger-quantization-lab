"""Command-line interface: `python -m hep_quant <command> ...`."""
from __future__ import annotations

import argparse
import json
import sys

from hep_quant.utils.logging import get_logger

logger = get_logger("hep_quant.cli")


def cmd_validate_data(args: argparse.Namespace) -> int:
    from hep_quant.config import apply_overrides, load_config

    cfg = load_config(args.config) if args.config else {"dataset": {}}
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    dataset_cfg = cfg.get("dataset", {})
    seed = cfg.get("seed", 42)

    if dataset_cfg.get("source") == "hls4ml_jets":
        from hep_quant.data.real_validation import validate_real_dataset, write_report

        report = validate_real_dataset(dataset_cfg, seed=seed)
        json_path, md_path = write_report(report, out_dir=args.output_dir, stem=args.stem)
        print(json.dumps({k: report[k] for k in ("dataset", "status")}, indent=2))
        print(f"Wrote {json_path} and {md_path}")
        logger.info(f"Real dataset validation {report['status']}")
        return 0 if report["status"] == "PASS" else 1

    from hep_quant.data.loader import load_dataset

    ds = load_dataset(
        dataset_cfg.get("path"),
        n_samples=dataset_cfg.get("n_samples", 20000),
        n_features=dataset_cfg.get("n_features", 16),
        n_classes=dataset_cfg.get("n_classes", 2),
        seed=seed,
    )
    report = ds.validate(raise_on_error=True)
    print(json.dumps(report.to_dict(), indent=2))
    logger.info("Dataset validation PASSED")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from hep_quant.experiments.runner import run_experiment

    record = run_experiment(args.config, overrides=args.override)
    print(json.dumps(record, indent=2, default=str))
    return 0 if record["status"] == "completed" else 1


def cmd_quantize(args: argparse.Namespace) -> int:
    # Quantization happens as part of the experiment run when the config's
    # quantization.method is "ptq" or "qat" — same entry point as `train`.
    return cmd_train(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from hep_quant.experiments.registry import ExperimentRegistry

    registry = ExperimentRegistry()
    record = registry.load(args.experiment)
    print(json.dumps(record, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    return cmd_evaluate(args)


def cmd_report(args: argparse.Namespace) -> int:
    from hep_quant.reporting import generate_report

    generate_report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hep_quant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-data", help="Load and validate a dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--override", action="append", default=[], help="key.path=value")
    p.add_argument("--output-dir", default="reports", help="where real-dataset validation reports are written")
    p.add_argument("--stem", default="real_data_validation", help="report file stem for real-dataset validation")
    p.set_defaults(func=cmd_validate_data)

    p = sub.add_parser("train", help="Run an experiment (train, optionally quantize, evaluate)")
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[], help="key.path=value")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("quantize", help="Alias for train with a quantization-enabled config")
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[])
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("evaluate", help="Print a saved experiment record")
    p.add_argument("--experiment", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="Print a saved experiment's latency/hardware record")
    p.add_argument("--experiment", required=True)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("report", help="Aggregate all experiment results into tables/plots/report")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
