"""Aggregate all experiment records into results_summary.csv, tables, and Pareto plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hep_quant.experiments.registry import ExperimentRegistry
from hep_quant.utils.logging import get_logger
from hep_quant.visualization.pareto import find_pareto_frontier, plot_pareto_frontier
from hep_quant.visualization.plots import plot_latency_vs_accuracy, plot_model_size_vs_precision

logger = get_logger(__name__)

RESULTS_COLUMNS = [
    "experiment_id", "architecture", "dataset", "precision", "method", "parameters",
    "model_size_mb", "accuracy", "precision_score", "recall", "f1", "roc_auc", "pr_auc",
    "inference_latency_ms", "latency_unit", "lut", "dsp", "bram", "ff",
    "hardware_measurement_tier", "resource_source",
    "training_time_s", "seed", "git_commit",
]


def _record_to_row(record: dict) -> dict:
    if record.get("status") != "completed":
        return None
    metrics = record.get("metrics", {})
    test_metrics = metrics.get("test", {})
    model_info = record.get("model_info", {})
    hardware = record.get("hardware", {})
    repro = record.get("reproducibility", {})

    history_duration = None  # populated by caller if available

    return {
        "experiment_id": record["experiment_id"],
        "architecture": model_info.get("architecture"),
        "dataset": record.get("config", {}).get("dataset", {}).get("name"),
        "precision": model_info.get("precision"),
        "method": model_info.get("method"),
        "parameters": model_info.get("n_parameters"),
        "model_size_mb": metrics.get("model_size", {}).get("estimated_size_mb"),
        "accuracy": test_metrics.get("accuracy"),
        "precision_score": test_metrics.get("precision"),
        "recall": test_metrics.get("recall"),
        "f1": test_metrics.get("f1"),
        "roc_auc": test_metrics.get("roc_auc"),
        "pr_auc": test_metrics.get("pr_auc"),
        "inference_latency_ms": metrics.get("latency", {}).get("mean_ms_per_sample"),
        "latency_unit": "ms/sample (software, CPU/GPU)",
        "lut": hardware.get("lut"),
        "dsp": hardware.get("dsp"),
        "bram": hardware.get("bram"),
        "ff": hardware.get("ff"),
        "hardware_measurement_tier": hardware.get("measurement_tier", "unavailable"),
        "resource_source": hardware.get("resource_source"),
        "training_time_s": history_duration,
        "seed": repro.get("seed"),
        "git_commit": repro.get("system", {}).get("git_commit"),
    }


def build_results_dataframe(results_dir: str = "experiments/results") -> pd.DataFrame:
    registry = ExperimentRegistry(results_dir)
    records = registry.load_all()
    rows = [row for row in (_record_to_row(r) for r in records) if row is not None]
    if not rows:
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def generate_report(
    results_dir: str = "experiments/results",
    out_csv: str = "results_summary.csv",
    figures_dir: str = "reports/figures",
    tables_dir: str = "reports/tables",
) -> pd.DataFrame:
    """Write the combined results CSV, then per-dataset tables/plots.

    Plots and Pareto analysis are generated SEPARATELY for each distinct
    `dataset` value and filename-tagged with it (e.g.
    `pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.png`). This project uses more
    than one dataset (a synthetic pipeline-validation set and the real
    hls4ml jet-tagging benchmark), with very different accuracy scales and
    physical meaning — a single combined plot would visually conflate them,
    which is exactly the silent-merging the project's dataset card policy
    forbids. The combined CSV is fine as a single file since every row already
    carries an explicit `dataset` column.
    """
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    df = build_results_dataframe(results_dir)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote {len(df)} experiment rows to {out_csv}")

    if df.empty:
        logger.warning("No completed experiments found; skipping plots/tables.")
        return df

    # Staged validation runs (e.g. EXP_010_JET_VALIDATE_10K) are deliberately
    # under-trained pipeline/training sanity checks at a smaller data subset
    # -- not comparable points in the precision-comparison matrix they
    # preceded. They stay in results_summary.csv (the full raw record of
    # everything that ran) but are excluded here so they don't get plotted
    # or Pareto-ranked alongside the actual experiment matrix, which would
    # misleadingly make e.g. a 10k/8-epoch FP32 run look "dominated" by a
    # 100k/30-epoch PTQ run on an unrelated axis (training budget).
    comparable_df = df[~df["experiment_id"].str.contains("VALIDATE", na=False)]

    datasets = sorted(d for d in comparable_df["dataset"].dropna().unique())
    for dataset in datasets:
        subset = comparable_df[comparable_df["dataset"] == dataset]
        _generate_dataset_report(subset, dataset, figures_dir, tables_dir)

    return df


def _generate_dataset_report(df: pd.DataFrame, dataset: str, figures_dir: str, tables_dir: str) -> None:
    markdown_table = df[
        ["experiment_id", "precision", "method", "accuracy", "f1", "roc_auc", "model_size_mb", "inference_latency_ms"]
    ].to_markdown(index=False, floatfmt=".4f")
    with open(Path(tables_dir) / f"quantization_results_{dataset}.md", "w") as f:
        f.write(f"# Quantization Results Summary — {dataset}\n\n" + markdown_table + "\n")

    try:
        plot_model_size_vs_precision(df, Path(figures_dir) / f"model_size_vs_precision_{dataset}.png")
    except Exception as exc:
        logger.warning(f"Could not generate model-size plot for {dataset}: {exc}")

    try:
        plot_latency_vs_accuracy(df, Path(figures_dir) / f"latency_vs_accuracy_{dataset}.png")
    except Exception as exc:
        logger.warning(f"Could not generate latency-vs-accuracy plot for {dataset}: {exc}")

    pareto_df = df.dropna(subset=["accuracy", "model_size_mb"])
    if len(pareto_df) >= 2:
        mask = find_pareto_frontier(pareto_df, maximize=["accuracy"], minimize=["model_size_mb"])
        plot_pareto_frontier(
            pareto_df, "model_size_mb", "accuracy", mask,
            str(Path(figures_dir) / f"pareto_accuracy_vs_size_{dataset}.png"),
            x_label="Estimated model size (MB)", y_label="Accuracy",
        )
        pareto_df.assign(pareto_optimal=mask).to_csv(
            Path(tables_dir) / f"pareto_accuracy_vs_size_{dataset}.csv", index=False
        )
