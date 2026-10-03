"""Aggregate all experiment records into results_summary.csv, tables, and Pareto plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hep_quant.experiments.registry import ExperimentRegistry
from hep_quant.utils.logging import get_logger
from hep_quant.visualization.pareto import find_pareto_frontier, plot_pareto_frontier
from hep_quant.visualization.plots import plot_latency_vs_accuracy, plot_model_size_vs_precision, plot_multiseed

logger = get_logger(__name__)

REPLICATE_PATTERN = r"^EXP_[SC]\d+_"  # EXP_S<seed>_ = seed replicates, EXP_C<seed>_ = compute-matched controls
SUMMARY_METRICS = ["accuracy", "f1", "roc_auc", "pr_auc"]

RESULTS_COLUMNS = [
    "experiment_id", "architecture", "dataset", "dataset_version", "subset_size", "precision", "method", "parameters",
    "theoretical_weight_storage_mb", "accuracy", "precision_score", "recall", "f1", "roc_auc", "pr_auc",
    "inference_latency_ms", "inference_device", "latency_unit", "lut", "dsp", "bram", "ff",
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
        "dataset_version": record.get("config", {}).get("dataset", {}).get("openml_version"),
        "subset_size": record.get("config", {}).get("dataset", {}).get("subset_size"),
        "precision": model_info.get("precision"),
        "method": model_info.get("method"),
        "parameters": model_info.get("n_parameters"),
        "theoretical_weight_storage_mb": metrics.get("model_size", {}).get("estimated_size_mb"),
        "accuracy": test_metrics.get("accuracy"),
        "precision_score": test_metrics.get("precision"),
        "recall": test_metrics.get("recall"),
        "f1": test_metrics.get("f1"),
        "roc_auc": test_metrics.get("roc_auc"),
        "pr_auc": test_metrics.get("pr_auc"),
        "inference_latency_ms": metrics.get("latency", {}).get("mean_ms_per_sample"),
        "inference_device": metrics.get("latency", {}).get("device"),
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
    # Multi-seed replicates (EXP_S<seed>_...) are aggregated into mean/std
    # summaries below, not plotted as individual points next to single runs.
    is_replicate = df["experiment_id"].str.match(REPLICATE_PATTERN, na=False)
    comparable_df = df[~df["experiment_id"].str.contains("VALIDATE", na=False) & ~is_replicate]

    datasets = sorted(d for d in comparable_df["dataset"].dropna().unique())
    for dataset in datasets:
        subset = comparable_df[comparable_df["dataset"] == dataset]
        _generate_dataset_report(subset, dataset, figures_dir, tables_dir)

    for dataset in sorted(d for d in df[is_replicate]["dataset"].dropna().unique()):
        summary = summarize_seeds(df[is_replicate & (df["dataset"] == dataset)])
        summary.to_csv(Path(tables_dir) / f"multiseed_summary_{dataset}.csv", index=False)
        with open(Path(tables_dir) / f"multiseed_summary_{dataset}.md", "w") as f:
            f.write(f"# Multi-seed summary — {dataset}\n\n" + _summary_markdown(summary) + "\n")
        plot_multiseed(summary, dataset, figures_dir)

    return df


def _generate_dataset_report(df: pd.DataFrame, dataset: str, figures_dir: str, tables_dir: str) -> None:
    markdown_table = df[
        ["experiment_id", "precision", "method", "accuracy", "f1", "roc_auc", "theoretical_weight_storage_mb", "inference_latency_ms"]
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

    pareto_df = df.dropna(subset=["accuracy", "theoretical_weight_storage_mb"])
    if len(pareto_df) >= 2:
        mask = find_pareto_frontier(pareto_df, maximize=["accuracy"], minimize=["theoretical_weight_storage_mb"])
        plot_pareto_frontier(
            pareto_df, "theoretical_weight_storage_mb", "accuracy", mask,
            str(Path(figures_dir) / f"pareto_accuracy_vs_size_{dataset}.png"),
            x_label="Theoretical weight storage (MB)", y_label="Accuracy",
        )
        pareto_df.assign(pareto_optimal=mask).to_csv(
            Path(tables_dir) / f"pareto_accuracy_vs_size_{dataset}.csv", index=False
        )


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) across seeds per (precision, method) configuration.

    Std is NaN (shown as N/A) when only one seed exists for a configuration.
    Model size, parameter count and the hardware columns are deterministic
    functions of architecture + precision, so the first value is carried.
    """
    rows = []
    for (precision, method), g in df.groupby(["precision", "method"], sort=False):
        row = {"precision": precision, "method": method, "n_seeds": int(g["seed"].nunique()),
               "seeds": ",".join(str(x) for x in sorted(g["seed"].unique())),
               "subset_size": g["subset_size"].iloc[0], "parameters": g["parameters"].iloc[0],
               "theoretical_weight_storage_mb": g["theoretical_weight_storage_mb"].iloc[0]}
        for m in SUMMARY_METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else float("nan")
        row["latency_ms_mean"] = g["inference_latency_ms"].mean()
        row["latency_ms_std"] = g["inference_latency_ms"].std(ddof=1) if len(g) > 1 else float("nan")
        row["inference_device"] = ",".join(sorted(g["inference_device"].dropna().unique()))
        for col in ("lut", "dsp", "bram", "ff"):
            row[col] = g[col].iloc[0]
        row["hardware_measurement_tier"] = g["hardware_measurement_tier"].iloc[0]
        rows.append(row)
    order = {"fp32": 0, "int8": 1, "int6": 2, "int4": 3}
    out = pd.DataFrame(rows)
    out["_o"] = out["precision"].map(order)
    return out.sort_values(["method", "_o"]).drop(columns="_o").reset_index(drop=True)


def _summary_markdown(summary: pd.DataFrame) -> str:
    view = pd.DataFrame({
        "precision": summary["precision"], "method": summary["method"], "seeds": summary["seeds"],
        "accuracy (mean ± std)": [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f} ± N/A" for m, s in zip(summary["accuracy_mean"], summary["accuracy_std"])],
        "F1 macro": [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}" for m, s in zip(summary["f1_mean"], summary["f1_std"])],
        "ROC-AUC macro": [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}" for m, s in zip(summary["roc_auc_mean"], summary["roc_auc_std"])],
        "theoretical weight storage MB": summary["theoretical_weight_storage_mb"].map(lambda v: f"{v:.5f}"),
        "latency ms/sample (software, device)": [f"{m:.3f} ({d})" for m, d in zip(summary["latency_ms_mean"], summary["inference_device"])],
    })
    return view.to_markdown(index=False)
