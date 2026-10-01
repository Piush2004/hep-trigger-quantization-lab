"""Reporting: per-dataset output splitting (never silently merge datasets in a plot)."""
from pathlib import Path

import pandas as pd

import hep_quant.reporting as reporting
from hep_quant.reporting import _generate_dataset_report, generate_report


def _fake_df(dataset: str, accuracies: list[float]) -> pd.DataFrame:
    n = len(accuracies)
    return pd.DataFrame(
        {
            "experiment_id": [f"{dataset}_{i}" for i in range(n)],
            "precision": ["fp32", "int8", "int6", "int4"][:n],
            "method": ["none", "ptq", "ptq", "ptq"][:n],
            "accuracy": accuracies,
            "f1": accuracies,
            "roc_auc": accuracies,
            "model_size_mb": [0.05, 0.02, 0.015, 0.01][:n],
            "inference_latency_ms": [0.5, 1.0, 1.0, 1.0][:n],
        }
    )


def test_dataset_reports_are_filename_tagged_and_separate(tmp_path):
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    figures_dir.mkdir()
    tables_dir.mkdir()

    synthetic_df = _fake_df("synthetic", [0.98, 0.98, 0.97, 0.96])
    real_df = _fake_df("hls4ml_lhc_jets_hlf", [0.75, 0.75, 0.74, 0.65])

    _generate_dataset_report(synthetic_df, "synthetic", str(figures_dir), str(tables_dir))
    _generate_dataset_report(real_df, "hls4ml_lhc_jets_hlf", str(figures_dir), str(tables_dir))

    # Each dataset gets its own figures/tables -- never one file holding both.
    assert (figures_dir / "pareto_accuracy_vs_size_synthetic.png").exists()
    assert (figures_dir / "pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.png").exists()
    assert (tables_dir / "quantization_results_synthetic.md").exists()
    assert (tables_dir / "quantization_results_hls4ml_lhc_jets_hlf.md").exists()

    # The two datasets' numbers must not appear in each other's table.
    synthetic_table = (tables_dir / "quantization_results_synthetic.md").read_text()
    real_table = (tables_dir / "quantization_results_hls4ml_lhc_jets_hlf.md").read_text()
    assert "0.9800" in synthetic_table
    assert "0.9800" not in real_table
    assert "0.6500" in real_table
    assert "0.6500" not in synthetic_table


def test_staged_validation_runs_excluded_from_comparison_plots(tmp_path, monkeypatch):
    """EXP_0xx_..._VALIDATE_10K/50K runs are deliberately under-trained
    pipeline/training sanity checks at a smaller subset -- they must stay in
    results_summary.csv (the full record) but must NOT appear in the
    comparison Pareto table, since they're not comparable points in the
    precision-comparison matrix."""
    mixed_df = pd.DataFrame(
        {
            "experiment_id": ["EXP_010_JET_VALIDATE_10K", "EXP_012_JET_FP32", "EXP_013_JET_INT8_PTQ"],
            "dataset": ["hls4ml_lhc_jets_hlf"] * 3,
            "precision": ["fp32", "fp32", "int8"],
            "method": ["none", "none", "ptq"],
            "accuracy": [0.55, 0.75, 0.75],
            "f1": [0.54, 0.75, 0.75],
            "roc_auc": [0.82, 0.93, 0.93],
            "model_size_mb": [0.017, 0.017, 0.013],
            "inference_latency_ms": [0.1, 0.1, 0.1],
        }
    )
    monkeypatch.setattr(reporting, "build_results_dataframe", lambda results_dir: mixed_df)

    out_csv = tmp_path / "results_summary.csv"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"

    generate_report(results_dir="unused", out_csv=str(out_csv), figures_dir=str(figures_dir), tables_dir=str(tables_dir))

    full_csv = pd.read_csv(out_csv)
    assert "EXP_010_JET_VALIDATE_10K" in full_csv["experiment_id"].values  # kept in the raw record

    pareto_csv = pd.read_csv(tables_dir / "pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.csv")
    assert "EXP_010_JET_VALIDATE_10K" not in pareto_csv["experiment_id"].values  # excluded from comparison
