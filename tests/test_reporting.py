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
            "theoretical_weight_storage_mb": [0.05, 0.02, 0.015, 0.01][:n],
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
            "theoretical_weight_storage_mb": [0.017, 0.017, 0.013],
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


def _replicate_df() -> pd.DataFrame:
    rows = []
    for seed, acc in [(42, 0.750), (123, 0.760), (2026, 0.740)]:
        for prec, method, a, size in [("fp32", "none", acc, 0.0167), ("int4", "ptq", acc - 0.10, 0.0025), ("int4", "qat", acc - 0.02, 0.0025)]:
            rows.append({
                "experiment_id": f"EXP_S{seed}_JET_{prec.upper()}_{method.upper()}", "dataset": "hls4ml_lhc_jets_hlf",
                "precision": prec, "method": method, "seed": seed, "subset_size": 100000, "parameters": 4389,
                "accuracy": a, "f1": a, "roc_auc": a + 0.18, "pr_auc": a, "theoretical_weight_storage_mb": size,
                "inference_latency_ms": 0.5, "inference_device": "cuda", "lut": 1000 * (4 if prec == "int4" else 32),
                "dsp": 12000, "bram": 0, "ff": 10, "hardware_measurement_tier": "estimated",
            })
    return pd.DataFrame(rows)


def test_summarize_seeds_mean_and_sample_std():
    summary = reporting.summarize_seeds(_replicate_df())
    fp32 = summary[(summary.precision == "fp32") & (summary.method == "none")].iloc[0]
    assert fp32["n_seeds"] == 3
    assert abs(fp32["accuracy_mean"] - 0.750) < 1e-12
    assert abs(fp32["accuracy_std"] - 0.010) < 1e-12  # sample std (ddof=1) of 0.74/0.75/0.76
    assert fp32["hardware_measurement_tier"] == "estimated"


def test_summarize_seeds_single_seed_has_nan_std():
    one = _replicate_df().query("seed == 42")
    summary = reporting.summarize_seeds(one)
    assert summary["accuracy_std"].isna().all()
    assert "N/A" in reporting._summary_markdown(summary)


def test_generate_report_aggregates_replicates_separately(tmp_path, monkeypatch):
    df = _replicate_df()
    for col in ("architecture", "recall", "precision_score", "latency_unit", "resource_source", "training_time_s", "git_commit", "dataset_version"):
        df[col] = None
    monkeypatch.setattr(reporting, "build_results_dataframe", lambda results_dir: df)
    figs, tabs = tmp_path / "f", tmp_path / "t"
    generate_report(results_dir="x", out_csv=str(tmp_path / "r.csv"), figures_dir=str(figs), tables_dir=str(tabs))

    assert (tabs / "multiseed_summary_hls4ml_lhc_jets_hlf.csv").exists()
    for name in ("precision", "size", "latency", "estimated_lut"):
        assert (figs / f"multiseed_accuracy_vs_{name}_hls4ml_lhc_jets_hlf.png").exists()
    # replicates must NOT also appear as individual single-run Pareto points
    assert not (tabs / "pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.csv").exists()


def test_control_runs_are_treated_as_replicates():
    import re

    assert re.match(reporting.REPLICATE_PATTERN, "EXP_S7_JET_FP32")
    assert re.match(reporting.REPLICATE_PATTERN, "EXP_C99_JET_FP32_FT")
    assert not re.match(reporting.REPLICATE_PATTERN, "EXP_012_JET_FP32")
