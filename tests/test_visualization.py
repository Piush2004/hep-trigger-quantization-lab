"""Visualization: plot generation must not crash on realistic result shapes."""
import pandas as pd

from hep_quant.visualization.pareto import find_pareto_frontier
from hep_quant.visualization.plots import plot_latency_vs_accuracy, plot_metric_vs_precision


def _sample_results_df() -> pd.DataFrame:
    # Mirrors reality: the same precision ("int8") appears for both PTQ and
    # QAT, which previously broke plot_metric_vs_precision's reindex (it
    # requires a unique index) -- see analysis.py / reporting.py bugfix.
    return pd.DataFrame(
        {
            "precision": ["fp32", "int8", "int6", "int4", "int8", "int6", "int4"],
            "method": ["none", "ptq", "ptq", "ptq", "qat", "qat", "qat"],
            "theoretical_weight_storage_mb": [0.048, 0.0126, 0.0097, 0.0068, 0.0126, 0.0097, 0.0068],
            "accuracy": [0.985, 0.985, 0.984, 0.983, 0.986, 0.986, 0.986],
            "inference_latency_ms": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        }
    )


def test_plot_metric_vs_precision_handles_duplicate_precisions(tmp_path):
    df = _sample_results_df()
    out_path = tmp_path / "size.png"
    plot_metric_vs_precision(df, "theoretical_weight_storage_mb", out_path, ylabel="Theoretical weight storage (MB)")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_latency_vs_accuracy_runs(tmp_path):
    df = _sample_results_df()
    out_path = tmp_path / "latency.png"
    plot_latency_vs_accuracy(df, out_path)
    assert out_path.exists()


def test_find_pareto_frontier_identifies_non_dominated_rows():
    df = _sample_results_df()
    mask = find_pareto_frontier(df, maximize=["accuracy"], minimize=["theoretical_weight_storage_mb"])
    # QAT int8 (acc=0.986, size=0.0126) dominates PTQ int8 (acc=0.985, size=0.0126):
    # same size, strictly higher accuracy -> PTQ int8 (row index 1) must NOT be on the frontier.
    assert mask.loc[1] == False  # noqa: E712
