"""Publication-quality research figures. All axes are labeled with units."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")


def plot_training_curves(history: dict, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epochs"], history["train_loss"], label="train")
    axes[0].plot(history["epochs"], history["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(history["epochs"], history["val_accuracy"], label="accuracy")
    axes[1].plot(history["epochs"], history["val_f1"], label="F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, auc: float, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AUC = {auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_vs_precision(df, metric_col: str, out_path: str | Path, ylabel: str | None = None) -> None:
    """Line plot of `metric_col` across precisions, one line per method.

    Multiple rows can share a precision (e.g. "int8" appears for both PTQ and
    QAT), so sorting is done per-method-group by precision order rather than
    via a single reindex over the whole (non-unique) precision column.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["fp32", "int8", "int6", "int4"]
    order_index = {p: i for i, p in enumerate(order)}

    methods = df["method"].unique() if "method" in df.columns else [None]
    for method in methods:
        subset = df[df["method"] == method] if method else df
        subset = subset.sort_values(by="precision", key=lambda col: col.map(order_index))
        ax.plot(subset["precision"], subset[metric_col], marker="o", label=method or metric_col)

    ax.set_xlabel("Quantization precision")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(f"{ylabel or metric_col} vs Precision")
    if "method" in df.columns and df["method"].nunique() > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_size_vs_precision(df, out_path: str | Path) -> None:
    plot_metric_vs_precision(df, "model_size_mb", out_path, ylabel="Model size (MB)")


def plot_latency_vs_accuracy(df, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for precision in df["precision"].unique():
        subset = df[df["precision"] == precision]
        ax.scatter(subset["inference_latency_ms"], subset["accuracy"], label=precision, s=80)
    ax.set_xlabel("Mean inference latency (ms/sample, software-measured)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Latency vs Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hardware_resource_vs_accuracy(df, resource_col: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = df.dropna(subset=[resource_col])
    if valid.empty:
        ax.text(0.5, 0.5, "No hardware resource data available\n(hls4ml backend unavailable)",
                ha="center", va="center", transform=ax.transAxes)
    else:
        for precision in valid["precision"].unique():
            subset = valid[valid["precision"] == precision]
            ax.scatter(subset[resource_col], subset["accuracy"], label=precision, s=80)
        ax.legend()
    ax.set_xlabel(f"Estimated {resource_col.upper()} usage")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Estimated {resource_col.upper()} vs Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
