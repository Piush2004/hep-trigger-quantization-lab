"""Classification metrics for binary and multiclass HEP-trigger-style classification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.preprocessing import label_binarize


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    averaging: str = "binary"  # "binary" | "macro" -- which definition produced precision/recall/f1/roc_auc/pr_auc

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "averaging": self.averaging,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> ClassificationMetrics:
    """Compute accuracy/precision/recall/F1 (+ ROC-AUC/PR-AUC when `y_prob` is given).

    Binary (exactly 2 classes present in `y_true`): `y_prob` is the
    positive-class probability (1D array), and precision/recall/F1/ROC-AUC/
    PR-AUC use the standard binary definitions.

    Multiclass (>2 classes): `y_prob` must be the full `(n_samples, n_classes)`
    probability matrix. precision/recall/F1 use macro-averaging (unweighted
    mean across classes, so a rare class counts as much as a common one — the
    right choice for a 5-way jet-tagging problem where classes needn't be
    balanced). ROC-AUC uses one-vs-rest macro-averaging; PR-AUC is
    macro-averaged `average_precision_score` over one-hot-encoded labels.
    Which definition was used is recorded in `averaging` so a reported number
    is never ambiguous.
    """
    n_classes = len(np.unique(y_true))
    is_multiclass = n_classes > 2
    average = "macro" if is_multiclass else "binary"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    roc_auc = None
    pr_auc = None
    if y_prob is not None and n_classes > 1:
        try:
            if is_multiclass:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")
            else:
                roc_auc = roc_auc_score(y_true, y_prob)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = sk_auc(recall_curve, precision_curve)
        except ValueError:
            # e.g. a class entirely absent from a small eval batch -- leave as
            # None rather than crash the whole evaluation over it.
            pass

    return ClassificationMetrics(accuracy, precision, recall, f1, roc_auc, pr_auc, averaging=average)


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary-only: ROC curve points. For multiclass, compute one-vs-rest per class instead."""
    return roc_curve(y_true, y_prob)


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary-only: PR curve points. For multiclass, compute one-vs-rest per class instead."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred)
