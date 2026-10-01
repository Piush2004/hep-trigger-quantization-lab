"""Classification metrics for binary HEP-trigger-style classification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> ClassificationMetrics:
    """Compute accuracy/precision/recall/F1, plus ROC-AUC/PR-AUC when `y_prob` (positive-class
    probability) is given and both classes are present in `y_true`."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = sk_auc(recall_curve, precision_curve)

    return ClassificationMetrics(accuracy, precision, recall, f1, roc_auc, pr_auc)


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y_true, y_prob)


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred)
