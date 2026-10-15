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


def background_rejection_at_efficiency(
    y_true_binary: np.ndarray, y_score: np.ndarray, efficiency: float = 0.70
) -> float:
    """1 / FPR at the given signal efficiency (TPR) working point, binary-only.

    Standard HEP convention: at a fixed signal efficiency, how much background is
    rejected. `np.interp` on the ROC curve's (TPR -> FPR) gives the FPR at exactly
    `efficiency` (ROC curves are monotonically non-decreasing in both axes as the
    threshold sweeps, so interpolation is well-defined); returns `inf` if FPR is 0
    at that efficiency (perfect rejection — reported as-is, not silently capped).
    """
    if not 0 < efficiency < 1:
        raise ValueError(f"efficiency must be in (0, 1), got {efficiency}")
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    fpr_at_eff = float(np.interp(efficiency, tpr, fpr))
    return float("inf") if fpr_at_eff == 0 else 1.0 / fpr_at_eff


def macro_background_rejection_at_efficiency(
    y_true: np.ndarray, y_prob: np.ndarray, efficiency: float = 0.70
) -> dict:
    """One-vs-rest background rejection at a fixed signal efficiency, per class and macro-averaged.

    This is a one-vs-rest *adaptation* of a concept defined for binary signal-vs-
    background searches: this is a 5-way jet-tagging task with no single "signal"
    class, so each class is treated as signal against the other four as background
    in turn, and the macro mean is reported alongside the per-class breakdown so the
    adaptation is visible, not hidden behind one averaged number.
    """
    classes = np.unique(y_true)
    if len(classes) < 3:
        raise ValueError("macro_background_rejection_at_efficiency assumes a multiclass (>=3 class) "
                          "task; for a true binary problem, call background_rejection_at_efficiency directly")
    y_true_bin = label_binarize(y_true, classes=classes)  # (n_samples, n_classes) one-hot, unambiguous for >=3 classes
    per_class = {}
    for i, c in enumerate(classes):
        per_class[int(c)] = background_rejection_at_efficiency(y_true_bin[:, i], y_prob[:, i], efficiency)
    finite = [v for v in per_class.values() if np.isfinite(v)]
    macro = float(np.mean(finite)) if finite else float("inf")
    return {"efficiency": efficiency, "per_class": per_class, "macro": macro, "n_classes_finite": len(finite), "n_classes_total": len(classes)}
