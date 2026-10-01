"""Multiclass generalization of compute_metrics (Phase 2: 5-class jet tagging)."""
import numpy as np
import pytest
import torch

from hep_quant.models.mlp import MLPClassifier
from hep_quant.training.evaluation import evaluate_model
from hep_quant.training.metrics import compute_metrics


def test_binary_path_unchanged():
    """Regression guard: the pre-existing binary behavior must not change."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(size=200)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics.averaging == "binary"
    assert metrics.roc_auc is not None
    assert metrics.pr_auc is not None
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1


def test_multiclass_uses_macro_averaging():
    rng = np.random.default_rng(42)
    n_classes = 5
    y_true = rng.integers(0, n_classes, size=500)
    logits = rng.normal(size=(500, n_classes))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, probs)
    assert metrics.averaging == "macro"
    assert metrics.roc_auc is not None
    assert metrics.pr_auc is not None
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1
    assert 0 <= metrics.roc_auc <= 1
    assert 0 <= metrics.pr_auc <= 1


def test_multiclass_perfect_predictions_score_one():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = y_true.copy()
    probs = np.eye(5)[y_true]  # one-hot "perfectly confident, perfectly correct"

    metrics = compute_metrics(y_true, y_pred, probs)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.pr_auc == 1.0


def test_multiclass_macro_weights_rare_class_equally():
    """A model that gets a rare class entirely wrong should be penalized just
    as much (macro) as getting a common class wrong -- NOT washed out by
    micro/weighted averaging, which is the whole point of using macro here.
    Needs >=3 classes: with exactly 2, compute_metrics intentionally uses
    sklearn's binary-averaging path (preserves the original Phase-1 binary
    behavior unchanged) rather than macro -- see compute_metrics' docstring."""
    # 90 of class 0 (all correct), 5 of class 1 (all correct), 5 of class 2 (all wrong).
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    y_pred = np.array([0] * 90 + [1] * 5 + [0] * 5)  # class 2 never predicted

    metrics = compute_metrics(y_true, y_pred, None)
    # Macro recall = mean(recall_class0=1.0, recall_class1=1.0, recall_class2=0.0) = 2/3,
    # which would be ~0.95 under micro/weighted averaging -- macro must differ.
    assert metrics.recall == pytest.approx(2 / 3)


def test_evaluate_model_multiclass_end_to_end():
    model = MLPClassifier(input_dim=16, hidden_dims=[32, 16], n_classes=5)
    X = np.random.default_rng(42).normal(size=(200, 16)).astype(np.float32)
    y = np.random.default_rng(42).integers(0, 5, size=200).astype(np.int64)

    result = evaluate_model(model, X, y, device="cpu")
    assert result.y_prob.shape == (200, 5)  # full probability matrix, not positive-class-only
    assert result.metrics.averaging == "macro"
    assert 0 <= result.metrics.accuracy <= 1
