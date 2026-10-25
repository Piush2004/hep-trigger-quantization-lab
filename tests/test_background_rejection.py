"""Background-rejection-at-fixed-signal-efficiency metric (Phase 13 baseline work).

Tests the binary primitive directly and the one-vs-rest multiclass adaptation on
synthetic data -- no dataset or checkpoint touched here.
"""
import numpy as np
import pytest

from hep_quant.training.metrics import background_rejection_at_efficiency, macro_background_rejection_at_efficiency


def test_perfect_separation_gives_infinite_rejection():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])  # perfectly separated
    rejection = background_rejection_at_efficiency(y_true, y_score, efficiency=0.70)
    assert rejection == float("inf")


def test_random_scores_give_finite_low_rejection():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000)
    y_score = rng.uniform(size=2000)  # no signal at all
    rejection = background_rejection_at_efficiency(y_true, y_score, efficiency=0.70)
    assert np.isfinite(rejection)
    assert 0 < rejection < 5  # near-random classifier: close to 1/(1-efficiency) ballpark


def test_efficiency_out_of_range_raises():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError):
        background_rejection_at_efficiency(y_true, y_score, efficiency=1.5)


def test_macro_rejects_binary_task():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError):
        macro_background_rejection_at_efficiency(y_true, y_prob, efficiency=0.70)


def test_macro_multiclass_reports_per_class_and_macro():
    rng = np.random.default_rng(1)
    n_classes = 5
    y_true = rng.integers(0, n_classes, size=3000)
    # construct probabilities correlated with the true label so results are non-trivial
    logits = rng.normal(size=(3000, n_classes)) * 0.3
    logits[np.arange(3000), y_true] += 2.0
    y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    result = macro_background_rejection_at_efficiency(y_true, y_prob, efficiency=0.70)
    assert result["efficiency"] == 0.70
    assert set(result["per_class"].keys()) == {0, 1, 2, 3, 4}
    assert result["n_classes_total"] == 5
    assert result["macro"] > 1  # better than random given the injected signal
