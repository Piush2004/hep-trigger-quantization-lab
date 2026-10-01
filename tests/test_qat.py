"""Gate 5: Quantization-Aware Training tests."""
import numpy as np
import pytest
import torch

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.qat import QATActivation, QATLinear, prepare_qat_model
from hep_quant.training.evaluation import evaluate_model
from hep_quant.training.trainer import Trainer
from hep_quant.utils.seeds import set_seed


def _make_trained_model_and_data():
    set_seed(42)
    ds = load_synthetic(n_samples=1500, n_features=10, seed=42)
    splits = split_dataset(ds.X, ds.y, seed=42)
    pre = Preprocessor.fit(splits.X_train)
    X_train = pre.transform(splits.X_train)
    X_val = pre.transform(splits.X_val)
    X_test = pre.transform(splits.X_test)
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    trainer = Trainer(model, {"epochs": 3, "batch_size": 128, "learning_rate": 0.01, "device": "cpu"}, "/tmp")
    trainer.fit(X_train, splits.y_train, X_val, splits.y_val)
    return model, X_train, splits.y_train, X_val, splits.y_val, X_test, splits.y_test


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_prepare_qat_model_starts_from_fp32_weights(precision):
    model, *_ = _make_trained_model_and_data()
    qat_model = prepare_qat_model(model, precision)

    orig_linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    qat_linears = [m for m in qat_model.net if isinstance(m, QATLinear)]
    assert len(orig_linears) == len(qat_linears)
    for orig, qat in zip(orig_linears, qat_linears):
        assert torch.allclose(orig.weight, qat.weight)


def test_prepare_qat_model_rejects_fp32():
    model, *_ = _make_trained_model_and_data()
    with pytest.raises(ValueError):
        prepare_qat_model(model, "fp32")


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_qat_finetuning_runs_without_nans(precision):
    model, X_train, y_train, X_val, y_val, X_test, y_test = _make_trained_model_and_data()
    qat_model = prepare_qat_model(model, precision)

    trainer = Trainer(qat_model, {"epochs": 3, "batch_size": 128, "learning_rate": 0.001, "device": "cpu"}, "/tmp")
    history = trainer.fit(X_train, y_train, X_val, y_val)

    assert all(np.isfinite(loss) for loss in history.train_loss)
    assert all(np.isfinite(loss) for loss in history.val_loss)


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_qat_model_evaluates_with_finite_metrics(precision):
    model, X_train, y_train, X_val, y_val, X_test, y_test = _make_trained_model_and_data()
    qat_model = prepare_qat_model(model, precision)
    result = evaluate_model(qat_model, X_test, y_test, device="cpu")
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_qat_activation_tracks_range_only_in_training_mode():
    act = QATActivation(torch.nn.ReLU(), bits=8)
    act.train()
    act(torch.tensor([[1.0, 5.0, -2.0]]))
    assert bool(act.initialized)
    recorded_max = act.running_max.item()

    act.eval()
    act(torch.tensor([[100.0]]))  # should NOT update range in eval mode
    assert act.running_max.item() == recorded_max
