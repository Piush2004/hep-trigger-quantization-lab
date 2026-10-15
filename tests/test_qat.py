"""Gate 5: Quantization-Aware Training tests."""
import numpy as np
import pytest
import torch

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.models.cnn import JetImageCNN
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.analysis import count_parameters, estimate_model_size
from hep_quant.quantization.qat import QATActivation, QATConv2d, QATLinear, prepare_qat_model
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


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_qat_model_size_matches_fp32_parameter_count(precision):
    """Regression test: QATLinear isn't an nn.Linear instance, so a naive
    isinstance-based parameter counter silently reports 0 weight params for
    QAT models (caught via EXP_005_INT8_QAT reporting estimated_size_mb=0.0)."""
    model, *_ = _make_trained_model_and_data()
    qat_model = prepare_qat_model(model, precision)

    fp32_total, fp32_weights, fp32_bias = count_parameters(model)
    qat_total, qat_weights, qat_bias = count_parameters(qat_model)
    assert qat_total == fp32_total
    assert qat_weights == fp32_weights
    assert qat_bias == fp32_bias

    size_report = estimate_model_size(qat_model, precision)
    assert size_report.estimated_size_mb > 0


def test_qat_activation_tracks_range_only_in_training_mode():
    act = QATActivation(torch.nn.ReLU(), bits=8)
    act.train()
    act(torch.tensor([[1.0, 5.0, -2.0]]))
    assert bool(act.initialized)
    recorded_max = act.running_max.item()

    act.eval()
    act(torch.tensor([[100.0]]))  # should NOT update range in eval mode
    assert act.running_max.item() == recorded_max


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_prepare_qat_model_works_for_cnn_two_block_structure(precision):
    """prepare_qat_model previously only handled a single flat model.net Sequential --
    JetImageCNN has two separate blocks (model.conv, model.classifier). This is the Phase 14
    generalization, via QUANT_BLOCKS, mirrored from the equivalent ptq.py fix."""
    set_seed(42)
    model = JetImageCNN(image_size=8, in_channels=1, conv_channels=[4, 8], n_classes=3)
    qat_model = prepare_qat_model(model, precision)

    conv_qat_layers = [m for m in qat_model.conv if isinstance(m, QATConv2d)]
    classifier_qat_layers = [m for m in qat_model.classifier if isinstance(m, QATLinear)]
    orig_convs = [m for m in model.conv if isinstance(m, torch.nn.Conv2d)]
    orig_linears = [m for m in model.classifier if isinstance(m, torch.nn.Linear)]

    assert len(conv_qat_layers) == len(orig_convs) == 2
    assert len(classifier_qat_layers) == len(orig_linears) == 2
    for orig, qat in zip(orig_convs, conv_qat_layers):
        assert torch.allclose(orig.weight, qat.weight)

    x = torch.randn(3, 1, 8, 8)
    out = qat_model(x)
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()


def test_prepare_qat_model_mlp_path_unchanged_after_cnn_generalization():
    """Regression guard: generalizing prepare_qat_model for JetImageCNN's QUANT_BLOCKS must not
    change behavior for MLPClassifier (QUANT_BLOCKS=("net",) should reduce to the original
    single-block logic exactly)."""
    model, *_ = _make_trained_model_and_data()
    qat_model = prepare_qat_model(model, "int8")
    qat_linears = [m for m in qat_model.net if isinstance(m, QATLinear)]
    orig_linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(qat_linears) == len(orig_linears)
