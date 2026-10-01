"""Gate 4: PTQ tests across FP32/INT8/INT6/INT4."""
import numpy as np
import pytest
import torch

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.analysis import estimate_model_size
from hep_quant.quantization.calibration import calibrate_activation_ranges
from hep_quant.quantization.ptq import SUPPORTED_PRECISIONS, quantize_model_ptq
from hep_quant.training.evaluation import evaluate_model
from hep_quant.utils.seeds import set_seed


def _make_model_and_data():
    set_seed(42)
    ds = load_synthetic(n_samples=1500, n_features=10, seed=42)
    splits = split_dataset(ds.X, ds.y, seed=42)
    pre = Preprocessor.fit(splits.X_train)
    X_train = pre.transform(splits.X_train)
    X_test = pre.transform(splits.X_test)
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    return model, X_train, X_test, splits.y_test


def test_calibration_produces_finite_ranges():
    model, X_train, _, _ = _make_model_and_data()
    ranges = calibrate_activation_ranges(model, X_train[:200])
    assert len(ranges) > 0
    for lo, hi in ranges.values():
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi >= lo


@pytest.mark.parametrize("precision", list(SUPPORTED_PRECISIONS))
def test_quantize_model_runs_and_produces_finite_output(precision):
    model, X_train, X_test, y_test = _make_model_and_data()
    quantized = quantize_model_ptq(model, precision, X_train[:200])

    quantized.eval()
    with torch.no_grad():
        out = quantized(torch.from_numpy(X_test[:32]))
    assert out.shape == (32, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("precision", list(SUPPORTED_PRECISIONS))
def test_quantize_model_metrics_computable(precision):
    model, X_train, X_test, y_test = _make_model_and_data()
    quantized = quantize_model_ptq(model, precision, X_train[:200])
    result = evaluate_model(quantized, X_test, y_test, device="cpu")
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_fp32_quantize_is_identity():
    model, X_train, X_test, _ = _make_model_and_data()
    quantized = quantize_model_ptq(model, "fp32", X_train[:200])
    model.eval()
    quantized.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_test[:16])
        assert torch.allclose(model(x), quantized(x))


def test_lower_precision_has_smaller_estimated_size():
    model, _, _, _ = _make_model_and_data()
    sizes = {p: estimate_model_size(model, p).estimated_size_mb for p in SUPPORTED_PRECISIONS}
    assert sizes["int8"] < sizes["fp32"]
    assert sizes["int6"] < sizes["int8"]
    assert sizes["int4"] < sizes["int6"]


def test_quantization_reproducible():
    """Running PTQ twice with the same calibration data must give identical quantized weights."""
    model, X_train, _, _ = _make_model_and_data()
    q1 = quantize_model_ptq(model, "int8", X_train[:200])
    q2 = quantize_model_ptq(model, "int8", X_train[:200])
    for p1, p2 in zip(q1.parameters(), q2.parameters()):
        assert torch.allclose(p1, p2)
