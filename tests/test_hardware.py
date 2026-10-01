"""Gate 6: hardware backend abstraction must degrade gracefully when hls4ml is
unavailable, AND must actually work when it is (this environment has hls4ml +
g++ installed, so both paths are exercised for real, not just mocked)."""
import shutil

import pytest

from hep_quant.hardware.latency import benchmark_inference_latency
from hep_quant.hardware.resource_estimator import HLSBackend, UnavailableBackend, get_hardware_backend
from hep_quant.models.mlp import MLPClassifier


def test_unavailable_backend_reports_na_fields():
    backend = UnavailableBackend(reason="test")
    report = backend.analyze(model=None, precision="int8", hardware_cfg={})
    assert report.status == "unavailable"
    assert report.lut is None
    assert report.dsp is None
    assert "test" in report.resource_source


def test_get_hardware_backend_defaults_to_unavailable():
    backend = get_hardware_backend({"backend": "unavailable"})
    assert isinstance(backend, UnavailableBackend)
    assert backend.is_available() is False


def test_get_hardware_backend_unknown_name_falls_back():
    backend = get_hardware_backend({"backend": "something_bogus"})
    assert isinstance(backend, UnavailableBackend)


def test_latency_benchmark_produces_finite_positive_values():
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    report = benchmark_inference_latency(model, input_dim=10, device="cpu", n_warmup=2, n_repeats=5)
    assert report.mean_ms > 0
    assert report.mean_ms_per_sample > 0
    assert report.source == "measured_software"


def _hls4ml_usable() -> bool:
    try:
        import hls4ml  # noqa: F401
    except ImportError:
        return False
    return shutil.which("g++") is not None


@pytest.mark.skipif(not _hls4ml_usable(), reason="hls4ml and/or a C++ toolchain not available")
def test_hls_backend_converts_and_csims_real_model():
    """Exercises the real hls4ml conversion + C-simulation path end-to-end
    (not mocked) against an actual trained-shape MLP. Does NOT assert on
    resource numbers (lut/dsp/bram/ff) since those require Vivado synthesis,
    which this environment does not have — only that conversion/csim succeed
    and the report says so honestly."""
    model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], n_classes=2)
    backend = HLSBackend()
    assert backend.is_available()

    report = backend.analyze(model, "int8", {"reuse_factor": 1})
    assert report.status in ("csim", "converted")
    assert report.lut is None  # no Vivado synthesis in this environment
    assert "hls4ml" in report.resource_source


@pytest.mark.skipif(not _hls4ml_usable(), reason="hls4ml and/or a C++ toolchain not available")
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="no CUDA device to test the GPU-model conversion path"
)
def test_hls_backend_handles_cuda_resident_model():
    """Regression test: hls4ml introspects weights via NumPy, which fails on a
    CUDA tensor with 'can't convert cuda:0 device type tensor to numpy' unless
    HLSBackend converts a CPU copy first (caught via EXP_008_FP32_HLS4ML, whose
    model was trained on CUDA and failed conversion before this fix)."""
    import torch

    model = MLPClassifier(input_dim=16, hidden_dims=[128, 64, 32], n_classes=2).cuda()
    backend = HLSBackend()
    report = backend.analyze(model, "int8", {"reuse_factor": 1})
    assert report.status in ("csim", "converted")
    assert next(model.parameters()).is_cuda  # original model object must be untouched
