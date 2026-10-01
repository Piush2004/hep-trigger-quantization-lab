"""Gate 6: hardware backend abstraction must degrade gracefully when hls4ml is unavailable."""
from hep_quant.hardware.latency import benchmark_inference_latency
from hep_quant.hardware.resource_estimator import UnavailableBackend, get_hardware_backend
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
