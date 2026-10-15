"""Phase 11: wa-hls4ml data adapter schema checks + the surrogate's feature-key contract.

The synthetic fixture (tests/fixtures/wa_hls4ml_sample.json, 2 hand-built records matching the
real schema confirmed by downloading exemplar_models.json) means these tests need no network
access and never touch the real multi-hundred-MB dataset. Real-download tests are gated behind
RUN_NETWORK_TESTS=1, the project's existing convention (same as the OpenML network tests).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hep_quant.data.wa_hls4ml import SchemaMismatchError, check_sample_schema, filter_jet_exemplar
from hep_quant.hardware.resource_estimator import SurrogateEstimateBackend, get_hardware_backend
from hep_quant.hardware.surrogate import features_from_pytorch_model, features_from_wa_hls4ml_record
from hep_quant.models.mlp import MLPClassifier

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wa_hls4ml_sample.json"


def _fixture_records() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_fixture_records_pass_schema_check():
    for record in _fixture_records():
        check_sample_schema(record)  # must not raise


def test_schema_check_fails_loudly_on_missing_keys():
    bad = {"meta_data": {}, "model_config": []}
    with pytest.raises(SchemaMismatchError):
        check_sample_schema(bad)


def test_filter_jet_exemplar_matches_jet_prefixed_names():
    records = _fixture_records()
    jet = filter_jet_exemplar(records)
    assert len(jet) == 1
    assert jet[0]["meta_data"]["model_name"].startswith("Jet")


def test_features_from_wa_hls4ml_record_extracts_expected_values():
    record = _fixture_records()[0]
    feats = features_from_wa_hls4ml_record(record)
    assert feats["n_dense_layers"] == 2.0
    assert feats["reuse_factor"] == 1.0
    assert feats["strategy_is_resource"] == 1.0
    assert feats["width_slot_0"] == 32.0
    assert feats["width_slot_1"] == 5.0
    assert feats["bits_slot_0"] == 16.0
    assert feats["bits_slot_1"] == 16.0


def test_features_from_wa_hls4ml_record_strategy_latency_flag():
    record = _fixture_records()[1]
    feats = features_from_wa_hls4ml_record(record)
    assert feats["strategy_is_resource"] == 0.0
    assert feats["reuse_factor"] == 128.0
    assert feats["n_dense_layers"] == 3.0


def test_features_from_both_sources_share_identical_key_set():
    """The whole backend depends on this: a trained-on-wa-hls4ml model must accept features
    built from this project's own live MLPClassifier, or predictions are meaningless."""
    wa_feats = features_from_wa_hls4ml_record(_fixture_records()[0])

    model = MLPClassifier(input_dim=16, hidden_dims=[32], n_classes=5)
    pt_feats = features_from_pytorch_model(model, {0: 8, 1: 8}, hardware_cfg={"reuse_factor": 1})

    assert set(wa_feats.keys()) == set(pt_feats.keys())


def test_surrogate_backend_falls_back_when_model_dir_missing(tmp_path):
    backend = SurrogateEstimateBackend(model_dir=tmp_path / "does_not_exist")
    assert backend.is_available() is False
    report = backend.analyze(MLPClassifier(input_dim=16, hidden_dims=[32], n_classes=5), "int8", {})
    assert report.measurement_tier == "unavailable"


def test_get_hardware_backend_surrogate_dispatch_falls_back_honestly():
    backend = get_hardware_backend({"backend": "surrogate"})
    # No trained surrogate present in a fresh checkout -> honest fallback, never a crash.
    assert backend.name in ("software_estimate", "surrogate")


@pytest.mark.skipif(not os.environ.get("RUN_NETWORK_TESTS"), reason="requires network + large download")
def test_real_exemplar_download_matches_schema():
    from hep_quant.data.wa_hls4ml import load_exemplar_records

    records = load_exemplar_records()
    assert len(records) > 0
    jet = filter_jet_exemplar(records)
    assert len(jet) > 0
