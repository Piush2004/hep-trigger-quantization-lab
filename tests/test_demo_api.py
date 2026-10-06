"""Demo layer: the result explorer serves exported frozen results only and keeps the scientific rules visible.

Skips cleanly when the optional demo dependencies (FastAPI, httpx) are not installed (see demo/requirements.txt).
"""
import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo" / "app"))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)
DATA = ROOT / "demo" / "data"
EXPLORATORY_ROC = 0.9278


def test_health_declares_no_runtime_training():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["training_at_runtime"] is False


def test_all_exported_files_exist_and_parse():
    for name in ["metadata", "baseline", "quantization", "pareto", "architecture", "calibration", "hls4ml"]:
        json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def test_index_and_static_assets_are_served():
    assert client.get("/").status_code == 200
    assert "HEP-Trigger Quantization Lab" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_exploratory_biased_mixed_result_never_public():
    rows = client.get("/api/results").json()["rows"]
    for r in rows:
        assert "EXPLORATORY" not in json.dumps(r)
        roc = r.get("roc_auc_mean")
        if isinstance(roc, float) and "Mixed" in r["configuration"]:
            assert not math.isclose(roc, EXPLORATORY_ROC, abs_tol=1e-9)
    pareto = client.get("/api/pareto").json()["points"]
    assert all("EXPLORATORY" not in json.dumps(p) for p in pareto)


def test_every_result_row_has_device_and_tier():
    rows = client.get("/api/results").json()["rows"]
    assert rows
    for r in rows:
        assert r["device"]
        assert r["tier"] in {"MEASURED", "SIMULATED", "ESTIMATED", "NOT AVAILABLE"}


def test_architecture_devices_are_not_merged():
    body = client.get("/api/architecture-comparison").json()
    labels = [p["label"] for p in body["panels"]]
    assert len(body["panels"]) == 3
    assert any("CUDA" in l for l in labels) and any("CPU" in l for l in labels)
    cuda = next(p for p in body["panels"] if "CUDA" in p["label"])
    cpu = next(p for p in body["panels"] if "CPU re-run" in p["label"])
    assert cuda["device"].startswith("CUDA") and cpu["device"] == "CPU"
    assert cuda["rows"] and cpu["rows"]
    assert "observational" in body["caveat"]
    assert "not established" in body["caveat"]


def test_hls4ml_marks_synthesis_and_physical_as_unavailable():
    rows = client.get("/api/hls4ml").json()["rows"]
    stages = {r["stage"]: r["tier"] for r in rows if "stage" in r}
    assert stages["HLS synthesis"] == "NOT AVAILABLE"
    assert stages["Physical FPGA"] == "NOT AVAILABLE"
    assert all(r["tier"] == "SIMULATED" for r in rows if r.get("stage") == "hls4ml C-simulation")


def test_metadata_states_no_runtime_training_and_no_cern_claim():
    meta = client.get("/api/metadata").json()
    assert "No training happens at runtime" in meta["demo_note"]
    assert meta["dataset"]["nature"].startswith("public simulated")
