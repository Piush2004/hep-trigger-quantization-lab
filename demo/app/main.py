"""HEP-Trigger Quantization Lab — result explorer (single FastAPI process).

Serves the static frontend at / and the JSON API at /api/*. Reads only the exported, frozen research results
in demo/data/ (produced by demo/scripts/export_results.py). Never trains, never runs experiments, never edits data.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
STATIC_DIR = APP_DIR / "static"

# Public charts never show the biased, test-selected exploratory mixed-precision result.
EXCLUDED_MARKERS = ("EXPLORATORY", "TEST-SET SELECTED", "TEST_SET_SELECTED")
EXCLUDED_ROC_AUC = 0.9278  # the biased exploratory value; must never be shown as a result

app = FastAPI(title="HEP-Trigger Quantization Lab — result explorer", version="1.0.0")


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"missing exported file: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_public(row: dict) -> bool:
    text = " ".join(str(v) for v in row.values())
    if any(marker in text for marker in EXCLUDED_MARKERS):
        return False
    roc = row.get("roc_auc_mean")
    if isinstance(roc, (int, float)) and abs(roc - EXCLUDED_ROC_AUC) < 1e-9 and "Mixed" in str(row.get("configuration", "")):
        return False
    return True


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "data_dir": str(DATA_DIR.relative_to(APP_DIR.parent.parent)).replace("\\", "/"),
            "training_at_runtime": False}


@app.get("/api/metadata")
def metadata() -> dict:
    meta = dict(_load("metadata.json"))
    meta["repository_url"] = os.environ.get("DEMO_REPO_URL") or None
    meta["demo_note"] = ("Visualization of frozen research results. No training happens at runtime. "
                         "Cross-device comparisons are observational.")
    return meta


@app.get("/api/results")
def results() -> dict:
    rows = [r for r in _load("baseline.json")["rows"] if _is_public(r)]
    return {"rows": rows, "count": len(rows)}


@app.get("/api/configurations")
def configurations() -> dict:
    rows = [r for r in _load("baseline.json")["rows"] if _is_public(r)]
    seen = []
    for r in rows:
        key = (r["architecture"], r["configuration"], r["device"])
        if key not in seen:
            seen.append(key)
    return {"configurations": [{"architecture": a, "configuration": c, "device": d} for a, c, d in seen]}


@app.get("/api/quantization")
def quantization() -> dict:
    return _load("quantization.json")


@app.get("/api/pareto")
def pareto() -> dict:
    data = dict(_load("pareto.json"))
    data["points"] = [p for p in data["points"] if _is_public({k: v for k, v in p.items()})
                      and "EXPLORATORY" not in str(p.get("configuration", ""))]
    return data


@app.get("/api/architecture-comparison")
def architecture_comparison() -> dict:
    """Architecture A on CUDA and on CPU are returned side by side, never merged.

    The CUDA rows are the frozen primary records; the CPU rows are the Phase 6 re-run. The comparison
    between them is observational: a controlled CUDA reproduction of the Phase 6 protocol was unavailable.
    """
    baseline = [r for r in _load("baseline.json")["rows"] if _is_public(r)]
    cuda_a = [r for r in baseline if r["architecture"].startswith("A primary") and "Mixed" not in r["configuration"]]
    arch = _load("architecture.json")["rows"]

    def _arch(row: dict) -> str:
        # the Phase 6 export spells architectures with underscores (A_[64,32,32]); normalise for matching
        return str(row["architecture"]).replace("_", " ")

    cpu_a = [r for r in arch if _arch(r) == "A [64,32,32]"]
    cpu_b = [r for r in arch if _arch(r) == "B [128,64,64,32]"]
    return {
        "panels": [
            {"label": "Architecture A · CUDA-era frozen records", "device": "CUDA-era (frozen)",
             "architecture": "A [64,32,32]", "parameters": 4389, "rows": cuda_a},
            {"label": "Architecture A · CPU re-run", "device": "CPU",
             "architecture": "A [64,32,32]", "parameters": 4389, "rows": cpu_a},
            {"label": "Architecture B · CPU", "device": "CPU",
             "architecture": "B [128,64,64,32]", "parameters": 16837, "rows": cpu_b},
        ],
        "caveat": ("The CUDA and CPU results for architecture A are shown side by side and are not merged. "
                   "The cross-device comparison is observational: a controlled CUDA reproduction of the Phase 6 "
                   "protocol was unavailable during final validation, so device dependence is not established."),
    }


@app.get("/api/calibration-sensitivity")
def calibration() -> dict:
    return _load("calibration.json")


@app.get("/api/hls4ml")
def hls4ml() -> dict:
    return _load("hls4ml.json")


# ---- frontend ----
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
