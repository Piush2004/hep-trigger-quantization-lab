"""HEP-Trigger Quantization Lab — result explorer (single FastAPI process).

Serves the static frontend at / and the JSON API at /api/*. Reads only the exported, frozen research results
in demo/data/ (produced by demo/scripts/export_results.py). Never trains, never runs experiments, never edits data.
"""
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
STATIC_DIR = APP_DIR / "static"
START_TIME = time.time()

# Public charts never show the biased, test-selected exploratory mixed-precision result.
EXCLUDED_MARKERS = ("EXPLORATORY", "TEST-SET SELECTED", "TEST_SET_SELECTED")
EXCLUDED_ROC_AUC = 0.9278  # the biased exploratory value; must never be shown as a result

# Bounded cache so a deployed Space never serves indefinitely stale scientific results: short enough
# that a redeploy (which replaces demo/data/*.json) is visible within minutes, long enough to cut
# repeat-request load. Static assets (JS/CSS) can cache longer; they are versioned by redeploy, not by URL.
API_CACHE_CONTROL = "public, max-age=300, must-revalidate"
STATIC_CACHE_CONTROL = "public, max-age=3600, must-revalidate"

app = FastAPI(title="HEP-Trigger Quantization Lab — result explorer", version="1.0.0",
              description="Visualization of frozen research results. No training or experiments run at request time.")


@app.middleware("http")
async def add_cache_and_timing_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = API_CACHE_CONTROL
    elif request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = STATIC_CACHE_CONTROL
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


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


def _health_payload() -> dict:
    missing = [n for n in ("metadata", "baseline", "quantization", "pareto", "architecture", "calibration", "hls4ml")
               if not (DATA_DIR / f"{n}.json").exists()]
    return {
        "status": "ok" if not missing else "degraded",
        "missing_data_files": missing,
        "data_dir": str(DATA_DIR.relative_to(APP_DIR.parent.parent)).replace("\\", "/"),
        "training_at_runtime": False,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.get("/health")
def health_root():
    """Root-level health endpoint for Docker HEALTHCHECK / orchestration probes."""
    payload = _health_payload()
    return JSONResponse(payload, status_code=200 if payload["status"] == "ok" else 503)


@app.get("/api/health")
def health() -> dict:
    return _health_payload()


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
# StaticFiles only ever serves files under STATIC_DIR and rejects path traversal by construction
# (Starlette resolves and checks the path stays within the mounted directory).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/results")
def results_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "results.html")


@app.get("/architecture")
def architecture_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "architecture.html")


@app.get("/hardware")
def hardware_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "hardware.html")


@app.get("/methodology")
def methodology_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "methodology.html")


@app.get("/research")
def research_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "research.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    # Browsers request this path by convention; serve the SVG icon for it rather than 404-spamming logs.
    return FileResponse(STATIC_DIR / "icons" / "favicon.svg", media_type="image/svg+xml")


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(STATIC_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    # Served at the origin root (not /static/) so its default scope covers the whole app.
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})
