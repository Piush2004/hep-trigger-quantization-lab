---
title: HEP-Trigger Quantization Lab
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive explorer for a frozen HEP jet-classification quantization benchmark
tags:
  - research
  - quantization
  - high-energy-physics
  - hls4ml
license: mit
---

# HEP-Trigger Quantization Lab — result explorer (demo layer)

An interactive, installable (PWA) explorer for the frozen research results. It is a presentation layer only:

- It **does not train** models, **run experiments**, or **change results**.
- It reads only the exported JSON in `demo/data/`, produced by `demo/scripts/export_results.py` from the research tables at commit `970f44a` on the research branch.
- It serves the frontend at `/` and the JSON API at `/api/*` from one FastAPI process. No separate backend service.

The research itself, its reproduction instructions, and the dataset provenance live in the research repository (`docs/REPRODUCIBILITY.md`). The public demo does not contain the dataset, checkpoints, or research code — only the exported summary JSON (see `demo/data/`, a few KB each).

> **Scientific scope.** This dashboard demonstrates a reproducible research benchmark. It does **not** demonstrate a production HEP trigger, a peer-reviewed publication, or any measured FPGA speedup. See [Evidence and limitations](#evidence-and-limitations) below.

## Run locally

```bash
pip install -r demo/requirements.txt
cd demo/app
uvicorn main:app --host 127.0.0.1 --port 7860
```

Open http://127.0.0.1:7860/.

### Configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `DEMO_REPO_URL` | unset (GitHub link hidden) | Shows a GitHub link in the header/hero when set to a repository URL. |

No other environment variables are read. No secrets or credentials are required to run the demo.

### Endpoints

| Path | Purpose |
|---|---|
| `GET /` | Frontend (single page, anchored sections) |
| `GET /health`, `GET /api/health` | Liveness/readiness: `{"status": "ok"\|"degraded", "missing_data_files": [...], "training_at_runtime": false, "uptime_seconds": ...}`. Returns HTTP 503 if any exported data file is missing. |
| `GET /api/metadata` | Dataset, seeds, device notes, evidence-tier definitions |
| `GET /api/results` | Canonical per-configuration table (device + tier on every row) |
| `GET /api/configurations` | Distinct (architecture, configuration, device) tuples |
| `GET /api/quantization` | Per-seed values for the primary architecture |
| `GET /api/pareto` | Storage-vs-accuracy points with dominance flags |
| `GET /api/architecture-comparison` | Architecture A (CUDA + CPU, side by side) and B (CPU), with the observational-comparison caveat |
| `GET /api/calibration-sensitivity` | INT4/6/8 PTQ accuracy vs. calibration-set size |
| `GET /api/hls4ml` | PyTorch → hls4ml C-simulation → (unavailable) synthesis/physical pipeline rows |
| `GET /manifest.webmanifest`, `GET /sw.js`, `GET /favicon.ico` | PWA assets |
| `GET /static/*` | CSS/JS/icons |

An unknown route returns FastAPI's default JSON 404; no route serves files outside `demo/app/static/` or `demo/data/`.

## Test

```bash
python -m pytest -q tests/test_demo_api.py   # demo-only
python -m pytest -q                          # full suite, including the research tests
```

The demo tests cover: the exploratory biased mixed-precision result never appears in public data; every row has a device and an evidence tier; architecture A's CUDA and CPU results are separate panels with the observational caveat; HLS synthesis and physical FPGA are marked `NOT AVAILABLE`; the health, manifest, service-worker and favicon routes; and that the page and static assets are served.

## PWA (installable on Android/desktop)

The app is an installable Progressive Web App: `manifest.webmanifest`, a set of generated icons (`demo/scripts/generate_icons.py`, including maskable variants for Android's adaptive-icon mask), and a minimal service worker (`sw.js`) that caches only the static shell (HTML/CSS/JS/icons) for offline opening.

**`/api/*` responses are never cached by the service worker.** Scientific results are always fetched fresh; if the device is offline the fetch simply fails and the page says so per section — it never shows old numbers as if they were current.

**Install on Android (Chrome):** open the site, then Menu (⋮) → "Add to Home screen" / "Install app". **Install on desktop (Chrome/Edge):** the address-bar install icon, or Menu → "Install HEP-Trigger Quantization Lab". iOS Safari: Share → "Add to Home Screen" (manifest support is partial on iOS; the app still works as a normal web page there).

A native Android app was evaluated and rejected for this release: it would need a second build toolchain, signing, and a store listing for no capability the installable web app doesn't already provide (no camera/background/push requirement exists here). A PWA is also directly servable from the same Hugging Face Space with no extra infrastructure.

## Docker (Hugging Face Space, Docker SDK)

```bash
docker build -f demo/Dockerfile -t hep-trigger-lab-demo .
docker run -p 7860:7860 hep-trigger-lab-demo
# curl http://localhost:7860/health
```

- Binds `0.0.0.0:7860` (the port Hugging Face's Docker SDK requires; declared via `app_port: 7860` in this file's front matter).
- Runs as a non-root user (UID 1000, matching the platform's runtime user).
- Has a Docker `HEALTHCHECK` against `/health`.
- `.dockerignore` (repo root) keeps the research code, dataset, `.venv*`, and `.git` out of the build context.
- Copies only `demo/app`, `demo/data` and `demo/requirements.txt` — no research pipeline, no raw dataset, no secrets.
- No GPU, no CUDA, no model download, and no training occurs in the container.

See `reports/deployment_report.md` (in the research tree) for whether this image has actually been built and run, and the exact commands and results.

## Evidence and limitations

- Every value carries its **device** (CUDA-era frozen or CPU re-run) and an **evidence tier**: `MEASURED` (PyTorch evaluation), `SIMULATED` (hls4ml C-simulation), `ESTIMATED` (software hardware-resource heuristic), `NOT AVAILABLE` (HLS synthesis, physical FPGA — neither exists for this project).
- Storage is **theoretical** (analytical bit-width arithmetic), not file size, not measured memory traffic, and not measured hardware resource consumption.
- Architecture A on CUDA and on CPU is shown **side by side and never merged**. The difference between them is **observational**: a controlled CUDA reproduction of the Phase 6 protocol was unavailable during final validation, so device dependence is not established as a mechanism.
- The earlier exploratory mixed-precision result (ROC-AUC 0.9278) was selected using test-split scores and is **excluded from every public chart, table and headline metric** (see `tests/test_demo_api.py::test_exploratory_biased_mixed_result_never_public`).
- Calibration-sensitivity results cover **min/max calibration only**; other calibration schemes were not evaluated.
- The hls4ml study shows that rounding and saturation together remove most of the narrow-width accuracy collapse; it does **not** claim their individual contributions were isolated.
- Auxiliary metrics not recorded for a configuration (e.g. PR-AUC/F1/accuracy for some mixed-precision rows) are shown as **"not recorded"**, never estimated.
- The dataset is a **publicly available simulated** HEP jet-tagging dataset (hls4ml LHC jet dataset). It contains **no detector-simulated or real detector data**, and the project makes no CERN-affiliation or production-readiness claim.

Independent research project. Not an official or CERN-affiliated project.
