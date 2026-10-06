# HEP-Trigger Quantization Lab — result explorer (demo layer)

A static interactive view of the frozen research results. It is a presentation layer only:

- It **does not train** models, **run experiments**, or **change results**.
- It reads only the exported JSON in `demo/data/`, produced by `demo/scripts/export_results.py` from the research tables at commit `970f44a`.
- It serves the frontend at `/` and the JSON API at `/api/*` from one FastAPI process.

The research itself, its reproduction instructions, and the dataset provenance live in the research repository (`docs/REPRODUCIBILITY.md`). The public demo does not contain the dataset.

## Run locally

```bash
pip install -r demo/requirements.txt
cd demo/app
uvicorn main:app --host 127.0.0.1 --port 7860
```

Then open http://127.0.0.1:7860/. Endpoints: `/api/health`, `/api/metadata`, `/api/results`, `/api/configurations`, `/api/quantization`, `/api/pareto`, `/api/architecture-comparison`, `/api/calibration-sensitivity`, `/api/hls4ml`.

Set `DEMO_REPO_URL` to show a GitHub link in the header.

## Test

```bash
python -m pytest -q tests/test_demo_api.py
```

These checks cover: the exploratory biased mixed-precision result never appears in public data; every row has a device and an evidence tier; architecture A's CUDA and CPU results are separate panels; HLS synthesis and physical FPGA are marked NOT AVAILABLE; and the page and assets are served.

## Docker (Hugging Face Space, Docker SDK)

```bash
docker build -f demo/Dockerfile -t hep-trigger-lab-demo .
docker run -p 7860:7860 hep-trigger-lab-demo
```

The image copies only `demo/app`, `demo/data` and `demo/requirements.txt`. **The Docker build has not been executed in this environment** (Docker Desktop was unavailable); the local `uvicorn` run and the test client were executed instead.

## Scientific labelling
- Every value carries its device (CUDA-era frozen or CPU) and an evidence tier (MEASURED, SIMULATED, ESTIMATED, NOT AVAILABLE).
- Storage is theoretical (analytical), not file size.
- Architecture A on CUDA and on CPU is shown side by side and is **not merged**. The comparison between them is **observational**: a controlled CUDA reproduction of the Phase 6 protocol was unavailable, so device dependence is not established.
- The earlier exploratory mixed-precision result (0.9278) selected on test data is excluded from every public chart.
- No FPGA synthesis or physical measurement exists. Hardware resource figures are estimates.

Independent research project. Not an official or CERN-affiliated project.
