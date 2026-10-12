# Release readiness audit — demo/web-release

Audit date: 2026-10-13 (IST). Branch `demo/web-release`, built on frozen research commit `970f44a` (unchanged). This audit precedes the test report and deployment report; see those for execution evidence.

## 1. Scope decision
The release is the **presentation/demo layer only** — a static-frontend + single-process FastAPI app that reads the exported, frozen research JSON in `demo/data/`. No research code, pipeline, dataset, or checkpoint ships in the demo image. The research itself remains frozen and is not part of this audit.

## 2. Technical decisions and why

| Decision | Reasoning |
|---|---|
| Static HTML/CSS/vanilla JS + Plotly (no Next.js/Node build) | Matches the explicit instruction to avoid a frontend build system; keeps the container small and the deploy surface simple. |
| Single FastAPI process serving `/`, `/static/*`, `/api/*` | One Hugging Face Space, one container, one port — simplest reliable deployment target. |
| `sdk: docker`, `app_port: 7860`, non-root UID 1000 | Confirmed against the current Hugging Face Hub "Spaces Configuration Reference" and community Docker-Space conventions (checked 2026-10-13; see the deployment report for the exact fields used). |
| Service worker caches the static shell only, never `/api/*` | Prevents "stale scientific data shown as current" — offline users get an explicit load error per section, not silently old numbers. |
| PWA over native Android app | No capability needed here (camera, push, background) justifies a second toolchain, signing pipeline, and store listing; a PWA is installable from the same Space with no extra infrastructure. |
| Cache-Control: max-age=300 on `/api/*`, max-age=3600 on `/static/*` | Bounded caching — fast repeat loads without indefinitely stale results after a redeploy. |
| SRI hash (`integrity=`) pinned on the Plotly CDN `<script>` tag | Defends against a compromised/altered CDN file being executed silently. |
| HTML-escaping helper (`esc()`) applied to every API-derived string interpolated into `innerHTML` | Defense-in-depth: the data is self-controlled (our own export script), but nothing API-derived reaches `innerHTML` unescaped. |
| `.dockerignore` at repo root | The Dockerfile's `COPY` list was already selective; this additionally keeps the *build context* from uploading `.git`, `.venv*`, the dataset, and research artifacts. |

## 3. What is explicitly preserved (scientific integrity)
Verified by `tests/test_demo_api.py` (see the test report) and by manual review of `demo/app/main.py` and `demo/scripts/export_results.py`:
- The exploratory, test-set-selected ROC-AUC `0.9278` is excluded from every public API response (`_is_public()` filter + `test_exploratory_biased_mixed_result_never_public`).
- Architecture A's CUDA-era and CPU results are two separate panels in `/api/architecture-comparison`, never averaged; the response carries an explicit "observational, not established" caveat string.
- Every row in `/api/results` and `/api/architecture-comparison` carries a `device` and a `tier` (`MEASURED`/`SIMULATED`/`ESTIMATED`/`NOT AVAILABLE`).
- `/api/hls4ml` marks `HLS synthesis` and `Physical FPGA` as `NOT AVAILABLE`, with no synthesized or physical value anywhere in the export.
- Storage values are labelled "theoretical"/"analytical" everywhere they appear in the frontend copy.
- Missing auxiliary metrics (e.g. PR-AUC for some mixed-precision rows) render as "not recorded" (`fmt()` in `app.js`), never estimated.
- No CERN-affiliation or production-trigger claim exists anywhere in `demo/app/static/index.html` or `demo/README.md`; both explicitly disclaim it.

## 4. Known gaps going into testing (resolved or documented in the test/deployment reports)
- Docker image had not yet been built in this environment at audit time (Docker Desktop engine was not running on this machine — see the deployment report for the attempt and the exact blocker).
- No Hugging Face account/token is configured in this environment — live deployment requires the user's credentials.
- GitHub CLI (`gh`) is authenticated as `Piush2004` with `repo` scope, but **no public repository has been created or pushed to** without the user's explicit go-ahead (publishing a new public repository is an outward-facing, hard-to-reverse action).

## 5. Out of scope for this release
- A native Android app (decision documented above; not built).
- Multi-route (non-anchor) navigation structure (kept as a single page with anchors — simpler for static+FastAPI, no build step needed).
- Percentile/MSE calibration, rounding/saturation isolation, device-causality experiments — these are research gaps, not demo-layer gaps, and are out of scope for a presentation-layer release.
