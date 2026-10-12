# Release test report — demo/web-release

All commands below were actually executed in this session; exit codes and counts are copied from their real output, not estimated. Environment: WSL2 Ubuntu, Python 3.11.16, PyTorch 2.11.0+cu128, GPU: NVIDIA GeForce RTX 5060 Laptop (CUDA **available** during this test pass — see §A note).

## A. Regression testing (full suite, including demo tests)

| Run | Command | Result | Duration |
|---|---|---|---|
| 1 | `.venv-wsl/bin/python -m pytest -q -rs` | **109 passed, 4 skipped** | 286.4s |
| 2 | `.venv-wsl/bin/python -m pytest -q -rs` (independent) | **109 passed, 4 skipped** | 226.3s |

Both runs are identical in pass/skip count; the 4 skips in both are the opt-in network tests in `tests/test_openml_loader.py` (`RUN_NETWORK_TESTS=1` not set).

**Count changed from the previously recorded 102 passed / 5 skipped — investigated, not assumed valid:**
- +7 passed: 6 new demo tests added this session (`test_root_health_endpoint_matches_docker_healthcheck_expectations`, `test_pwa_assets_are_served_with_correct_types`, `test_api_responses_have_bounded_cache_control_not_indefinite`, `test_unknown_route_returns_404_not_a_file_leak`, `test_no_permissive_cors_header_on_api`, `test_no_obvious_secrets_committed_in_demo_tree`) plus **1 previously-skipped test now passing**: `torch.cuda.is_available()` returned `True` in this WSL session (`python -c "import torch; print(torch.cuda.is_available())"` → `True`), so `tests/test_hardware.py`'s CUDA-only GPU-model-conversion test ran instead of skipping. CUDA availability on this machine's WSL2 is documented elsewhere in this project as intermittent; it happened to be up for both runs (both gave the identical 109/4), so this is explained and stable, not a flaky result.
- −1 skip: the same CUDA test moving from skip to pass.
- Net: 102 + 7 passed − 1 skip = 109 passed, 4 skipped. Exact match.

No assertion was weakened and no test was removed to obtain a passing run.

**Frozen Phase 2 snapshot, re-verified after all demo-layer work:** `scripts/freeze_phase2.py verify` → `{"files_checked": 84, "changed": []}`.

## B. API tests (14 demo tests, `tests/test_demo_api.py`, all passing in both runs above)

| Test | Verifies |
|---|---|
| `test_health_declares_no_runtime_training` | `/api/health` reports `training_at_runtime: false` |
| `test_all_exported_files_exist_and_parse` | every `demo/data/*.json` file exists and is valid JSON |
| `test_index_and_static_assets_are_served` | `/`, `/static/app.js`, `/static/styles.css` return 200 |
| `test_exploratory_biased_mixed_result_never_public` | the biased `0.9278` mixed-precision result never appears in `/api/results` or `/api/pareto` |
| `test_every_result_row_has_device_and_tier` | every row in `/api/results` has a non-empty `device` and a valid tier |
| `test_architecture_devices_are_not_merged` | `/api/architecture-comparison` returns 3 distinct panels (CUDA + CPU for A, CPU for B), each with its own rows, and the "observational"/"not established" caveat text |
| `test_hls4ml_marks_synthesis_and_physical_as_unavailable` | `HLS synthesis` and `Physical FPGA` are `NOT AVAILABLE`; all C-simulation rows are `SIMULATED` |
| `test_metadata_states_no_runtime_training_and_no_cern_claim` | `/api/metadata` text matches the no-training, public-simulated-dataset facts |
| `test_root_health_endpoint_matches_docker_healthcheck_expectations` *(new)* | `/health` returns 200 with `status: ok`, no missing data files |
| `test_pwa_assets_are_served_with_correct_types` *(new)* | `/manifest.webmanifest`, `/sw.js`, `/favicon.ico` return 200 with correct content types; the service worker text references `/api/` (the path it deliberately excludes from caching) |
| `test_api_responses_have_bounded_cache_control_not_indefinite` *(new)* | `/api/results` has a `Cache-Control` header with `max-age`, and it is not an indefinite/immutable value |
| `test_unknown_route_returns_404_not_a_file_leak` *(new)* | a path-traversal attempt and an unknown API route both return 404/400, not a file |
| `test_no_permissive_cors_header_on_api` *(new)* | `/api/results` carries no `Access-Control-Allow-Origin` header |
| `test_no_obvious_secrets_committed_in_demo_tree` *(new)* | no AWS/private-key/OpenAI/GitHub/Hugging-Face token-shaped string anywhere under `demo/` |

Malformed-data handling: `_load()` raises an HTTP 500 with a clear `detail` if an expected `demo/data/*.json` file is missing (exercised indirectly by `/health`'s `missing_data_files` field, which was empty in this run since all 7 files are present).

## C. Frontend tests (manual, via the built-in browser pane; `http://localhost:7860/`, local `uvicorn` run)

- **Initial load:** page renders, hero/credibility chips/caution banner present.
- **Navigation:** anchor links to all 6 sections confirmed present with `aria-label="Primary"` on `<nav>`; a skip-to-content link is present and becomes visible on focus.
- **Charts:** all 8 Plotly charts (`chart-main`, `chart-seeds`, `chart-ptq-qat`, `chart-calib`, `chart-hls`, 3× `arch-chart-N`) render with `role="img"` set after draw, confirmed via a script querying `.main-svg` element counts (3 per chart, Plotly's normal output) — all 8 present.
- **Filters/controls:** precision/method segmented buttons and the architecture-configuration `<select>` were exercised programmatically; `aria-pressed` toggles correctly; the select has an explicit `<label>` and `aria-label`.
- **Tooltip/headline readability:** the computed headline text was read back and matches the underlying data (`INT6 PTQ reaches ROC-AUC 0.9329 ± 0.0014 ... 78.8% below FP32 (17.6 KB)` — checked against `reports/tables/final_results.csv`).
- **Empty/error states:** `boot()` uses `Promise.allSettled`; a failing endpoint shows an inline `role="alert"` message naming the section and suggesting a reload, without blanking the other sections or the static Methodology/Reproducibility/Research text.
- **Desktop and mobile layout:** checked at the pane's default desktop width and at an emulated 375×812 (mobile) viewport — screenshot confirms no horizontal overflow, header wraps, hero buttons stack, badges remain legible.
- **Console errors:** one error was found and investigated (service-worker registration — see below); no other console errors.
- **Horizontal overflow / clipped content:** `body { max-width: 100vw; overflow-x: hidden }` added; confirmed visually at mobile width.
- **Accessibility basics:** skip link, `:focus-visible` outline, `prefers-reduced-motion` handling, labelled nav/select, `aria-label`/`role="img"` on charts, `aria-pressed` on segmented controls.
- **Links/documentation:** the GitHub link element exists but stays `hidden` unless `DEMO_REPO_URL` is set (verified absent by default, since no such URL has been confirmed — see the deployment report).
- **Visualization-library failure:** `plot()` checks `window.Plotly` and, if absent, replaces the chart with a plain-text note (`role="note"`) instead of leaving a blank or broken element; the surrounding headline/metrics text (which doesn't depend on Plotly) stays visible either way.
- **Subresource Integrity:** the pinned Plotly CDN `<script>` tag carries a `sha384` `integrity` attribute computed from the exact pinned file (`openssl dgst -sha384`), with `crossorigin="anonymous"`; confirmed the script still loads correctly with it in place (`typeof window.Plotly === "object"`).

**Cross-browser testing was not performed.** Only the built-in browser-pane's Chromium-based engine was exercised; no claim is made about Firefox/Safari/other engines.

**Service worker — investigated, reported honestly:**
- `sw.js` fetches with HTTP 200 and the correct `application/javascript` content type (`fetch('/sw.js')` confirmed this from within the page).
- Loading `sw.js` as a plain `Worker` (not a `ServiceWorker`) ran for 1.2s with no `onerror` fired — the script is syntactically valid and its top-level `addEventListener` calls execute without throwing.
- `navigator.serviceWorker.register('/sw.js')` failed with `TypeError: ... An unknown error occurred when fetching the script.` in this session's browser-pane tool, despite the page being a secure context (`window.isSecureContext === true`), not inside an iframe (`window.top === window`), and the API being present (`'serviceWorker' in navigator === true`).
- This combination (file fetches fine directly, syntax is valid, yet `register()` fails generically) is a known symptom of Service Worker registration being restricted inside CDP-automated/embedded Chrome profiles, not an application defect. **This is reported as unverified in this tool, not as a passing test** — see `reports/final_release_checklist.md` for the exact recommended real-browser check.

## D. Container and live-deployment tests

**Not run.** Docker engine unavailable in this environment (see `reports/deployment_report.md` §1). No live Hugging Face Space exists (§2). Nothing is claimed as tested in a container or in production.

## E. Security and privacy

| Check | Result |
|---|---|
| No committed secrets/tokens | Scanned `demo/` for AWS/private-key/OpenAI/GitHub/Hugging-Face token patterns — none found (`test_no_obvious_secrets_committed_in_demo_tree`, passing) |
| No secrets embedded in frontend JS | `demo/app/static/app.js` reviewed — contains no credentials, only public API paths |
| No unnecessary file-serving routes | Only `/static` (mounted to `demo/app/static/` only, via Starlette `StaticFiles`, which rejects `..` traversal by construction) plus explicit single-file routes (`/`, `/favicon.ico`, `/manifest.webmanifest`, `/sw.js`) — nothing serves `demo/data/` or the repo root directly |
| No directory traversal | `GET /../../etc/passwd` → 404/400, not a file (`test_unknown_route_returns_404_not_a_file_leak`) |
| No unsafe dynamic code execution | No `eval`, `exec`, `innerHTML` of unescaped API strings (an `esc()` HTML-escape helper is applied everywhere API-derived text reaches `innerHTML`), no server-side templating of request input |
| No unnecessary permissive CORS | No CORS middleware is configured at all; `/api/results` carries no `Access-Control-Allow-Origin` header (`test_no_permissive_cors_header_on_api`) |
| No accidental dataset/env-file exposure | `.dockerignore` added; the Dockerfile's `COPY` list only ever includes `demo/requirements.txt`, `demo/app`, `demo/data` |
| No unnecessary user-information collection | The app collects nothing from visitors — no cookies, no analytics, no forms |
| No unexpected external dependency | Exactly one third-party network dependency: the pinned, SRI-protected Plotly CDN script. No other external call exists in the frontend or backend. |

All findings above were fixed inline during this session (escaping helper, CORS absence confirmed, `.dockerignore` added, SRI added) rather than left as review comments.

**Post-commit automated security review (triggered after the first four release commits) found one additional issue, fixed immediately:**
- `demo/app/static/sw.js`'s navigate-request handler wrote the network response into Cache Storage under the key `"/"` without first checking `response.ok` — unlike the static-asset handler a few lines below it, which already gated on `res.ok`. A transient error page (or an opaque/tampered response) could therefore have been cached and replayed to the user on a future offline load: a cache-poisoning risk, even though the same-origin guard already present meant the practical exposure was narrow.
- Fixed by adding an `isCacheable(res)` helper (`res.ok && res.type === "basic"`) and applying it to both cache-write sites. `test_pwa_assets_are_served_with_correct_types` now asserts the guard text is present in the served `sw.js`. Re-ran the 14 demo tests: 14 passed.

## F. Scope note
This report covers the demo/presentation layer only. The research test suite (94 of the 109 passing tests) is unchanged by this work and is documented separately in the research tree's own reports (`reports/phase4_build_report.md` and others).
