# Experiment Log

Running log of what was built, tested, what failed, what was fixed, and what passed —
per the project's engineering-transparency requirement. Chronological.

## Gate 0 — Environment

**Built:** Python 3.11 project (via `uv`), WSL2-based environment after native Windows
and Docker Desktop both hit blockers.

**Failure 1 — Windows Smart App Control blocked matplotlib.**
- Component: native-Windows `.venv`, `matplotlib` import.
- Error: `ImportError: DLL load failed while importing _path: An Application Control
  policy has blocked this file.`
- Root cause: Windows Smart App Control reputation-blocked
  `matplotlib/_path.cp311-win_amd64.pyd`, confirmed via the CodeIntegrity operational
  event log (Code Integrity Event ID 3077/3089/3118).
- Fix: switched execution environment from native Windows to a Linux environment.
  First attempted Docker Desktop (user's choice between the two reasonable options);
  see Failure 2.
- Test performed: re-ran core-library import test inside the new environment.
- Result: PASS (see Failure 2's resolution).

**Failure 2 — Docker Desktop backend crash on orphaned socket file.**
- Component: Docker Desktop backend service.
- Error: `backend crashed ...: initializing Ingest server: listening on
  unix://...\run\sailor-ingest.sock: ... The file cannot be accessed by the system.`
- Root cause: an orphaned Unix-domain-socket file under
  `%LOCALAPPDATA%\Docker\run\` left behind by an earlier crash, which Windows reports
  as inaccessible even after killing all Docker processes and `wsl --shutdown`.
  Clearing it looked like it would need a full system reboot.
- Fix: rather than ask for a disruptive reboot mid-session, switched to running
  directly inside the machine's existing WSL2 Ubuntu distro (which already had GPU
  passthrough working out of the box), avoiding Docker Desktop's broken backend
  entirely while still sidestepping Smart App Control (Linux binaries aren't subject
  to it).
- Test performed: imported torch/numpy/pandas/sklearn/matplotlib/seaborn/yaml/h5py and
  ran a CUDA matmul + saved a matplotlib figure, inside WSL2.
- Result: PASS. Full environment report in `environment_report.md`.

**Gate 0 acceptance:** `pytest -q` run twice against `tests/test_environment.py` plus
the full suite — 43 tests passed on both runs (see Gate 1-4 below; the suite was
written and run together, see `git log` for commit-by-commit staging).

## Gate 1/2 — Data pipeline + synthetic validation

**Built:** `hep_quant.data` (loader, validation, splits, preprocessing, synthetic
generator).

**Tested:** `tests/test_data.py` (13 tests) — shape/dtype validation, NaN/Inf
rejection, deterministic + stratified + leakage-free splitting, preprocessing
fit/transform correctness.

**Result:** PASS, twice (`pytest -q` run twice, see Gate 0 acceptance above — all
tests across all gates were run together each time since they were implemented in the
same session).

## Gate 3 — FP32 baseline

**Built:** configurable `MLPClassifier`, `Trainer` (early stopping, checkpointing),
`evaluate_model`.

**Tested:** `tests/test_models.py`, `tests/test_training.py` — forward pass shape,
checkpoint save/reload producing identical predictions (`atol=1e-6`), two independent
training runs with the same seed converging to identical test accuracy.

**Result:** PASS, twice.

## Gate 4 — PTQ

**Built:** `hep_quant.quantization.{calibration,ptq,analysis}` — uniform affine
fake-quantization for FP32/INT8/INT6/INT4 (see `docs/quantization.md` for why this
approach, not PyTorch's native INT8-only quantized kernels).

**Tested:** `tests/test_quantization.py` — all four precisions produce finite outputs
and computable metrics; FP32 quantization is a no-op (identity); size estimates
strictly decrease INT8 > INT6 > INT4; quantization is deterministic given the same
calibration data.

**Result:** PASS, twice.

## Gate 6 — Hardware-aware analysis (built ahead of QAT, since it only needed the
abstraction, not a trained model)

**Built:** `HardwareBackend` abstraction with `HLSBackend` (real hls4ml attempt) and
`UnavailableBackend` (explicit None-fields fallback).

**Tested:** `tests/test_hardware.py` — `UnavailableBackend` returns well-formed `None`
fields with a documented reason; unknown backend names fall back safely; software
latency benchmarking produces finite, positive values.

**Result:** PASS, twice.

*(Entries for QAT, hls4ml live-attempt, and the full experiment matrix are appended
below as they're completed.)*

## Phase 2 audit and Phase 3 (real-data strengthening)

**Built:** dedicated dataset adapter with schema enforcement; real-data validation report; 21-run multi-seed matrix (frozen, hash-verified); results audit; hls4ml C-simulation sweep; reproducibility command; Phase 3 studies (calibration, component, per-unit, mixed precision, per-class/calibration/error analysis, numerical consistency, controlled latency); compute-matched FP32 fine-tune control; seeds 7/99; provenance fields; environment lock.

**Failures and corrections (kept visible):**
- *Premise errors in the prompts.* The audit prompt assumed the matrix had finished (it had not; 10 of 21 at the time) and the Phase 3 prompt assumed the Phase 2 audit and frozen results existed (they did not). The matrix was allowed to finish, then the audit and freeze were performed first.
- *Unsupported claims in earlier documents, corrected after checking primary sources:* generator "Pythia8 + Delphes" (the dataset paper describes MadGraph5 + Pythia8 + FastJet, no detector simulation); size "~880,000" (830,000); "within 1 pp of the original hls4ml paper" (reference is arXiv:2003.06308 Table 5, 75%, different split/budget); attribution of the first CUDA loss to `wsl --shutdown` (the timeline does not support it).
- *Single-seed overreach:* the earlier "INT4 PTQ −9.9 points, QAT recovers most" was one draw of a highly variable quantity (n = 5: ROC-AUC 0.835–0.909).
- *Tooling mistakes caught by tests/inspection:* a row-builder bug that let a metric named `precision` overwrite the config `precision`; a CRLF/encoding mangling of one source line (repaired); the C-simulation accumulator override initially applied to only two of four Dense layers (fixed after inspecting the generated `parameters.h`); a misleading regex-based `sed` edit that silently did not apply (caught by re-reading the file).
- *CUDA lost mid-run* (`EXP_C7_JET_FP32_FT`): job aborted, recovered with one WSL restart, resumed; the failed record was overwritten by the completed re-run (log: `experiments/logs/phase3_runs.log`).
- *Provenance gap:* the 21 frozen runs recorded commit `4e9b673` while the tree was dirty; documented in the audit; `git_dirty` is now recorded.
