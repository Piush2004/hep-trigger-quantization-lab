# Reproducibility

This document lists how to reproduce every result in the final report, what is deterministic, what is not, and what cannot be reproduced here at all.

## 1. Environment (as tested)
| Item | Value |
|---|---|
| Host | Windows 11; runs inside WSL2 Ubuntu (`/mnt/d/Projects/CERN`, venv `.venv-wsl`) |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 (built with CUDA 12.8) |
| CUDA available at run time | **False** in the WSL session used for Phases 4–10. The primary architecture's frozen results were produced on CUDA in an earlier session; they are not reproduced in this environment. |
| GPU | RTX 5060 Laptop (from `environment_report.md`); not usable from WSL during Phases 4–10 |
| NumPy / scikit-learn | 2.4.6 / 1.9.1 |
| hls4ml | 1.3.0 (C-simulation only; no HLS synthesis tool installed) |
| Pinned lock | `requirements-lock.txt`, `environment.yml` |

Recorded in each run: `utils/system.py` writes the environment hash, package versions and git-dirty flag into every result record.

## 2. Dataset
- Name: OpenML `hls4ml_lhc_jets_hlf`, version 1. Zenodo DOI 10.5281/zenodo.3602260, CC BY 4.0.
- Content: 830,000 jets, 16 high-level features, 5 classes (g, q, t, W, Z; about 20% each).
- Provenance: particle-level Monte Carlo (MadGraph5 + Pythia8 + FastJet anti-kT R = 0.8, jet pT 0.8–1.6 TeV, 13 TeV); no detector simulation is described. **This is a publicly available simulated dataset, not detector data.**
- Full-dataset fingerprint (SHA-256 of the validated arrays, from `reports/real_data_validation.json`):
  - X: `b2863e6c92578c774dc4b677f88a8f054906226197e2ade5a938d3f2531f096c`
  - y: `ec22bad2a3d3786323994c56e99dd6cbbf13b968c876ebda8d7a0ef7462b82fd`
- The file is fetched by `hep_quant.data.hls4ml_jets` (wraps OpenML). The checks in `real_data_validation.py` verify schema, classes, NaN/Inf, duplicates and the leakage screen.

## 3. Splits and seeds
- Subset: 100,000 jets, stratified, seed-controlled.
- Split: 70 / 15 / 15 train / validation / test, stratified, seed-controlled.
- Seeds: 42, 123, 2026, 7, 99. A seed controls subsample, split, initialisation and shuffling.
- Preprocessing is fitted on the training split only.
- Test data is never used for model, layout or checkpoint selection (see `reports/phase5_mixed_precision_gate.md` for the protocol and the leakage correction).

## 4. Commands
All commands are run from the repository root inside the WSL virtualenv. On Windows PowerShell, prefix with `wsl -d Ubuntu -e bash -c "cd /mnt/d/Projects/CERN && ..."`.

```bash
# environment (once) -- needs uv installed first: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv-wsl --python 3.11
# requirements-lock.txt pins a CUDA build of torch (cu128); both --extra-index-url AND
# --index-strategy unsafe-best-match are required, verified directly: --extra-index-url alone
# still fails ("No solution found... certifi was found on <cu128 index>, but not at the
# requested version") because uv's default dependency-confusion protection restricts every
# package to versions on the FIRST index that has it, once one package needs a non-default
# index. unsafe-best-match lifts that restriction (safe here since both indices are
# first-party: pypi.org and PyTorch's own wheel host). On a CPU-only machine, edit the
# torch==... line in requirements-lock.txt to the CPU wheel first instead.
uv pip install --python .venv-wsl --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r requirements-lock.txt

# tests (full suite; ~3 minutes; network tests are opt-in)
.venv-wsl/bin/python -m pytest -q

# frozen Phase 2 snapshot integrity (84 files)
.venv-wsl/bin/python scripts/freeze_phase2.py verify

# primary architecture: frozen results are read, not re-run
.venv-wsl/bin/python scripts/audit_phase2.py

# Phase 3 PTQ/QAT studies on frozen FP32 checkpoints (no training)
.venv-wsl/bin/python scripts/phase3_ptq_studies.py

# Phase 4 INT4 diagnosis on frozen checkpoints (no training)
.venv-wsl/bin/python scripts/phase4_int4_stability.py

# Phase 5 / 5B mixed-precision gates (validation selects; test once)
.venv-wsl/bin/python scripts/phase5_mixed_precision_gate.py
.venv-wsl/bin/python scripts/phase5b_size_constrained_gate.py

# Phase 6 architecture generalization (50 experiments, CPU, ~1 hour per run-tag)
.venv-wsl/bin/python scripts/phase6_arch_generalization.py --run-tag run1
.venv-wsl/bin/python scripts/phase6_analysis.py run1
.venv-wsl/bin/python scripts/phase6_compare_runs.py      # needs run1 and run2 (run2 with --run-tag run2)
.venv-wsl/bin/python scripts/phase6_export_results.py

# Phase 7 hls4ml C-simulation consistency (rebuilds C-simulation projects; minutes)
.venv-wsl/bin/python scripts/phase3_numerical_consistency.py

# Phase 8 canonical final table (reads existing records; no training)
.venv-wsl/bin/python scripts/phase8_final_audit.py

# Phase 10 final figures 01–08 (reads existing tables; no training)
.venv-wsl/bin/python scripts/phase10_final_figures.py

# single-experiment reproduction (isolated re-run, bit-compare)
.venv-wsl/bin/python -m hep_quant reproduce --experiment EXP_S42_JET_INT4_QAT

# Phase 11 surrogate hardware validation (one-time training; large download, needs
# `pip install -e ".[surrogate]"` and ~700MB from Hugging Face -- see the phase report)
.venv-wsl/bin/python scripts/phase11_train_surrogate.py
.venv-wsl/bin/python scripts/phase11_apply_surrogate_hardware.py

# Phase 12 near-exhaustive mixed-precision search (PTQ: ~20 min full grid on CPU;
# QAT frontier fine-tuning: ~15-20 min for 23 layouts x 5 seeds)
.venv-wsl/bin/python scripts/phase12_exhaustive_mixed_precision_pareto.py
.venv-wsl/bin/python scripts/phase12_qat_frontier.py

# Phase 13 non-NN baselines + background rejection at fixed efficiency (a few minutes)
.venv-wsl/bin/python scripts/phase13_baselines_and_efficiency.py
```

## 5. What is deterministic
- **Phase 5 and 5B gates:** two independent runs produced byte-identical outputs (`cmp`).
- **Phase 6:** two independent runs of all 50 experiments on CPU produced identical per-seed metrics (ROC-AUC, PR-AUC, accuracy, F1) and byte-identical checkpoints.
- **Phase 4 INT4 diagnosis:** two runs produced byte-identical CSVs.
- **Single-experiment reproduction:** EXP_S42_JET_INT4_QAT and EXP_S42_JET_INT8_PTQ were re-run in isolation with bit-identical metrics, weights and predictions (`reports/reproducibility_real_*.md`).

## 6. What is NOT reproducible across devices
- **INT4 PTQ point estimates differ between CPU and CUDA.** Seed 42: accuracy 0.651 on CPU, 0.572 on CUDA. The frozen primary records are CUDA-era; the Phase 6 re-runs are CPU. The primary architecture's CPU re-run has INT4 PTQ 0.8904 ± 0.0122, versus the frozen CUDA 0.8819 ± 0.0308. **Compare only within one device.** Pooled statistics never mix devices.
- The magnitude of the INT4 PTQ seed spread depends on device (see `reports/final_statistical_audit.md`).

## 7. Checkpoints and generated artifacts
- Checkpoints under `experiments/checkpoints/` are regenerable and excluded from git (`.gitignore`). The Phase 6 checkpoint folders (`experiments/checkpoints_phase6_run*/`) are excluded for the same reason.
- Per-experiment figures (`reports/figures/EXP_*`) are regenerable by the experiment runner and are not the final figure set.
- The final figure set is `reports/figures/01_baseline_quantization.png` … `08_final_summary.png`, built by `scripts/phase10_final_figures.py`.

## 8. Measurement tiers (must be kept separate)
| Tier | Meaning | Where it appears |
|---|---|---|
| MEASURED | actual software measurement (PyTorch accuracy, ROC-AUC, timings in emulation) | all accuracy and ROC-AUC numbers |
| SIMULATED | hls4ml C-simulation (fixed-point arithmetic, no hardware) | `reports/numerical_consistency.json`, figure 07 |
| ESTIMATED | heuristic resource estimate (linear software model) | `hardware_summary` tables |
| SURROGATE | ML regressor trained on real external hls4ml synthesis records (fastmachinelearning/wa-hls4ml), with its own held-out error reported alongside every number | `reports/tables/phase11_surrogate_hardware.csv`, `reports/phase11_surrogate_hardware_validation.md` |
| SYNTHESIZED | actual HLS synthesis | **not available** |
| PHYSICAL | actual FPGA measurement | **not available** |

HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT. No Vivado, Vitis HLS or Quartus is installed (checked 2026-10-05). No SYNTHESIZED or PHYSICAL number exists in this repository.

## 9. Known limits of reproduction
- CUDA cannot be exercised in this WSL session; the CUDA test is skipped and reported as skipped.
- Four network tests (OpenML fetch) are opt-in (`RUN_NETWORK_TESTS=1`).
- Phase 7 C-simulation rebuilds hls4ml projects; results were compared across two independent runs (see `reports/phase7_hls_numerical_validation.md`).
