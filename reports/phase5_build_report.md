# Phase 5 build report

**Status: PARTIALLY COMPLETE.** The validation-selected mixed-precision gate was executed and reproduced twice, and the regression suite passed twice. Phase 5 is not closed, for the reasons in §9.

## 1. Environment (as executed)
- Host: Windows 11; execution in WSL2 Ubuntu 26.04 (`/mnt/d/Projects/CERN`).
- Python 3.11.16; PyTorch 2.11.0+cu128 (CUDA runtime 12.8); NumPy 2.4.6; scikit-learn 1.9.1.
- `torch.cuda.is_available()` = **False** in WSL during this phase. All Phase 5 runs were **CPU-only**. The GPU recorded in `environment_report.md` (RTX 5060 Laptop) was not used. CUDA reproducibility was not tested.
- Git HEAD `b413636`. Phase 5 changes are uncommitted (see §6).

## 2. Files inspected
- `scripts/phase5_mixed_precision_gate.py` (audited for leakage, ordering, seed and paths)
- `scripts/phase3_ptq_studies.py` (origin of the biased S3 selection: per-unit ΔROC-AUC on the TEST split)
- `src/hep_quant/analysis/ptq_studies.py`, `src/hep_quant/quantization/ptq.py`, `src/hep_quant/data/splits.py`
- `reports/exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md`, `research_report.md`, `REAL_HEP_RESULTS.md`, `contribution_statement.md`, `phase3_experiment_plan.md`
- Frozen records (`experiments/results/frozen_phase2/`, read only) and the seed 7/99 records in `experiments/results/`

## 3. Files changed or created in Phase 5

| File | Change |
|---|---|
| `scripts/phase5_mixed_precision_gate.py` | Created. Fixed two defects during audit: stage A now loads with `include_test=False` (test split never preprocessed or held there); the hard-coded `test_set_used_for_selection: False` was replaced by a description of the inputs, so it is not an asserted flag. |
| `tests/test_phase5_selection.py` | Created. 3 unit tests on synthetic data: highest mean validation AUC among mixed layouts wins; FP32 and INT4 base are excluded; tie-break; selection record declares validation-only input. |
| `reports/exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md` | Created. Labels 0.9278 as biased and excluded. |
| `reports/research_report.md`, `reports/REAL_HEP_RESULTS.md`, `reports/contribution_statement.md`, `reports/phase3_experiment_plan.md` | Edited: the 0.9278 figure is marked exploratory/biased, not deleted. |
| `reports/phase5_mixed_precision_gate.md` | Created. Full gate report. |
| `reports/tables/phase5_mixed_*` (4 files) | Created by the gate. |
| `reports/phase5_build_report.md` | This file. |

Not changed: anything under `experiments/results/frozen_phase2/`, the Phase 3 tables, and the Phase 4 artifacts.

**Working-tree note:** `git status` lists `configs/jet_base.yaml`, `scripts/generate_dataset_schema.py`, `src/hep_quant/__main__.py`, `src/hep_quant/data/hls4ml_jets.py` and `src/hep_quant/data/real_validation.py` as modified. `git diff --ignore-cr-at-eol` is empty for all of them: the only difference is CRLF line endings in the working copy against LF in the index. I did not change their content and did not revert them. They should be normalised before committing.

## 4. Experiments executed

| Run | What | Result |
|---|---|---|
| Gate, run 1 | Full protocol: 9 candidates × 5 seeds on validation, selection, one test pass | Selected L0L1A0_int8; test ROC-AUC 0.9281 ± 0.0026 |
| Gate, run 2 | Same script, same code, independent process | **Byte-identical** to run 1 on all four artifacts (`cmp`) |
| Gate, earlier attempt (discarded) | Run before the stage-A loader fix | Stopped by me, outputs deleted before the official runs. Not used for any result. |

## 5. Tests executed

| Suite | Run 1 | Run 2 | Notes |
|---|---|---|---|
| `pytest -q -rs` (full) | 94 passed, 5 skipped (186 s) | 94 passed, 5 skipped (176 s) | Skips: 1 CUDA-only test (CUDA unavailable) + 4 opt-in network tests |
| `tests/test_phase5_selection.py` | 3 passed | — | Run once separately before the gate; included in full runs |
| `scripts/freeze_phase2.py verify` | 84 files, 0 changed | — | Frozen Phase 2 unchanged |

Full suite is 94 = 91 (Phase 0) + 3 new Phase 5 tests.

## 6. Git
- HEAD `b413636`; Phase 5 work uncommitted (31 entries in `git status`, including the CRLF-only items in §3).
- I have **not** committed. The stop rule requires commits only after all validation passes, and Phase 5 is not yet fully validated (§9).
- Nothing destructive was run: no reset, no deletion of experiments or reports.

## 7. Dependency versions (as tested)
Python 3.11.16, PyTorch 2.11.0+cu128, CUDA 12.8 (runtime build; not available in this WSL session), NumPy 2.4.6, scikit-learn 1.9.1, pandas and SciPy as in `requirements-lock.txt`.

## 8. Reproducibility status
- **Two independent gate runs were byte-identical** on CPU. This is the evidence for reproducibility claimed here.
- **Not tested:** GPU or cross-device reproducibility. Phase 4 found that per-seed INT4 PTQ point estimates on CPU are deterministic; the earlier CPU-vs-CUDA gap remains unexplained and is not resolved here.
- No nondeterminism source was found in the gate code. Calibration uses the first 20,480 training jets (deterministic order); splits are seeded; the candidate list is fixed.

## 9. Blockers and open items

1. **Phase 5 is not closed.** The validation-selected layout is dominated by uniform INT6 PTQ on both test ROC-AUC (0.9281 vs 0.9329) and storage (4,196 B vs 3,724 B). The selection rule maximised ROC-AUC alone with no size constraint; that was part of the pre-declared rule and I have kept it, but a size-constrained selection would be a different experiment, which needs its own pre-declaration.
2. **Selection margin is within noise.** L0L1A0 leads L0A0 by 0.0008 on mean validation ROC-AUC. Selection between these two is not reliable.
3. **Marginal advantage over INT4 QAT.** +0.0034 on test, 95% CI [+0.0001, +0.0067], positive in all seeds. Not robust.
4. **Shell execution was blocked for a period** during this phase by the environment's safety classifier (every Bash and PowerShell call returned "no verdict"). I did not work around it. Execution resumed later without any change on my side. The blocked period produced no results and no fabricated output.
5. **Latency and checkpoint size** of the mixed layout are NOT MEASURED. Theoretical storage only.
6. **Hardware-oriented estimate** is NOT MEASURED. No synthesis tool is available (see `docs/hardware.md`).
7. **Previous statement that Phase 5 was "done"** (in Phase 3 documentation and the exploratory labels) is superseded by this report.

## 10. Evidence labelling
- **Experimentally verified (this phase):** gate outputs (two identical runs); test-set performance of the selected layout; baseline comparison using recorded experiment metrics; full-suite results.
- **Exploratory:** the biased 0.9278 layout result. Kept, labelled, not used.
- **Not measured:** latency, checkpoint size, hardware resources, GPU behaviour.

## 11. Next action
Decide whether to pre-declare a size-constrained selection rule (for example, select the highest-validation-AUC layout no larger than INT6 PTQ's theoretical storage) and run it as a new, separately labelled gate. Without that, the Phase 5 result stands as: a validation-selected mixed layout that recovers most of the INT4 PTQ loss but does not beat INT6 PTQ.
