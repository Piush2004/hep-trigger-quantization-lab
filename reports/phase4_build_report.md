# Phase 4 build report (partial)

Status: **PARTIAL.** Phase 4 is not complete. This report covers what was built, what was re-tested, and what remains. It does not claim the Phase 4 final validation.

## 1. Environment (as tested)
- Host: Windows 11; analysis in WSL2 Ubuntu.
- Python 3.11.16; PyTorch 2.11.0+cu128; NumPy/scikit-learn as pinned in `requirements-lock.txt`.
- `torch.cuda.is_available()` = **False** in this WSL session. All Phase 4 computation was CPU-only. The CUDA skip in `tests/test_hardware.py:120` is therefore expected and is reported, not hidden.
- Git HEAD `b413636`. Phase 4 files are uncommitted (see §7).

## 2. Test results (re-run for this report)

| Check | Run 1 | Run 2 | Notes |
|---|---|---|---|
| `pytest -q` | 91 passed, 5 skipped (162.8 s) | 91 passed, 5 skipped (167.0 s) | 5 skips = 1 CUDA-only test + 4 opt-in network tests (`RUN_NETWORK_TESTS=1`) |
| `scripts/freeze_phase2.py verify` | 84 files, 0 changed | (verified once; run before and after Phase 4 work, both 0 changed) | Frozen Phase 2 snapshot unchanged |
| `scripts/phase4_novelty_check.py` | PASS | PASS | Every novelty-bearing line in `research_novelty_analysis.md` is cited or qualified |
| INT4 diagnosis `scripts/phase4_int4_stability.py` | completed | completed, all 5 CSVs byte-identical to run 1 (`cmp`) | Phase 4 Phase-2 analysis |
| Phase 0 `pytest -q` (earlier, same environment) | 91 / 5 | 91 / 5 | Consistent with this report |

**Two-test rule:** met for every check above. Each test passed on two independent runs with matching results.

**Not yet verified by a test:** the report numbers in `int4_ptq_stability_analysis.md` were cross-checked against the saved CSVs (all match). They have no automated test; that check is manual.

## 3. What was built

| Artifact | Purpose | Verified |
|---|---|---|
| `reports/phase4_evidence_lock.md` | Phase 0 state, claims, unsupported claims, open questions | Yes (content reviewed) |
| `reports/related_work_review.md` | Literature review with per-paper verification level (F/S/N) | Partly. F = fetched; S = snippet; author lists of some papers **not verified** (marked) |
| `reports/research_novelty_analysis.md` | Novelty classification: **C (benchmark)**, B only for surviving analyses | Consistency check PASS (TEST #1). Skeptical re-read recorded (TEST #2) |
| `scripts/phase4_novelty_check.py` | Structured check: cited sections exist; priority claims are cited or qualified | Passes twice |
| `scripts/phase4_int4_stability.py` | Causal diagnosis of INT4 PTQ on frozen FP32 checkpoints (no training) | Two runs byte-identical |
| `reports/tables/phase4_int4_*.csv` (5 files) | Raw results: A subsets, B model seeds, C determinism, D calibration size, weight steps | Two runs byte-identical |
| `reports/int4_ptq_stability_analysis.md` | Interpretation of the INT4 diagnosis | Numbers cross-checked against CSVs |
| `reports/phase4_int4_summary.json` | Determinism flag and device info | Written by the script |

## 4. Key results from this build (INT4 PTQ, CPU)

- Calibration size 100 → 10,000 lowers INT4 ROC-AUC from 0.907 to 0.886 (mean over 5 seeds × 5 draws). Paired per-seed difference +0.021, 95% t-interval [+0.010, +0.033]; all five seeds in the same direction.
- Mechanism: the first hidden activation's calibrated range widens (hi 4.9 → 8.1), so the 4-bit step grows 65%. Clip rate at test is ~0 under min/max calibration, so **clipping is not the mechanism**.
- Model seed dominates the spread: between-seed SD of per-seed means 0.029, versus within-seed SD across calibration draws 0.008.
- Determinism: five rebuilds and three repeated inferences on this CPU were identical.
- INT8 and INT6 PTQ are flat across calibration size.

## 5. What is NOT established (do not cite as results)
- **Device dependence** of INT4 PTQ: not tested (no CUDA). Withdrawn as an explanation.
- **Architecture generalization**: not run.
- **Calibration sensitivity beyond min/max** (percentile, MSE, per-channel): not run.
- **Mixed-precision result (0.9278 ± 0.0037, n = 3)**: **test-set selection leakage.** The configuration was chosen by a sensitivity study scored on the test split (`scripts/phase3_ptq_studies.py`, S3). The reported figure is optimistically biased. Fix (re-select on validation, evaluate on test, 5 seeds) is not done.
- **hls4ml numerical consistency** (Phase 6), **hardware synthesis** (Phase 7; toolchain unavailable, not fabricated), **final statistics audit** (Phase 8), **hostile review** (Phase 9), **final research report rewrite** (Phase 10), **docs/REPRODUCIBILITY.md** (Phase 4 item 18): not done.
- **INT4 figures** (performance vs calibration size, variance across subsets, saturation/error, QAT comparison): **not produced.** The tables exist; the figures do not.
- **Zero-point not rounded to an integer** in the fake-quantizer. Recorded as a limitation; not changed.
- **Dataset size discrepancy**: a fetched summary of arXiv:2102.11289 says 870,000 jets; the verified OpenML dataset has 830,000. Unresolved.
- **Two prior-work fetches** returned PDF binary and could not be parsed. Their numbers stay at verification level S.

## 6. Known issues found while building
1. The first diagnosis run failed: seeds 7 and 99 are not in the frozen directory. Fixed with a read-only loader (`load_seed_any`). The frozen directory was not written.
2. The determinism table's inference-hash column is computed on raw logits, while the build-hash column is computed on probabilities. The two are not comparable. Both are stable; the mismatch is a labelling issue in the log, not a result.
3. The initial Phase 0 test attempt in the background was cut off when its shell exited; it was rerun in full and both runs are recorded above.

## 7. Git state
- Phase 4 files are **uncommitted**. I did not commit because the phase is incomplete and the stop rule requires commits only after all validation passes.
- Untracked and not created by this project: `docs/CERN_Study_Guide.pdf` (left alone).
- No secrets, credentials, or temporary datasets were written to the repository. Checkpoints for seeds 7/99 remain under `experiments/checkpoints/` as before.

## 8. Recommended next action
Fix the mixed-precision selection leakage: re-select the L0/A0 configuration using the validation split, then evaluate it on test with the existing frozen checkpoints. This is cheap (no training) and determines whether the 0.9278 figure stands.
