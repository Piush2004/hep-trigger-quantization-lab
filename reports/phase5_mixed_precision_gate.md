# Phase 5 — Validation-selected mixed-precision PTQ (gate report)

**Status: executed, reproduced twice, selection and test performed once.** Scope: post-training quantization on the frozen FP32 checkpoints; one simulated dataset; one 4,389-parameter MLP; five seeds (42, 123, 2026, 7, 99).

## 1. Objective
Replace the test-selected exploratory mixed-precision result (L0 + A0 at INT8, ROC-AUC 0.9278 ± 0.0037, n = 3) with a layout chosen on validation data only, then evaluate that single layout once on the test set.

## 2. Experimental design
Validation selects; the test set is evaluated once, after selection, for the selected layout only.

## 3. Candidate layouts (pre-declared in `scripts/phase5_mixed_precision_gate.py`)
Units: L0–L3 are the four Linear weight tensors; A0–A2 are the three hidden ReLU activations. Every unit is INT4 unless listed.
- Single INT8 upgrades: L0, L1, L2, L3, A0, A1, A2 (7)
- Pairs and triples: L0+A0; L0+L1+A0 (2)
- Total mixed candidates: **9**. Reference (not a candidate): all-INT4 base.

## 4. Dataset split
Stratified 70/15/15 train/validation/test split, seeded per model seed (`split_dataset`, `Preprocessor` fit on train only). Dataset: OpenML `hls4ml_lhc_jets_hlf` v1, 100,000-jet subset.

## 5. Validation protocol (stage A)
For each seed, each candidate is built from the FP32 checkpoint with calibration on the first 20,480 **training** jets, then scored on the **validation** split. Stage A loads with `include_test=False`, so the test split is not preprocessed or held in memory there.

## 6. Selection rule (fixed before running)
Maximise mean validation macro ROC-AUC over the five seeds among mixed layouts. Tie-break: fewer INT8 units, then candidate name. The selection is written to `reports/tables/phase5_mixed_selection.json` before stage B starts.

## 7. Test protocol (stage B)
Evaluate only the selected layout on the test split, once, for the same five seeds.

## 8. Baselines
Recorded test metrics from the experiment records (not used for selection): FP32, INT8 PTQ, INT6 PTQ, INT4 PTQ, INT4 QAT. Same dataset, split procedure, metric definition (macro one-vs-rest ROC-AUC) and seed set.

## 9. Seed policy
Five seeds, same as the primary matrix. Seeds control subsample, split, initialisation and shuffling. Calibration subset is the first 20,480 training jets for every candidate.

## 10. Selected configuration
**L0 + L1 + A0 at INT8; all other units INT4.** Theoretical weight storage 4,196 B.

## 11. Validation results (mean over 5 seeds; all 9 candidates)

| Candidate | Mean val ROC-AUC | SD | INT8 units | Theoretical storage (B) |
|---|---:|---:|---:|---:|
| **L0L1A0_int8 (selected)** | **0.9272** | 0.0022 | 3 | 4,196 |
| L0A0_int8 | 0.9265 | 0.0023 | 2 | 3,172 |
| A0_int8 | 0.9078 | 0.0120 | 1 | 2,660 |
| A1_int8 | 0.8921 | 0.0216 | 1 | 2,660 |
| A2_int8 | 0.8821 | 0.0270 | 1 | 2,660 |
| L3_int8 | 0.8817 | 0.0266 | 1 | 2,740 |
| L1_int8 | 0.8804 | 0.0290 | 1 | 3,684 |
| L2_int8 | 0.8802 | 0.0279 | 1 | 3,172 |
| L0_int8 | 0.8726 | 0.0183 | 1 | 3,172 |

Reference rows on validation: FP32 0.9333 (SD 0.0013); INT4 base 0.8800 (SD 0.0285).

**The selection margin is small.** L0L1A0 leads L0A0 by 0.0008 in mean validation ROC-AUC. Given the per-seed SDs (~0.002), the choice between these two is not reliable; it is the rule's output, not evidence that one layout is clearly better.

## 12. Final test result (selected layout, one evaluation)

| Seed | ROC-AUC | F1 | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.9267 | 0.7363 | 0.7324 |
| 123 | 0.9324 | 0.7556 | 0.7519 |
| 2026 | 0.9269 | 0.7467 | 0.7429 |
| 7 | 0.9259 | 0.7325 | 0.7299 |
| 99 | 0.9284 | 0.7458 | 0.7428 |
| **Mean ± SD** | **0.9281 ± 0.0026** | | |

Range 0.9259–0.9324.

## 13. Baseline comparison (test; same seeds)

| Configuration | Test ROC-AUC (mean ± SD, n = 5) | Theoretical weight storage (B) |
|---|---:|---:|
| FP32 | 0.9343 ± 0.0011 | 17,556 |
| INT8 PTQ | 0.9342 ± 0.0011 | 4,788 |
| INT6 PTQ | 0.9329 ± 0.0014 | 3,724 |
| INT4 QAT | 0.9246 ± 0.0018 | 2,660 |
| **Mixed, validation-selected (L0+L1+A0 INT8)** | **0.9281 ± 0.0026** | **4,196** |
| INT4 PTQ | 0.8819 ± 0.0308 | 2,660 |

Paired differences (mixed minus comparator, per seed, 95% t-interval, n = 5):

| Comparison | Mean difference | 95% CI | Per-seed sign |
|---|---:|---|---|
| vs FP32 | −0.0063 | [−0.0087, −0.0038] | 5/5 negative |
| vs INT4 PTQ | +0.0461 | [+0.0096, +0.0827] | 5/5 positive |
| vs INT4 QAT | +0.0034 | [+0.0001, +0.0067] | 5/5 positive |

### What the comparison supports
- The selected mixed layout recovers most of the INT4 PTQ loss (+0.046 AUC over INT4 PTQ, interval excludes zero).
- Against INT4 QAT it is higher by 0.0034 in every seed. The interval only just excludes zero, and this is a single-layout comparison with five seeds. **Treat as marginal, not as a demonstrated advantage.** INT4 QAT is an untuned recipe (Phase 3 control), and the mixed layout is PTQ with no training.
- The mixed layout is **dominated by uniform INT6 PTQ**: INT6 PTQ is both higher on test (0.9329 vs 0.9281) and smaller (3,724 B vs 4,196 B). The selection rule maximised ROC-AUC alone and had no storage constraint, so it chose a larger layout. **This layout is not on the storage–accuracy Pareto frontier of the tested configurations.**

## 14. Reproducibility check
Two independent runs of `scripts/phase5_mixed_precision_gate.py` from the same code produced **byte-identical** outputs: `phase5_mixed_candidates_validation.csv`, `phase5_mixed_selection.json`, `phase5_mixed_final_test.csv` and `phase5_mixed_reference_test.csv` (verified with `cmp`). Both runs were CPU-only (`cuda_available = False`). This matches the Phase 4 finding that PTQ on this CPU is deterministic for fixed inputs. It says nothing about GPU reproducibility, which was not tested.

## 15. Statistical uncertainty
Descriptive only: sample SD (ddof = 1) and 95% t-intervals with n = 5. No significance tests are claimed. The INT4 QAT comparison is marginal (interval lower bound +0.0001).

## 16. Limitations
- One simulated particle-level dataset and one MLP.
- Post-training quantization only; the mixed layout was not quantization-aware trained.
- Simulated (fake) quantization: no bit-packed artifact, no integer kernels, zero-point not rounded.
- The selection margin over the runner-up is 0.0008, within selection noise.
- The rule had no size constraint, which is why the selected layout is dominated.
- Latency and checkpoint size: **NOT MEASURED** for this layout (theoretical storage only).
- Hardware-oriented estimate: **NOT MEASURED.** No synthesis; see `docs/hardware.md`.

## 17. Historical biased result
The earlier **exploratory** result (L0 + A0 INT8; ROC-AUC 0.9278 ± 0.0037, n = 3) chose its layout using test-split scores and is **not** a confirmatory result. It is preserved and labelled in `reports/exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md`. The validation-selected layout is a different configuration (L0 + L1 + A0) and its 0.9281 is not comparable to the biased figure.

## 18. Scientific interpretation
Validation-selected sensitivity-guided mixed precision at INT8 on three units recovers most of the INT4 PTQ loss on this task. However, on this task it does not beat uniform INT6 PTQ at either accuracy or storage, and its marginal edge over INT4 QAT is not robust enough to claim. The mixed-precision benefit is therefore not established as a distinct advantage here; a size-constrained selection rule would be needed to say whether any mixed layout is on the frontier, and that was not part of the pre-declared protocol.

## 19. Artifacts
- `scripts/phase5_mixed_precision_gate.py` (gate)
- `reports/tables/phase5_mixed_candidates_validation.csv`
- `reports/tables/phase5_mixed_selection.json`
- `reports/tables/phase5_mixed_final_test.csv`
- `reports/tables/phase5_mixed_reference_test.csv`
- `tests/test_phase5_selection.py` (selection-rule unit tests)

## 20. Reproduction command
```bash
wsl -d Ubuntu -e bash -c "cd /mnt/d/Projects/CERN && .venv-wsl/bin/python scripts/phase5_mixed_precision_gate.py"
```
Expected: the four CSV/JSON artifacts above, byte-identical across runs on the same CPU and software environment.
