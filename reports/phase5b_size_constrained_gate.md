# Phase 5B — Size-constrained, validation-selected mixed-precision gate

## 1. Scientific question
Can a mixed-precision PTQ layout reach better validation-selected accuracy than uniform INT6 PTQ while using no more theoretical weight storage than INT6 PTQ?

## 2. Pre-declared selection rule
Among mixed layouts with theoretical weight storage ≤ SIZE_LIMIT_BYTES, select the layout with the highest mean validation macro ROC-AUC over the five seeds (42, 123, 2026, 7, 99). Tie band TIE_EPS = 0.001: layouts within it are treated as indistinguishable and the smaller one is preferred, then candidate name. No test metric participates. Both constants were fixed in `scripts/phase5b_size_constrained_gate.py` before any run and were not changed afterwards.

## 3. Size threshold
SIZE_LIMIT_BYTES = **3,724 B**: theoretical weight storage of uniform INT6 PTQ (4,389 parameters; weights at 6 bits, biases at 4 bytes).

## 4. Candidate space
The Phase 5 space, unchanged: 9 mixed layouts (single INT8 upgrades of L0, L1, L2, L3, A0, A1, A2; pairs and triples L0+A0 and L0+L1+A0). Units not listed are INT4. Nothing was added or removed.

## 5. Eligible candidates
8 of 9 are eligible. The only ineligible one is L0+L1+A0 (4,196 B).

| Candidate | Bit widths | Storage (B) | Eligible | Val ROC-AUC mean | SD | Status |
|---|---|---:|:---:|---:|---:|---|
| **L0A0_int8** | L0 = INT8, A0 = INT8, rest INT4 | 3,172 | yes | **0.9265** | 0.0023 | **SELECTED** |
| L0L1A0_int8 | L0, L1, A0 = INT8 | 4,196 | **no** | 0.9272 | 0.0022 | not eligible |
| A0_int8 | A0 = INT8 | 2,660 | yes | 0.9078 | 0.0120 | eligible, not selected |
| L1_int8 | L1 = INT8 | 3,684 | yes | 0.8804 | 0.0290 | eligible, not selected |
| L2_int8 | L2 = INT8 | 3,172 | yes | 0.8802 | 0.0279 | eligible, not selected |
| L3_int8 | L3 = INT8 | 2,740 | yes | 0.8817 | 0.0266 | eligible, not selected |
| A1_int8 | A1 = INT8 | 2,660 | yes | 0.8921 | 0.0216 | eligible, not selected |
| A2_int8 | A2 = INT8 | 2,660 | yes | 0.8821 | 0.0270 | eligible, not selected |
| L0_int8 | L0 = INT8 | 3,172 | yes | 0.8726 | 0.0183 | eligible, not selected |

The per-seed validation values for every candidate are in `reports/tables/phase5b_mixed_candidates_validation.csv`.

## 6. Selected candidate
**L0A0_int8**: L0 and A0 at INT8, the rest INT4. Theoretical weight storage 3,172 B. The tie band (0.001) contained only this candidate, so no tie-break was applied. The runner-up eligible layout, A0_int8, is 0.0187 lower on validation.

## 7. Validation results
Mean validation ROC-AUC 0.9265, SD 0.0023 (five seeds). Selection was written to `reports/tables/phase5b_mixed_selection.json` before the test set was evaluated.

## 8. Test results (selected layout, one evaluation, five seeds)

| Seed | Test ROC-AUC |
|---:|---:|
| 42 | 0.9245 |
| 123 | 0.9319 |
| 2026 | 0.9271 |
| 7 | 0.9278 |
| 99 | 0.9266 |
| **Mean ± SD** | **0.9276 ± 0.0027** |

Per-seed values are in `reports/tables/phase5b_mixed_final_test.csv`.

## 9. Baseline comparison
Recorded test values for the same five seeds (not used for selection):

| Configuration | Test ROC-AUC (mean ± SD) | Theoretical weight storage (B) |
|---|---:|---:|
| FP32 | 0.9343 ± 0.0011 | 17,556 |
| INT8 PTQ | 0.9342 ± 0.0011 | 4,788 |
| **INT6 PTQ** | **0.9329 ± 0.0014** | **3,724** |
| Mixed L0L1A0 (Phase 5; not size-constrained) | 0.9281 ± 0.0026 | 4,196 |
| **Mixed L0A0 (Phase 5B; size-constrained, selected)** | **0.9276 ± 0.0027** | **3,172** |
| INT4 QAT | 0.9246 ± 0.0018 | 2,660 |
| INT4 PTQ | 0.8819 ± 0.0308 | 2,660 |

Paired differences, mixed L0A0 minus comparator (same seeds, 95% t-interval, n = 5):

| Comparator | Mean difference | 95% CI | Seeds positive |
|---|---:|---|---:|
| **INT6 PTQ** | **−0.0053** | **[−0.0078, −0.0029]** | **0 / 5** |
| INT4 QAT | +0.0030 | [+0.0006, +0.0053] | 5 / 5 |
| INT4 PTQ | +0.0456 | [+0.0099, +0.0814] | 5 / 5 |
| FP32 | −0.0067 | [−0.0093, −0.0041] | 0 / 5 |

## 10. Pareto analysis
Axes: theoretical weight storage (x) and mean test ROC-AUC (y). A point is dominated if another point is no larger and no worse, with at least one strict improvement.

- **INT6 PTQ** (3,724 B, 0.9329): not dominated. The selected layout is smaller and cannot dominate it.
- **Selected L0A0** (3,172 B, 0.9276): **not dominated.** It is smaller than INT6 PTQ, and the only configurations that are no larger than it (INT4 QAT, INT4 PTQ) have lower ROC-AUC. It lies between INT4 QAT and INT6 PTQ on the frontier of the evaluated set.
- **Mixed L0L1A0** (4,196 B, 0.9281): **dominated** by INT6 PTQ (smaller and higher). Confirms the Phase 5 result.

So the selected layout is a non-dominated point, but it is an intermediate trade-off, not an improvement on INT6. It does not beat INT6 on the criterion the question set.

## 11. Statistical comparison
- Against INT6 PTQ: −0.0053, CI excludes zero, negative in all five seeds. **INT6 PTQ is statistically better in ROC-AUC.**
- Against INT4 QAT: +0.0030, CI [+0.0006, +0.0053], positive in all five seeds. This is a real but small gain over INT4 QAT at 19% more storage, and it does not change the INT6 conclusion.
- Descriptive intervals only; no significance tests are claimed.

## 12. Leakage audit
- Stage A (validation scoring) uses `include_test=False`. The test split is not loaded for candidate scoring. The candidate table is asserted to contain no column whose name includes "test".
- Selection was computed and written to `phase5b_mixed_selection.json` before stage B ran. The script then evaluated the test set once, for the selected layout only.
- **Disclosed caveat:** the layout L0A0 was the 0.9278 exploratory layout, chosen earlier on test-set scores. Its present selection is made on validation (it is the highest-scoring eligible layout on validation), so the selection itself is clean. Its 5-seed test value (0.9276) is, however, the first confirmatory test evaluation of this layout under the corrected protocol. It should be read as such and not as independent of the earlier test-set exploration.
- **Disclosed mechanics:** the selection JSON was rewritten after stage B to add the final test fields and `test_set_used_after_selection: true`. The pre-test selection content was serialized first, as the code order enforces, but the final file contains the post-test state.
- The historical 0.9278 result remains labelled `EXPLORATORY — TEST-SET SELECTED — BIASED — EXCLUDED FROM CONFIRMATORY ANALYSIS` (`reports/exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md`) and is kept separate from this gate.

## 13. Reproducibility
- Gate run twice from the same code (TEST #1 and TEST #2). All five output files were **byte-identical** (`cmp`): `phase5b_mixed_candidates_validation.csv`, `phase5b_mixed_selection.json`, `phase5b_mixed_final_test.csv`, `phase5b_paired_comparisons.csv`, `phase5b_pareto.csv`.
- CPU only (`cuda_available = False`). GPU reproducibility was not tested.
- Regression suite, run twice: **94 passed, 5 skipped** on both runs, matching the expected baseline. No tests were added in this gate.
- Frozen Phase 2 snapshot: `scripts/freeze_phase2.py verify` → 84 files checked, 0 changed (run after the gate).

## 14. Limitations
- One simulated particle-level dataset and one 4,389-parameter MLP.
- PTQ only; the layout was not quantization-aware trained. INT4 QAT is an untuned recipe, so the comparison with it is not a tuned-versus-tuned one.
- Theoretical storage is analytical, not a checkpoint size. Checkpoints are float32 and no packed artifact exists.
- Latency and hardware estimates for the layout are **NOT MEASURED** and are not claimed here.
- The search space is nine layouts. Mixed precision was not searched exhaustively, so "best within the evaluated candidate space" is the strongest available claim, and it says nothing about the best possible mixed layout.
- Small validation margins: the eligible runner-up is 0.0187 lower, but the top layouts' SDs are about 0.002.

## 15. Final conclusion
Under the pre-declared 3,724 B constraint, the validation-selected mixed layout (L0 and A0 at INT8, rest INT4) reaches 0.9276 ± 0.0027 test ROC-AUC at 3,172 B. It is a non-dominated intermediate point between INT4 QAT and INT6 PTQ, and it is marginally better than INT4 QAT. It does **not** improve on uniform INT6 PTQ, which is both more accurate (0.9329 vs 0.9276, CI excludes zero in every seed) and only 552 B larger.

**Mixed-precision status: NO ADVANTAGE OVER INT6.**

Per the stop condition, this gate is closed. No further mixed-precision layout search, size-threshold change or candidate expansion is warranted by this result.
