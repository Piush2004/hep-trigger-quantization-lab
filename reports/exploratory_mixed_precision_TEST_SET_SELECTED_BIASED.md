# Exploratory mixed-precision result — TEST_SET_SELECTED / BIASED

**Status: excluded from the confirmatory analysis due to test-set selection leakage.**

## What the number is
- Layout: first Linear weight tensor (L0) and first hidden activation (A0) at INT8, all other units INT4, PTQ on the frozen FP32 checkpoints of seeds 42, 123, 2026.
- Reported as ROC-AUC 0.9278 ± 0.0037 (n = 3), theoretical weight storage 3,172 B.
- Originally produced by `scripts/phase3_ptq_studies.py` (per-unit study S3), whose per-unit ΔROC-AUC was computed on the **TEST** split.

## Why it is biased
The layout was chosen after looking at test-split ROC-AUC for each unit. Any layout picked this way will look better on test than it will on new data. The 0.9278 figure therefore does not estimate the generalisation performance of the chosen layout, and must not be compared with the confirmatory results.

## What is preserved
- The original tables (`reports/tables/phase3_ptq_s3_units.csv`, related outputs) are kept unchanged as the historical record. They are not deleted.
- Mentions of 0.9278 elsewhere in the reports are retained for traceability and marked as exploratory.

## What replaces it
The validation-selected, test-once analysis: `scripts/phase5_mixed_precision_gate.py`. Its candidates, validation scores, selection rule and final untouched test score are in `reports/tables/phase5_mixed_*` and `reports/phase5_mixed_precision_gate.md`.
