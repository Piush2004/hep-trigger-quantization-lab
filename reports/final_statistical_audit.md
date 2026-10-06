# Final statistical audit

Canonical table: `reports/tables/final_results.csv` (built by `scripts/phase8_final_audit.py` from existing per-seed records; no new training). Test-set macro ROC-AUC unless stated.

## Rules applied
- **n** is the number of seeds. Every configuration here has n = 5 (seeds 42, 123, 2026, 7, 99).
- **Mean, SD (sample, ddof = 1), min, max** are descriptive.
- **95% CI** is a t-interval on the mean over n = 5 seeds. It describes uncertainty over seeds; it is not a significance test.
- **Degradation** is measured against the same architecture's FP32 on the same seeds: absolute (ROC-AUC points) and percentage of the FP32 mean.
- **Statistical significance** is claimed only where a paired interval excludes zero. No p-values were computed, and "significant" is not used for descriptive intervals.
- **Practical significance** is judged separately, with a stated threshold: a change of 0.001 ROC-AUC or less is treated as practically negligible on this task; this threshold is a judgement, not a statistical result.

## Canonical results (n = 5, ROC-AUC)

| Architecture | Configuration | Mean | SD | Min | Max | 95% CI | Δ vs FP32 | % | Storage (B) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| A primary (frozen, CUDA-era) | FP32 | 0.9343 | 0.0011 | — | — | — | — | — | 17,556 |
| A primary | INT8 PTQ | 0.9342 | 0.0011 | — | — | — | −0.0001 | −0.01 | 4,788 |
| A primary | INT6 PTQ | 0.9329 | 0.0014 | — | — | — | −0.0014 | −0.15 | 3,724 |
| A primary | INT4 PTQ | 0.8819 | 0.0308 | 0.835 | 0.909 | see final_results.csv | −0.0524 | −5.6 | 2,660 |
| A primary | INT4 QAT | 0.9246 | 0.0018 | — | — | — | −0.0097 | −1.0 | 2,660 |
| A primary | Mixed L0+A0 INT8 (Phase 5B, size-constrained) | 0.9276 | 0.0027 | — | — | — | −0.0067 | −0.7 | 3,172 |
| A primary | Mixed L0+L1+A0 INT8 (Phase 5, validation-selected) | 0.9281 | 0.0026 | — | — | — | −0.0063 | −0.7 | 4,196 |
| A re-run on CPU | FP32 | 0.9343 | 0.0015 | 0.9322 | 0.9357 | [0.9324, 0.9361] | — | — | 17,556 |
| A re-run on CPU | INT8 PTQ | 0.9342 | 0.0016 | 0.9320 | 0.9357 | [0.9322, 0.9362] | −0.0001 | −0.01 | 4,788 |
| A re-run on CPU | INT6 PTQ | 0.9333 | 0.0017 | 0.9310 | 0.9349 | [0.9312, 0.9353] | −0.0010 | −0.11 | 3,724 |
| A re-run on CPU | INT4 PTQ | 0.8904 | 0.0122 | 0.8807 | 0.9075 | [0.8752, 0.9056] | −0.0438 | −4.7 | 2,660 |
| A re-run on CPU | INT4 QAT | 0.9249 | 0.0014 | 0.9232 | 0.9267 | [0.9231, 0.9266] | −0.0094 | −1.0 | 2,660 |
| B [128,64,64,32] | FP32 | 0.9376 | 0.0009 | 0.9366 | 0.9389 | [0.9365, 0.9388] | — | — | 67,348 |
| B | INT8 PTQ | 0.9376 | 0.0009 | 0.9366 | 0.9388 | [0.9364, 0.9387] | −0.0000 | 0.00 | 17,716 |
| B | INT6 PTQ | 0.9366 | 0.0010 | 0.9351 | 0.9379 | [0.9353, 0.9379] | −0.0010 | −0.11 | 13,580 |
| B | INT4 PTQ | 0.9112 | 0.0045 | 0.9048 | 0.9156 | [0.9056, 0.9168] | −0.0264 | −2.8 | 9,444 |
| B | INT4 QAT | 0.9307 | 0.0018 | 0.9284 | 0.9326 | [0.9285, 0.9329] | −0.0070 | −0.7 | 9,444 |

Min, max and CI are shown where per-seed values are in the recorded Phase 6 table; for the frozen primary rows they are in `final_results.csv`, and the primary INT4 PTQ range (0.835–0.909) is quoted from earlier audits.

**Note on the primary rows:** the mixed-precision rows have only ROC-AUC recorded in their source tables, so F1, accuracy and PR-AUC are blank for them rather than estimated.

## Paired comparisons (same seeds)
These come from the Phase 6 and Phase 5B paired tables.

| Comparison | Mean difference | 95% CI | Seeds in same direction | Reading |
|---|---:|---|---|---|
| A (CPU): INT8 PTQ − FP32 | −0.0001 | [−0.0002, 0.0000] | 4/5 negative | practically zero (≤ 0.001) |
| A (CPU): INT6 PTQ − FP32 | −0.0010 | [−0.0013, −0.0007] | 5/5 negative | statistically established, practically small |
| A (CPU): INT4 PTQ − FP32 | −0.0438 | [−0.0576, −0.0301] | 5/5 negative | statistically and practically large |
| A (CPU): INT4 QAT − INT4 PTQ | +0.0344 | [+0.0189, +0.0499] | 5/5 positive | QAT recovery, established |
| B: INT4 PTQ − FP32 | −0.0264 | [−0.0322, −0.0206] | 5/5 negative | established, practically large |
| B: INT4 QAT − INT4 PTQ | +0.0194 | [+0.0129, +0.0259] | 5/5 positive | QAT recovery, established |
| Degradation B − A, INT4 PTQ | +0.0175 | [−0.0009, +0.0358] | 4/5 positive | **not statistically established** |
| Mixed L0A0 (5B) − INT6 PTQ | −0.0053 | [−0.0078, −0.0029] | 0/5 positive | INT6 statistically better |
| Mixed L0A0 (5B) − INT4 QAT | +0.0030 | [+0.0006, +0.0053] | 5/5 positive | statistically positive, practically small |

## What is and is not established
- **Established:** INT6 PTQ is slightly but reliably below FP32; INT4 PTQ is reliably and substantially below FP32 in both architectures; INT4 QAT recovers most of that loss in both; INT6 is statistically better than the size-constrained mixed layout.
- **Not established:** that the INT4 PTQ loss is smaller for architecture B (interval includes zero); that INT4 PTQ instability (seed SD) is a stable property (A's SD differs by device: 0.0308 on CUDA, 0.0122 on CPU); that mixed precision beats INT6 under the 3,724 B constraint.
- **Practically negligible:** INT8 PTQ versus FP32 (Δ ≈ 0.0001).

## Caveats that affect every interval
- n = 5 seeds is a small sample. The intervals describe seed-to-seed variation for this dataset and these architectures only.
- The primary architecture's frozen numbers come from CUDA; the Phase 6 re-run came from CPU. Device is therefore not held constant across the whole table, and cross-row absolute comparisons carry that caveat.
- Storage is theoretical (analytical). It is not checkpoint size; checkpoints are float32.
- Latency is not included: the only latency numbers in the project are software-emulation timings, which are not hardware measurements.
