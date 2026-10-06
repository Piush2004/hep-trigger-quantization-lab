# Final results

All accuracy numbers are **MEASURED** (PyTorch, test split, macro one-vs-rest ROC-AUC, five seeds). Storage is **analytical** (theoretical weight storage), not checkpoint size. Hardware numbers are **SIMULATED** (hls4ml C-simulation) or **ESTIMATED** (software heuristic); none is synthesized or physical. Canonical table: `reports/tables/final_results.csv`. Statistics and rules: `reports/final_statistical_audit.md`.

## Primary architecture A (4,389 parameters), frozen records, n = 5

| Configuration | ROC-AUC mean ± SD | Δ vs FP32 | Storage (B) |
|---|---:|---:|---:|
| FP32 | 0.9343 ± 0.0011 | — | 17,556 |
| INT8 PTQ | 0.9342 ± 0.0011 | −0.0001 | 4,788 |
| INT6 PTQ | 0.9329 ± 0.0014 | −0.0014 | 3,724 |
| INT4 PTQ | 0.8819 ± 0.0308 | −0.0524 | 2,660 |
| INT4 QAT | 0.9246 ± 0.0018 | −0.0097 | 2,660 |

## Mixed precision (no longer a headline result)
- **Exploratory, biased, excluded:** L0+A0 INT8, 0.9278 ± 0.0037 (n = 3). Layout chosen on test-split scores.
- **Validation-selected, not size-constrained (Phase 5):** L0+L1+A0 INT8, 0.9281 ± 0.0026, 4,196 B. Dominated by INT6 PTQ.
- **Validation-selected under 3,724 B (Phase 5B):** L0+A0 INT8, 0.9276 ± 0.0027, 3,172 B. Below INT6 PTQ by 0.0053 (CI excludes zero). Non-dominated between INT4 QAT and INT6 PTQ.
- **Conclusion:** no evaluated mixed layout outperformed uniform INT6 PTQ under the constraint.

## Architecture generalization (Phase 6, CPU, same protocol, n = 5)

| Architecture | FP32 | INT8 PTQ Δ | INT6 PTQ Δ | INT4 PTQ Δ | INT4 QAT Δ | QAT share of INT4 loss recovered |
|---|---:|---:|---:|---:|---:|---:|
| A [64,32,32], 4,389 params (CPU) | 0.9343 ± 0.0015 | −0.0001 | −0.0010 | −0.0438 (SD 0.0122) | −0.0094 | 78.5% |
| B [128,64,64,32], 16,837 params (CPU) | 0.9376 ± 0.0009 | 0.0000 | −0.0010 | −0.0264 (SD 0.0045) | −0.0070 | 73.6% |

Architecture B's INT4 PTQ loss is smaller by 0.0175 (CI includes zero; not established). INT8 and INT6 degradation are the same in both.

## hls4ml numerical consistency (SIMULATED C-simulation, 15,000 test jets)
- Default arithmetic (truncation, wrap, accumulator = W bits): accuracy 0.2021 / 0.3541 / 0.7255 / 0.7481 / 0.7511 at W = 8 / 10 / 12 / 14 / 16 (FP32 PyTorch 0.7495).
- Rounding and saturation, accumulator = W: 0.5034 / 0.7343 / 0.7489 / 0.7505 / 0.7497.
- Wide accumulator with default rounding: 0.2017 / 0.5486 / 0.7462 / 0.7507 / 0.7500.
- Rounding, saturation and wide accumulator: 0.6508 / 0.7471 / 0.7500 / 0.7502 / 0.7498.
- Interpretation in `reports/phase7_hls_numerical_validation.md`.

## Hardware evidence
- MEASURED: PyTorch accuracy and ROC-AUC (all configurations).
- SIMULATED: hls4ml C-simulation accuracy and agreement (table above).
- ESTIMATED: LUT/DSP/BRAM/FF from a linear software heuristic (not validated against synthesis).
- SYNTHESIZED: **not available.** HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT.
- PHYSICAL: **not available.**

## Key limitations
- One simulated particle-level dataset (830,000 jets; subset 100,000); two small MLPs.
- Simulated (fake) quantization; zero-point not rounded; storage analytical; no integer deployment.
- INT4 PTQ magnitude and seed spread are device-dependent (CPU vs CUDA).
- Five seeds; intervals are descriptive.
- No hardware synthesis or physical measurement.
