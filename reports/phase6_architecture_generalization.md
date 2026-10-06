# Phase 6 — architecture generalization

## 1. Scientific question
Are the observed precision–performance relationships specific to the primary MLP, or do they hold for a second lightweight architecture?

## 2. Architecture selection
Selected before any run (`reports/phase6_architecture_selection.md`).
- **Architecture A (primary):** 16 → 64 → 32 → 32 → 5 (hidden [64, 32, 32], ReLU, dropout 0.1). **4,389 parameters.**
- **Architecture B (second):** 16 → 128 → 64 → 64 → 32 → 5 (hidden [128, 64, 64, 32], ReLU, dropout 0.1). **16,837 parameters**, one more hidden layer and roughly 3.8× the parameters.

B differs from A only in `model.hidden_dims`. Same input, output, activation, dropout and training protocol.

## 3. Experimental methodology
- 2 architectures × 5 configurations (FP32, INT8 PTQ, INT6 PTQ, INT4 PTQ, INT4 QAT) × 5 seeds (42, 123, 2026, 7, 99) = **50 experiments per run**.
- **Device control:** all Phase 6 runs are on **CPU**. Architecture A was re-run under this protocol rather than reusing the frozen CUDA results, because Phase 4 showed INT4 PTQ values change with device. The frozen CUDA results are reported only as a reference.
- Driver: `scripts/phase6_arch_generalization.py --run-tag run1|run2`. Each run writes to its own result and checkpoint folders (`experiments/results_phase6_<tag>/`, `experiments/checkpoints_phase6_<tag>/`). Phase 2–5 result folders were not written.
- Two independent full runs were executed (TEST #1 and TEST #2).

## 4. Dataset and split
`hls4ml_lhc_jets_hlf` (OpenML v1), 100,000-jet stratified subset per seed. Train/validation/test split 70/15/15, seed-controlled, identical procedure for A and B. Preprocessing fitted on the training split only. The dataset is a publicly available simulated particle-level Monte Carlo dataset, not detector data.

## 5. Training methodology (identical for A and B)
- Optimizer: Adam; learning rate 1e-3; weight decay 1e-4; batch size 1024.
- Epochs: at most 30, early stopping with patience 6 on validation loss. Checkpoint chosen by validation loss.
- FP32 baseline is trained per experiment, from the seed.
- **QAT:** 7 fine-tuning epochs (`max(5, E//4)`) at 0.1× learning rate, straight-through estimator, starting from the same-seed FP32 model. Same recipe for A and B.
- No hyperparameter was tuned on any split for either architecture.

## 6. Quantization methodology (identical for A and B)
Simulated uniform affine (fake) quantization, as in the primary study: Linear weight tensors and hidden ReLU outputs are quantized per tensor; inputs, logits, biases and accumulators are not. PTQ calibration uses the first 20,480 training jets. Caveat carried from earlier phases: the zero-point is not rounded to an integer.

## 7. Results

### Storage and parameters
| | Architecture A | Architecture B |
|---|---:|---:|
| Parameters | 4,389 | 16,837 |
| FP32 weight storage | 17,556 B | 67,348 B |
| INT8 weight storage | 4,788 B | 17,716 B |
| INT6 weight storage | 3,724 B | 13,580 B |
| INT4 weight storage | 2,660 B | 9,444 B |

Theoretical storage: quantized weights at their bit-width, biases at 4 bytes. Analytical, not checkpoint sizes (checkpoints are float32; no packed artifact exists). Formula checked against the primary architecture's documented values.

### Test metrics, mean ± SD over 5 seeds (Run 1; Run 2 identical, see §13)

| Architecture | Precision | Method | ROC-AUC mean | ROC-AUC SD | Min | Max | 95% CI (ROC-AUC) | Δ vs own FP32 | Δ% |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| A | FP32 | — | 0.9343 | 0.0015 | 0.9322 | 0.9357 | [0.9324, 0.9361] | — | — |
| A | INT8 | PTQ | 0.9342 | 0.0016 | 0.9320 | 0.9357 | [0.9322, 0.9362] | −0.0001 | −0.01% |
| A | INT6 | PTQ | 0.9333 | 0.0017 | 0.9310 | 0.9349 | [0.9312, 0.9353] | −0.0010 | −0.11% |
| A | INT4 | PTQ | 0.8904 | 0.0122 | 0.8807 | 0.9075 | [0.8752, 0.9056] | −0.0438 | −4.7% |
| A | INT4 | QAT | 0.9249 | 0.0014 | 0.9232 | 0.9267 | [0.9231, 0.9266] | −0.0094 | −1.0% |
| B | FP32 | — | 0.9376 | 0.0009 | 0.9366 | 0.9389 | [0.9365, 0.9388] | — | — |
| B | INT8 | PTQ | 0.9376 | 0.0009 | 0.9366 | 0.9388 | [0.9364, 0.9387] | −0.0000 | 0.00% |
| B | INT6 | PTQ | 0.9366 | 0.0010 | 0.9351 | 0.9379 | [0.9353, 0.9379] | −0.0010 | −0.11% |
| B | INT4 | PTQ | 0.9112 | 0.0045 | 0.9048 | 0.9156 | [0.9056, 0.9168] | −0.0264 | −2.8% |
| B | INT4 | QAT | 0.9307 | 0.0018 | 0.9284 | 0.9326 | [0.9285, 0.9329] | −0.0070 | −0.7% |

Min, max and per-seed values, PR-AUC, accuracy and macro F1 for every configuration and seed are in `reports/tables/phase6_run1_per_seed.csv` and `reports/tables/phase6_run1_summary.csv`. 95% CIs are t-intervals over five seeds (descriptive).

Reference only, not used in the comparison: the frozen CUDA primary results were FP32 0.9343 ± 0.0011, INT4 PTQ 0.8819 ± 0.0308 and INT4 QAT 0.9246 ± 0.0018. The CPU re-run of A reproduces the FP32 level but gives a smaller INT4 PTQ spread (0.0122).

## 8. Statistical analysis
Paired differences (same seed), mean with 95% t-interval, n = 5 (`reports/tables/phase6_run1_paired.csv`):

| Comparison | Mean difference | 95% CI | Seeds in same direction |
|---|---:|---|---|
| A: INT6 PTQ − FP32 | −0.0010 | [−0.0013, −0.0007] | 5/5 negative |
| A: INT4 PTQ − FP32 | −0.0438 | [−0.0576, −0.0301] | 5/5 negative |
| A: INT4 QAT − FP32 | −0.0094 | [−0.0119, −0.0069] | 5/5 negative |
| A: QAT − PTQ (recovery) | +0.0344 | [+0.0189, +0.0499] | 5/5 positive |
| B: INT6 PTQ − FP32 | −0.0010 | [−0.0014, −0.0007] | 5/5 negative |
| B: INT4 PTQ − FP32 | −0.0264 | [−0.0322, −0.0206] | 5/5 negative |
| B: INT4 QAT − FP32 | −0.0070 | [−0.0083, −0.0056] | 5/5 negative |
| B: QAT − PTQ (recovery) | +0.0194 | [+0.0129, +0.0259] | 5/5 positive |
| **Degradation B − A, INT8 PTQ** | 0.0000 | [−0.0001, +0.0002] | 2 / 3 split |
| **Degradation B − A, INT6 PTQ** | −0.0000 | [−0.0005, +0.0004] | 2 / 3 split |
| **Degradation B − A, INT4 PTQ** | **+0.0175** | **[−0.0009, +0.0358]** | 4 / 1 |
| **Degradation B − A, INT4 QAT** | +0.0025 | [−0.0005, +0.0055] | 4 / 1 |

Statistical versus practical significance:
- **INT8 and INT6 degradation is identical across architectures** (differences ≈ 0; practically none).
- **INT4 PTQ is less degraded in B by 0.0175**, but the interval includes zero and only 4 of 5 seeds agree. This is **not statistically established** at n = 5. It is practically notable (about 40% smaller loss in B) but should be read as suggestive.
- **QAT recovery of the INT4 PTQ loss:** A recovers **78.5%** of its loss; B recovers **73.6%**. Both recover reliably (5/5 seeds).
- Between-seed spread of INT4 PTQ: A SD 0.0122, B SD 0.0045 (descriptive; no test of variance equality is claimed).

## 9. Architecture comparison
- The qualitative pattern is the same in both architectures: INT8 lossless, INT6 close to FP32, INT4 PTQ clearly degraded, INT4 QAT recovers most of the loss.
- The **magnitude** of INT4 PTQ degradation and its seed spread are smaller for B. This is consistent with a deeper/wider model being more robust at 4 bits, but the difference is not established at five seeds.
- B has a higher FP32 baseline (0.9376 vs 0.9343), so absolute INT4 PTQ values are not comparable across architectures. The comparison above uses degradation relative to each architecture's own FP32.

## 10. INT4 behaviour
- INT4 PTQ degrades both architectures on every seed (5/5 negative).
- The spread is larger for A (0.0122) than for B (0.0045) on CPU.
- **Device dependence:** the frozen CUDA INT4 PTQ spread for A was 0.0308, versus 0.0122 on CPU. The magnitude of INT4 PTQ instability therefore depends on device, as found in Phase 4. The qualitative degradation does not.
- Phase 4's calibration analysis was not repeated here, per the phase rules. The calibration-size mechanism reported in Phase 4 was established for architecture A only.

## 11. QAT recovery
- A: QAT recovers 78.5% of the INT4 PTQ loss (+0.0344 ROC-AUC, 5/5 seeds).
- B: QAT recovers 73.6% (+0.0194 ROC-AUC, 5/5 seeds).
- QAT remains 0.0094 (A) and 0.0070 (B) below its own FP32. Recovery is substantial but incomplete in both.
- The compute-matched control from Phase 3 (FP32 with the same extra epochs) was not repeated for B; it is therefore not established for architecture B whether QAT's gain comes from quantization-awareness or from extra training.

## 12. Limitations
- Two architectures, both small MLPs of the same family (B is deeper and wider, not a different topology class).
- One simulated particle-level dataset and one subset size (100,000 jets).
- Simulated (fake) quantization only; storage is analytical; no latency or hardware measurements.
- CPU only. The primary architecture's frozen CUDA results were not reproduced in the same environment, so cross-device absolute values are not compared.
- The INT4 PTQ magnitude difference between architectures is not statistically established at n = 5.
- The compute-matched control was not repeated for architecture B.
- The INT4 instability (large seed spread) is not clearly generalizing: its magnitude is smaller for B and device-dependent for A.

## 13. Reproducibility
- **TEST #1 and TEST #2** (independent full runs, 50 experiments each, same code and configuration): all 50 per-seed metric rows match **exactly** on ROC-AUC, PR-AUC, accuracy and macro F1 (`scripts/phase6_compare_runs.py`).
- All **50 saved checkpoints are byte-identical** between the two runs.
- Summary tables and figures are generated from the same inputs by `scripts/phase6_analysis.py`. Result JSON files were compared on their metric fields; registry metadata (timestamps, durations) was not compared byte-wise.
- The frozen Phase 2 snapshot checks clean: **84 files, 0 changed**.

## 14. Regression tests
`pytest -q` run twice after both experiment runs: **94 passed, 5 skipped** on both runs (expected baseline). The 5 skips are the CUDA-only test and four opt-in network tests. No test count change.

## 15. Scientific interpretation
The precision–performance pattern generalizes across two MLP sizes:
- INT8 PTQ is practically lossless in both (|Δ| ≤ 0.0001).
- INT6 PTQ costs about 0.001 ROC-AUC in both.
- INT4 PTQ degrades both models on every seed, by 0.044 (A) and 0.026 (B).
- INT4 QAT recovers 74–79% of that loss in both.

Two aspects do not generalize cleanly: the size of the INT4 PTQ loss (smaller for B, not established at n = 5) and the INT4 seed spread (smaller for B, and device-dependent for A). They should be reported as architecture- and device-sensitive magnitudes, not as a stable property.

**Outcome: QUANTIZATION PATTERN GENERALIZES**, with the qualifications above. The claim is limited to two small MLPs on one simulated dataset.
