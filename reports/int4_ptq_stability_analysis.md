# INT4 PTQ stability analysis (Phase 4, Phase 2)

Script: `scripts/phase4_int4_stability.py`. Tables: `reports/tables/phase4_int4_*.csv`.
Inputs: the five FP32 checkpoints (seeds 42, 123, 2026 from the frozen set; 7 and 99 from `experiments/checkpoints/`). **No training was run and nothing under `experiments/results/frozen_phase2/` was written.**
Device: **CPU only.** `torch.cuda.is_available()` = False in this session, so **no device comparison was made.**

## Question
The INT4 PTQ result (ROC-AUC 0.8819 ± 0.0308) has a large spread. Which factor explains it: calibration-set selection, calibration-set size, model seed, a numerical effect, or determinism? Previous reports described it as "device-dependent"; that wording is **not established by this analysis** and is withdrawn here as an explanation.

## Test-of-hypothesis design
| Exp | Design | What it isolates |
|---|---|---|
| A | Fixed model (seed 42); 10 calibration subsets of 1000 training jets | calibration-subset variance, fixed model |
| B | Each of 5 seeds; 5 subsets of 1000 each | model-seed variance vs calibration variance |
| C | Seed 42, one calibration set, 5 rebuilds + 3 repeated inferences | determinism |
| D | Calibration size 100 / 500 / 1000 / 5000 / 10000; 5 draws each; 5 seeds; INT8, INT6, INT4 | calibration-size effect |

Diagnostics: calibrated activation range [lo, hi] for each hidden ReLU; quantization step (hi−lo)/(2^b−1); clip rate (fraction of test pre-quantizer activations outside the range); argmax disagreement and mean |Δp| against the FP32 model; per-layer weight min/max and step.

## Results

**C. Determinism (TEST #1 and TEST #2 both identical).** Five independent rebuilds of the INT4 variant produced bit-identical probabilities (SHA-256 prefix `ce931c87ac49fd72`). Three repeated inferences on the same built model also matched. The inference hash `d913a0ac15c78b15` is computed on raw logits rather than probabilities, so it is not comparable to the build hash; both hashes are stable, and this is a definitional difference in the log, not a failure. **Conclusion: on this CPU, INT4 PTQ is deterministic for a fixed model and calibration set.**

**D. Calibration-size sweep (INT4, mean over 5 seeds × 5 draws).**

| Calibration size | ROC-AUC | F1 | Accuracy | A0 hi | A0 step | Clip rate (A0) | Argmax disagreement vs FP32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.9073 ± 0.0214 | 0.688 | 0.690 | 4.88 | 0.325 | 0.0005 | 0.128 |
| 500 | 0.8997 ± 0.0257 | 0.675 | 0.679 | 6.33 | 0.422 | 0.0001 | 0.146 |
| 1000 | 0.8946 ± 0.0283 | 0.666 | 0.670 | 6.91 | 0.461 | 0.0000 | 0.158 |
| 5000 | 0.8881 ± 0.0290 | 0.652 | 0.658 | 7.82 | 0.521 | 0.0000 | 0.176 |
| 10000 | 0.8862 ± 0.0279 | 0.648 | 0.654 | 8.05 | 0.537 | 0.0000 | 0.181 |

Reference (unchanged): INT8 and INT6 PTQ are flat across calibration sizes (INT8 0.9343 → 0.9342; INT6 0.9335 → 0.9330).

**Paired effect of calibration size (per seed, mean over draws).** AUC(n=100) − AUC(n=10000) = +0.022, +0.036, +0.013, +0.013, +0.021 for seeds 42, 123, 2026, 7, 99. Mean +0.021, 95% t-interval (n = 5) [+0.010, +0.033]. **All five seeds move in the same direction.** This is a descriptive interval; no significance test is claimed.

**B. Model seed vs calibration draw (INT4, n = 1000, 5 draws per seed).**

| Seed | Mean ROC-AUC | SD across calibration draws |
|---:|---:|---:|
| 42 | 0.848 | 0.007 |
| 123 | 0.921 | 0.001 |
| 2026 | 0.884 | 0.011 |
| 7 | 0.911 | 0.019 |
| 99 | 0.910 | 0.003 |

Between-seed SD of the per-seed means: **0.029**. Mean within-seed SD across calibration draws: **0.008**. **The dominant source of variation is the model seed, not the calibration subset.** The 0.835–0.909 range of the 5-seed result is therefore mostly a property of the trained model.

**A. Fixed model (seed 42), 10 calibration subsets of 1000.** ROC-AUC 0.840–0.856; the spread is small and is consistent with B.

**Weight quantization is calibration-free.** Weight ranges come from each tensor's min/max, so weight steps do not depend on the calibration set. For seed 42 at INT4 the steps are L0 0.139, L1 0.128, L2 0.099, L3 0.099. Any calibration effect must therefore act on the activation side.

## Causal interpretation

1. **Calibration size acts through the activation range, not through clipping.** Larger calibration sets capture more extreme activations, so the first hidden activation's range widens (A0 hi 4.9 → 8.1). The 16-level grid step then grows by 65% (0.33 → 0.54). The clip rate stays at essentially 0 at every size, because a min/max range covers the test activations by construction. **Saturation of test inputs is therefore not the mechanism.** It would become relevant with percentile or MSE clipping, which was not run.
2. **The size effect is real but small relative to the INT4 gap.** The calibration effect over 100 → 10000 is ~0.02 AUC. The INT4 gap to FP32 (0.934) is already ~0.027 at n = 100 and ~0.05 at n = 10000. Calibration explains part of the gap; most of it is intrinsic to 4-bit weights and activations on this model.
3. **Model seed dominates the spread.** Seed 42 is the weakest model and seed 123 the strongest, both at fixed calibration procedure. The 0.029 between-seed SD is about four times the within-seed calibration SD. The pooled correlation between AUC and A0 step across INT4 rows is −0.59, but it is pooled over seeds and therefore partly confounded by seed; it is not used as causal evidence on its own.
4. **Numerical and determinism effects are not the driver on this CPU.** The variant build and inference are bit-stable (C). Whatever numerical difference exists between backends was not measured here.

## What is NOT established
- **Device dependence.** Not tested: no CUDA in this session. The earlier single-seed observation (seed 42: 0.651 on CPU vs 0.572 on CUDA) is still unexplained, and this analysis does not reproduce or explain it. It must not be described as a mechanism.
- **Zero-point rounding.** `fake_quantize_tensor` does not round the zero-point to an integer. The effect of this on INT4 was not isolated and remains a limitation.
- **Percentile, MSE or per-channel calibration.** Not run. The clipping-versus-resolution trade-off, which is the textbook explanation for calibration sensitivity, can only be tested with them.
- **Causal attribution to a specific weight tensor.** Phase 3 per-unit sensitivity (L0 weights and A0 activation dominate) was not repeated here.

## Consequence for claims
- Supported: INT4 PTQ point estimates depend mainly on the trained model (seed); calibration size shifts them downward by about 0.02 AUC on average, consistently across seeds; the mechanism is activation range widening.
- Not supported: "INT4 is device dependent"; "saturation causes the INT4 loss"; "INT4 is unstable" as a stand-alone statement (the 0.835–0.909 spread is a property of the seeds, and the design has 5 seeds).

## Figures
**Pending.** The requested figures (performance vs calibration size, variance across calibration subsets, saturation/error behaviour, comparison with QAT) are not yet produced. The tables above contain the underlying numbers.

## Verification
- TEST #1 and TEST #2: `scripts/phase4_int4_stability.py` was run twice from the same inputs; all five output CSVs are byte-identical (`cmp`).
- The first attempt failed because seeds 7 and 99 are not in the frozen directory (fixed with a read-only loader; the frozen directory was not written).
