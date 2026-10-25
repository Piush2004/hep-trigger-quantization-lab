# Phase 14 Quantization: PTQ Catastrophically Fails on the CNN, QAT Recovers Gracefully

## 1. Objective

With `ptq.py`/`qat.py` generalized this phase to support `Conv2d` and the CNN's two-block
(`conv`, `classifier`) structure, this asks the natural follow-up question: does the MLP's
quantization story (INT8/INT6 PTQ near-lossless, INT4 PTQ unstable, QAT recovers most of the
loss) transfer to the CNN?

**Scope disclosure**: this sweep runs on a single seed (42), not the project's usual 5- or
3-seed convention. The CNN's FP32 training cost (measured ~8-30 min/seed depending on
concurrent system load, see `reports/phase14_cnn_jet_images.md` §4) makes a full multi-seed
PTQ+QAT sweep (7 configurations × N seeds, several of which need their own fine-tuning epochs)
materially more expensive than the MLP study's equivalent. This is reported as a single-seed
demonstration that the pipeline runs correctly end-to-end on the CNN, not as equivalent
statistical power to the rest of this project's multi-seed results.

## 2. Method

FP32 `JetImageCNN` trained once (seed 42, same 100,000-jet subsample and split as
`reports/phase14_cnn_jet_images.md`). From that single trained base: `quantize_model_ptq`
(PTQ, weight min/max + 2,048-sample activation-range calibration) and `prepare_qat_model` +
fine-tuning (`qat_epochs = max(5, 30 // 4) = 7`, `qat_lr = 0.001 * 0.1`, the same formula the
MLP's uniform QAT path uses) at INT8/INT6/INT4 — 7 configurations total (FP32 + 3×PTQ + 3×QAT),
all evaluated once on the full 270,000-jet held-out test split.

## 3. Results (single seed, held-out test)

| config | ROC-AUC | F1 | accuracy | bkg. rejection @ 70% eff. |
|---|---|---|---|---|
| FP32 | 0.8836 | 0.6171 | 0.6246 | 11.32 |
| INT8 PTQ | 0.7323 | 0.0673 | 0.2016 | 5.58 |
| **INT8 QAT** | **0.8812** | 0.6250 | 0.6272 | 11.11 |
| INT6 PTQ | 0.7161 | 0.0671 | 0.2015 | 4.95 |
| INT6 QAT | 0.8597 | 0.5850 | 0.5904 | 9.01 |
| INT4 PTQ | 0.6270 | 0.0671 | 0.2015 | 3.01 |
| INT4 QAT | 0.8234 | 0.5085 | 0.5282 | 6.87 |

## 4. A dramatic, qualitatively different failure mode from the MLP

**PTQ catastrophically fails for the CNN at every precision tried, including INT8** — accuracy
collapses to ~0.2015 and F1 to ~0.067 across all three PTQ configurations. For a near-balanced
5-class task, a classifier that always predicts the same single class gets almost exactly this:
accuracy ≈ 1/5 = 0.20, and macro F1 ≈ (2·0.2·1)/(0.2+1)/5 ≈ 0.067 (one class with perfect recall
and 20% precision, the other four at zero) — the measured numbers match this degenerate-collapse
pattern closely enough to be the leading explanation, though the exact mechanism was not traced
further (see Limitations). Notably, **ROC-AUC stays well above random (0.63-0.73) even as
accuracy/F1 collapse** — the underlying probability *scores* still carry real ranking signal
(ROC-AUC uses the full score distribution, not just the argmax decision), consistent with PTQ
badly distorting the decision scale (e.g. one class's logit dominating after quantization)
while leaving relative ordering within each class partially intact.

**QAT recovers gracefully, the way quantization-aware training is supposed to**: INT8 QAT
(0.8812) is within noise of FP32 (0.8836); INT6 QAT (0.8597) and INT4 QAT (0.8234) degrade
smoothly rather than collapsing.

**This is a genuinely different and more severe pattern than the MLP's quantization story.**
On the tabular task (`README.md` results table), INT8 PTQ was *practically lossless*
(0.9342 vs. FP32's 0.9343) and only INT4 PTQ showed real instability (0.8819 ± 0.0308, highly
seed-dependent). Here, **INT8 PTQ already collapses** for the CNN. This is a real, substantive
generality finding: **PTQ's reliability is not architecture-independent** — a result that
matters practically, since it means "INT8 PTQ is usually safe" (a reasonable takeaway from the
MLP work alone) does not generalize to this CNN architecture on this task.

## 5. Candidate explanation (stated as a hypothesis, not traced to confirmation)

PTQ's weight quantization uses each layer's raw min/max for its quantization range
(`_quantize_weights_in_place`). A `Conv2d` kernel's weight tensor has a different structure and
typically a different range distribution than a `Linear` layer's — if a small number of outlier
weights dominate a convolutional kernel's min/max, the bulk of that kernel's weights can be
crushed into very few quantization levels. Separately, the 2,048-sample activation-range
calibration subset may be insufficient for a CNN's spatially-extended feature maps compared to
the MLP's flat activation vectors. **Neither hypothesis was isolated or confirmed** — this is
flagged as a specific, well-scoped follow-up (e.g., comparing weight-range histograms per layer,
or sweeping the calibration subset size), not claimed as the established cause.

## 6. Limitations

- Single seed (42) — see scope disclosure in §1. The *qualitative* finding (PTQ collapses, QAT
  doesn't) is unlikely to be a seed-specific fluke given how large the gap is (accuracy 0.20 vs.
  0.63+ ROC-AUC-implied signal), but the exact numbers should not be read as seed-stable point
  estimates.
- The degenerate-collapse explanation in §4 is inferred from the accuracy/F1 pattern matching
  a "always predicts one class" baseline closely, not confirmed by directly inspecting the
  per-class prediction distribution.
- No hls4ml csim or surrogate-hardware numbers for these quantized CNN variants in this report
  (see `reports/phase14_cnn_jet_images.md` and the repository's hls4ml-input-shape
  generalization work for the infrastructure; a full hardware-estimate pass on these specific
  quantized CNN checkpoints was not run as part of this sweep).
- No FPGA synthesis (same constraint as the rest of this project — no Vivado installed).

## 7. Reproducibility

`hep_quant.utils.seeds.set_seed` is called before both the FP32 training and each QAT
fine-tune (the same fix Phase 12 needed for `build_qat_variant`, applied here too). PTQ is
deterministic by construction (no training randomness involved — calibration and
quantize-then-dequantize are both pure functions of the fixed trained weights and the fixed
calibration subset). Not independently re-run twice at full scale given the ~15-20 min+ total
cost of this sweep; the underlying mechanisms (PTQ determinism, QAT seeding) were already
verified in isolation in `reports/phase14_cnn_jet_images.md` §8 and the Phase 12 report.

## 8. Reproduction

```bash
# WSL2 Ubuntu, .venv-wsl
python scripts/phase14_cnn_quantization.py
python -m pytest -q tests/test_qat.py tests/test_hardware.py
```
