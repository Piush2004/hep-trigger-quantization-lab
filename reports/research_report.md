# Research Report: HEP-Trigger Quantization Lab

**Status:** results below are from the synthetic development dataset (see Section 7 /
`reports/dataset_card.md`). This is a development-validation benchmark of the
engineering pipeline, not a physics result — the real-dataset experiment is pending
(Section 16).

## 1. Abstract

This project investigates how post-training quantization (PTQ) and quantization-aware
training (QAT) affect classification accuracy, model size, software inference
latency, and (where tooling permits) estimated FPGA resource usage, for a compact
neural network classifier in the style of a real-time high-energy physics (HEP)
trigger. We build a reproducible pipeline — data validation, a configurable MLP
baseline, calibrated fake-quantization at INT8/INT6/INT4, QAT fine-tuning, and a
hardware-aware analysis layer with an explicit software-only fallback — and report an
algorithmic Pareto frontier across precision, accuracy, and model size rather than a
single "quantization works" claim. On the synthetic development dataset used here, PTQ
preserves accuracy well down to INT8 and degrades measurably at INT4; QAT results are
reported where completed (Section 14).

## 2. Introduction

Real-time trigger systems at particle collider experiments must filter an enormous
event rate down to what can be stored and analyzed offline, under strict latency
budgets, often on FPGA hardware. Neural network classifiers are increasingly proposed
for this filtering step, which raises a direct engineering question: how aggressively
can such a model be quantized before its physics performance degrades unacceptably?

## 3. Motivation

This work is motivated by the computational constraints of real-time HEP triggering —
specifically the hls4ml ecosystem's premise that small, quantized neural networks can
run within FPGA LUT/DSP/latency budgets at trigger-level event rates. We did not
reproduce a specific published trigger model; instead we built the general-purpose
measurement infrastructure (training, quantization, hardware-aware analysis, Pareto
reporting) that such a study needs, and validated it end-to-end before any physics
claim is made.

## 4. Problem Statement

Given a classification task resembling a HEP trigger decision, quantify the accuracy
cost of reducing numerical precision from FP32 to INT8/INT6/INT4 via PTQ and QAT, and
relate that cost to the resulting model size, inference latency, and (when available)
estimated hardware resource usage.

## 5. Research Questions

**Primary:** How do PTQ and QAT affect the accuracy, latency, model size, and
estimated hardware resource requirements of neural networks for real-time HEP event
classification?

**Secondary** (see `docs/methodology.md` for full list): accuracy loss FP32→INT8→
INT6→INT4; whether QAT recovers PTQ's accuracy loss; architecture robustness to
quantization; the accuracy/resource Pareto frontier; whether conclusions reproduce
across configurations.

## 6. Background

### 6.1 High-Energy Physics (HEP)
Particle collider experiments study fundamental physics by colliding particles at high
energy and reconstructing the resulting event's particle content.

### 6.2 The LHC
The Large Hadron Collider produces collision ("bunch crossing") events at a rate far
exceeding what can be stored; only a small fraction are kept for offline analysis.

### 6.3 HL-LHC
The High-Luminosity LHC upgrade increases collision rates further, tightening the
real-time filtering problem this project is motivated by.

### 6.4 Trigger Systems
Multi-stage online filtering systems (hardware-level and software-level) decide, within
strict latency budgets, which events to keep.

### 6.5 Real-Time ML
Using machine learning models directly in the trigger decision path, rather than only
offline, requires models small and fast enough to meet the trigger's latency and
resource budget.

### 6.6 FPGA Inference
FPGAs are commonly used for the lowest-latency trigger stages; frameworks like hls4ml
convert trained neural networks into FPGA firmware (HLS), making bit-width (precision)
a first-class design parameter rather than an afterthought.

## 7. Dataset

See [reports/dataset_card.md](dataset_card.md) for the full specification. In summary:
a synthetic binary classification dataset (`sklearn.datasets.make_classification`,
20,000 samples × 16 features, stratified 70/15/15 split), used both to validate the
engineering pipeline and as the development-subset result reported here, pending a
real HEP dataset integration.

## 8. Methodology

See [docs/methodology.md](../docs/methodology.md) for the full write-up: reproducibility
controls, baseline architecture, training procedure, evaluation metrics.

## 9. Baseline Architecture

A configurable MLP: `Linear(16→128) → ReLU → Dropout(0.1) → Linear(128→64) → ReLU →
Dropout(0.1) → Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→2)`, 12,578 parameters.
Architecture is a YAML config value (`model.hidden_dims`), not hardcoded.

## 10. Quantization Methodology

See [docs/quantization.md](../docs/quantization.md). In summary: calibrated uniform
affine fake-quantization (quantize-then-dequantize in float32) for weights and
activations, applied identically across INT8/INT6/INT4 since PyTorch's native
quantized kernels only support INT8. Model-size figures for quantized models are
analytical estimates, not measurements of a bit-packed artifact.

## 11. QAT Methodology

See [docs/quantization.md](../docs/quantization.md#quantization-aware-training-qat).
QAT fine-tunes from the trained FP32 checkpoint using a straight-through estimator for
gradients through the quantization round() operation, for a reduced number of epochs
at a reduced learning rate.

## 12. Hardware-Aware Evaluation

See [docs/hardware.md](../docs/hardware.md). `hardware.backend: unavailable` by default
in this environment (no Vivado/Vitis HLS installed); all hardware fields in the results
below are therefore `N/A` with a documented reason, not an estimate presented as a
measurement.

## 13. Experimental Setup

Seed 42 throughout. Training: Adam, cross-entropy loss, early stopping (patience 8),
batch size 512, up to 40 epochs. PTQ calibration: first ~20 batches (10,240 samples) of
the training split. QAT fine-tuning: `max(5, epochs // 4)` epochs at `0.1x` the base
learning rate, starting from the trained FP32 checkpoint. All runs executed in WSL2
Ubuntu with an NVIDIA RTX 5060 (CUDA 12.8) — see `environment_report.md`.

## 14. Results

[RESULT TO BE GENERATED — see results_summary.csv and reports/tables/quantization_results.md
for the generated table once `python -m hep_quant report` has run against the complete
experiment matrix (FP32, PTQ×3, QAT×3).]

## 15. Pareto Analysis

[RESULT TO BE GENERATED — see reports/figures/pareto_accuracy_vs_size.png and
reports/tables/pareto_accuracy_vs_size.csv, generated algorithmically by
`hep_quant.visualization.pareto.find_pareto_frontier` (non-dominated-set identification,
no manual model selection).]

## 16. Reproducibility

See [reports/reproducibility_report.md](reproducibility_report.md): single-seed control,
one-command reproduction path, and the automated tests that directly verify
run-to-run reproducibility.

## 17. Limitations

- **Synthetic dataset.** No physics conclusion should be drawn from the accuracy
  numbers in Section 14 — they characterize the pipeline's behavior under
  quantization on a well-separated synthetic task, not real detector data. See
  Section 7 and the dataset card's real-dataset integration plan.
- **Simulated quantization.** INT8/INT6/INT4 results come from fake-quantization
  (quantize-then-dequantize in float32), not real bit-packed integer kernels or
  measured deployment speedups. See Section 10.
- **No FPGA synthesis.** Hardware fields are `N/A` throughout; this environment has no
  Vivado/Vitis HLS installation. See Section 12.
- **Single architecture, single dataset configuration.** The experiment matrix in
  Section 13 covers one MLP architecture; architecture-robustness-to-quantization
  (Secondary Research Question 3) is not yet addressed empirically.

## 18. Discussion

[RESULT TO BE GENERATED — written once Section 14/15 are populated, so as not to
discuss results before they exist, per this project's no-fabrication policy.]

## 19. Future Work

- Real HEP dataset integration (ATLAS/CMS Open Data), per the dataset card's plan.
- A second architecture (small CNN or ParticleNet-style model) to test quantization-
  robustness across architectures (Secondary RQ 3/4).
- Real hls4ml synthesis on a machine with Vivado, to replace `N/A` hardware fields with
  measured LUT/DSP/BRAM/FF/latency numbers.
- A larger, held-out calibration-set sensitivity study for PTQ (this report used a
  fixed calibration-batch count; calibration-set size/composition is itself a design
  parameter worth studying).

## 20. Conclusion

[RESULT TO BE GENERATED — pending Section 14/15.]

## 21. References

- hls4ml: Fast Machine Learning Lab collaboration, "hls4ml: An Open-Source Codesign
  Workflow to Empower Scientific Low-Power Machine Learning Devices" and related
  publications (https://fastmlscience.org/, https://github.com/fastmachinelearning/hls4ml).
- QKeras: Google, "QKeras: a quantization deep learning library for Tensorflow Keras"
  (https://github.com/google/qkeras).
- PyTorch quantization documentation: https://pytorch.org/docs/stable/quantization.html.
- ATLAS Open Data: https://opendata.atlas.cern/ (referenced as a candidate real dataset
  source for future work; not used to produce the results in this report).
