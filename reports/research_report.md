# Research Report: HEP-Trigger Quantization Lab

**Status:** results below are from the synthetic development dataset (see Section 7 /
`reports/dataset_card.md`). This is a development-validation benchmark of the
engineering pipeline, not a physics result — the real-dataset experiment is pending
(Section 19).

## 1. Abstract

This project investigates how post-training quantization (PTQ) and quantization-aware
training (QAT) affect classification accuracy, model size, software inference
latency, and (where tooling permits) estimated FPGA resource usage, for a compact
neural network classifier in the style of a real-time high-energy physics (HEP)
trigger. We build a reproducible pipeline — data validation, a configurable MLP
baseline, calibrated fake-quantization at INT8/INT6/INT4, QAT fine-tuning, and a
hardware-aware analysis layer with a real (not mocked) hls4ml conversion/C-simulation
path and an explicit software-only fallback — and report an algorithmic Pareto frontier
across precision, accuracy, and model size rather than a single "quantization works"
claim. On the synthetic development dataset used here (a 12,578-parameter MLP), PTQ
accuracy degrades gently from FP32 (98.53%) to INT4 (98.33%, −0.20 pp); QAT recovers
this loss at every precision tested, and the Pareto frontier identifies QAT-INT6 as the
best accuracy/size trade-off. A notable methodological finding: simulated
fake-quantization measurably *increases* software inference latency relative to FP32
(no real integer kernel speedup is available in this software-only setup) — a result we
report explicitly to avoid the common misreading that smaller numerical precision
implies faster software inference.

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
in this environment (no Vivado/Vitis HLS installed); all hardware fields for the seven
training-matrix experiments (Section 14) are therefore `N/A` with a documented reason,
not an estimate presented as a measurement. One additional experiment (EXP_008, same
FP32 model, Section 14) opted into `hardware.backend: hls4ml`: hls4ml 1.3.0 successfully
converted the trained model and compiled it for C-simulation (`status: csim`),
confirming the model is HLS-compatible — but LUT/DSP/BRAM/FF fields are `N/A` there too,
since csim validates functional correctness, not resource usage, which requires Vivado
synthesis this environment doesn't have.

## 13. Experimental Setup

Seed 42 throughout. Training: Adam, cross-entropy loss, early stopping (patience 8),
batch size 512, up to 40 epochs. PTQ calibration: first ~20 batches (10,240 samples) of
the training split. QAT fine-tuning: `max(5, epochs // 4)` epochs at `0.1x` the base
learning rate, starting from the trained FP32 checkpoint. All runs executed in WSL2
Ubuntu with an NVIDIA RTX 5060 (CUDA 12.8) — see `environment_report.md`.

## 14. Results

Two datasets, reported separately per the dataset card's policy of never merging a
pipeline-validation result with a physics-relevant one into a single table. Read
Section 14.2 for the dataset this project's actual scientific conclusions should be
drawn from — 14.1 is retained because it's what validated the pipeline in the first
place (Gate 2), not because its numbers mean anything physically.

### 14.1 Synthetic development dataset (pipeline validation only)

Full experiment matrix (8 experiments: FP32 baseline, PTQ×3, QAT×3, and one FP32 run
with the hls4ml backend enabled), on the synthetic development dataset:

| Experiment | Precision | Method | Accuracy | F1 | ROC-AUC | Model size (MB, estimated for quantized) | Latency (ms/sample, software) |
|---|---|---|---:|---:|---:|---:|---:|
| EXP_001_FP32_MLP | FP32 | — | 0.9853 | 0.9853 | 0.9918 | 0.0480 | 0.445 |
| EXP_002_INT8_PTQ | INT8 | PTQ | 0.9850 | 0.9850 | 0.9918 | 0.0126 | 2.099 |
| EXP_003_INT6_PTQ | INT6 | PTQ | 0.9843 | 0.9843 | 0.9917 | 0.0097 | 2.159 |
| EXP_004_INT4_PTQ | INT4 | PTQ | 0.9833 | 0.9833 | 0.9918 | 0.0068 | 1.821 |
| EXP_005_INT8_QAT | INT8 | QAT | 0.9863 | 0.9863 | 0.9915 | 0.0126 | 5.497 |
| EXP_006_INT6_QAT | INT6 | QAT | 0.9863 | 0.9863 | 0.9913 | 0.0097 | 4.947 |
| EXP_007_INT4_QAT | INT4 | QAT | 0.9857 | 0.9856 | 0.9917 | 0.0068 | 4.151 |
| EXP_008_FP32_HLS4ML | FP32 | — (hls4ml csim) | 0.9853 | 0.9853 | 0.9918 | 0.0480 | 0.473 |

(Full machine-readable table: [results_summary.csv](../results_summary.csv); same table
rendered: [reports/tables/quantization_results_synthetic.md](tables/quantization_results_synthetic.md).)

**Accuracy vs. precision (PTQ).** PTQ accuracy degrades monotonically and gently as
precision drops: FP32 0.9853 → INT8 0.9850 (−0.03 pp) → INT6 0.9843 (−0.10 pp) → INT4
0.9833 (−0.20 pp). On this dataset/architecture, even INT4 PTQ loses less than a
quarter of a percentage point of accuracy versus FP32 — this task is not strongly
precision-sensitive at this model size, which is itself a finding (see Discussion).

**QAT recovers, and slightly exceeds, the PTQ accuracy loss.** QAT at every precision
(0.9863 / 0.9863 / 0.9857 for INT8/INT6/INT4) matches or exceeds the FP32 baseline's
0.9853, and is strictly higher than the corresponding PTQ accuracy at every precision
tested. This answers Secondary Research Question 2 affirmatively for this task: QAT
not only recovers PTQ's accuracy loss here, it slightly improves on the FP32 baseline,
plausibly because the additional fine-tuning epochs act as extra training on top of an
already-converged model rather than purely a quantization-recovery effect — this
dataset is small and well-separated, so some of the QAT "gain" may be noise rather than
a robust effect (see Limitations).

**Latency does NOT improve with lower precision — it gets worse.** This is the most
important methodological result in this report: INT8/INT6/INT4 (PTQ and QAT alike) are
all slower per-sample than FP32, by roughly 4–12x. This is a direct, expected
consequence of the simulated fake-quantization approach documented in Section 10 — the
extra round()/clamp() operations inserted at every weight and activation add real
compute, with none of the integer-kernel speedup real hardware INT8 execution would
provide. **Latency in this report measures the cost of simulating quantization in
software, not the deployment speedup quantization is meant to achieve** — that question
belongs to the hardware-aware analysis (Section 12), which on this environment only
reaches HLS conversion/C-simulation, not synthesis-level latency numbers.

**Model size scales exactly as expected.** Estimated size drops linearly with
bit-width within each method (FP32 0.048 MB → INT8 0.0126 MB → INT6 0.0097 MB → INT4
0.0068 MB), consistent with `estimate_model_size`'s analytical formula (Section 10);
PTQ and QAT produce identical size estimates at a given precision since both quantize
the same architecture.

### 14.2 Real dataset: hls4ml LHC jet tagging (the result that matters)

Architecture `hidden_dims: [64, 32, 32]` (4,389 parameters) deliberately mirrors the
reference dense network from the original hls4ml paper on this exact dataset, as a
sanity check: the FP32 baseline below (74.95% accuracy) lands within 1 percentage
point of that paper's own reported baseline for the same architecture on the same
data, which is strong evidence the pipeline (data handling, training, evaluation) is
implemented correctly — not just "runs without crashing."

Staged validation (per the project's risk-management plan, run before the full matrix):

| Stage | Config | Subset size | Accuracy | ROC-AUC (macro) |
|---|---|---:|---:|---:|
| 1: pipeline validation | `jet_validate_10k.yaml` | 10,000 | 0.5507 | 0.8235 |
| 2: training validation | `jet_validate_50k.yaml` | 50,000 | 0.7381 | 0.9237 |

Stage 1 used only 8 epochs (by design, to validate the pipeline runs end-to-end, not
to converge) — 55% accuracy against a 20% random baseline for 5 classes, with
ROC-AUC 0.82, confirms the pipeline is learning real signal before any time is spent
on larger runs. Stage 2's 73.8% accuracy at 50k events, close to the eventual 100k
result below, gave confidence to proceed to the full matrix.

**Full experiment matrix, 100,000 events** (FP32 baseline, PTQ×3, QAT×3):

| Experiment | Precision | Method | Accuracy | F1 (macro) | ROC-AUC (macro) |
|---|---|---|---:|---:|---:|
| EXP_012_JET_FP32 | FP32 | — | 0.7495 | 0.7527 | 0.9335 |
| EXP_013_JET_INT8_PTQ | INT8 | PTQ | 0.7503 | 0.7535 | 0.9333 |
| EXP_014_JET_INT6_PTQ | INT6 | PTQ | 0.7501 | 0.7533 | 0.9323 |
| EXP_015_JET_INT4_PTQ | INT4 | PTQ | 0.6507 | 0.6473 | 0.8832 |
| EXP_016_JET_INT8_QAT | INT8 | QAT | 0.7523 | 0.7555 | 0.9344 |
| EXP_017_JET_INT6_QAT | INT6 | QAT | 0.7517 | 0.7550 | 0.9336 |
| EXP_018_JET_INT4_QAT | INT4 | QAT | 0.7291 | 0.7324 | 0.9232 |

**This is the central finding of this project, and it is qualitatively different from
the synthetic-dataset result in 14.1.** On real physics data, INT8 and INT6 PTQ are
essentially lossless (+0.08 pp and +0.06 pp — within noise of the FP32 baseline), but
**INT4 PTQ collapses by 9.9 percentage points** (74.95% → 65.07%) and ROC-AUC drops
from 0.933 to 0.883. This is a genuine precision cliff, not the gentle monotonic
degradation seen on the synthetic dataset — confirming the synthetic result in 14.1
understated real quantization sensitivity, almost certainly because that task was too
easy (near-ceiling accuracy, wide margins) for 4-bit rounding error to flip any
decisions. The real jet-tagging task, with five overlapping physics classes (note W/Z
confusion in the confusion matrices), has much narrower decision margins, and INT4's
~16-level weight/activation quantization is coarse enough to meaningfully blur them.

**QAT substantially recovers the INT4 cliff, but does not fully close it.**
INT4-QAT reaches 72.91% accuracy — a 7.8 percentage-point recovery from INT4-PTQ's
65.07%, closing roughly 79% of the 9.9 pp gap to the FP32 baseline — but still lands
2.0 pp below FP32 (74.95%) and below INT8/INT6-QAT (~75.2%, both of which essentially
match or marginally exceed FP32, consistent with INT8/INT6 already being
near-lossless before any fine-tuning). **This is a materially different, more
qualified answer to Secondary Research Question 2 than the synthetic dataset gave.**
On the synthetic task (Section 14.1), QAT fully recovered — even slightly exceeded —
FP32 accuracy at every precision including INT4. On real data, QAT recovers *most* of
a genuine accuracy cliff but leaves a real residual gap at the most aggressive
precision tested. The correct general statement, based on both datasets together, is:
QAT reliably closes small-to-moderate PTQ accuracy gaps, and substantially — but not
necessarily completely — closes large ones; whether it closes a gap *fully* appears to
depend on how severe the underlying precision cliff is, which itself depends on task
difficulty and decision-margin width, not on the quantization method alone.

## 15. Pareto Analysis

### 15.1 Synthetic development dataset

Pareto frontier computed algorithmically (`find_pareto_frontier`, non-dominated-set
identification — no manual model selection) over (accuracy↑, model_size_mb↓) across all
seven training-matrix experiments:

| Experiment | Precision | Method | Accuracy | Size (MB) | Pareto-optimal |
|---|---|---|---:|---:|:---:|
| EXP_001_FP32_MLP | FP32 | — | 0.9853 | 0.0480 | No |
| EXP_002_INT8_PTQ | INT8 | PTQ | 0.9850 | 0.0126 | No |
| EXP_003_INT6_PTQ | INT6 | PTQ | 0.9843 | 0.0097 | No |
| EXP_004_INT4_PTQ | INT4 | PTQ | 0.9833 | 0.0068 | No |
| EXP_005_INT8_QAT | INT8 | QAT | 0.9863 | 0.0126 | No |
| EXP_006_INT6_QAT | INT6 | QAT | 0.9863 | 0.0097 | **Yes** |
| EXP_007_INT4_QAT | INT4 | QAT | 0.9857 | 0.0068 | **Yes** |

(Figure: [reports/figures/pareto_accuracy_vs_size_synthetic.png](figures/pareto_accuracy_vs_size_synthetic.png);
full data: [reports/tables/pareto_accuracy_vs_size_synthetic.csv](tables/pareto_accuracy_vs_size_synthetic.csv).)

Only QAT-INT6 and QAT-INT4 are non-dominated. QAT-INT8 is dominated by QAT-INT6: they
tie on accuracy (0.9863) but QAT-INT6 is smaller — so there is no reason, on this
accuracy/size trade-off alone, to prefer QAT-INT8 over QAT-INT6. Every PTQ
configuration is dominated by at least one QAT configuration at equal-or-smaller size
with equal-or-higher accuracy. The frontier's practical reading: **QAT-INT6 is the
best accuracy/size trade-off found in this experiment matrix**, with QAT-INT4
available as a smaller-footprint alternative at a small (0.06 pp) accuracy cost.

### 15.2 Real dataset: hls4ml LHC jet tagging

| Experiment | Precision | Method | Accuracy | Size (MB) | Pareto-optimal |
|---|---|---|---:|---:|:---:|
| EXP_012_JET_FP32 | FP32 | — | 0.7495 | 0.0167 | No |
| EXP_013_JET_INT8_PTQ | INT8 | PTQ | 0.7503 | 0.0045 | No |
| EXP_014_JET_INT6_PTQ | INT6 | PTQ | 0.7501 | 0.0035 | No |
| EXP_015_JET_INT4_PTQ | INT4 | PTQ | 0.6507 | 0.0025 | No |
| EXP_016_JET_INT8_QAT | INT8 | QAT | 0.7523 | 0.0045 | **Yes** |
| EXP_017_JET_INT6_QAT | INT6 | QAT | 0.7517 | 0.0035 | **Yes** |
| EXP_018_JET_INT4_QAT | INT4 | QAT | 0.7291 | 0.0025 | **Yes** |

(Figure: [reports/figures/pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.png](figures/pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.png);
full data: [reports/tables/pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.csv](tables/pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.csv).
Staged-validation runs, EXP_010/EXP_011, are excluded from this analysis — see
`hep_quant.reporting.generate_report`'s docstring for why.)

Here, all three QAT configurations are Pareto-optimal, and the FP32 baseline and all
three PTQ configurations are dominated. This is a cleaner, more decisive frontier than
the synthetic dataset's: on real data, QAT strictly dominates both FP32 and PTQ at
every tested precision — even INT4-QAT, despite not fully closing the gap to FP32 in
absolute accuracy (Section 14.2), is Pareto-optimal because nothing in this matrix
matches its combination of small size and recovered accuracy. INT4-PTQ is the clearest
loser on this frontier: worse accuracy than everything except nothing, at only
marginally smaller size than INT6-PTQ's already-dominated point. **The practical
reading for real deployment: if INT4 precision is required by a resource budget, QAT is
not optional — PTQ at INT4 on this task is sufficiently worse that only
fine-tuning-aware quantization is viable.**

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

**This task, at this model size, is not strongly quantization-sensitive.** The gap
between FP32 and INT4-PTQ accuracy (0.20 percentage points) is small relative to typical
run-to-run training variance for a model this size. That is itself informative: a
12,578-parameter MLP on a well-separated binary classification task has enough
redundancy that aggressive uniform quantization barely perturbs its decision boundary.
Whether this holds for a harder, more class-overlapping task (closer to a real trigger
background-rejection problem) is exactly the kind of question the real-dataset
experiment (Section 17/19) would answer — this result should not be read as "HEP
trigger models are insensitive to quantization" in general.

**QAT's apparent improvement over FP32 deserves a skeptical reading, not just an
optimistic one.** QAT-INT8/INT6 both reach 0.9863 accuracy, 0.10 percentage points above
the FP32 baseline's 0.9853. The mechanism is plausible — QAT fine-tunes for additional
epochs beyond the FP32 training run, so part of the "QAT gain" may simply be extra
optimization steps rather than a quantization-specific effect — but with a single seed
and a test set of 3,000 synthetic samples, a 0.10 pp difference is close to what sampling
noise alone could produce. The Pareto conclusion (QAT-INT6 is non-dominated) is robust
to this caveat since it's a size argument first and an accuracy argument second, but the
specific claim "QAT exceeds FP32 accuracy" is not asserted here as a general result.

**The latency result is the one most worth internalizing methodologically.** It would
be easy to mis-read a "model size shrinks with lower precision" result as "lower
precision also runs faster," and this report's own latency numbers show the opposite
(4–12x slower, Section 14) for exactly the reason explained in Section 10: this
project's PTQ/QAT is simulated fake-quantization in float32, not real integer hardware
execution. Anyone extending this project to claim a deployment speedup must do so from
real hardware (FPGA synthesis, Section 12) or a real bit-packed integer kernel — not
from this report's software latency numbers.

**Relating back to the research questions (Section 5):** on this synthetic task, PTQ
accuracy loss FP32→INT4 is small (RQ1); QAT fully recovers it, within the noise caveat
above (RQ2); architecture-robustness-across-quantization (RQ3) and cross-dataset
reproducibility (RQ7) are not yet addressed — both require the real-dataset and
second-architecture work in Section 19; the accuracy/size Pareto frontier (RQ4) is
answered concretely in Section 15; model-complexity-vs-quantization-sensitivity (RQ5)
and lightweight-model-viability (RQ6) are suggested but not conclusively shown by one
12k-parameter architecture on one dataset.

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

This project built and validated, gate-by-gate with a two-test acceptance rule at each
step, a reproducible pipeline for measuring the accuracy/size/latency/hardware
trade-offs of quantizing a neural network classifier for a HEP-trigger-style task. On
the synthetic development dataset used here: PTQ degrades accuracy gently and
monotonically from FP32 to INT4 (−0.20 pp at worst); QAT recovers that loss at every
precision tested; and the algorithmic Pareto analysis identifies QAT-INT6 as the best
accuracy/size trade-off in the matrix, with QAT-INT4 as a smaller alternative at
negligible further cost. A real hls4ml conversion and C-simulation (EXP_008) confirmed
the trained architecture is HLS-compatible, though FPGA resource numbers remain
unmeasured pending Vivado synthesis. The central engineering lesson —
that simulated fake-quantization makes software inference *slower*, not faster, and
must not be mistaken for a measured hardware speedup — is, in this project's view, as
important a result as the accuracy numbers themselves. These findings are scoped
explicitly to a synthetic, well-separated classification task; the project's own
Limitations (Section 17) and Future Work (Section 19) describe exactly what would be
needed to extend them into a physics-relevant claim.

## 21. References

- hls4ml: Fast Machine Learning Lab collaboration, "hls4ml: An Open-Source Codesign
  Workflow to Empower Scientific Low-Power Machine Learning Devices" and related
  publications (https://fastmlscience.org/, https://github.com/fastmachinelearning/hls4ml).
- QKeras: Google, "QKeras: a quantization deep learning library for Tensorflow Keras"
  (https://github.com/google/qkeras).
- PyTorch quantization documentation: https://pytorch.org/docs/stable/quantization.html.
- ATLAS Open Data: https://opendata.atlas.cern/ (referenced as a candidate real dataset
  source for future work; not used to produce the results in this report).
