# Methodology

## Experimental design

Each experiment is defined by a YAML config (`configs/*.yaml`) specifying: dataset,
model architecture, training hyperparameters, quantization method/precision, and
hardware backend. One config = one `experiment_id` = one JSON record in
`experiments/results/`.

## Reproducibility controls

- A single `seed` (default 42) seeds Python's `random`, NumPy, and PyTorch (CPU and
  CUDA), with cuDNN determinism requested (`hep_quant/utils/seeds.py`).
- Train/val/test splitting is deterministic given the seed and is stratified to
  preserve class balance; `check_no_leakage` verifies zero row-level overlap between
  splits on every run.
- Every experiment record captures Python/PyTorch/CUDA versions, GPU name, OS, and the
  current git commit (`hep_quant/experiments/reproducibility.py`), so a result can
  always be traced back to the exact code and environment that produced it.
- `tests/test_training.py::test_training_reproducible_with_same_seed` and
  `tests/test_pipeline.py::test_full_pipeline_runs_twice_consistently` directly verify
  that two independent runs with the same seed converge to the same test accuracy.

## Baseline architecture

A configurable MLP (`hep_quant/models/mlp.py`): `Linear -> activation -> [Dropout] ->
... -> Linear`. Hidden layer sizes come entirely from `model.hidden_dims` in the
config — the class never hardcodes an architecture. The default
(`configs/base.yaml`) is `[128, 64, 32]`, intentionally compact per this project's
scope (quantization sensitivity, not maximizing raw model capacity).

## Training

Adam optimizer, cross-entropy loss, early stopping on validation loss (default
patience 8 epochs), best-validation-loss checkpoint restored at the end of training.
Metrics tracked per epoch: train/val loss, val accuracy, val F1, val ROC-AUC, learning
rate, epoch duration (`hep_quant/training/trainer.py`).

## Evaluation metrics

Accuracy, precision, recall, F1, ROC-AUC, PR-AUC (binary classification) via
`hep_quant/training/metrics.py`; ROC-AUC/PR-AUC are computed only when the positive-
class probability is available and both classes are present in the evaluation split.

## Quantization

See [docs/quantization.md](quantization.md) for the full PTQ/QAT methodology.

## Hardware-aware evaluation

See [docs/hardware.md](hardware.md). In short: a best-effort hls4ml integration that
reports conversion/C-simulation status when hls4ml is usable, and explicit `None`/"N/A"
hardware fields with a documented reason when it is not — the rest of the pipeline
never depends on hardware numbers being present.

## What this methodology does NOT claim

- No claim of novelty, "first", or exclusivity — this is an independent research
  investigation inspired by publicly available HEP/trigger literature (e.g. the
  hls4ml ecosystem), not an official or CERN-affiliated effort.
- No physics conclusions — the primary dataset used to produce the results in this
  repository is synthetic (see `reports/dataset_card.md`); conclusions here are about
  the *engineering pipeline's* behavior under quantization, pending a real-dataset run.
- No FPGA hardware measurements — only software-measured CPU/GPU latency, and (when
  available) hls4ml conversion/C-simulation status, never physical synthesis/place-
  and-route numbers from real Vivado hardware runs, since Vivado is not installed in
  this environment.
