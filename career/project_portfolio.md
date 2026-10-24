# Project portfolio — HEP-Trigger Quantization Lab

Two formats below: a **short application-form version** (fits a "describe a project" box) and a
**longer case-study version** (for a portfolio page or LinkedIn "Featured" section).

---

## Short version (application forms, ~100 words)

Independent research project benchmarking how aggressively neural networks can be quantized (INT8 →
INT4, post-training and quantization-aware training) before accuracy breaks down, on a simulated
high-energy-physics jet-classification task. Ran a 5-seed, pre-registered statistical protocol across
two architectures, implemented a PyTorch-to-fixed-point `hls4ml` conversion path, and trained an ML
surrogate for FPGA resource estimation. Found that a mixed-precision layout from a near-exhaustive
search reliably beats uniform INT6 at lower storage. Shipped as a public GitHub repository with a
FastAPI/Plotly results dashboard, 141 automated tests, and full reproduction documentation. Not
CERN-affiliated; uses only public, simulated data.

## Case-study version (portfolio / LinkedIn)

**Problem.** Real-time particle-physics trigger systems must decide what data is worth keeping within
a strict, hardware-bound time and resource budget. Quantizing a neural network — running it at lower
numerical precision — is one way to fit it into that budget, but how much accuracy that costs, and
whether a simpler model would do just as well, isn't obvious without actually measuring it.

**Research question.** How much does post-training quantization (PTQ) and quantization-aware training
(QAT) cost in accuracy at INT8/INT6/INT4, for a jet-classification model, and does a search over
per-layer mixed precision beat the best uniform precision at the same storage budget?

**Dataset.** The public `hls4ml_lhc_jets_hlf` dataset (Zenodo, CC BY 4.0): ~830K simulated jets, 16
high-level physics features, 5 classes. Particle-level Monte Carlo — explicitly **not** real detector
data, a distinction preserved throughout the project rather than blurred for effect.

**Method.** FP32 baseline vs. INT8/INT6/INT4 PTQ and QAT, evaluated across two MLP architectures with
5 random seeds each and a pre-registered test-set-selection protocol. A near-exhaustive search (2,187
layouts) over per-unit mixed precision. A PyTorch → `hls4ml` fixed-point C-simulation path for
hardware-realistic numerical behavior, plus an ML surrogate (trained on the public `wa-hls4ml` dataset)
for FPGA resource estimation, since no local synthesis toolchain or physical FPGA was available.

**Key results.**
- INT8 PTQ is close to lossless; INT4 PTQ degrades substantially but QAT recovers most of the loss.
- A mixed-precision layout found by the exhaustive search reliably beats uniform INT6 PTQ at 78% of
  its storage (all 5 seeds, 95% CI entirely positive).
- A default-hyperparameter gradient-boosted-tree baseline beats the FP32 MLP on raw accuracy — reported
  openly: the neural network's case is FPGA deployability via `hls4ml`, not raw accuracy.

**Architecture and tools.** Python, PyTorch, scikit-learn, `hls4ml`, FastAPI, Plotly, pytest, Docker,
GitHub Actions CI.

**Validation and reproducibility.** 141 automated tests (data pipeline, quantization, hardware
estimation, API layer), run twice independently with identical results; every number traces to a
committed report/table; full environment, dataset-fingerprint, and reproduction-command documentation.

**Limitations, stated plainly.** No HLS synthesis was run and no physical FPGA was measured — all
hardware-resource numbers are either a software heuristic or an ML surrogate, both explicitly labelled
as such, never presented as measured. The CPU-vs-CUDA device comparison is observational, not causal.
This is a reproducible benchmark, not a production trigger system or a peer-reviewed publication.

**Links.** Code, reports, and reproduction instructions:
https://github.com/Piush2004/hep-trigger-quantization-lab
*(Live interactive dashboard link to be added once a public deployment is verified — see
`reports/deployment_report.md` for current status.)*
