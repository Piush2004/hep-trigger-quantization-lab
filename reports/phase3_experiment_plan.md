# Phase 3 Experiment Plan

Written after the gap analysis (`research_gap_analysis.md`) and the Phase 2 audit
(`phase2_results_audit.md`). Note: this plan was formalised retroactively as a single
document; the decisions below were made, and the experiments run, during Phase 3
(commit `8377844`). Nothing here claims an experiment was run that was not.

Rule applied: an experiment is run only if it answers a stated research question
that existing evidence cannot, at an acceptable cost. Every executed experiment got
a new ID; `experiments/results/frozen_phase2/` was never modified.

| Experiment | Reason | Research question | Expected value | Cost | Priority | Decision |
|---|---|---|---|---|---|---|
| Seeds 7, 99 x 7 configs (n = 3 -> 5) | INT4 PTQ spread dominated n = 3 uncertainty | Is the INT4 PTQ collapse a stable effect or a seed artifact? | High | ~1 GPU-hour | CRITICAL | **Done** (14 runs, `EXP_S7_*`, `EXP_S99_*`) |
| Compute-matched FP32 fine-tune control | QAT gets extra epochs PTQ does not | Is QAT's edge due to quantization-awareness or extra training? | High | 5 short runs | CRITICAL | **Done** (`EXP_C*`; QAT edge at INT8/INT6 not separable from extra epochs) |
| Calibration-size / method study (INT4 PTQ) | INT4 PTQ unstable | Is the failure a calibration artifact? | High | Minutes (checkpoint-level, no retraining) | HIGH | **Done** (more calibration data -> lower AUC) |
| Weights-vs-activations, per-unit sensitivity | Locate the INT4 damage | Which tensors drive INT4 loss? | High | Minutes | HIGH | **Done** (L0 weights + A0 activation dominate) |
| Mixed precision (L0+A0 INT8, rest INT4), PTQ | Follows from sensitivity result | Can a cheap sensitivity-guided assignment avoid the cliff? | Medium | Minutes, n = 3 | HIGH | **Done, exploratory only** — layout chosen on test data (TEST_SET_SELECTED / BIASED); see `exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md` |
| Per-class / ECE / error analysis | Macro AUC hides class structure | Who is hurt by INT4 PTQ? | Medium | Minutes | HIGH | **Done** (W/Z recall, ECE 0.0085 -> 0.0888) |
| hls4ml C-sim fixed-point mode ablation | Default-mode collapse suspicious | Is the C-sim collapse arithmetic-mode or model-driven? | High | Minutes | HIGH | **Done** (rounding + saturation removes most of it) |
| Controlled latency study (CPU 1 thread, CUDA events) | Earlier latency numbers uncontrolled | Does fake-quant latency reflect anything but emulation overhead? | Medium | Minutes | MEDIUM | **Done** (emulation overhead only) |
| Numerical consistency (CPU vs CUDA, repeat runs) | INT4 seed-42 differed by device | What is and is not reproducible? | High | Minutes | HIGH | **Done** (same-device bit-identical; INT4 PTQ not cross-device) |
| Reproducibility command + 2 re-runs | Phase 2 prompt requirement | Are recorded results reproducible in isolation? | High | Minutes | HIGH | **Done** (bit-identical) |
| FPGA synthesis (Vivado/Vitis HLS) | Only source of measured hardware numbers | What are real LUT/DSP/BRAM/latency? | Very high | Needs licensed tools | HIGH | **Not possible**: toolchain check 2026-10-05 found no Vivado/Vitis/Quartus on PATH or in standard install dirs (WSL or Windows). Documented as the top remaining gap. Hardware numbers stay `estimated`/`simulated`/`unavailable`. |
| Percentile/MSE/per-channel calibration, learned quantizers | Separates "INT4 is hard" from "min/max is weak" | Is INT4 PTQ failure specific to naive calibration? | Medium | Moderate (new code + tests) | MEDIUM | **Deferred**; listed as remaining gap #2. Not needed to answer the stated questions. |
| Mixed precision under QAT, more seeds | Extends mixed-precision finding | Does the mixed assignment help QAT / hold at n = 5? | Low-Medium | Moderate | LOW | **Deferred** |
| Second architecture (e.g. wider MLP / CNN) | Generality | Do findings hold beyond one 4,389-parameter MLP? | Medium | New models + full matrix | LOW | **Not run**: no concrete gap in the stated questions requires it; would add scope, not rigor. |
| Second dataset (e.g. JetSet / other) | Generality | Do findings hold beyond one simulated dataset? | Medium | Download + adapter + matrix | LOW | **Not run**: ATLAS JetSet evaluated in Phase 2 and not adopted; generality claim is explicitly not made. |
| ParticleNet / Transformer | Model-class breadth | Not a stated research question | Low | High | NOT REQUIRED | **Not run** |
| Power / energy measurement | Hardware realism | Needs physical device | n/a | n/a | NOT REQUIRED | **Not possible** |

## A/B two-run rule
Studies that depend on training were executed per seed on a single device; the
reproducibility command re-ran two experiments twice each (bit-identical). Pooled
statistics never mix CPU and CUDA runs.

## Stop criterion
The research questions (accuracy / storage / inference cost / hardware-oriented cost of
PTQ vs QAT at INT8/6/4 for a small HEP jet-tagging MLP) are adequately answered in
simulation. Remaining gaps require external tooling (synthesis) or are generality
extensions; neither is attempted without a concrete need.
