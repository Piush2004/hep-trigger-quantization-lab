# Final Project Status

## Overall Status

**RESEARCH PROTOTYPE** (see Research Readiness below for the distinction).

## Completed

- Project scaffold, YAML config system, CLI (`python -m hep_quant ...`), seed/logging/
  system-info utilities.
- Data pipeline: HDF5/CSV/Parquet/NumPy loader, shape/dtype/NaN/Inf/duplicate/class-
  balance validation (fails loudly), deterministic stratified leakage-checked
  train/val/test split, synthetic dataset generator.
- Configurable MLP baseline, training loop (early stopping, checkpointing, per-epoch
  metric logging), evaluation (accuracy/precision/recall/F1/ROC-AUC/PR-AUC).
- PTQ and QAT via calibrated uniform affine fake-quantization at INT8/INT6/INT4
  (documented explicitly as simulated, not real bit-packed kernels).
- Hardware-aware analysis abstraction, three tiers: `HLSBackend` (real hls4ml 1.3.0
  conversion + C-simulation, `measurement_tier="simulated"`), `SoftwareEstimateBackend`
  (parameter-count x bit-width x reuse-factor heuristic, `measurement_tier="estimated"`,
  always available, used as the automatic fallback when hls4ml isn't usable), and
  `UnavailableBackend` (`measurement_tier="unavailable"`) — the pipeline works
  end-to-end on any of the three.
- Experiment registry (self-contained traceable JSON per run), reproducibility capture
  (seed, library versions, GPU, git commit).
- Visualization (training curves, ROC/PR, confusion matrix, size/latency-vs-precision)
  and an algorithmic (non-manual) Pareto-frontier finder.
- Full experiment matrix executed: FP32 baseline, PTQ×3 (INT8/INT6/INT4), QAT×3
  (INT8/INT6/INT4), + 1 hls4ml-backend run — 8 experiments total.
- Documentation: README, architecture/methodology/quantization/hardware/
  troubleshooting docs, dataset card, reproducibility report, experiment log,
  research report (all sections populated with real results, no placeholders).

## Tested

43 tests at the initial environment+data+model+PTQ+hardware-stub milestone, growing to
**60 tests** after QAT and the real hls4ml integration were added. Every gate was run
twice (two-test rule) before moving to the next; see `reports/experiment_log.md` for
the full build-order log, including two environment-setup failures and how each was
resolved.

## Test Results

- Test suite: 60/60 passed, run twice independently (see `reports/experiment_log.md`
  and the final quality-check run below).
- End-to-end: full synthetic pipeline (data → preprocessing → train → eval → quantize
  → eval) verified to run twice with identical results
  (`tests/test_pipeline.py::test_full_pipeline_runs_twice_consistently`).
- Reproducibility: verified via dedicated tests (`test_training_reproducible_with_same_seed`,
  `test_quantization_reproducible`, `test_split_dataset_deterministic`) rather than
  asserted — see `reports/reproducibility_report.md`.

## Experiments Completed

| ID | Precision | Method |
|---|---|---|
| EXP_001_FP32_MLP | FP32 | — |
| EXP_002_INT8_PTQ | INT8 | PTQ |
| EXP_003_INT6_PTQ | INT6 | PTQ |
| EXP_004_INT4_PTQ | INT4 | PTQ |
| EXP_005_INT8_QAT | INT8 | QAT |
| EXP_006_INT6_QAT | INT6 | QAT |
| EXP_007_INT4_QAT | INT4 | QAT |
| EXP_008_FP32_HLS4ML | FP32 | — (hls4ml backend) |
| EXP_009_FP32_ESTIMATE | FP32 | — (software_estimate backend) |

Results: `results_summary.csv`, `reports/tables/quantization_results.md`,
`reports/figures/*.png`, full discussion in `reports/research_report.md`.

## Experiments Not Completed

- **Real HEP dataset run.** Everything above is on the synthetic development dataset
  (see `reports/dataset_card.md`). The loader already supports HDF5/CSV/Parquet, so
  this is a config change, not a code change, once a dataset is selected and
  downloaded — but doing that download/validation carefully was out of scope for this
  session's time budget.
- **FPGA synthesis.** hls4ml conversion + C-simulation succeeded (EXP_008), but LUT/
  DSP/BRAM/FF/synthesis-latency numbers require Vivado/Vitis HLS, which is not
  installed in this environment.
- **Second architecture** (CNN / ParticleNet-style) for cross-architecture
  quantization-robustness comparison — scoped as future work (Section 19 of the
  research report), not attempted here per the project's "start simple" priority.

## Known Issues

Two real bugs were found and fixed during the build (not hidden — both are in
`reports/experiment_log.md` and the git history with full root-cause writeups):

1. `estimate_model_size`'s parameter counter matched by `isinstance(m, nn.Linear)`,
   which silently returned 0 for every QAT model (QATLinear isn't an nn.Linear
   subclass) — every QAT experiment initially reported `estimated_size_mb: 0.0`.
   Fixed by matching on parameter name instead; verified with a regression test and
   by re-running the affected experiments.
2. hls4ml's PyTorch conversion needs a config built via `config_from_pytorch_model`
   (not a hand-rolled dict) with a tuple `input_shape`, and needs a CPU-resident model
   (it introspects weights via NumPy) — the first live attempt against a CUDA-resident
   trained model failed with `can't convert cuda:0 device type tensor to numpy`. Fixed
   by converting a CPU copy inside `HLSBackend.analyze`, with a regression test
   covering the CUDA-resident case specifically.
3. `plot_metric_vs_precision` used `reindex` over a precision column that isn't unique
   once both PTQ and QAT exist at the same precision (e.g. two "int8" rows) — crashed
   generating `model_size_vs_precision.png` once the QAT experiments existed. Fixed by
   sorting per method-group; covered by `tests/test_visualization.py`.

No open/unresolved code issues at time of writing — all three were caught, fixed,
verified, and the affected experiments re-run. One environment issue remains open (see
"CUDA availability regression" below) — not a code bug, and explicitly not chased down
per instruction.

## Hardware

Host: Windows 11 Home, AMD Ryzen 7, 24 GB RAM, NVIDIA RTX 5060 Laptop GPU (8 GB VRAM),
429 GB disk. Execution environment: WSL2 Ubuntu 26.04.1 LTS (not native Windows — see
`environment_report.md` for the two environment-setup blockers that led to this
choice: Windows Smart App Control blocking matplotlib's native extension, and a
separately-broken Docker Desktop backend). PyTorch 2.11.0+cu128. hls4ml 1.3.0 installed
and verified working for conversion + C-simulation (no Vivado for synthesis).

**CUDA availability regression:** CUDA worked earlier in the build (GPU-accelerated
training, a CUDA matmul, and a CUDA-resident hls4ml conversion all succeeded — see git
history, e.g. `EXP_001`–`EXP_008`'s `reproducibility.system.cuda_available: true`).
After a `wsl --shutdown` run during Docker Desktop troubleshooting, CUDA stopped
initializing in WSL2 (`nvidia-smi` → `Failed to initialize NVML`, reproduced on both
WSL2 and the Windows host). One bounded fix attempt (shutdown + retry) didn't resolve
it; per explicit instruction, further NVIDIA/WSL debugging was not pursued. All
experiments after this point (`EXP_009` onward) ran on CPU — same code path, since
device selection is `device="auto"`, not a code change — just slower. Likely fixable
with a full machine reboot, left to the user's discretion.

## Dataset

Synthetic (`sklearn.datasets.make_classification`, 20,000 samples × 16 features,
binary classification). Full specification, rationale, and real-dataset integration
plan in `reports/dataset_card.md`. **No real HEP dataset was used to produce any
result in this repository** — this is stated explicitly and repeatedly throughout the
documentation to prevent any result here being mistaken for a physics result.

## Results Available

Accuracy/F1/ROC-AUC/PR-AUC, estimated model size, software inference latency, and
hls4ml conversion/C-simulation status, across FP32/INT8/INT6/INT4 × {baseline, PTQ,
QAT}, plus an algorithmic Pareto frontier — all in `results_summary.csv` and discussed
in `reports/research_report.md`.

## Results Not Yet Available

FPGA resource utilization (LUT/DSP/BRAM/FF) and synthesis-level latency — require
Vivado/Vitis HLS, not installed here. Any physics-relevant accuracy conclusion —
requires the real-dataset experiment.

## Scientific Limitations

See `reports/research_report.md` Section 17 (Limitations) in full. Summary: synthetic
dataset only; quantization is simulated fake-quantization, not real integer hardware
execution (latency numbers reflect simulation overhead, not a deployment speedup); no
FPGA synthesis; single architecture; single random seed (reproducibility is verified
run-to-run for that one seed, not characterized across seeds).

## Recommended Next Step

Integrate one real public HEP dataset (ATLAS or CMS Open Data) per the plan in
`reports/dataset_card.md`, and re-run the existing experiment matrix unchanged against
it (the loader/config system already supports this as a config change) — this is the
single step that would convert the engineering conclusions in this report into a
physics-relevant one.

## Research Readiness

**Research Prototype.** The engineering pipeline (data validation, training,
quantization, hardware-aware analysis, experiment tracking, reporting) is complete,
tested twice at every gate, and scientifically honest about what it does and doesn't
measure. It is not yet a **Reproducible Benchmark** in the sense of having run against
a real, citable HEP dataset, and it is not a **Publication Candidate** without that
real-dataset run and (ideally) FPGA synthesis numbers to complete the hardware-aware
story. Calling it more than "Research Prototype" would overstate what has actually
been measured here.
