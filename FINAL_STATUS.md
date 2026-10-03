# Final Project Status

## Overall Status

**REPRODUCIBLE BENCHMARK** (upgraded from Research Prototype — see Research Readiness
below for exactly what changed and what's still missing for Publication Candidate).

## Completed

- Project scaffold, YAML config system, CLI (`python -m hep_quant ...`), seed/logging/
  system-info utilities.
- Data pipeline: HDF5/CSV/Parquet/NumPy/OpenML loader, shape/dtype/NaN/Inf/duplicate/
  class-balance validation (fails loudly), deterministic stratified leakage-checked
  train/val/test split, synthetic dataset generator, and a real-dataset loader with a
  parsed-data cache (avoids re-paying a ~2-3 minute ARFF parse on every run).
- Configurable MLP baseline, training loop (early stopping, checkpointing, per-epoch
  metric logging), evaluation generalized to both binary and multiclass (macro-averaged
  precision/recall/F1/ROC-AUC/PR-AUC for >2 classes, original binary behavior unchanged
  for 2).
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
  and an algorithmic (non-manual) Pareto-frontier finder, generated **separately per
  dataset** (never silently merging two datasets' results into one plot).
- **Real dataset integrated and fully experimented on**: OpenML `hls4ml_lhc_jets_hlf`
  (the hls4ml team's own published jet-tagging benchmark, CC BY, DOI
  10.5281/zenodo.3602260) — 5-class, 16 features. Staged validation (10k pipeline
  check, 50k training check) followed by the full 100k-event matrix: FP32 baseline,
  PTQ×3, QAT×3 (7 experiments).
- Full experiment matrix also still executed on the synthetic development dataset: FP32
  baseline, PTQ×3, QAT×3, + 1 hls4ml-backend run, + 1 software-estimate-backend run (9
  experiments) — kept as the pipeline-validation reference, never conflated with the
  real-dataset results.
- Documentation: README, architecture/methodology/quantization/hardware/
  troubleshooting docs, dataset card (both datasets, with explicit real-vs-synthetic-
  vs-simulated-MC distinctions), reproducibility report, experiment log, research
  report (every section populated with real results, no placeholders).
- Git tag `v0.1.0-research-prototype` created at the end of Phase 1, marking the
  checkpoint before real-dataset work began (see "Git tag" note below — not pushed, no
  remote configured).

## Tested

43 tests at the initial environment+data+model+PTQ+hardware-stub milestone, growing
through 60 (QAT + real hls4ml), 66 (SoftwareEstimateBackend), 71 (multiclass metrics),
to **73 tests** after the per-dataset reporting fix — plus 4 opt-in, network-gated
OpenML tests (not counted in the default `pytest -q` run; run separately with
`RUN_NETWORK_TESTS=1`). Every gate was run twice (two-test rule) before moving to the
next; see `reports/experiment_log.md` for the full build-order log.

## Test Results

- Test suite: 73/73 passed (+5 skipped: 1 CUDA-only, 4 network-gated), run twice
  independently at every milestone in this session.
- OpenML loader: 4/4 passed twice independently (schema, stratified-subsample balance,
  validation, parsed-data cache correctness).
- End-to-end: full synthetic pipeline verified to run twice with identical results
  (`tests/test_pipeline.py`); PTQ determinism, training determinism, and split
  determinism each independently verified.

## Experiments Completed

**Real dataset** (`hls4ml_lhc_jets_hlf`, 100k events unless noted):

| ID | Precision | Method | Notes |
|---|---|---|---|
| EXP_010_JET_VALIDATE_10K | FP32 | — | Stage 1: pipeline validation (10k, 8 epochs) |
| EXP_011_JET_VALIDATE_50K | FP32 | — | Stage 2: training validation (50k, 15 epochs) |
| EXP_012_JET_FP32 | FP32 | — | Stage 3 baseline |
| EXP_013/014/015_JET_INTx_PTQ | INT8/6/4 | PTQ | |
| EXP_016/017/018_JET_INTx_QAT | INT8/6/4 | QAT | |

**Synthetic dataset** (pipeline validation only):

| ID | Precision | Method |
|---|---|---|
| EXP_001_FP32_MLP – EXP_007_INT4_QAT | FP32/INT8/INT6/INT4 | baseline/PTQ/QAT |
| EXP_008_FP32_HLS4ML | FP32 | — (hls4ml backend) |
| EXP_009_FP32_ESTIMATE | FP32 | — (software_estimate backend) |

Results: `results_summary.csv` (18 rows, one `dataset` column distinguishing them),
per-dataset tables in `reports/tables/`, per-dataset figures in `reports/figures/`,
full discussion in `reports/research_report.md`.

## Experiments Not Completed

- **FPGA synthesis.** hls4ml conversion + C-simulation succeeded, but LUT/DSP/BRAM/FF/
  synthesis-latency numbers require Vivado/Vitis HLS, which is not installed in this
  environment. `SoftwareEstimateBackend` provides a labeled heuristic in the meantime.
- **Second architecture** (CNN / ParticleNet-style) for cross-architecture
  quantization-robustness comparison — scoped as future work, not attempted here.
- **Multi-seed statistics.** All results use a single seed (42). Reproducibility is
  verified run-to-run for that seed, but no confidence interval or variance estimate
  across seeds exists — notably relevant for the real dataset's INT4-QAT result (a 2.0
  pp gap to FP32), which would benefit from knowing its seed-to-seed variance before
  treating it as a precise number rather than an estimate.
- **hls4ml resource/latency estimate reconciliation.** `SoftwareEstimateBackend`'s
  heuristic has never been cross-checked against a real Vivado synthesis report, since
  none exists in this environment — its ballpark is internally consistent but
  unvalidated against ground truth.

## Known Issues

Four real bugs were found and fixed during the build (not hidden — all are in
`reports/experiment_log.md` and the git history with full root-cause writeups):

1. `estimate_model_size`'s parameter counter matched by `isinstance(m, nn.Linear)`,
   which silently returned 0 for every QAT model (QATLinear isn't an nn.Linear
   subclass) — every QAT experiment initially reported `estimated_size_mb: 0.0`.
   Fixed by matching on parameter name instead.
2. hls4ml's PyTorch conversion needs a config built via `config_from_pytorch_model`
   (not a hand-rolled dict) with a tuple `input_shape`, and needs a CPU-resident model
   (it introspects weights via NumPy) — the first live attempt against a CUDA-resident
   trained model failed with `can't convert cuda:0 device type tensor to numpy`. Fixed
   by converting a CPU copy inside `HLSBackend.analyze`.
3. `plot_metric_vs_precision` used `reindex` over a precision column that isn't unique
   once both PTQ and QAT exist at the same precision — crashed generating
   `model_size_vs_precision.png` once the QAT experiments existed. Fixed by sorting per
   method-group.
4. `generate_report()` was plotting both the synthetic and real datasets' results
   together in one Pareto/size/latency figure with no visual distinction — exactly the
   silent-dataset-merging this project's own dataset card forbids. Fixed: figures/
   tables are now generated separately per dataset, filename-tagged, with the
   deliberately-under-trained staged-validation runs (EXP_010/011) excluded from the
   precision-comparison analysis entirely.

Every one of these has a regression test and was caught by actually exercising the
code against real multi-experiment data, not just unit-level mocks. No open/unresolved
code issues at time of writing. One environment issue remains open (CUDA regression,
below) — not a code bug, explicitly not chased down per instruction.

## Hardware

Host: Windows 11 Home, AMD Ryzen 7, 24 GB RAM, NVIDIA RTX 5060 Laptop GPU (8 GB VRAM),
429 GB disk. Execution environment: WSL2 Ubuntu 26.04.1 LTS (see `environment_report.md`
for why: Windows Smart App Control blocking matplotlib's native extension, and a
separately-broken Docker Desktop backend). PyTorch 2.11.0+cu128. hls4ml 1.3.0 installed
and verified working for conversion + C-simulation (no Vivado for synthesis).

**CUDA availability regression:** CUDA worked earlier in the build (GPU-accelerated
training through `EXP_008`). After a `wsl --shutdown` run during Docker Desktop
troubleshooting, CUDA stopped initializing in WSL2 (`nvidia-smi` → `Failed to
initialize NVML`, reproduced on both WSL2 and the Windows host). Per explicit
instruction, this was not debugged further — all experiments from `EXP_009` onward
(including the entire real-dataset matrix) ran on CPU. This did not block anything:
device selection is `device="auto"`, so no code changed, just wall-clock time. Likely
fixable with a full machine reboot, left to the user's discretion.

## Dataset

**Primary (Phase 2):** OpenML `hls4ml_lhc_jets_hlf` — the hls4ml team's own published
jet-tagging benchmark (16 features, 5 classes: gluon/quark/top/W/Z jets), Monte
Carlo-simulated (MadGraph5+Pythia8+FastJet, no detector simulation described), CC BY licensed, DOI 10.5281/zenodo.3602260. This is
**simulated** jet data, not recorded ATLAS/CMS detector data — see
`reports/dataset_card.md`'s "What 'real' means here" for why that's the correct and
standard thing for this kind of study, and why it's still meaningfully different from
this project's own synthetic (`make_classification`) data.

**Secondary (Phase 1, retained):** Synthetic (`sklearn.datasets.make_classification`,
20,000 samples × 16 features, binary) — pipeline-validation only, no physics meaning.

Full specification, license, citation, and schema for both in `reports/dataset_card.md`.

## Results Available

**Real dataset:** FP32 baseline 74.95% accuracy / 0.9335 ROC-AUC (consistent with the
75% reported for the same architecture in arXiv:2003.06308 Table 5, which uses a different
split and training budget — a sanity check, not a reproduction). INT8/INT6 PTQ essentially lossless. **INT4 PTQ collapses 9.9
percentage points** (a genuine precision cliff). QAT recovers most of it (+7.8pp) but
not completely (still -2.0pp vs FP32). All three QAT configs are Pareto-optimal on
accuracy-vs-size; PTQ and FP32 are all dominated. Full discussion:
`reports/research_report.md` §14.2/§15.2.

**Synthetic dataset:** PTQ barely degrades even at INT4; QAT fully recovers (slightly
exceeds) FP32 at every precision — a materially easier, less realistic picture than the
real dataset, which is itself a documented finding (§14.1 vs §14.2 comparison).

## Results Not Yet Available

FPGA resource utilization (LUT/DSP/BRAM/FF) and synthesis-level latency from real
Vivado/Vitis HLS synthesis — not installed in this environment (software-heuristic
estimates are available via `SoftwareEstimateBackend` as a labeled stand-in). Multi-seed
variance/confidence intervals on any reported accuracy number.

## Scientific Limitations

See `reports/research_report.md` Section 17 in full. Summary: the real dataset is
simulated (not recorded detector data); quantization is simulated fake-quantization,
not real integer hardware execution (latency numbers reflect simulation overhead, not a
deployment speedup); no FPGA synthesis; single architecture; single random seed.

## Git tag

`v0.1.0-research-prototype` was created locally at the Phase 1 → Phase 2 boundary, per
request, to preserve that checkpoint. **It has not been pushed** — this repository has
no configured git remote (`git remote -v` returns empty), so there is nowhere to push
it to yet. If you add a remote, push it with `git push origin v0.1.0-research-prototype`.

## Incidental finding

A file `docs/CERN_Study_Guide.pdf` (22 pages) appeared in the working tree during this
session that was not created by this work — it is untracked and was not touched,
staged, or committed. Flagging it in case it's something you placed there that you'd
like added to the repo, or something unexpected worth checking.

## Recommended Next Step

With the real dataset now integrated and the full experiment matrix run, the highest-
value next steps are: (1) run on a machine with Vivado/Vitis HLS to get real FPGA
synthesis numbers, replacing `SoftwareEstimateBackend`'s heuristic with ground truth;
(2) repeat the real-dataset matrix across 3-5 seeds to put a confidence interval on the
INT4-QAT residual gap (currently a single-seed point estimate); (3) a second
architecture on the same real dataset, to test whether the INT4 precision cliff is
architecture-specific or general to this task.

## Research Readiness

**Reproducible Benchmark.** This is an upgrade from "Research Prototype": the pipeline
now runs end-to-end against a real, citable, properly licensed HEP dataset (not just
synthetic data), with a documented staged-validation process, full PTQ/QAT experiment
matrix, and a central finding (the INT4 precision cliff and QAT's partial recovery of
it) that is specific to real data and would not have been discovered from the synthetic
dataset alone. It is not yet a **Publication Candidate**: that would need FPGA synthesis
numbers (not just the software estimate), multi-seed statistical rigor on the headline
INT4-QAT result, and ideally a second architecture or dataset to establish generality.
Calling it more than "Reproducible Benchmark" would overstate what has actually been
measured here.
