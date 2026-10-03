# HEP-Trigger Quantization Lab

A reproducible research platform for benchmarking quantized neural networks for
real-time high-energy physics (HEP) triggering.

> **Independent research project.** This investigates quantization trade-offs
> motivated by the computational constraints of real-time HEP trigger systems; it is
> not an official or CERN-affiliated project, and makes no claim of novelty or
> exclusivity. See [docs/methodology.md](docs/methodology.md) for scope and honesty
> notes.

## What is this?

A pipeline that trains a neural network classifier, quantizes it to INT8/INT6/INT4 via
both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT), and
measures the resulting trade-off between accuracy, model size, software inference
latency, and (where tooling permits) estimated FPGA resource usage — producing a
Pareto analysis rather than a single "quantization works" demo.

## Why does it matter?

Real-time trigger systems at collider experiments (e.g. the LHC) must decide, in
nanoseconds to microseconds, which collision events to keep — typically on FPGA
hardware with tight LUT/DSP/BRAM/latency budgets. Quantization is one of the primary
levers for fitting a neural classifier into that budget, but how much accuracy it
costs, and whether QAT can recover PTQ's losses, is architecture- and task-dependent.
This project builds the engineering infrastructure to measure that trade-off
rigorously and reproducibly, rather than asserting it.

## Research question

> How do post-training quantization and quantization-aware training affect the
> accuracy, latency, model size, and estimated hardware resource requirements of
> neural networks for real-time HEP event classification?

See [docs/methodology.md](docs/methodology.md) for the full set of secondary
questions.

## Architecture

```
Dataset (synthetic | HDF5 | CSV | Parquet)
   |
   v
Validation -> Stratified split (leakage-checked) -> Preprocessing
   |
   v
Configurable MLP -> Training (early stopping, checkpointing)
   |
   +--> FP32 baseline
   +--> PTQ (INT8 / INT6 / INT4)
   +--> QAT (INT8 / INT6 / INT4, fine-tuned from the FP32 checkpoint)
   |
   v
Evaluation (accuracy/F1/ROC-AUC/PR-AUC, software latency)
   |
   v
Hardware-aware analysis (hls4ml if usable, else explicit "unavailable")
   |
   v
Experiment registry (traceable JSON per run) -> Pareto analysis + report
```

Full detail in [docs/architecture.md](docs/architecture.md).

## Dataset

**Primary (real-HEP-benchmark) dataset:** the hls4ml LHC jet-tagging benchmark, OpenML `hls4ml_lhc_jets_hlf` v1
(Zenodo 10.5281/zenodo.3602260, CC BY 4.0; arXiv:1804.06913): **830,000 jets, 16 high-level substructure features, 5 classes** (g, q, t, W, Z; ≈20% each).
Provenance (verified in the dataset paper): **particle-level Monte Carlo** — MadGraph5 + Pythia8 + FastJet anti-kT R = 0.8, jet pT 0.8–1.6 TeV, 13 TeV.
It is **simulated physics data, not detector-level and not recorded collision data** (the paper describes no detector simulation). It is **not** the ATLAS JetSet dataset (evaluated and not adopted).
The benchmark has been validated on a publicly available simulated HEP dataset; it has *not* been validated on real detector data.
See [reports/dataset_selection.md](reports/dataset_selection.md), [reports/dataset_schema.md](reports/dataset_schema.md), [reports/real_data_validation.md](reports/real_data_validation.md).
A synthetic dataset is retained only for pipeline tests (see [reports/dataset_card.md](reports/dataset_card.md)).

## Quantization methodology

FP32/INT8/INT6/INT4 are all implemented as calibrated uniform-affine
fake-quantization (quantize-then-dequantize in float32), since PyTorch's native
quantized kernels only cover INT8. This is explained in full, including what it does
and doesn't measure, in [docs/quantization.md](docs/quantization.md) — read this
before interpreting model-size or latency numbers.

## Experiments

Real-data (primary) configs are `configs/jet_*.yaml` (`jet_base.yaml`, staged `jet_validate_*`, the 7-configuration matrix and `jet_fp32_finetune_control.yaml`); the table lists the synthetic-pipeline configs.

| Config | Method | Precision |
|---|---|---|
| `configs/fp32.yaml` | — | FP32 baseline |
| `configs/int8.yaml` | PTQ | INT8 |
| `configs/int6.yaml` | PTQ | INT6 |
| `configs/int4.yaml` | PTQ | INT4 |
| `configs/int8_qat.yaml` | QAT | INT8 |
| `configs/int6_qat.yaml` | QAT | INT6 |
| `configs/int4_qat.yaml` | QAT | INT4 |

## Results

### Real dataset: hls4ml jet tagging (5-class), audited multi-seed results

MLP 16-64-32-32-5 (4,389 parameters), 100,000-jet subsets, **5 seeds** (42, 123, 2026 frozen; 7, 99 added), all on one device. Macro one-vs-rest ROC-AUC, mean ± SD (n = 5, descriptive):

| Precision | Method | ROC-AUC | ΔAUC vs same-seed FP32 | Theoretical weight storage |
|---|---|---:|---:|---:|
| FP32 | — | 0.9343 ± 0.0011 | — | 17,556 B |
| INT8 | PTQ | 0.9342 ± 0.0011 | −0.0001 | 4,788 B |
| INT6 | PTQ | 0.9329 ± 0.0014 | −0.0014 | 3,724 B |
| **INT4** | **PTQ** | **0.8819 ± 0.0308** (0.835–0.909) | **−0.052** | 2,660 B |
| INT8 | QAT | 0.9348 ± 0.0017 | +0.0005 | 4,788 B |
| INT6 | QAT | 0.9342 ± 0.0015 | −0.0001 | 3,724 B |
| INT4 | QAT | 0.9246 ± 0.0018 | −0.0097 | 2,660 B |

What the evidence supports (simulation, this simulated dataset, this MLP): INT8/INT6 PTQ are practically lossless; **INT4 PTQ is unreliable** (seed- and device-dependent, worst on W/Z jets and probability calibration, tied to calibration range and the first layer);
INT4 QAT is stable at about −1% AUC; and **a compute-matched FP32 fine-tune control shows QAT's INT8/INT6 edge over PTQ is not separable from extra training epochs.**
*Not* supported: any FPGA latency/resource claim or generality to other architectures or datasets.
Details: [reports/REAL_HEP_RESULTS.md](reports/REAL_HEP_RESULTS.md), [reports/phase2_results_audit.md](reports/phase2_results_audit.md). The FP32 baseline is consistent with (not a reproduction of) the 75% in arXiv:2003.06308 Table 5.

**Hardware measurement limitations — read before using any hardware number.** "Storage" above is *analytical* (bit-packed equivalent; no packed artifact exists; checkpoints are float32).
LUT/DSP/BRAM/FF are a **software heuristic (`measurement_tier = estimated`)**, never validated against synthesis. hls4ml **C-simulation** (`simulated`) measures accuracy under fixed-point arithmetic only (no latency or resources).
Software latency is emulation overhead (FP32 < PTQ < QAT) and not hardware latency. **Physical FPGA synthesis was not performed.** Tiers: measured / simulated / estimated / unavailable, never mixed in a figure.

Earlier single-seed numbers (INT4 PTQ "−9.9 points, QAT recovers most") came from one CPU run and are superseded by the table above.

### Synthetic development dataset (pipeline validation only — no physics meaning)

| Precision | Method | Accuracy | Model size (MB) | Latency (ms/sample, software) |
|---|---|---:|---:|---:|
| FP32 | — | 0.9853 | 0.0480 | 0.445 |
| INT8 | PTQ | 0.9850 | 0.0126 | 2.099 |
| INT6 | PTQ | 0.9843 | 0.0097 | 2.159 |
| INT4 | PTQ | 0.9833 | 0.0068 | 1.821 |
| INT8 | QAT | 0.9863 | 0.0126 | 5.497 |
| **INT6** | **QAT** | **0.9863** | **0.0097** | 4.947 |
| INT4 | QAT | 0.9857 | 0.0068 | 4.151 |

On this easier synthetic task PTQ barely degrades even at INT4 and QAT fully recovers
at every precision — contrast this with the real dataset above, where INT4 shows a
real cliff. This gap between the two datasets is itself a finding (see the research
report's Discussion): the synthetic task understated real quantization sensitivity.

Notably, latency gets *worse*, not better, at lower precision on both datasets — this
is a real measurement, not a mistake: it's the cost of simulating quantization in
float32 (see [docs/quantization.md](docs/quantization.md)), not a deployment speedup,
which would require real integer hardware kernels or FPGA synthesis.

Full data: [results_summary.csv](results_summary.csv) (every experiment, one `dataset`
column distinguishing them — never silently merged).
Full discussion: [reports/research_report.md](reports/research_report.md).

## How to run

Requires Python 3.11 (PyTorch does not yet support 3.14; see
[environment_report.md](environment_report.md) for why this project runs inside
WSL2 rather than native Windows on this machine).

```bash
# Set up environment
uv venv .venv --python 3.11
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128  # or /cpu
uv pip install --python .venv -e .

# Validate the data pipeline
python -m hep_quant validate-data --config configs/fp32.yaml

# Run the full experiment matrix + aggregate report
python scripts/run_experiment.py

# Or run one experiment at a time
python -m hep_quant train --config configs/fp32.yaml
python -m hep_quant train --config configs/int8.yaml
python -m hep_quant report
```

## Testing

```bash
pytest -q
```

90+ tests across environment, data pipeline, schema enforcement, models, training, PTQ/QAT (including equivalence of the study builder), hardware abstraction, reporting and full-pipeline reproducibility
(a few network/CUDA-dependent tests are opt-in or auto-skipped). Every gate in this project's build was
run twice (two-test rule) before moving to the next — see
[reports/experiment_log.md](reports/experiment_log.md) for the full build log,
including two environment-setup failures and how they were resolved.

## Reproducibility

The 21 primary runs are **frozen** (`experiments/results/frozen_phase2/`, SHA-256 manifest; verify with `python scripts/freeze_phase2.py verify`).
Re-run any recorded experiment in isolation and compare metrics, weights and predictions:

```bash
python -m hep_quant reproduce --experiment EXP_S42_JET_INT4_QAT
```

Two such reproductions were bit-identical on the same device; INT4 PTQ is **not** reproducible across devices (CPU vs CUDA differ by 8 accuracy points at one seed), so the distribution over seeds is the reproducible quantity.
Pinned environment: `requirements-lock.txt`, `environment.yml`; the CUDA availability timeline is in [environment_report.md](environment_report.md).
Known provenance gap: the 21 frozen runs recorded commit `4e9b673` while the working tree was dirty (see the audit). See also [reports/reproducibility_report.md](reports/reproducibility_report.md).

## Limitations

- The dataset is simulated particle-level MC (not detector-level, not recorded data); one dataset, one 4,389-parameter MLP, 5 seeds.
- Quantization is simulated (fake-quantization), not real bit-packed integer kernels
  — see [docs/quantization.md](docs/quantization.md).
- No FPGA synthesis numbers (Vivado not installed in this environment) — see
  [docs/hardware.md](docs/hardware.md) for exactly what the hardware-aware analysis
  does and doesn't measure.

## Future work

- Percentile/MSE calibration for INT4 PTQ; mixed precision under QAT.
- A second architecture and dataset, only if a generality claim is wanted.
- Run real hls4ml synthesis on a machine with Vivado installed, to replace the
  "unavailable"/"csim" hardware fields with actual LUT/DSP/BRAM/FF numbers.

## License

MIT — see [LICENSE](LICENSE).
