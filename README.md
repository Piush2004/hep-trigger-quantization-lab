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

Runs by default on a **synthetic** classification dataset (see
[reports/dataset_card.md](reports/dataset_card.md) for why, and the real-dataset
integration plan). Every result in this repository is a development-validation
result unless the dataset card says otherwise.

## Quantization methodology

FP32/INT8/INT6/INT4 are all implemented as calibrated uniform-affine
fake-quantization (quantize-then-dequantize in float32), since PyTorch's native
quantized kernels only cover INT8. This is explained in full, including what it does
and doesn't measure, in [docs/quantization.md](docs/quantization.md) — read this
before interpreting model-size or latency numbers.

## Experiments

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

### Real dataset: hls4ml LHC jet tagging (5-class) — the result that matters

100,000-event experiment matrix. FP32 baseline (74.95% accuracy) is consistent with the 75% reported
for the same 16-64-32-32-5 architecture in arXiv:2003.06308 (Table 5; different split and training
budget, so a sanity check rather than a reproduction).

| Precision | Method | Accuracy | ROC-AUC (macro) | Model size (MB) |
|---|---|---:|---:|---:|
| FP32 | — | 0.7495 | 0.9335 | 0.0167 |
| INT8 | PTQ | 0.7503 | 0.9333 | 0.0045 |
| INT6 | PTQ | 0.7501 | 0.9323 | 0.0035 |
| **INT4** | **PTQ** | **0.6507** | **0.8832** | 0.0025 |
| INT8 | QAT | 0.7523 | 0.9344 | 0.0045 |
| INT6 | QAT | 0.7517 | 0.9336 | 0.0035 |
| INT4 | QAT | 0.7291 | 0.9232 | 0.0025 |

**INT4 PTQ collapses by 9.9 percentage points** — a genuine precision cliff, not a
gentle degradation. **QAT recovers most but not all of it** (65.07% → 72.91%, still
2.0 pp below FP32). All three QAT configurations are Pareto-optimal; PTQ and FP32 are
all dominated
([figure](reports/figures/pareto_accuracy_vs_size_hls4ml_lhc_jets_hlf.png)). See
[reports/research_report.md §14.2](reports/research_report.md) for the full discussion,
and [reports/dataset_card.md](reports/dataset_card.md) for dataset provenance/license
(OpenML `hls4ml_lhc_jets_hlf`, CC BY, DOI 10.5281/zenodo.3602260).

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

43+ tests across environment, data pipeline, models, training, PTQ, QAT, hardware
abstraction, and full-pipeline reproducibility. Every gate in this project's build was
run twice (two-test rule) before moving to the next — see
[reports/experiment_log.md](reports/experiment_log.md) for the full build log,
including two environment-setup failures and how they were resolved.

## Reproducibility

Single seed (42) controls Python/NumPy/PyTorch/CUDA and the data split. Every
experiment record captures the exact git commit, library versions, and GPU used. See
[reports/reproducibility_report.md](reports/reproducibility_report.md).

## Limitations

- Primary results are on a synthetic dataset, not real collision data (see dataset
  card).
- Quantization is simulated (fake-quantization), not real bit-packed integer kernels
  — see [docs/quantization.md](docs/quantization.md).
- No FPGA synthesis numbers (Vivado not installed in this environment) — see
  [docs/hardware.md](docs/hardware.md) for exactly what the hardware-aware analysis
  does and doesn't measure.

## Future work

- Integrate a real public HEP dataset (ATLAS/CMS Open Data) per the plan in the
  dataset card.
- Extend beyond the MLP to a small CNN / ParticleNet-style architecture.
- Run real hls4ml synthesis on a machine with Vivado installed, to replace the
  "unavailable"/"csim" hardware fields with actual LUT/DSP/BRAM/FF numbers.

## License

MIT — see [LICENSE](LICENSE).
