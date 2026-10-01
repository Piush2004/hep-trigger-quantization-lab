# Architecture

```
Dataset (synthetic | HDF5 | CSV | Parquet)
   |
   v
Validation (shape, dtype, NaN/Inf, label range, duplicates, class balance)
   |
   v
Stratified train/val/test split (deterministic seed, leakage-checked)
   |
   v
Preprocessing (standardization, fit on train only)
   |
   v
Model (configurable MLP; hidden_dims from YAML, never hardcoded)
   |
   v
Training (early stopping, checkpointing, per-epoch metric logging)
   |
   +--> FP32 baseline evaluation
   |
   +--> PTQ: calibrate activation ranges -> fake-quantize weights+activations
   |         (INT8 / INT6 / INT4, uniform affine quantize-dequantize)
   |
   +--> QAT: replace Linear/activation layers with fake-quant equivalents,
             fine-tune from the FP32 checkpoint with a straight-through estimator
   |
   v
Evaluation (accuracy/precision/recall/F1/ROC-AUC/PR-AUC, software latency)
   |
   v
Hardware-aware analysis (HLSBackend if hls4ml usable, else UnavailableBackend
                          with every resource field reported as None + reason)
   |
   v
Experiment registry (self-contained, traceable JSON per run: config +
                      reproducibility info + metrics + hardware)
   |
   v
Reporting (results_summary.csv, research figures, algorithmic Pareto frontier)
```

## Package layout

- `hep_quant.data` — loading, validation, splitting, preprocessing, synthetic generator.
- `hep_quant.models` — configurable MLP, model factory.
- `hep_quant.training` — training loop, metrics, evaluation.
- `hep_quant.quantization` — PTQ, QAT, calibration, size/parameter analysis.
- `hep_quant.hardware` — software latency benchmarking, hls4ml backend abstraction.
- `hep_quant.experiments` — orchestration (`runner.py`), registry, reproducibility capture.
- `hep_quant.visualization` — research figures, Pareto frontier.
- `hep_quant.reporting` — aggregates all experiment records into tables/plots/CSV.

## Design decisions worth knowing

- **Fake-quantization for all precisions, including INT8.** PyTorch's native
  quantized kernels only support INT8 (via fbgemm/qnnpack); there is no built-in
  INT6/INT4 kernel. To compare FP32/INT8/INT6/INT4 on equal footing, all four are
  implemented as calibrated quantize-then-dequantize in float32 (see
  `hep_quant/quantization/ptq.py`). This is standard research methodology, but it
  means model-size numbers for quantized models are *estimated* (bit-packed-equivalent
  size), not measured from an actually bit-packed artifact.
- **Hardware backend degrades gracefully.** `HardwareBackend` has two
  implementations: `HLSBackend` (real hls4ml conversion + C-simulation, best-effort)
  and `UnavailableBackend` (every field `None`, reason documented). The rest of the
  pipeline — training, quantization, experiment tracking, Pareto analysis — works
  completely on `UnavailableBackend` alone; hardware numbers are a bonus layer, not a
  dependency.
- **WSL2, not native Windows.** See `environment_report.md` for why: Windows Smart
  App Control blocks matplotlib's native extension on this machine, and Docker
  Desktop's backend was separately broken by an orphaned socket file. The code itself
  is platform-agnostic (no Windows- or WSL-specific logic in `src/hep_quant`).
