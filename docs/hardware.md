# Hardware-Aware Analysis

## Current status

`hardware.backend` defaults to `unavailable` in `configs/base.yaml` (`UnavailableBackend`),
so by default every experiment reports hardware fields (`lut`, `dsp`, `bram`, `ff`,
`latency_estimate_ns`) as `None`/`N/A`, with `resource_source` stating why.

For experiments that opt in with `hardware.backend: hls4ml` (see `configs/fp32_hls.yaml`),
**hls4ml 1.3.0 is installed and working in this environment**: `HLSBackend` performs a
real `hls4ml.converters.convert_from_pytorch_model` conversion of the trained MLP and
compiles it for **C-simulation** (`status: "csim"`), verified against the actual
trained-shape model in `tests/test_hardware.py::test_hls_backend_converts_and_csims_real_model`
and recorded as a real experiment in `experiments/results/EXP_008_FP32_HLS4ML.json`.
LUT/DSP/BRAM/FF resource counts are still `None` even in this successful path, because
those numbers come only from Vivado/Vitis HLS synthesis (see below) — conversion and
C-simulation validate that the model structure is HLS-compatible and numerically
correct, not how many FPGA resources it would consume.

## Why

Getting real FPGA resource/latency numbers out of hls4ml requires Xilinx Vivado/Vitis
HLS installed and licensed. This environment does not have Vivado installed (it's a
multi-tens-of-GB proprietary toolchain requiring a separate install and, for some
device families, a license), so FPGA *synthesis* numbers are out of scope for this
build. What IS in scope, and implemented in `hep_quant/hardware/resource_estimator.py`,
is everything short of synthesis:

- `HLSBackend.analyze()` attempts a real `hls4ml.converters.convert_from_pytorch_model`
  call, and — if a C++ compiler (`g++`) is present — compiles the generated project for
  **C-simulation** (`hls_model.compile()`). This exercises the actual hls4ml conversion
  path and would catch real conversion incompatibilities (unsupported layer types,
  shape issues), but it does not produce LUT/DSP/BRAM/FF numbers, since those come only
  from Vivado HLS synthesis.
- If hls4ml cannot even be imported (e.g. incompatible with the installed NumPy/
  TensorFlow versions, or not installed at all), `get_hardware_backend` returns
  `UnavailableBackend`, which reports `status="unavailable"` and a `resource_source`
  string naming the import error.

## What a result looks like, honestly

| status | meaning | fields populated |
|---|---|---|
| `unavailable` | hls4ml not usable, or backend explicitly disabled | none (all `None`) |
| `converted` | hls4ml accepted the model, no C++ toolchain for csim | none (all `None`), notes say conversion succeeded |
| `csim` | hls4ml conversion + C-simulation both succeeded | none (all `None` — csim validates functional correctness, not resource usage); notes explain Vivado synthesis is still required for LUT/DSP/BRAM/FF |
| `synthesized` | (not reachable in this environment — would require Vivado) | would be populated from the Vivado HLS report |

## If you have Vivado available

Set `hardware.backend: hls4ml` in a config and extend `HLSBackend.analyze()` to call
`hls_model.build(csim=False, synth=True)` and parse
`<output_dir>/myproject_prj/solution1/syn/report/myproject_csynth.rpt` for the
resource/latency table. This project's abstraction (`HardwareBackend` /
`HardwareReport`) is already shaped to carry those numbers once a real synthesis step
produces them — only the `HLSBackend.analyze()` body needs extending, nothing else in
the pipeline (`runner.py`, `reporting.py`, Pareto analysis) needs to change, since they
already treat hardware fields as nullable.
