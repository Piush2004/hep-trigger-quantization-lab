# Hardware-Aware Analysis

## Four backends, four measurement tiers

Every `HardwareReport` carries a `measurement_tier` field so a consumer never has to
guess how trustworthy a number is:

| Backend | `measurement_tier` | What it actually does |
|---|---|---|
| `UnavailableBackend` | `"unavailable"` | Nothing — every field `None`. Used when `hardware.backend: unavailable` in config. |
| `SoftwareEstimateBackend` | `"estimated"` | Pure arithmetic heuristic (parameter count × bit-width × reuse factor). No external tool, always available. See formula below. |
| `SurrogateEstimateBackend` | `"surrogate"` | ML regressor trained on real hls4ml synthesis records from the public `fastmachinelearning/wa-hls4ml` dataset (see `reports/phase11_surrogate_hardware_validation.md`). Reports its own held-out error with every prediction. **Not validated against this project's own synthesis** — only against wa-hls4ml's held-out data. |
| `HLSBackend` | `"simulated"` on success (`"unavailable"` if hls4ml itself fails) | Real hls4ml conversion + C-simulation. Would be `"measured"` if extended to run Vivado synthesis (see below) — not reachable in this environment. |

`get_hardware_backend` requesting `hls4ml` (or `surrogate`) but finding it unusable falls
back to `SoftwareEstimateBackend`, not straight to `UnavailableBackend` — a labeled estimate
is preferred over nothing, but it is never silently presented as hls4ml, Vivado, or
surrogate-model output it did not actually produce.

**Important limitation disclosed in Phase 11's report:** the trained v1 surrogate gives
essentially zero feature importance to per-layer bit-width or width/depth — its LUT/DSP/FF/
BRAM/latency/II predictions are driven almost entirely by total parameter count and reuse
factor, meaning two models that differ only in quantization precision currently get the
**same** surrogate prediction. For precision-sensitive comparisons, `SoftwareEstimateBackend`'s
cruder heuristic (which does scale with bit-width) remains the more informative of the two —
use the surrogate for its validated latency/II numbers, not yet for precision-driven LUT/DSP/FF
comparisons. See the phase report for the full feature-importance analysis.

## Current status

`hardware.backend` defaults to `unavailable` in `configs/base.yaml` (`UnavailableBackend`),
so by default every experiment reports hardware fields (`lut`, `dsp`, `bram`, `ff`,
`latency_estimate_ns`) as `None`/`N/A`, with `resource_source` stating why.

`configs/fp32_estimate.yaml` opts into `hardware.backend: software_estimate`:
`SoftwareEstimateBackend` always runs (no external dependency) and reports non-null
`lut`/`dsp`/`ff` from the heuristic below, explicitly tagged `measurement_tier:
"estimated"`. The formula (see `SoftwareEstimateBackend`'s docstring in
`resource_estimator.py`):

```
dsp  = ceil(weight_params / reuse_factor)       # one DSP per MAC "slot"
lut  = ceil(dsp * bits * 1.5)                   # fixed, documented constant — not device-calibrated
ff   = total_params * bits
bram = 0                                        # models this small don't need block RAM (documented simplification)
```

This is deliberately simple and explicitly NOT calibrated against any real Vivado run —
it exists to give an always-available, clearly-labeled ballpark, never a stand-in for
real synthesis.

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

| status | meaning | measurement_tier | fields populated |
|---|---|---|---|
| `unavailable` | hls4ml not usable, or backend explicitly disabled | `unavailable` | none (all `None`) |
| `estimated` | `SoftwareEstimateBackend` heuristic | `estimated` | `lut`/`dsp`/`ff` (non-null), `bram=0` |
| `surrogate` | `SurrogateEstimateBackend` ML regressor (wa-hls4ml-trained) | `surrogate` | `lut`/`dsp`/`bram`/`ff`/`latency_estimate_ns`/`initiation_interval` (non-null); `notes`/`configuration` carry the model's own held-out R²/MAE so trust is self-reported, not assumed |
| `converted` | hls4ml accepted the model, no C++ toolchain for csim | `simulated` | none (all `None`), notes say conversion succeeded |
| `csim` | hls4ml conversion + C-simulation both succeeded | `simulated` | none (all `None` — csim validates functional correctness, not resource usage); notes explain Vivado synthesis is still required for LUT/DSP/BRAM/FF |
| `synthesized` | (not reachable in this environment — would require Vivado) | `measured` | would be populated from the Vivado HLS report |

## If you have Vivado available

Set `hardware.backend: hls4ml` in a config and extend `HLSBackend.analyze()` to call
`hls_model.build(csim=False, synth=True)` and parse
`<output_dir>/myproject_prj/solution1/syn/report/myproject_csynth.rpt` for the
resource/latency table. This project's abstraction (`HardwareBackend` /
`HardwareReport`) is already shaped to carry those numbers once a real synthesis step
produces them — only the `HLSBackend.analyze()` body needs extending, nothing else in
the pipeline (`runner.py`, `reporting.py`, Pareto analysis) needs to change, since they
already treat hardware fields as nullable.

## Provenance fields on every hardware result (Phase 3)

Each `HardwareReport` now carries, besides `measurement_tier`: `source` (backend class), `tool`, `tool_version` and `configuration` (the settings the numbers depend on, e.g. reuse factor, precision, heuristic constants,
whether the heuristic was validated against synthesis — it was not). Example for the heuristic backend: `measurement_tier: estimated`, `source: SoftwareEstimateBackend`, `tool: analytical heuristic (hep_quant)`.
The 21 frozen Phase-2 records predate these fields; `reports/tables/hardware_summary.csv` fills them with N/A where the record cannot say.

## C-simulation precision (audited)

`hep_quant.hardware.hls_csim.csim_predict` documents and exposes the fixed-point mode: default is `ap_fixed<W,I>` for input, weights, biases, accumulator, layer outputs and result with the C++ defaults (truncation `AP_TRN`, wrap-around `AP_WRAP`);
optional `rounding`/`saturation` and a separate wide accumulator. `reports/numerical_consistency.md` shows the mode changes results far more than the width does at 8–12 bits. C-simulation produces **no** latency, II or resources; those need HLS synthesis (not performed).

