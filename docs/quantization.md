# Quantization Methodology

## What's actually implemented

Four precisions are compared: **FP32** (unmodified), **INT8**, **INT6**, **INT4**.

PyTorch's native `torch.ao.quantization` module provides real bit-packed integer
kernels, but only for INT8 (via the fbgemm/qnnpack backends) — there is no built-in
support for INT6 or INT4. Rather than special-case INT8 with real kernels and
INT6/INT4 with something else (which would make the three-way comparison apples-to-
oranges), this project implements **uniform affine fake-quantization** for all three
quantized precisions, applied identically:

```
quantize:    q = clamp(round(x / scale + zero_point), 0, 2^bits - 1)
dequantize:  x_hat = (q - zero_point) * scale
```

Both the forward weight tensors and activation outputs are passed through this
quantize-then-dequantize op, so the numerical effect of N-bit quantization (rounding
error, clipping at the calibrated range) is faithfully simulated, but the result is
still stored and computed in ordinary float32 — there is no bit-packed storage and no
integer-only compute kernel. This is standard practice in the quantization research
literature (it's also how PyTorch's own `FakeQuantize`/QAT machinery works
internally), but it means:

- **Model size for INT8/INT6/INT4 is an analytical estimate**, not a measurement of an
  actually-packed artifact (see `hep_quant/quantization/analysis.py`). Biases are kept
  at FP32 in this estimate, consistent with the implementation (only `nn.Linear`
  weights and activation outputs are fake-quantized; biases are not).
- **Inference latency is measured on the fake-quantized float32 model**, not on a true
  integer kernel, so it will NOT show the speedup real INT8 hardware execution would
  provide. Latency numbers in this project characterize accuracy-vs-precision
  trade-offs, not deployment-speed trade-offs — the deployment-speed question is what
  Gate 6 (hardware-aware / hls4ml analysis) is for.

## Post-Training Quantization (PTQ)

1. Train an FP32 model to convergence.
2. Calibrate: run a subset of training data through the trained model, recording
   observed min/max at each activation (`hep_quant/quantization/calibration.py`).
3. Quantize: fake-quantize each `nn.Linear` layer's weights (range = that layer's own
   min/max), and insert a fake-quant op after each activation using the calibrated
   range (`hep_quant/quantization/ptq.py::quantize_model_ptq`).
4. Evaluate the quantized model on the held-out test split — no further training.

## Quantization-Aware Training (QAT)

1. Train an FP32 baseline model (same as PTQ step 1 — QAT fine-tunes from an FP32
   checkpoint, it does not train a quantized model from scratch).
2. Replace `nn.Linear` layers with `QATLinear` (weight range recomputed from the live
   weight tensor every forward pass) and activation layers with `QATActivation` (range
   tracked via an exponential moving average observer, updated only in training mode).
3. Both use a straight-through estimator (STE) for the backward pass through
   `round()`, since it has zero gradient almost everywhere otherwise
   (`hep_quant/quantization/qat.py`).
4. Fine-tune for a reduced number of epochs at a reduced learning rate (a smaller
   nudge from the FP32 solution, not a from-scratch training run).

## Supported precision x method matrix

|       | FP32 | INT8 | INT6 | INT4 |
|-------|:----:|:----:|:----:|:----:|
| PTQ   |  —   |  ✓   |  ✓   |  ✓   |
| QAT   |  —   |  ✓   |  ✓   |  ✓   |

FP32 has no "quantized" row — it's the unmodified baseline both PTQ and QAT start
from.

## Honesty notes

- Nothing in this project claims INT8 PTQ results came from PyTorch's native quantized
  kernels. They did not — see above.
- "Model size" fields marked `is_estimated: true` in experiment records are exactly
  that: computed from parameter counts and target bit-width, not measured from a
  packed file on disk.
