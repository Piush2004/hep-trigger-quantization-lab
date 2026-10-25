"""hls4ml C-simulation accuracy under bit-accurate fixed-point arithmetic.

This is the one hardware-oriented quantity obtainable without Vivado: hls4ml
compiles the converted model to C++ with `ap_fixed<W,I>` types and
`hls_model.predict` executes it bit-accurately on the host CPU. Reported with
`measurement_tier="simulated"`. It does NOT yield latency, LUT/DSP/BRAM/FF
(those need HLS synthesis, not performed), and it is a different quantization
scheme from this project's calibrated uniform-affine fake-quantization (global
fixed-point format, no per-layer calibration), so the two are not numerically
interchangeable.
"""
from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path

import numpy as np
import torch.nn as nn


def csim_predict(
    model: nn.Module,
    X: np.ndarray,
    total_bits: int,
    int_bits: int = 6,
    reuse_factor: int = 1,
    output_dir: str | Path = "experiments/hls4ml_csim",
    rounding: str | None = None,
    saturation: str | None = None,
    accum_bits: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return logits from the compiled ap_fixed<total_bits,int_bits[,rounding,saturation]> C simulation of `model`.

    `rounding`/`saturation` (e.g. "AP_RND", "AP_SAT") are appended to every type; left as None the C++
    ap_fixed defaults (truncation AP_TRN, wrap-around AP_WRAP) apply. `accum_bits=(W,I)` gives every
    Dense layer a separate, wider accumulator type; None keeps accumulator = the default type.
    """
    import hls4ml

    if shutil.which("g++") is None:
        raise RuntimeError("g++ not found: C-simulation cannot be compiled")
    cpu_model = copy.deepcopy(model).cpu().eval()
    extra = "".join(f",{m}" for m in (rounding, saturation) if m) if (rounding or saturation) else ""
    if saturation and not rounding:
        extra = f",AP_TRN,{saturation}"  # ap_fixed template needs the rounding slot before the overflow slot
    config = hls4ml.utils.config_from_pytorch_model(
        cpu_model,
        input_shape=cpu_model.hls4ml_input_shape,
        default_precision=f"ap_fixed<{total_bits},{int_bits}{extra}>",
        default_reuse_factor=reuse_factor,
    )
    if accum_bits:  # model-level override: one wider accumulator type for every layer
        config["Model"]["Precision"] = {
            "default": f"ap_fixed<{total_bits},{int_bits}{extra}>",
            "accum": f"ap_fixed<{accum_bits[0]},{accum_bits[1]}{extra}>",
        }
    tag = f"w{total_bits}_i{int_bits}_{rounding or 'trn'}_{saturation or 'wrap'}_{'acc' + str(accum_bits[0]) if accum_bits else 'noacc'}"
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        cpu_model, hls_config=config, output_dir=str(Path(output_dir) / tag), backend="Vivado"
    )
    hls_model.compile()
    return np.asarray(hls_model.predict(np.ascontiguousarray(X, dtype=np.float64)))


def csim_accuracy_sweep(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    widths: list[int],
    int_bits: int = 6,
    reference_pred: np.ndarray | None = None,
) -> list[dict]:
    """Accuracy (and agreement with the floating-point model, if given) per fixed-point width."""
    rows = []
    for w in widths:
        t0 = time.perf_counter()
        logits = csim_predict(model, X, total_bits=w, int_bits=int_bits)
        pred = logits.argmax(axis=1)
        row = {
            "ap_fixed": f"<{w},{int_bits}>",
            "total_bits": w,
            "int_bits": int_bits,
            "frac_bits": w - int_bits,
            "n_eval": int(len(y)),
            "accuracy": float((pred == y).mean()),
            "finite_outputs": bool(np.isfinite(logits).all()),
            "wall_seconds_compile_and_predict": round(time.perf_counter() - t0, 1),
            "measurement_tier": "simulated",
        }
        if reference_pred is not None:
            row["agreement_with_fp32_predictions"] = float((pred == reference_pred).mean())
        rows.append(row)
    return rows
