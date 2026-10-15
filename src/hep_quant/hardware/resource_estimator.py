"""Hardware-aware analysis abstraction (Gate 6).

Defines `HardwareBackend` with three implementations, each tagging its report
with an explicit `measurement_tier` so a consumer never has to guess how
trustworthy a number is:

  - `HLSBackend` (`measurement_tier="simulated"` when it succeeds): a real
    hls4ml conversion + C-simulation. FPGA *synthesis* (LUT/DSP/BRAM/FF
    numbers, `measurement_tier="measured"`) additionally requires a Vivado/
    Vitis HLS installation, which this project does not assume is present;
    csim validates that the model is HLS-compatible and numerically correct,
    it does NOT produce resource numbers, so those fields stay `None` even on
    a successful csim run.
  - `SoftwareEstimateBackend` (`measurement_tier="estimated"`): a pure
    analytical heuristic (parameter count x bit-width x reuse factor) that
    needs no external tool and is always available. Used as the fallback when
    hls4ml is requested but not usable, so a resource ballpark is still
    reported instead of silently giving up — but every field it produces is
    explicitly labeled an estimate, never presented as measured or simulated.
  - `UnavailableBackend` (`measurement_tier="unavailable"`): every hardware
    field `None`, used when the config explicitly disables hardware analysis.

The project must work end-to-end with `UnavailableBackend` — hardware
numbers are a bonus, not a requirement for the rest of the pipeline (PTQ/QAT,
experiment tracking, Pareto analysis all operate on software-measured
metrics + whatever hardware fields are available, using None/"N/A" otherwise).
"""
from __future__ import annotations

import copy
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

def _hep_quant_version() -> str:
    import hep_quant

    return hep_quant.__version__


_BITS_MAP = {"fp32": 32, "int8": 8, "int6": 6, "int4": 4}


@dataclass
class HardwareReport:
    backend: str
    status: str  # "unavailable" | "converted" | "csim" | "synthesized" | "estimated"
    resource_source: str  # human-readable statement of what was/wasn't measured
    measurement_tier: str = "unavailable"  # "measured" | "simulated" | "estimated" | "unavailable"
    latency_estimate_ns: float | None = None
    initiation_interval: int | None = None
    lut: int | None = None
    dsp: int | None = None
    bram: int | None = None
    ff: int | None = None
    notes: list[str] = field(default_factory=list)
    source: str = "none"  # which backend class produced the numbers
    tool: str | None = None  # underlying tool (None for pure arithmetic / nothing)
    tool_version: str | None = None
    configuration: dict = field(default_factory=dict)  # settings the numbers depend on

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "status": self.status,
            "resource_source": self.resource_source,
            "measurement_tier": self.measurement_tier,
            "latency_estimate_ns": self.latency_estimate_ns,
            "initiation_interval": self.initiation_interval,
            "lut": self.lut,
            "dsp": self.dsp,
            "bram": self.bram,
            "ff": self.ff,
            "notes": self.notes,
            "source": self.source,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "configuration": self.configuration,
        }


class HardwareBackend:
    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def analyze(self, model: nn.Module, precision: str, hardware_cfg: dict) -> HardwareReport:
        raise NotImplementedError


class UnavailableBackend(HardwareBackend):
    name = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def analyze(self, model: nn.Module, precision: str, hardware_cfg: dict) -> HardwareReport:
        return HardwareReport(
            backend=self.name,
            status="unavailable",
            resource_source=f"hls4ml unavailable: {self.reason}",
            measurement_tier="unavailable",
            notes=[self.reason],
            source="UnavailableBackend",
        )


class HLSBackend(HardwareBackend):
    """Best-effort hls4ml integration: conversion + C-simulation only (no Vivado synthesis)."""

    name = "hls4ml"

    def __init__(self):
        self._hls4ml = None
        self._import_error: str | None = None
        try:
            import hls4ml  # noqa: F401

            self._hls4ml = hls4ml
        except ImportError as exc:
            self._import_error = str(exc)

    def is_available(self) -> bool:
        return self._hls4ml is not None

    def analyze(self, model: nn.Module, precision: str, hardware_cfg: dict) -> HardwareReport:
        if not self.is_available():
            return HardwareReport(
                backend=self.name,
                status="unavailable",
                resource_source=f"hls4ml import failed: {self._import_error}",
                measurement_tier="unavailable",
                notes=[self._import_error or "unknown import error"],
            )

        notes = []
        bits = _BITS_MAP.get(precision, 16)

        try:
            import hls4ml

            # hls4ml introspects weight tensors via NumPy, which requires CPU
            # tensors; the model may have been trained/evaluated on CUDA, so
            # convert a CPU copy rather than require (or mutate) the caller's
            # model device.
            cpu_model = copy.deepcopy(model).cpu()

            precision_str = f"ap_fixed<{bits + 6},{6}>" if precision != "fp32" else "ap_fixed<32,16>"
            config = hls4ml.utils.config_from_pytorch_model(
                cpu_model,
                input_shape=cpu_model.hls4ml_input_shape,
                default_precision=precision_str,
                default_reuse_factor=hardware_cfg.get("reuse_factor", 1),
            )
            hls_model = hls4ml.converters.convert_from_pytorch_model(
                cpu_model,
                hls_config=config,
                output_dir="experiments/hls4ml_prj",
                backend="Vivado",
            )
            notes.append("Model converted to an HLS project via hls4ml.")

            if shutil.which("g++") is None:
                notes.append("No g++ found: skipping C-simulation (compile/csim step).")
                return HardwareReport(
                    backend=self.name,
                    status="converted",
                    resource_source="hls4ml conversion only (no C++ toolchain for csim; no Vivado for synthesis)",
                    measurement_tier="simulated",
                    notes=notes,
                    source="HLSBackend", tool="hls4ml", tool_version=getattr(hls4ml, "__version__", None),
                    configuration={"backend": "Vivado", "io_type": "io_parallel", "default_precision": precision_str, "reuse_factor": hardware_cfg.get("reuse_factor", 1), "stage": "conversion only"},
                )

            hls_model.compile()
            notes.append("C-simulation compiled successfully.")
            notes.append(
                "LUT/DSP/BRAM/FF resource counts require Vivado/Vitis HLS synthesis, "
                "which is not installed in this environment — reporting conversion/csim status only."
            )
            return HardwareReport(
                backend=self.name,
                status="csim",
                resource_source="hls4ml C-simulation (no Vivado synthesis available for resource/latency numbers)",
                measurement_tier="simulated",
                notes=notes,
                source="HLSBackend", tool="hls4ml", tool_version=getattr(hls4ml, "__version__", None),
                configuration={"backend": "Vivado", "io_type": "io_parallel", "default_precision": precision_str, "reuse_factor": hardware_cfg.get("reuse_factor", 1), "stage": "C-simulation compile only (no synthesis)"},
            )
        except Exception as exc:  # hls4ml's conversion surface is broad; fail soft, don't crash the pipeline
            notes.append(f"hls4ml conversion/compile failed: {exc}")
            return HardwareReport(
                backend=self.name,
                status="unavailable",
                resource_source=f"hls4ml conversion failed: {exc}",
                measurement_tier="unavailable",
                notes=notes,
            )


class SoftwareEstimateBackend(HardwareBackend):
    """Pure analytical resource heuristic — no external tool, always available.

    Used when hls4ml is requested but not usable, so a ballpark resource
    number is still reported instead of silently giving up. Every field is
    explicitly `measurement_tier="estimated"`: this is NOT hls4ml, NOT a
    simulation, and NOT a Vivado synthesis result — it is arithmetic over the
    model's parameter count, target bit-width, and reuse factor, using the
    same rough rule of thumb hls4ml's own documentation uses to explain its
    resource/reuse-factor trade-off (one multiply-accumulate per weight,
    `reuse_factor` of them sharing one DSP slot, fully unrolled i.e.
    `reuse_factor=1` by default):

        dsp  ~= ceil(n_weights / reuse_factor)         (one DSP per MAC "slot")
        lut  ~= dsp * bits * LUT_PER_BIT_PER_DSP        (control/routing logic
                                                          scales with DSP count and bit-width)
        ff   ~= n_params * bits                         (one register-worth of
                                                          storage per parameter, per bit)
        bram ~= 0 for models this small (weights fit in distributed LUTRAM/FF,
                not block RAM) — a documented simplification, not a claim that
                larger models would also need none.

    `LUT_PER_BIT_PER_DSP` is a fixed, documented constant (not fit to any
    particular device), chosen only to produce a self-consistent, order-of-
    magnitude ballpark across precisions — it is not calibrated against any
    real Vivado synthesis run and must not be read as device-specific.
    """

    name = "software_estimate"
    LUT_PER_BIT_PER_DSP = 1.5

    def is_available(self) -> bool:
        return True

    def analyze(self, model: nn.Module, precision: str, hardware_cfg: dict) -> HardwareReport:
        from hep_quant.quantization.analysis import count_parameters

        bits = _BITS_MAP.get(precision, 16)
        reuse_factor = max(1, int(hardware_cfg.get("reuse_factor", 1)))
        total_params, weight_params, _bias_params = count_parameters(model)

        dsp_estimate = math.ceil(weight_params / reuse_factor)
        lut_estimate = math.ceil(dsp_estimate * bits * self.LUT_PER_BIT_PER_DSP)
        ff_estimate = total_params * bits
        bram_estimate = 0

        notes = [
            f"Heuristic: dsp=ceil(weight_params({weight_params})/reuse_factor({reuse_factor})), "
            f"lut=ceil(dsp*bits({bits})*{self.LUT_PER_BIT_PER_DSP}), ff=total_params*bits, bram=0 "
            "(assumes a model small enough to avoid block RAM).",
            "This is an analytical estimate, NOT hls4ml output and NOT a Vivado synthesis "
            "result — it exists so a resource ballpark is available even when hls4ml itself "
            "cannot be used, and must not be read as device-specific or validated against "
            "real synthesis.",
        ]
        return HardwareReport(
            backend=self.name,
            status="estimated",
            resource_source="software heuristic estimate (parameter-count arithmetic, not hls4ml or Vivado)",
            measurement_tier="estimated",
            lut=lut_estimate,
            dsp=dsp_estimate,
            bram=bram_estimate,
            ff=ff_estimate,
            notes=notes,
            source="SoftwareEstimateBackend", tool="analytical heuristic (hep_quant)", tool_version=_hep_quant_version(),
            configuration={"precision": precision, "bits": bits, "reuse_factor": reuse_factor, "lut_per_bit_per_dsp": self.LUT_PER_BIT_PER_DSP,
                           "formula": "dsp=ceil(weights/reuse); lut=ceil(dsp*bits*c); ff=params*bits; bram=0", "validated_against_synthesis": False},
        )


class SurrogateEstimateBackend(HardwareBackend):
    """ML-surrogate resource/latency estimate, trained on fastmachinelearning/wa-hls4ml.

    Unlike `SoftwareEstimateBackend`'s hand-derived arithmetic heuristic, this loads a trained
    sklearn regressor (see hep_quant.hardware.surrogate) and reports its own held-out error
    (measured on wa-hls4ml's own test split, plus separately on its Jet-family exemplar slice)
    alongside every prediction, so a consumer can judge how much to trust the number -- never
    presented as validated against THIS project's own real synthesis, only against wa-hls4ml's.
    """

    name = "surrogate"

    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else Path("experiments/surrogate_models/v1")
        self._bundle = None
        self._load_error: str | None = None
        try:
            from hep_quant.hardware.surrogate import load_surrogate

            self._bundle = load_surrogate(self.model_dir)
        except Exception as exc:
            self._load_error = str(exc)

    def is_available(self) -> bool:
        return self._bundle is not None

    def analyze(self, model: nn.Module, precision: str, hardware_cfg: dict) -> HardwareReport:
        if not self.is_available():
            return HardwareReport(
                backend=self.name,
                status="unavailable",
                resource_source=f"surrogate model unavailable: {self._load_error}",
                measurement_tier="unavailable",
                notes=[self._load_error or "surrogate model not found"],
                source="SurrogateEstimateBackend",
            )

        from hep_quant.hardware.surrogate import features_from_pytorch_model, predict_resources

        bits = _BITS_MAP.get(precision, 16)
        weight_bits = {i: bits for i in range(sum(1 for m in model.net if isinstance(m, nn.Linear)))}
        feats = features_from_pytorch_model(model, weight_bits, hardware_cfg)
        pred = predict_resources(feats, self._bundle)
        held_out = self._bundle.get("held_out", {})
        jet_held_out = self._bundle.get("jet_exemplar_held_out", {})

        notes = [
            f"Held-out (wa-hls4ml test split) {t}: R2={m['r2']:.3f}, MAE={m['mae']:.1f} (n={m['n']})"
            for t, m in held_out.items()
        ]
        notes.append("Training data license: CC-BY-NC-4.0 (non-commercial).")
        notes.append("NOT validated against this project's own real synthesis -- only against "
                      "wa-hls4ml's held-out split and its Jet-family exemplar slice.")

        return HardwareReport(
            backend=self.name,
            status="surrogate",
            resource_source="ML surrogate (sklearn) trained on fastmachinelearning/wa-hls4ml "
                             "(resource/latency fully-connected splits); not hls4ml or Vivado output",
            measurement_tier="surrogate",
            lut=int(round(pred["lut"])), dsp=int(round(pred["dsp"])),
            bram=int(round(pred["bram"])), ff=int(round(pred["ff"])),
            latency_estimate_ns=pred["latency_ns"], initiation_interval=int(round(pred["ii"])),
            notes=notes,
            source="SurrogateEstimateBackend",
            tool="sklearn surrogate trained on wa-hls4ml",
            tool_version=self._bundle.get("sklearn_version"),
            configuration={
                "feature_spec_version": self._bundle.get("feature_spec_version"),
                "held_out_metrics": held_out,
                "jet_exemplar_held_out": jet_held_out,
                "trained_on": self._bundle.get("trained_on"),
                "validated_against_this_projects_own_synthesis": False,
            },
        )


def get_hardware_backend(hardware_cfg: dict) -> HardwareBackend:
    requested = hardware_cfg.get("backend", "unavailable")
    if requested == "unavailable":
        return UnavailableBackend(reason="backend explicitly set to 'unavailable' in config")

    if requested == "software_estimate":
        return SoftwareEstimateBackend()

    if requested == "hls4ml":
        backend = HLSBackend()
        if backend.is_available():
            return backend
        # Per project policy: if hls4ml isn't usable, fall back to an honest
        # heuristic estimate rather than giving up outright — never fabricate
        # hls4ml/Vivado-attributed numbers, but a labeled estimate is fine.
        return SoftwareEstimateBackend()

    if requested == "surrogate":
        backend = SurrogateEstimateBackend()
        if backend.is_available():
            return backend
        # Same honest-fallback policy as the hls4ml branch above: no trained
        # surrogate on disk -> fall back to the heuristic, never silently to
        # UnavailableBackend and never fabricate surrogate-attributed numbers.
        return SoftwareEstimateBackend()

    return UnavailableBackend(reason=f"unknown backend '{requested}'")
