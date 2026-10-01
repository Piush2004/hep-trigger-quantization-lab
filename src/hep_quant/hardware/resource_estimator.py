"""Hardware-aware analysis abstraction (Gate 6).

Defines `HardwareBackend` with two implementations:
  - `HLSBackend`: attempts a real hls4ml conversion + C-simulation. FPGA
    *synthesis* (LUT/DSP/BRAM/FF numbers) additionally requires a Vivado/
    Vitis HLS installation, which this project does not assume is present;
    when synthesis isn't available, resource fields are left `None` and
    `resource_source` says so explicitly.
  - `UnavailableBackend`: always returns a report with every hardware field
    `None` and a documented reason. Used whenever hls4ml itself cannot be
    imported/used (e.g. incompatible install, as commonly happens on Windows
    without WSL/Linux, or when no C++ toolchain is present for C-simulation).

The project must work end-to-end with `UnavailableBackend` — hardware
numbers are a bonus, not a requirement for the rest of the pipeline (PTQ/QAT,
experiment tracking, Pareto analysis all operate on software-measured
metrics + whatever hardware fields are available, using None/"N/A" otherwise).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class HardwareReport:
    backend: str
    status: str  # "unavailable" | "converted" | "csim" | "synthesized"
    resource_source: str  # human-readable statement of what was/wasn't measured
    latency_estimate_ns: float | None = None
    initiation_interval: int | None = None
    lut: int | None = None
    dsp: int | None = None
    bram: int | None = None
    ff: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "status": self.status,
            "resource_source": self.resource_source,
            "latency_estimate_ns": self.latency_estimate_ns,
            "initiation_interval": self.initiation_interval,
            "lut": self.lut,
            "dsp": self.dsp,
            "bram": self.bram,
            "ff": self.ff,
            "notes": self.notes,
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
            notes=[self.reason],
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
                notes=[self._import_error or "unknown import error"],
            )

        notes = []
        bits_map = {"fp32": 32, "int8": 8, "int6": 6, "int4": 4}
        bits = bits_map.get(precision, 16)

        try:
            import hls4ml

            precision_str = f"ap_fixed<{bits + 6},{6}>" if precision != "fp32" else "ap_fixed<32,16>"
            config = {
                "Model": {
                    "Precision": precision_str,
                    "ReuseFactor": hardware_cfg.get("reuse_factor", 1),
                    "Strategy": hardware_cfg.get("strategy", "Latency"),
                }
            }
            hls_model = hls4ml.converters.convert_from_pytorch_model(
                model,
                input_shape=[None, model.input_dim],
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
                    notes=notes,
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
                notes=notes,
            )
        except Exception as exc:  # hls4ml's conversion surface is broad; fail soft, don't crash the pipeline
            notes.append(f"hls4ml conversion/compile failed: {exc}")
            return HardwareReport(
                backend=self.name,
                status="unavailable",
                resource_source=f"hls4ml conversion failed: {exc}",
                notes=notes,
            )


def get_hardware_backend(hardware_cfg: dict) -> HardwareBackend:
    requested = hardware_cfg.get("backend", "unavailable")
    if requested == "unavailable":
        return UnavailableBackend(reason="backend explicitly set to 'unavailable' in config")

    if requested == "hls4ml":
        backend = HLSBackend()
        if backend.is_available():
            return backend
        return UnavailableBackend(reason=f"hls4ml requested but not usable: {backend._import_error}")

    return UnavailableBackend(reason=f"unknown backend '{requested}'")
