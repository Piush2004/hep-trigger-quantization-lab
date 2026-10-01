"""Reproducibility metadata captured with every experiment result."""
from __future__ import annotations

from dataclasses import dataclass

from hep_quant.utils.system import SystemInfo, get_system_info


@dataclass
class ReproducibilityInfo:
    seed: int
    system: SystemInfo

    def to_dict(self) -> dict:
        return {"seed": self.seed, "system": self.system.to_dict()}


def capture_reproducibility_info(seed: int) -> ReproducibilityInfo:
    return ReproducibilityInfo(seed=seed, system=get_system_info())
