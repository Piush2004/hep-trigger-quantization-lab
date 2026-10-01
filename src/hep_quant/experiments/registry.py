"""Experiment registry: every run is saved as a self-contained, traceable JSON record."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hep_quant.experiments.reproducibility import capture_reproducibility_info


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    config: dict[str, Any]
    reproducibility: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    model_info: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "reproducibility": self.reproducibility,
            "metrics": self.metrics,
            "model_info": self.model_info,
            "hardware": self.hardware,
            "status": self.status,
            "error": self.error,
        }


def create_experiment_record(
    experiment_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    model_info: dict[str, Any],
    hardware: dict[str, Any] | None = None,
    status: str = "completed",
    error: str | None = None,
) -> ExperimentRecord:
    seed = config.get("seed", 42)
    repro = capture_reproducibility_info(seed)
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        reproducibility=repro.to_dict(),
        metrics=metrics,
        model_info=model_info,
        hardware=hardware or {},
        status=status,
        error=error,
    )


class ExperimentRegistry:
    def __init__(self, results_dir: str | Path = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ExperimentRecord) -> Path:
        path = self.results_dir / f"{record.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
        return path

    def load(self, experiment_id: str) -> dict:
        path = self.results_dir / f"{experiment_id}.json"
        with open(path) as f:
            return json.load(f)

    def load_all(self) -> list[dict]:
        records = []
        for path in sorted(self.results_dir.glob("*.json")):
            with open(path) as f:
                records.append(json.load(f))
        return records
