#!/usr/bin/env python
"""Freeze and verify the Phase-2 real-data results (3 seeds x 7 configurations = 21 experiments).

`freeze`  copies each experiment's record, checkpoint, training history and metrics into
          experiments/results/frozen_phase2/ with SHA-256 hashes in MANIFEST.json, runs the
          per-experiment integrity checks, and marks the copies read-only. Refuses to
          overwrite an existing snapshot.
`verify`  recomputes every hash and reports any change.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEEDS = [42, 123, 2026]
TAGS = ["FP32", "INT8_PTQ", "INT6_PTQ", "INT4_PTQ", "INT8_QAT", "INT6_QAT", "INT4_QAT"]
FROZEN = Path("experiments/results/frozen_phase2")
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def check_experiment(exp_id: str, seed: int) -> dict:
    """Return {check_name: bool|str}; a failing/absent item is False, never silently ignored."""
    from hep_quant.analysis.phase2 import load_checkpoint_model

    res = {}
    rec_path = Path(f"experiments/results/{exp_id}.json")
    res["record_exists"] = rec_path.exists()
    if not res["record_exists"]:
        return res
    rec = json.loads(rec_path.read_text())
    ckpt = Path(f"experiments/checkpoints/{exp_id}/best_model.pt")
    hist = Path(f"experiments/checkpoints/{exp_id}/training_history.json")
    res["status_completed"] = rec.get("status") == "completed" and rec.get("error") is None
    res["training_history_exists"] = hist.exists()
    res["checkpoint_exists"] = ckpt.exists()
    try:
        info = rec["model_info"]
        load_checkpoint_model(ckpt, info["method"], info["precision"])
        res["checkpoint_loads"] = True
    except Exception as exc:  # recorded, not swallowed
        res["checkpoint_loads"] = f"FAILED: {exc}"
    test = rec.get("metrics", {}).get("test", {})
    res["metrics_present"] = all(k in test and test[k] is not None for k in METRIC_KEYS)
    res["metrics_finite"] = res["metrics_present"] and all(math.isfinite(test[k]) for k in METRIC_KEYS)
    res["config_present"] = bool(rec.get("config"))
    res["seed_recorded_and_matches"] = rec.get("reproducibility", {}).get("seed") == seed == rec.get("config", {}).get("seed")
    res["git_commit_recorded"] = bool(rec.get("reproducibility", {}).get("system", {}).get("git_commit"))
    res["timestamp_present"] = bool(rec.get("timestamp"))
    res["latency_present"] = rec.get("metrics", {}).get("latency", {}).get("mean_ms_per_sample") is not None
    res["model_size_present"] = rec.get("metrics", {}).get("model_size", {}).get("estimated_size_mb") is not None
    return res


def freeze() -> int:
    if FROZEN.exists():
        print(f"{FROZEN} already exists; refusing to overwrite a frozen snapshot", file=sys.stderr)
        return 2
    import numpy
    import sklearn
    import torch

    from hep_quant.data.hls4ml_jets import load_hls4ml_jets

    FROZEN.mkdir(parents=True)
    entries, checks = [], {}
    for seed in SEEDS:
        for tag in TAGS:
            exp_id = f"EXP_S{seed}_JET_{tag}"
            checks[exp_id] = check_experiment(exp_id, seed)
            if not checks[exp_id].get("record_exists"):
                continue
            files = {
                f"{exp_id}.json": Path(f"experiments/results/{exp_id}.json"),
                f"{exp_id}__best_model.pt": Path(f"experiments/checkpoints/{exp_id}/best_model.pt"),
                f"{exp_id}__training_history.json": Path(f"experiments/checkpoints/{exp_id}/training_history.json"),
                f"{exp_id}__metrics.json": Path(f"experiments/checkpoints/{exp_id}/metrics.json"),
            }
            hashes = {}
            for name, src in files.items():
                if src.exists():
                    shutil.copy2(src, FROZEN / name)
                    hashes[name] = sha256_file(FROZEN / name)
            rec = json.loads(Path(f"experiments/results/{exp_id}.json").read_text())
            hist = json.loads(Path(f"experiments/checkpoints/{exp_id}/training_history.json").read_text())
            entries.append({
                "experiment_id": exp_id, "seed": seed, "timestamp": rec["timestamp"],
                "architecture": rec["model_info"]["architecture"], "hidden_dims": rec["model_info"]["hidden_dims"],
                "method": rec["model_info"]["method"], "precision": rec["model_info"]["precision"],
                "dataset": rec["config"]["dataset"]["name"], "subset_size": rec["config"]["dataset"]["subset_size"],
                "training_config": rec["config"]["training"], "quantization_config": rec["config"]["quantization"],
                "git_commit_recorded": rec["reproducibility"]["system"]["git_commit"],
                "system_recorded": {k: rec["reproducibility"]["system"][k] for k in ("python_version", "platform", "torch_version", "cuda_available", "cuda_version", "gpu_name")},
                "epochs_run": len(hist["epochs"]), "train_seconds_sum_of_epochs": round(sum(hist["epoch_duration_s"]), 1),
                "file_sha256": hashes,
            })
    dataset_hashes = {}
    full = load_hls4ml_jets(subset_size=None, seed=42)
    dataset_hashes["full_X_float32_sha256"] = hashlib.sha256(full.X.tobytes()).hexdigest()
    dataset_hashes["full_n_samples"] = int(len(full.y))
    for seed in SEEDS:
        sub = load_hls4ml_jets(subset_size=100000, seed=seed)
        dataset_hashes[f"subset_100000_seed{seed}_X_sha256"] = hashlib.sha256(sub.X.tobytes()).hexdigest()
    status = git("status", "--porcelain")
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_expected": len(SEEDS) * len(TAGS), "n_present": len(entries),
        "git_head_at_freeze": git("rev-parse", "HEAD"),
        "git_working_tree_dirty_at_freeze": bool(status),
        "software_at_freeze_time": {
            "note": "numpy/scikit-learn versions are NOT stored in the experiment records; captured here at freeze time from the same virtualenv",
            "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        },
        "dataset": {"openml_name": "hls4ml_lhc_jets_hlf", "openml_version": 1, **dataset_hashes},
        "integrity_checks": checks, "experiments": entries,
    }
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for p in FROZEN.iterdir():
        os.chmod(p, stat.S_IREAD)
    failed = {k: [c for c, v in r.items() if v is not True] for k, r in checks.items() if any(v is not True for v in r.values())}
    print(json.dumps({"present": len(entries), "expected": manifest["n_expected"], "checks_failing": failed}, indent=2))
    return 0 if not failed and len(entries) == manifest["n_expected"] else 1


def verify() -> int:
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text())
    bad = []
    for e in manifest["experiments"]:
        for name, h in e["file_sha256"].items():
            if sha256_file(FROZEN / name) != h:
                bad.append(name)
    print(json.dumps({"files_checked": sum(len(e["file_sha256"]) for e in manifest["experiments"]), "changed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "verify"])
    sys.exit(freeze() if ap.parse_args().mode == "freeze" else verify())
