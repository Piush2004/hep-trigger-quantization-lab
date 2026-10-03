"""Re-run a recorded experiment from its stored configuration, in isolation, and compare with the stored reference.

Reference lookup order: experiments/results/frozen_phase2/<id>.json (+ its frozen checkpoint), then
experiments/results/<id>.json (+ experiments/checkpoints/<id>/best_model.pt). The re-run writes ONLY to a
scratch directory, so neither frozen nor live results are modified.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from hep_quant.analysis.phase2 import load_checkpoint_model, predict_proba, test_split_for_record
from hep_quant.experiments.runner import run_experiment

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
FROZEN_DIR = Path("experiments/results/frozen_phase2")


def _weight_diff(path_a: Path, path_b: Path) -> dict:
    a = torch.load(path_a, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(path_b, map_location="cpu", weights_only=False)["model_state_dict"]
    if a.keys() != b.keys():
        return {"same_keys": False}
    diffs = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a if a[k].dtype.is_floating_point}
    return {"same_keys": True, "max_abs_weight_diff": max(diffs.values()), "worst_tensor": max(diffs, key=diffs.get)}


def _reference(experiment_id: str) -> tuple[dict, Path]:
    if (FROZEN_DIR / f"{experiment_id}.json").exists():
        return json.loads((FROZEN_DIR / f"{experiment_id}.json").read_text()), FROZEN_DIR / f"{experiment_id}__best_model.pt"
    live = Path(f"experiments/results/{experiment_id}.json")
    if not live.exists():
        raise FileNotFoundError(f"no recorded experiment {experiment_id} (looked in {FROZEN_DIR} and experiments/results)")
    return json.loads(live.read_text()), Path(f"experiments/checkpoints/{experiment_id}/best_model.pt")


def reproduce(experiment_id: str, repeats: int = 2, metric_tol: float = 0.005, out_stem: str = "reports/reproducibility_real") -> dict:
    original, orig_ckpt = _reference(experiment_id)
    cfg = original["config"]
    scratch = Path(tempfile.mkdtemp(prefix="repro_"))
    cfg_path = scratch / "stored_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    runs = []
    for i in range(repeats):
        res, ckpt, fig = scratch / f"r{i}/results", scratch / f"r{i}/ckpt", scratch / f"r{i}/fig"
        rec = run_experiment(str(cfg_path), results_dir=str(res), checkpoint_dir=str(ckpt), figures_dir=str(fig))
        runs.append((rec, ckpt / cfg["experiment"]["id"] / "best_model.pt"))

    report = {"experiment": experiment_id, "metric_tolerance_abs": metric_tol, "repeats": []}
    ok = True
    info = original["model_info"]
    X_test = test_split_for_record(original)[0] if cfg["dataset"].get("source") == "hls4ml_jets" else None
    for i, (rec, ckpt_path) in enumerate(runs):
        o, n = original["metrics"]["test"], rec["metrics"]["test"]
        metric_diffs = {m: abs(o[m] - n[m]) for m in METRICS}
        meta_same = {
            "config": rec["config"] == original["config"],
            "seed": rec["reproducibility"]["seed"] == original["reproducibility"]["seed"],
            "parameters": rec["model_info"]["n_parameters"] == original["model_info"]["n_parameters"],
            "model_size": rec["metrics"]["model_size"] == original["metrics"]["model_size"],
            "hardware_estimate_fields": all(rec["hardware"].get(k) == original["hardware"].get(k) for k in ("lut", "dsp", "bram", "ff", "measurement_tier")),
        }
        entry = {
            "repeat": i + 1,
            "metric_abs_diff": metric_diffs,
            "metadata_identical": meta_same,
            "weights": _weight_diff(orig_ckpt, ckpt_path),
            "original_metrics": {m: o[m] for m in METRICS},
            "rerun_metrics": {m: n[m] for m in METRICS},
        }
        if X_test is not None:
            p_orig = predict_proba(load_checkpoint_model(orig_ckpt, info["method"], info["precision"]), X_test)
            p_new = predict_proba(load_checkpoint_model(ckpt_path, info["method"], info["precision"]), X_test)
            entry["predictions"] = {
                "n_test": int(len(X_test)),
                "class_agreement": float((p_orig.argmax(1) == p_new.argmax(1)).mean()),
                "max_abs_probability_diff": float(abs(p_orig - p_new).max()),
                "mean_abs_probability_diff": float(abs(p_orig - p_new).mean()),
            }
        entry["passed"] = all(d <= metric_tol for d in metric_diffs.values()) and all(meta_same.values())
        ok &= entry["passed"]
        report["repeats"].append(entry)

    report["excluded_fields"] = ["timestamp", "git_commit", "git_dirty", "latency (wall-clock)"]
    report["original_device"] = original["metrics"]["latency"]["device"]
    report["rerun_device"] = runs[0][0]["metrics"]["latency"]["device"]
    report["overall"] = "PASS" if ok else "FAIL"

    stem = Path(f"{out_stem}_{experiment_id}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = [
        f"# Reproducibility check — {experiment_id}",
        "",
        f"Re-ran from the recorded config, in isolation, {repeats}x. Absolute tolerance per metric: {metric_tol}. "
        f"Devices: original `{report['original_device']}`, re-run `{report['rerun_device']}`. **Overall: {report['overall']}**",
        "",
        "| repeat | " + " | ".join(f"|Δ{m}|" for m in METRICS) + " | max |Δweight| | class agreement | max |Δprob| | metadata identical | passed |",
        "|---|" + "---:|" * len(METRICS) + "---:|---:|---:|---|---|",
    ]
    for r in report["repeats"]:
        pr = r.get("predictions", {})
        lines.append(
            f"| {r['repeat']} | " + " | ".join(f"{r['metric_abs_diff'][m]:.2e}" for m in METRICS)
            + f" | {r['weights'].get('max_abs_weight_diff', float('nan')):.2e} | {pr.get('class_agreement', float('nan')):.6f}"
            + f" | {pr.get('max_abs_probability_diff', float('nan')):.2e} | {all(r['metadata_identical'].values())} | {r['passed']} |"
        )
    lines += ["", f"Excluded from comparison (legitimately differ): {', '.join(report['excluded_fields'])}."]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    return report
