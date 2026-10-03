#!/usr/bin/env python
"""Gate H: re-run a recorded experiment from its own stored configuration, in isolation, and compare.

The recorded config (from experiments/results/<id>.json) is written to a scratch
YAML and re-run with results/checkpoints/figures redirected to a scratch
directory, so nothing in the repository is overwritten. The re-run is repeated
`--repeats` times. Compared against the original record/checkpoint:
  metrics (accuracy, macro F1, macro ROC-AUC, macro PR-AUC), recorded metadata
  (config, seed, dataset subset, parameter count, model size, hardware block),
  and the saved weights (max absolute difference per tensor).
Fields that legitimately differ (timestamp, git commit, wall-clock latency) are
excluded and listed. Output: reports/reproducibility_real.{json,md}.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hep_quant.experiments.runner import run_experiment

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def _weight_diff(path_a: Path, path_b: Path) -> dict:
    a = torch.load(path_a, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(path_b, map_location="cpu", weights_only=False)["model_state_dict"]
    if a.keys() != b.keys():
        return {"same_keys": False}
    diffs = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a if a[k].dtype.is_floating_point}
    return {"same_keys": True, "max_abs_weight_diff": max(diffs.values()), "worst_tensor": max(diffs, key=diffs.get)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="recorded experiment id, e.g. EXP_S42_JET_INT4_PTQ")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--metric-tol", type=float, default=0.005, help="documented tolerance on each metric (absolute)")
    ap.add_argument("--out-stem", default="reports/reproducibility_real")
    args = ap.parse_args()

    original = json.loads(Path(f"experiments/results/{args.experiment}.json").read_text())
    cfg = original["config"]
    scratch = Path(tempfile.mkdtemp(prefix="repro_"))
    cfg_path = scratch / "stored_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    runs = []
    for i in range(args.repeats):
        res, ckpt, fig = scratch / f"r{i}/results", scratch / f"r{i}/ckpt", scratch / f"r{i}/fig"
        rec = run_experiment(str(cfg_path), results_dir=str(res), checkpoint_dir=str(ckpt), figures_dir=str(fig))
        exp_id = cfg["experiment"]["id"]
        runs.append((rec, ckpt / exp_id / "best_model.pt"))

    orig_ckpt = Path(f"experiments/checkpoints/{cfg['experiment']['id']}/best_model.pt")
    report = {"experiment": args.experiment, "metric_tolerance_abs": args.metric_tol, "repeats": []}
    ok = True
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
        weights = _weight_diff(orig_ckpt, ckpt_path)
        passed = all(d <= args.metric_tol for d in metric_diffs.values()) and all(meta_same.values())
        ok &= passed
        report["repeats"].append({
            "repeat": i + 1, "metric_abs_diff": metric_diffs, "metadata_identical": meta_same, "weights": weights,
            "original_metrics": {m: o[m] for m in METRICS}, "rerun_metrics": {m: n[m] for m in METRICS}, "passed": passed,
        })
    report["excluded_fields"] = ["timestamp", "git_commit", "latency (wall-clock)", "system.cuda_available if the device changed"]
    report["original_device"] = original["metrics"]["latency"]["device"]
    report["rerun_device"] = runs[0][0]["metrics"]["latency"]["device"]
    report["overall"] = "PASS" if ok else "FAIL"

    stem = Path(f"{args.out_stem}_{args.experiment}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = [f"# Reproducibility check — {args.experiment}", "",
             f"Re-ran from the recorded config, in isolation, {args.repeats}x. Absolute tolerance per metric: {args.metric_tol}. "
             f"Devices: original `{report['original_device']}`, re-run `{report['rerun_device']}`. **Overall: {report['overall']}**", "",
             "| repeat | " + " | ".join(f"|Δ{m}|" for m in METRICS) + " | max |Δweight| | metadata identical | passed |",
             "|---|" + "---:|" * len(METRICS) + "---:|---|---|"]
    for r in report["repeats"]:
        lines.append(f"| {r['repeat']} | " + " | ".join(f"{r['metric_abs_diff'][m]:.2e}" for m in METRICS)
                     + f" | {r['weights'].get('max_abs_weight_diff', float('nan')):.2e} | {all(r['metadata_identical'].values())} | {r['passed']} |")
    lines += ["", f"Excluded from comparison (legitimately differ): {', '.join(report['excluded_fields'])}."]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(scratch, ignore_errors=True)
    print(report["overall"], json.dumps([r["metric_abs_diff"]["accuracy"] for r in report["repeats"]]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
