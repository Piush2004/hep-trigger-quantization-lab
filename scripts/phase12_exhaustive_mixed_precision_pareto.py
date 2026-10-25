#!/usr/bin/env python
"""Phase 12: near-exhaustive mixed-precision PTQ search, then QAT on the Pareto frontier.

Generalizes Phase 5/5B (9 hand-picked layouts) to the full per-unit search space:
7 units (L0-L3 weights, A0-A2 activations) x {INT4, INT6, INT8} = 3**7 = 2187 layouts.
Reuses phase5_mixed_precision_gate.py's load_splits/score/build_ptq_variant plumbing
unchanged (imported, not copied) and the same five seeds/frozen checkpoints Phase 5/5B used.

Protocol (fixed before running):
  1. Timing benchmark on a small sample decides the path: if the estimated full 2187x5-seed
     grid is cheap (<= FULL_GRID_HOURS_THRESHOLD hours), run it directly, all 5 seeds, no
     single-seed screening approximation. Otherwise fall back to a staged screen (all 2187 @
     seed 42) -> confirm (frontier only @ 5 seeds) protocol. Printed/recorded, not assumed.
  2a. [Full-grid path] Stage 1: all 2187 layouts, validation only, all 5 seeds, directly.
  2b. [Staged path] Stage 0 (screen): all 2187 layouts, validation only, seed 42 only; Pareto
      frontier extracted (storage vs. validation ROC-AUC, hep_quant.visualization.pareto.
      find_pareto_frontier); Stage 1 (confirm): frontier layouts only, validation, all 5 seeds.
  3. Selection rule (pre-declared, mirrors Phase 5B exactly): among the Stage-1 layouts with
     theoretical_weight_storage_bytes <= a dynamically computed INT6-uniform limit, pick the
     highest mean validation ROC-AUC; ties within TIE_EPS broken by smaller storage, then name.
     Selection is written to disk before the test split is touched.
  4. Stage 2 (test, once): the single selected layout, all 5 seeds, test split.

Outputs: reports/tables/phase12_timing_benchmark.json, phase12_screen_validation.csv (staged
         path only), phase12_screen_frontier.json (staged path only),
         phase12_confirm_validation.csv (the Stage-1 table, either path), phase12_selection.json,
         phase12_final_test.csv
"""
import argparse
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "reports" / "tables"

SEEDS = [42, 123, 2026, 7, 99]
BIT_CHOICES = (4, 6, 8)
N_LINEAR = 4
N_ACT = 3
TIE_EPS = 0.001
FULL_GRID_HOURS_THRESHOLD = 2.0


def _load_phase5():
    spec = importlib.util.spec_from_file_location("phase5_gate", ROOT / "scripts" / "phase5_mixed_precision_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def all_layouts():
    """Deterministic order: itertools.product over the 7 units (L0..L3, A0..A2)."""
    for combo in itertools.product(BIT_CHOICES, repeat=N_LINEAR + N_ACT):
        w = {i: combo[i] for i in range(N_LINEAR)}
        a = {i: combo[N_LINEAR + i] for i in range(N_ACT)}
        yield w, a


def layout_name(w: dict, a: dict) -> str:
    parts = [f"L{i}-{w[i]}" for i in sorted(w)] + [f"A{i}-{a[i]}" for i in sorted(a)]
    return "_".join(parts)


def layouts_by_name() -> dict:
    return {layout_name(w, a): (w, a) for w, a in all_layouts()}


def size_limit_bytes(model, p5) -> float:
    """Dynamic INT6-uniform storage limit (architecture-agnostic; not hardcoded)."""
    w6, _ = p5.uniform_units(model, 6)
    return p5.theoretical_weight_storage_bytes(model, w6)


def benchmark(p5, n: int = 20, seed: int = 42) -> float:
    """Mean seconds per build_ptq_variant+score call, over n randomly sampled layouts, one seed."""
    model, tr_val, _ = p5.load_splits(seed, include_test=False)
    Xtr = tr_val["train"]
    Xva, yva = tr_val["val"]
    sample = random.Random(0).sample(list(itertools.islice(all_layouts(), 3 ** (N_LINEAR + N_ACT))), n)
    t0 = time.perf_counter()
    for w, a in sample:
        v = p5.build_ptq_variant(model, Xtr[:20480], w, a)
        p5.score(v, Xva, yva)
    return (time.perf_counter() - t0) / n


def stage0_screen(p5, seed: int = 42) -> pd.DataFrame:
    model, tr_val, _ = p5.load_splits(seed, include_test=False)
    Xtr = tr_val["train"]
    Xva, yva = tr_val["val"]
    rows = []
    for w, a in all_layouts():
        v = p5.build_ptq_variant(model, Xtr[:20480], w, a)
        rows.append({"seed": seed, "layout": layout_name(w, a), **p5.score(v, Xva, yva),
                     "theoretical_weight_storage_bytes": p5.theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def stage1_confirm(p5, by_name: dict, names: list[str], seeds: list[int] = SEEDS) -> pd.DataFrame:
    rows = []
    for s in seeds:
        model, tr_val, _ = p5.load_splits(s, include_test=False)
        Xtr = tr_val["train"]
        Xva, yva = tr_val["val"]
        for name in names:
            w, a = by_name[name]
            v = p5.build_ptq_variant(model, Xtr[:20480], w, a)
            rows.append({"seed": s, "layout": name, **p5.score(v, Xva, yva),
                         "theoretical_weight_storage_bytes": p5.theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def select(confirm_df: pd.DataFrame, limit: float) -> dict:
    summary = confirm_df.groupby("layout").agg(
        val_auc_mean=("roc_auc", "mean"), val_auc_sd=("roc_auc", "std"),
        storage=("theoretical_weight_storage_bytes", "first")).reset_index()
    base = {
        "selection_rule": ("among confirmed layouts with theoretical_weight_storage_bytes <= size_limit_bytes, "
                           "select highest mean validation macro ROC-AUC over the confirm-stage seeds; "
                           "candidates within tie_eps of the best are indistinguishable and the smallest "
                           "storage is preferred, then layout name"),
        "size_limit_bytes": float(limit),
        "tie_eps": TIE_EPS,
        "candidate_count": int(len(summary)),
        "selection_inputs": "validation split only (stage 1 confirm, include_test=False)",
        "selection_used_test_set": False,
    }
    elig = summary[summary.storage <= limit]
    if elig.empty:
        base.update({"selected_layout": None, "status": "NO ELIGIBLE CONFIGURATION"})
        return base
    best = elig.val_auc_mean.max()
    tied = elig[elig.val_auc_mean >= best - TIE_EPS].sort_values(["storage", "layout"])
    chosen = tied.iloc[0]
    base.update({
        "selected_layout": chosen["layout"],
        "selected_storage_bytes": float(chosen["storage"]),
        "validation_mean": float(chosen["val_auc_mean"]),
        "validation_sd": float(chosen["val_auc_sd"]) if pd.notna(chosen["val_auc_sd"]) else None,
        "indistinguishable_from_best": tied["layout"].tolist(),
        "status": "SELECTED",
        "test_set_used_after_selection": None,
    })
    return base


def stage2_test(p5, name: str, w: dict, a: dict, seeds: list[int] = SEEDS) -> pd.DataFrame:
    rows = []
    for s in seeds:
        model, tr_val, (Xte, yte) = p5.load_splits(s, include_test=True)
        v = p5.build_ptq_variant(model, tr_val["train"][:20480], w, a)
        rows.append({"seed": s, "layout": name, **p5.score(v, Xte, yte),
                     "theoretical_weight_storage_bytes": p5.theoretical_weight_storage_bytes(model, w)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true", help="Run the timing micro-benchmark and exit.")
    args = parser.parse_args()

    p5 = _load_phase5()
    TAB.mkdir(parents=True, exist_ok=True)
    device = {"cuda_available": torch.cuda.is_available(), "device": "cpu"}

    n_total = 3 ** (N_LINEAR + N_ACT)
    avg_s = benchmark(p5)
    est_full_hours = avg_s * n_total * len(SEEDS) / 3600
    bench = {"avg_seconds_per_eval": avg_s, "n_sampled": 20, "seed": 42,
             "n_layouts": n_total, "estimated_full_grid_hours": est_full_hours, **device}
    (TAB / "phase12_timing_benchmark.json").write_text(json.dumps(bench, indent=2))
    print(f"avg {avg_s:.3f} s/eval over 20 sampled layouts; "
          f"full {n_total}x{len(SEEDS)}-seed grid estimate: {est_full_hours:.1f} h")
    if args.benchmark_only:
        return 0

    model0, _tr_val, _ = p5.load_splits(42, include_test=False)
    limit = size_limit_bytes(model0, p5)
    by_name = layouts_by_name()

    if est_full_hours <= FULL_GRID_HOURS_THRESHOLD:
        # ---- Full grid: all 2187 layouts, all 5 seeds, directly (no screening approximation) ----
        protocol = {"path": "full_grid", "reason": f"estimated {est_full_hours:.2f}h <= "
                    f"{FULL_GRID_HOURS_THRESHOLD}h threshold; ran full 5-seed grid directly"}
        print(f"full-grid path: {est_full_hours:.2f}h estimated, running all {n_total} layouts x {len(SEEDS)} seeds")
        confirm = stage1_confirm(p5, by_name, list(by_name.keys()), seeds=SEEDS)
    else:
        # ---- Staged: screen (seed 42 only) -> frontier -> confirm (frontier only, all seeds) ----
        protocol = {"path": "staged_screen_confirm", "reason": f"estimated {est_full_hours:.2f}h > "
                    f"{FULL_GRID_HOURS_THRESHOLD}h threshold; used single-seed screen + frontier confirm"}
        screen = stage0_screen(p5, seed=42)
        screen.to_csv(TAB / "phase12_screen_validation.csv", index=False)

        from hep_quant.visualization.pareto import find_pareto_frontier
        mask = find_pareto_frontier(screen, maximize=["roc_auc"], minimize=["theoretical_weight_storage_bytes"])
        frontier_names = screen.loc[mask, "layout"].tolist()
        (TAB / "phase12_screen_frontier.json").write_text(json.dumps(
            {"size_limit_bytes": limit, "frontier_count": len(frontier_names),
             "frontier_layouts": frontier_names, **device}, indent=2))
        print(f"screen frontier: {len(frontier_names)} layouts (of {n_total})")
        confirm = stage1_confirm(p5, by_name, frontier_names, seeds=SEEDS)

    (TAB / "phase12_protocol.json").write_text(json.dumps({**protocol, **device}, indent=2))
    confirm.to_csv(TAB / "phase12_confirm_validation.csv", index=False)

    # ---- Selection, written before any test evaluation ----
    sel = select(confirm, limit)
    sel.update(device)
    (TAB / "phase12_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    if sel.get("selected_layout") is None:
        print(sel["status"])
        return 0
    name = sel["selected_layout"]
    w, a = by_name[name]
    print(f"selected: {name} (mean val ROC-AUC {sel['validation_mean']:.4f}, "
          f"storage {sel['selected_storage_bytes']:.0f} B)")

    # ---- Stage 2: test, once, selected layout only ----
    final = stage2_test(p5, name, w, a, seeds=SEEDS)
    final.to_csv(TAB / "phase12_final_test.csv", index=False)
    print(final[["seed", "roc_auc", "f1", "accuracy"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
