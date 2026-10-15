#!/usr/bin/env python
"""Phase 11, training stage: train the sklearn hardware-resource surrogate on wa-hls4ml.

One-time/offline entrypoint -- not part of the normal experiment pipeline. Trains on the
`resource`+`latency` fully-connected splits' train+val records (never 2layer/3layer/conv1d/
conv2d/2_20 -- excluded per the dataset card's own disclosed limitation that those are mostly
tiny, unactivated, non-representative networks), evaluates held-out error on the `test` split
(wa-hls4ml's own split, not a re-split), and separately against the Jet-family exemplar records
(architecture-matched to this project's own MLP, but small-n -- reported, never pooled with the
bulk held-out metric). Writes experiments/surrogate_models/v1/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hep_quant.data.wa_hls4ml import (  # noqa: E402
    FC_SUBSETS,
    filter_jet_exemplar,
    iter_records_stream,
    load_exemplar_records,
)
from hep_quant.hardware.surrogate import save_surrogate, train_surrogate  # noqa: E402

CACHE = ROOT / "data" / "raw" / "wa_hls4ml_cache"
MODEL_DIR = ROOT / "experiments" / "surrogate_models" / "v1"


def _load_split(split: str) -> list[dict]:
    records = []
    for subset in FC_SUBSETS:
        path = CACHE / split / f"{split}_{subset}_merged.json"
        records.extend(iter_records_stream(path))
    return records


def main() -> int:
    print("loading exemplar (Jet-family architecture-matched check)...")
    exemplar = load_exemplar_records(str(CACHE))
    jet_exemplar = filter_jet_exemplar(exemplar)
    print(f"  {len(jet_exemplar)} Jet-family exemplar records")

    print("streaming train+val (resource+latency)...")
    train_records = _load_split("train") + _load_split("val")
    print(f"  {len(train_records)} training records")

    print("streaming test (resource+latency)...")
    test_records = _load_split("test")
    print(f"  {len(test_records)} held-out test records")

    print("training per-target regressors...")
    bundle = train_surrogate(train_records, test_records, jet_exemplar_records=jet_exemplar)

    print("\nheld-out (wa-hls4ml test split):")
    for target, m in bundle["held_out"].items():
        print(f"  {target:12s} R2={m['r2']:.3f} MAE={m['mae']:.1f} (n={m['n']})")
    print("\nheld-out (Jet-family exemplar, small-n):")
    for target, m in bundle["jet_exemplar_held_out"].items():
        print(f"  {target:12s} R2={m['r2']:.3f} MAE={m['mae']:.1f} (n={m['n']})")

    save_surrogate(bundle, MODEL_DIR)
    print(f"\nsaved to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
