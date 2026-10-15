"""Download/cache adapter for the fastmachinelearning/wa-hls4ml Hugging Face dataset.

Raw per-model-type JSON files on the Hub (confirmed: NOT parquet-convertible, so
`datasets.load_dataset()` is not used here -- `huggingface_hub.hf_hub_download` instead).
We only ever pull the `resource`/`latency` fully-connected splits, never
`2layer`/`3layer`/`conv1d`/`conv2d`/`2_20` -- the dataset card itself flags those as mostly
tiny, unactivated, non-representative networks (and they are far larger downloads too).

Schema confirmed by downloading and inspecting `exemplar/exemplar_models.json` directly
(887 records, flat JSON list, every record has exactly these 8 top-level keys -- there is
no `hls_resource_report` field, unlike an earlier, unverified guess):

  meta_data          -- {uuid, model_name, artifacts_file}
  model_config       -- list of per-layer dicts (class_name, input_shape, output_shape,
                         parameters, neurons [Dense], activation [Activation], reuse_factor)
  hls_config         -- {"Model": {Precision, ReuseFactor, Strategy, ...},
                         "LayerName": {<layer_id>: {Precision: {weight/bias/accum/result},
                                                     ReuseFactor, ...}, ...}}
  resource_report    -- {bram, dsp, ff, lut, uram} (numeric strings)
  latency_report     -- {cycles_min, cycles_max, target_clock, estimated_clock,
                         interval_min, interval_max} (numeric strings; interval = II)
  target_part        -- e.g. "xcu200-fsgd2104-2-e"
  vivado_version      -- e.g. "2019.1"
  hls4ml_version      -- e.g. "0.8.1"

124 of the 887 exemplar records are a "Jet"-family model (model_name starting "Jet_") --
the same task family (LHC jet tagging) as this project's MLP -- across varying bit-widths,
reuse factors, and Resource/Latency strategies, giving an architecture-matched validation
slice in addition to the bulk `resource`/`latency` held-out split.

License: CC-BY-NC-4.0 (non-commercial). Disclose in any report; do not present a surrogate
model trained on this data as freely redistributable.
"""
from __future__ import annotations

import json
from pathlib import Path

HF_REPO = "fastmachinelearning/wa-hls4ml"
HF_REPO_TYPE = "dataset"
LICENSE = "cc-by-nc-4.0"
DEFAULT_CACHE_DIR = "data/raw/wa_hls4ml_cache"

FC_SUBSETS = ("resource", "latency")
EXCLUDED_SUBSETS = ("2layer", "3layer", "conv1d", "conv2d", "2_20")

EXPECTED_SAMPLE_KEYS = {
    "meta_data", "model_config", "hls_config",
    "resource_report", "latency_report",
    "target_part", "vivado_version", "hls4ml_version",
}


class SchemaMismatchError(Exception):
    """Raised when a downloaded wa-hls4ml record does not match the documented schema."""


def check_sample_schema(sample: dict) -> None:
    missing = EXPECTED_SAMPLE_KEYS - sample.keys()
    if missing:
        raise SchemaMismatchError(f"wa-hls4ml record missing expected keys: {sorted(missing)}")


def download_exemplar(data_home: str = DEFAULT_CACHE_DIR) -> Path:
    """Download the small (~6 MB) exemplar file -- used for the Jet-family validation slice."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
        filename="exemplar/exemplar_models.json", local_dir=data_home,
    ))


def download_subset(subset: str, split: str, data_home: str = DEFAULT_CACHE_DIR) -> Path:
    """Download one `resource`/`latency` split file (`split` in train/val/test). Large (hundreds
    of MB) -- caller is responsible for getting user permission before calling this."""
    if subset not in FC_SUBSETS:
        raise ValueError(f"subset '{subset}' not in allowed FC subsets {FC_SUBSETS} (excluded: {EXCLUDED_SUBSETS})")
    from huggingface_hub import hf_hub_download

    filename = f"{split}/{split}_{subset}_merged.json"
    return Path(hf_hub_download(repo_id=HF_REPO, repo_type=HF_REPO_TYPE, filename=filename, local_dir=data_home))


def load_exemplar_records(data_home: str = DEFAULT_CACHE_DIR) -> list[dict]:
    path = download_exemplar(data_home)
    records = json.loads(path.read_text())
    for r in records:
        check_sample_schema(r)
    return records


def filter_jet_exemplar(records: list[dict]) -> list[dict]:
    return [r for r in records if str(r.get("meta_data", {}).get("model_name", "")).lower().startswith("jet")]


def iter_records_stream(path: Path):
    """Stream-parse a large merged JSON file (confirmed flat top-level list, same shape as
    the exemplar file) without loading the whole file into memory."""
    import ijson

    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            check_sample_schema(item)
            yield item
