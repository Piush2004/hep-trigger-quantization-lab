"""ML surrogate for FPGA resource/latency estimation, trained on fastmachinelearning/wa-hls4ml.

Rather than loading the wa-hls4ml paper's own pretrained GNN/transformer surrogate (unknown
feature schema, unknown graph-library dependencies), this trains a small per-target sklearn
regressor on tabular, per-layer features extracted from the dataset's raw JSON records. The
same feature extractor (`features_from_wa_hls4ml_record` / `features_from_pytorch_model`) is
used on both the training side (wa-hls4ml records) and the inference side (this project's own
MLPClassifier + per-unit bit dicts) -- the two MUST return identical key sets, or predictions
are meaningless; this contract is enforced by a dedicated test.

Features are restricted to things BOTH a wa-hls4ml record and a live PyTorch model can supply:
per-Dense-layer width and weight bit-width (fixed-width, zero-padded to MAX_LAYERS slots),
total parameter count, a single top-level reuse factor, a Resource/Latency strategy flag, and a
target clock period. Board/Vivado-version/hls4ml-version are recorded as training-data
provenance (see `train_surrogate`'s `trained_on` metadata) but are NOT model features, since
this project's own `HLSBackend`/`SoftwareEstimateBackend` do not let a caller specify them.

Targets: lut, dsp, bram, ff (from `resource_report`), latency_ns, ii (from `latency_report`,
converted from cycles via the record's own clock period).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn

MAX_LAYERS = 7
TARGETS = ("lut", "dsp", "bram", "ff", "latency_ns", "ii")
FEATURE_SPEC_VERSION = "v1"
DEFAULT_CLOCK_NS = 10.0


def _fixed_bits(precision_str: str) -> float:
    """'fixed<16,6>' (or 'fixed<16, 6>') -> 16.0. Returns NaN if the string isn't a fixed<> spec
    (e.g. 'ap_fixed<...>' uses the same '<total,int>' shape, handled identically)."""
    try:
        inner = precision_str[precision_str.index("<") + 1 : precision_str.index(">")]
        return float(inner.split(",")[0].strip())
    except (ValueError, AttributeError):
        return float("nan")


def _ordered_layer_names(hls_config: dict, prefix: str) -> list[str]:
    """hls4ml's deterministic auto-naming: 'dense', 'dense_1', 'dense_2', ... in layer order."""
    names = [k for k in hls_config.get("LayerName", {}) if k == prefix or k.startswith(prefix + "_")]

    def _key(n: str) -> int:
        return 0 if n == prefix else int(n.rsplit("_", 1)[-1])

    return sorted(names, key=_key)


def _pad(values: list[float], length: int = MAX_LAYERS) -> dict[str, float]:
    padded = (values + [0.0] * length)[:length]
    return {f"slot_{i}": v for i, v in enumerate(padded)}


def features_from_wa_hls4ml_record(record: dict) -> dict:
    """Flatten one wa-hls4ml record into the fixed-width feature schema."""
    model_config = record["model_config"]
    hls_config = record["hls_config"]
    model_cfg = hls_config.get("Model", {})

    dense_layers = [layer for layer in model_config if layer.get("class_name") == "Dense"]
    widths = [float(layer.get("neurons", 0)) for layer in dense_layers]
    total_params = sum(float(layer.get("parameters", 0)) for layer in model_config)

    dense_names = _ordered_layer_names(hls_config, "dense")
    weight_bits = []
    for name in dense_names:
        prec = hls_config["LayerName"].get(name, {}).get("Precision", {})
        weight_bits.append(_fixed_bits(prec.get("weight", prec.get("result", ""))))

    latency = record.get("latency_report", {})
    clock_ns = float(latency.get("target_clock", DEFAULT_CLOCK_NS))

    features = {
        "n_dense_layers": float(len(dense_layers)),
        "total_params": total_params,
        "reuse_factor": float(model_cfg.get("ReuseFactor", 1)),
        "strategy_is_resource": 1.0 if model_cfg.get("Strategy") == "Resource" else 0.0,
        "target_clock_ns": clock_ns,
    }
    features.update({f"width_{k}": v for k, v in _pad(widths).items()})
    features.update({f"bits_{k}": v for k, v in _pad(weight_bits).items()})
    return features


def features_from_pytorch_model(
    model: nn.Module,
    weight_bits: dict[int, int | None],
    hardware_cfg: dict,
) -> dict:
    """Same feature schema as `features_from_wa_hls4ml_record`, built from a live MLPClassifier
    and the per-unit weight-bit dict `build_ptq_variant` already takes (act_bits is not part of
    this schema -- hls4ml resource counts are driven by weight/accumulator precision per Dense
    layer, not activation precision, matching what `hls_config`'s LayerName entries expose)."""
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    widths = [float(lin.out_features) for lin in linears]
    total_params = float(sum(p.numel() for p in model.parameters()))
    bits = [float(weight_bits.get(i) or 32) for i in range(len(linears))]

    features = {
        "n_dense_layers": float(len(linears)),
        "total_params": total_params,
        "reuse_factor": float(hardware_cfg.get("reuse_factor", 1)),
        "strategy_is_resource": 1.0 if hardware_cfg.get("strategy", "Resource") == "Resource" else 0.0,
        "target_clock_ns": float(hardware_cfg.get("clock_period_ns", DEFAULT_CLOCK_NS)),
    }
    features.update({f"width_{k}": v for k, v in _pad(widths).items()})
    features.update({f"bits_{k}": v for k, v in _pad(bits).items()})
    return features


def _targets_from_record(record: dict) -> dict[str, float] | None:
    res = record.get("resource_report", {})
    lat = record.get("latency_report", {})
    try:
        clock_ns = float(lat["target_clock"])
        cycles = float(lat["cycles_max"])
        return {
            "lut": float(res["lut"]), "dsp": float(res["dsp"]),
            "bram": float(res["bram"]), "ff": float(res["ff"]),
            "latency_ns": cycles * clock_ns,
            "ii": float(lat["interval_max"]),
        }
    except (KeyError, ValueError, TypeError):
        return None


def _records_to_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        t = _targets_from_record(r)
        if t is None:
            continue
        rows.append({**features_from_wa_hls4ml_record(r), **t})
    return pd.DataFrame(rows)


def _held_out_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {"r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred)), "n": int(len(y_true))}


def train_surrogate(
    train_records: list[dict],
    test_records: list[dict],
    jet_exemplar_records: list[dict] | None = None,
    model_type: str = "random_forest",
) -> dict:
    """One regressor per target (simpler and easier to report per-target held-out error than a
    multi-output model; GradientBoostingRegressor doesn't support multi-output natively anyway).

    Held-out evaluation uses wa-hls4ml's OWN train/test split (not a re-split), plus a separate,
    much smaller Jet-exemplar check -- reported distinctly, never pooled with the bulk metric,
    since that slice is small-n.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    train_df = _records_to_frame(train_records)
    test_df = _records_to_frame(test_records)
    feature_cols = [c for c in train_df.columns if c not in TARGETS]

    models, held_out = {}, {}
    for target in TARGETS:
        if model_type == "random_forest":
            reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        else:
            reg = GradientBoostingRegressor(random_state=42)
        reg.fit(train_df[feature_cols], train_df[target])
        pred = reg.predict(test_df[feature_cols])
        held_out[target] = _held_out_metrics(test_df[target].to_numpy(), pred)
        models[target] = reg

    jet_held_out = {}
    if jet_exemplar_records:
        jet_df = _records_to_frame(jet_exemplar_records)
        if len(jet_df) > 0:
            for target in TARGETS:
                pred = models[target].predict(jet_df[feature_cols])
                jet_held_out[target] = _held_out_metrics(jet_df[target].to_numpy(), pred)

    import sklearn

    return {
        "models": models,
        "feature_cols": feature_cols,
        "held_out": held_out,
        "jet_exemplar_held_out": jet_held_out,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "sklearn_version": sklearn.__version__,
        "trained_on": {"repo": "fastmachinelearning/wa-hls4ml", "subsets": ["resource", "latency"],
                       "license": "cc-by-nc-4.0", "n_train": len(train_df), "n_test": len(test_df)},
    }


def save_surrogate(bundle: dict, model_dir: str | Path) -> None:
    # joblib here only ever round-trips sklearn models this project trains and saves itself
    # (never a file from an external/untrusted source), so the usual pickle-load risk doesn't apply.
    import joblib

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for target, reg in bundle["models"].items():
        joblib.dump(reg, model_dir / f"{target}.joblib")
    metadata = {k: v for k, v in bundle.items() if k != "models"}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_surrogate(model_dir: str | Path) -> dict:
    import joblib

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    models = {target: joblib.load(model_dir / f"{target}.joblib") for target in TARGETS}
    return {**metadata, "models": models}


def predict_resources(features: dict, bundle: dict) -> dict[str, float]:
    row = pd.DataFrame([features])[bundle["feature_cols"]]
    return {target: float(bundle["models"][target].predict(row)[0]) for target in TARGETS}
