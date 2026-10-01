"""Dataset loading abstraction supporting HDF5, CSV, Parquet, and NumPy sources.

Also exposes `load_synthetic` as the documented development-subset path used
when a real HEP dataset is unavailable (see reports/dataset_card.md).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from hep_quant.data.synthetic import generate_synthetic_dataset
from hep_quant.data.validation import ValidationReport, validate_dataset


class HEPDataset:
    """Loaded dataset: feature matrix, integer labels, and feature names."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        source: str,
        class_names: list[str] | None = None,
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.source = source
        # Original class labels (e.g. "g"/"q"/"t"/"w"/"z"), index-aligned with
        # the integer-encoded `y` values, when the source dataset has named
        # classes. None for datasets with no meaningful class names (e.g. the
        # synthetic generator's plain 0/1 labels).
        self.class_names = class_names

    def validate(self, raise_on_error: bool = True) -> ValidationReport:
        return validate_dataset(self.X, self.y, raise_on_error=raise_on_error)

    def inspect(self) -> dict:
        report = self.validate(raise_on_error=False)
        return {"source": self.source, "feature_names": self.feature_names, **report.to_dict()}


def load_synthetic(
    n_samples: int = 20000, n_features: int = 16, n_classes: int = 2, seed: int = 42
) -> HEPDataset:
    X, y, feature_names = generate_synthetic_dataset(n_samples, n_features, n_classes, seed)
    return HEPDataset(X, y, feature_names, source="synthetic")


def load_hdf5(path: str | Path, feature_key: str = "features", label_key: str = "labels") -> HEPDataset:
    path = Path(path)
    with h5py.File(path, "r") as f:
        X = np.asarray(f[feature_key][:], dtype=np.float32)
        y = np.asarray(f[label_key][:], dtype=np.int64)
        feature_names = (
            [n.decode() if isinstance(n, bytes) else str(n) for n in f["feature_names"][:]]
            if "feature_names" in f
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
    return HEPDataset(X, y, feature_names, source=str(path))


def load_csv(path: str | Path, label_column: str = "label") -> HEPDataset:
    path = Path(path)
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not found in columns: {list(df.columns)}")
    y = df[label_column].to_numpy(dtype=np.int64)
    feature_df = df.drop(columns=[label_column])
    X = feature_df.to_numpy(dtype=np.float32)
    return HEPDataset(X, y, list(feature_df.columns), source=str(path))


def load_parquet(path: str | Path, label_column: str = "label") -> HEPDataset:
    path = Path(path)
    df = pd.read_parquet(path)
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not found in columns: {list(df.columns)}")
    y = df[label_column].to_numpy(dtype=np.int64)
    feature_df = df.drop(columns=[label_column])
    X = feature_df.to_numpy(dtype=np.float32)
    return HEPDataset(X, y, list(feature_df.columns), source=str(path))


def load_numpy(path: str | Path, feature_key: str = "X", label_key: str = "y") -> HEPDataset:
    path = Path(path)
    data = np.load(path)
    X = np.asarray(data[feature_key], dtype=np.float32)
    y = np.asarray(data[label_key], dtype=np.int64)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return HEPDataset(X, y, feature_names, source=str(path))


def load_openml(
    name: str = "hls4ml_lhc_jets_hlf",
    version: int | str = 1,
    subset_size: int | None = None,
    seed: int = 42,
    data_home: str | Path = "data/raw/openml_cache",
) -> HEPDataset:
    """Fetch a real dataset from OpenML, cached locally under `data_home`.

    Labels are integer-encoded for training, but the original class names
    (e.g. "g"/"q"/"t"/"w"/"z" for the hls4ml jet-tagging dataset) are kept on
    `HEPDataset.class_names` — never silently renamed to "0"/"1"/... — so
    reports and confusion-matrix axes stay physically meaningful.

    When `subset_size` is given, a *stratified* subsample is drawn (class
    proportions preserved) rather than a plain truncation, since OpenML's
    row ordering is not guaranteed to be shuffled or class-balanced. This is
    the mechanism behind the project's staged 10k -> 50k -> 100k+ validation
    sequence for real datasets (see configs/jet_tagging_*.yaml).
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    Path(data_home).mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(name=name, version=version, as_frame=True, data_home=str(data_home), parser="auto")

    feature_df = bunch.data
    y_raw = bunch.target
    if y_raw is None:
        raise ValueError(f"OpenML dataset '{name}' has no default target column configured")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw).astype(np.int64)
    X = feature_df.to_numpy(dtype=np.float32)
    feature_names = list(feature_df.columns)

    if subset_size is not None and subset_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=subset_size, random_state=seed, stratify=y)

    return HEPDataset(
        X, y, feature_names,
        source=f"openml:{name}:v{version}",
        class_names=list(encoder.classes_),
    )


_LOADERS = {
    ".h5": load_hdf5,
    ".hdf5": load_hdf5,
    ".csv": load_csv,
    ".parquet": load_parquet,
    ".npz": load_numpy,
}


def load_dataset(path: str | Path | None, **kwargs) -> HEPDataset:
    """Dispatch to the right loader based on file extension, or synthetic if `path` is None."""
    if path is None:
        return load_synthetic(**kwargs)
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported dataset format '{path.suffix}' for {path}")
    return loader(path, **kwargs)
