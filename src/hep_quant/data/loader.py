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

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], source: str):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.source = source

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
