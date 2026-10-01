"""Deterministic train/validation/test splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """Stratified, deterministic train/val/test split.

    Splitting is done in two stages (train+val vs test, then train vs val)
    using the same `seed` so the split is reproducible across runs. Stratified
    on `y` to preserve class balance across all three splits.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    relative_val_fraction = val_fraction / (1 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_fraction,
        random_state=seed,
        stratify=y_trainval,
    )
    return DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def check_no_leakage(splits: DatasetSplits) -> int:
    """Return the count of feature rows shared between any two splits (should be 0).

    Rows are compared by exact value equality via a hashable byte-view, which
    is sufficient for detecting accidental split-time duplication/leakage.
    """

    def row_set(X: np.ndarray) -> set[bytes]:
        return {row.tobytes() for row in X}

    train_set = row_set(splits.X_train)
    val_set = row_set(splits.X_val)
    test_set = row_set(splits.X_test)

    overlap = len(train_set & val_set) + len(train_set & test_set) + len(val_set & test_set)
    return overlap
