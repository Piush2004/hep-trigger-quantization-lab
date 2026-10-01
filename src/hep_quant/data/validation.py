"""Data integrity validation.

Fails loudly: `validate_dataset` raises `DataValidationError` rather than
silently continuing when the data is unsafe to train on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class DataValidationError(Exception):
    """Raised when a dataset fails an integrity check."""


@dataclass
class ValidationReport:
    n_samples: int
    n_features: int
    dtype_X: str
    dtype_y: str
    n_nan: int
    n_inf: int
    n_duplicate_rows: int
    class_counts: dict[int, int]
    class_balance: dict[int, float]
    feature_means: list[float]
    feature_stds: list[float]
    feature_mins: list[float]
    feature_maxs: list[float]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "dtype_X": self.dtype_X,
            "dtype_y": self.dtype_y,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_duplicate_rows": self.n_duplicate_rows,
            "class_counts": self.class_counts,
            "class_balance": self.class_balance,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_mins": self.feature_mins,
            "feature_maxs": self.feature_maxs,
            "warnings": self.warnings,
        }


def validate_dataset(
    X: np.ndarray, y: np.ndarray, *, raise_on_error: bool = True
) -> ValidationReport:
    """Validate shape, dtype, NaN/Inf, label range, duplicates, and class balance.

    Raises `DataValidationError` on fatal issues (NaN/Inf present, shape
    mismatch, empty dataset) when `raise_on_error=True`. Non-fatal findings
    (e.g. a duplicate row count, mild class imbalance) are recorded as
    warnings in the returned report rather than raising.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if X.ndim != 2:
        errors.append(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        errors.append(f"y must be 1D, got shape {y.shape}")
    if X.ndim == 2 and y.ndim == 1 and X.shape[0] != y.shape[0]:
        errors.append(f"X has {X.shape[0]} rows but y has {y.shape[0]} labels")
    if X.shape[0] == 0:
        errors.append("Dataset is empty (0 samples)")

    if errors:
        raise DataValidationError("; ".join(errors))

    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0:
        errors.append(f"X contains {n_nan} NaN values")
    if n_inf > 0:
        errors.append(f"X contains {n_inf} infinite values")

    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(np.mod(y, 1), 0)):
            errors.append("y contains non-integer class labels")

    if errors and raise_on_error:
        raise DataValidationError("; ".join(errors))

    classes, counts = np.unique(y, return_counts=True)
    class_counts = {int(c): int(n) for c, n in zip(classes, counts)}
    total = int(counts.sum())
    class_balance = {int(c): n / total for c, n in zip(classes, counts)}

    min_balance = min(class_balance.values()) if class_balance else 0.0
    if min_balance < 0.05:
        warnings.append(f"Severe class imbalance detected (min class fraction={min_balance:.3f})")

    n_duplicate_rows = int(X.shape[0] - np.unique(X, axis=0).shape[0])
    if n_duplicate_rows > 0:
        warnings.append(f"{n_duplicate_rows} duplicate feature rows detected")

    report = ValidationReport(
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        dtype_X=str(X.dtype),
        dtype_y=str(y.dtype),
        n_nan=n_nan,
        n_inf=n_inf,
        n_duplicate_rows=n_duplicate_rows,
        class_counts=class_counts,
        class_balance=class_balance,
        feature_means=X.mean(axis=0).tolist(),
        feature_stds=X.std(axis=0).tolist(),
        feature_mins=X.min(axis=0).tolist(),
        feature_maxs=X.max(axis=0).tolist(),
        warnings=warnings,
    )
    return report
