"""Synthetic classification dataset generator.

Used to validate the engineering pipeline (loader -> preprocessing -> split ->
model -> training -> quantization) independently of any real HEP dataset, and
as a documented development subset when a real dataset is unavailable (see
reports/dataset_card.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification


def generate_synthetic_dataset(
    n_samples: int = 20000,
    n_features: int = 16,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic binary/multi-class classification dataset.

    Loosely mimics the shape of a HEP tabular classification problem
    (continuous kinematic-like features, moderately separable classes).
    Returns (X, y, feature_names).
    """
    n_informative = max(2, int(n_features * 0.6))
    n_redundant = max(0, int(n_features * 0.2))
    n_clusters_per_class = 2 if n_classes == 2 else 1

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        n_classes=n_classes,
        flip_y=0.02,
        class_sep=1.2,
        random_state=seed,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names
