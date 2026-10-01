"""Gate 1/2: data pipeline and synthetic dataset tests."""
import numpy as np
import pytest

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import check_no_leakage, split_dataset
from hep_quant.data.synthetic import generate_synthetic_dataset
from hep_quant.data.validation import DataValidationError, validate_dataset


def test_generate_synthetic_shape():
    X, y, names = generate_synthetic_dataset(n_samples=1000, n_features=12, n_classes=2, seed=42)
    assert X.shape == (1000, 12)
    assert y.shape == (1000,)
    assert len(names) == 12
    assert set(np.unique(y)) <= {0, 1}


def test_generate_synthetic_deterministic():
    X1, y1, _ = generate_synthetic_dataset(n_samples=500, n_features=10, seed=42)
    X2, y2, _ = generate_synthetic_dataset(n_samples=500, n_features=10, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_validate_dataset_passes_on_clean_data():
    X, y, _ = generate_synthetic_dataset(n_samples=1000, n_features=10, seed=42)
    report = validate_dataset(X, y)
    assert report.n_nan == 0
    assert report.n_inf == 0
    assert report.n_samples == 1000
    assert report.n_features == 10


def test_validate_dataset_rejects_nan():
    X, y, _ = generate_synthetic_dataset(n_samples=100, n_features=5, seed=42)
    X[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        validate_dataset(X, y)


def test_validate_dataset_rejects_inf():
    X, y, _ = generate_synthetic_dataset(n_samples=100, n_features=5, seed=42)
    X[0, 0] = np.inf
    with pytest.raises(DataValidationError):
        validate_dataset(X, y)


def test_validate_dataset_rejects_shape_mismatch():
    X, y, _ = generate_synthetic_dataset(n_samples=100, n_features=5, seed=42)
    with pytest.raises(DataValidationError):
        validate_dataset(X, y[:-1])


def test_split_dataset_proportions():
    X, y, _ = generate_synthetic_dataset(n_samples=2000, n_features=10, seed=42)
    splits = split_dataset(X, y, val_fraction=0.15, test_fraction=0.15, seed=42)
    total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total == 2000
    assert abs(len(splits.X_test) / total - 0.15) < 0.02
    assert abs(len(splits.X_val) / total - 0.15) < 0.02


def test_split_dataset_no_leakage():
    X, y, _ = generate_synthetic_dataset(n_samples=2000, n_features=10, seed=42)
    splits = split_dataset(X, y, seed=42)
    assert check_no_leakage(splits) == 0


def test_split_dataset_deterministic():
    X, y, _ = generate_synthetic_dataset(n_samples=1000, n_features=10, seed=42)
    splits1 = split_dataset(X, y, seed=42)
    splits2 = split_dataset(X, y, seed=42)
    assert np.array_equal(splits1.X_train, splits2.X_train)
    assert np.array_equal(splits1.y_test, splits2.y_test)


def test_split_dataset_stratified():
    X, y, _ = generate_synthetic_dataset(n_samples=2000, n_features=10, n_classes=2, seed=42)
    splits = split_dataset(X, y, seed=42)
    full_balance = y.mean()
    for split_y in (splits.y_train, splits.y_val, splits.y_test):
        assert abs(split_y.mean() - full_balance) < 0.05


def test_preprocessor_fit_transform():
    X, y, _ = generate_synthetic_dataset(n_samples=1000, n_features=10, seed=42)
    splits = split_dataset(X, y, seed=42)
    pre = Preprocessor.fit(splits.X_train)
    X_train_scaled = pre.transform(splits.X_train)
    assert abs(X_train_scaled.mean()) < 0.1
    assert abs(X_train_scaled.std() - 1.0) < 0.1


def test_load_synthetic_dataset():
    ds = load_synthetic(n_samples=500, n_features=8, seed=42)
    assert ds.X.shape == (500, 8)
    report = ds.validate()
    assert report.n_nan == 0
