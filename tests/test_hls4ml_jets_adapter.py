"""Schema enforcement of the dedicated hls4ml jet-tagging adapter (no network)."""
import numpy as np
import pytest

from hep_quant.data.hls4ml_jets import EXPECTED_CLASSES, EXPECTED_FEATURES, FEATURE_DOCS, SchemaMismatchError, check_schema
from hep_quant.data.loader import HEPDataset


def _ds(features=None, classes=None, n_cols=16):
    X = np.zeros((10, n_cols), dtype=np.float32)
    return HEPDataset(X, np.zeros(10, dtype=np.int64), features or list(EXPECTED_FEATURES), "test", class_names=classes or list(EXPECTED_CLASSES))


def test_documented_schema_has_16_unique_features_and_5_classes():
    assert len(EXPECTED_FEATURES) == 16 == len(set(EXPECTED_FEATURES)) == len(FEATURE_DOCS)
    assert EXPECTED_CLASSES == ["g", "q", "t", "w", "z"]


def test_matching_schema_passes():
    check_schema(_ds())


def test_wrong_feature_names_fail_loudly():
    bad = list(EXPECTED_FEATURES)
    bad[0] = "something_else"
    with pytest.raises(SchemaMismatchError):
        check_schema(_ds(features=bad))


def test_wrong_classes_fail_loudly():
    with pytest.raises(SchemaMismatchError):
        check_schema(_ds(classes=["a", "b", "c", "d", "e"]))
