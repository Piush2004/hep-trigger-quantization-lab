"""Phase 14 baseline: hand-engineered per-image scalar feature extraction."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hep_quant.data.image_features import FEATURE_NAMES, extract_image_features


def test_output_shape_matches_feature_count():
    images = np.random.default_rng(0).random((5, 10, 10)).astype(np.float32)
    feats = extract_image_features(images)
    assert feats.shape == (5, len(FEATURE_NAMES))


def test_total_energy_matches_pixel_sum():
    images = np.random.default_rng(0).random((5, 10, 10)).astype(np.float32)
    feats = extract_image_features(images)
    np.testing.assert_allclose(feats[:, FEATURE_NAMES.index("total_energy")], images.sum(axis=(1, 2)))


def test_centroid_on_single_hot_pixel_matches_its_position():
    images = np.zeros((1, 10, 10), dtype=np.float32)
    images[0, 3, 7] = 5.0
    feats = extract_image_features(images)
    assert feats[0, FEATURE_NAMES.index("centroid_row")] == 3.0
    assert feats[0, FEATURE_NAMES.index("centroid_col")] == 7.0
    assert feats[0, FEATURE_NAMES.index("spread_row")] == 0.0
    assert feats[0, FEATURE_NAMES.index("spread_col")] == 0.0


def test_all_zero_image_does_not_crash_or_produce_nan():
    images = np.zeros((2, 10, 10), dtype=np.float32)
    feats = extract_image_features(images)
    assert np.all(np.isfinite(feats))
    assert feats[0, FEATURE_NAMES.index("centroid_row")] == 5.0  # image center, h/2


def test_nonzero_pixel_count_matches_manual_count():
    images = np.zeros((3, 4, 4), dtype=np.float32)
    images[0, 0, 0] = 1.0
    images[1, :2, :2] = 1.0
    feats = extract_image_features(images)
    idx = FEATURE_NAMES.index("nonzero_pixel_count")
    assert feats[0, idx] == 1
    assert feats[1, idx] == 4
    assert feats[2, idx] == 0
