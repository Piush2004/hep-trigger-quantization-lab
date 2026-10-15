"""Phase 14: jet-image loader schema checks and the max_jets subsampling contract.

Uses a small slice of the real downloaded dataset (tests/fixtures/jet_image_shard.h5, a copy of
one real val shard trimmed to a handful of jets) rather than a hand-built fixture, since the
whole point of this loader is matching a specific real HDF5 schema -- a synthetic fixture could
silently drift from the real one. Real-download tests are gated behind RUN_NETWORK_TESTS=1, the
project's existing convention.
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hep_quant.data.jet_images import (
    CLASS_NAMES,
    SchemaMismatchError,
    load_and_downsample_jet_images,
    load_jet_images,
    sum_pool,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "jet_images"


def test_fixture_shard_loads_and_matches_schema():
    ds = load_jet_images(str(FIXTURE_DIR))
    assert ds.images.ndim == 3
    assert ds.images.shape[1:] == (100, 100)
    assert ds.images.dtype == np.float32
    assert ds.y.shape == (ds.images.shape[0],)
    assert set(np.unique(ds.y)).issubset(set(range(5)))
    assert ds.class_names == CLASS_NAMES


def test_max_jets_subsample_is_smaller_and_deterministic():
    ds = load_jet_images(str(FIXTURE_DIR))
    n_total = len(ds.y)
    half = max(n_total // 2, 1)
    a = load_jet_images(str(FIXTURE_DIR), max_jets=half, seed=42)
    b = load_jet_images(str(FIXTURE_DIR), max_jets=half, seed=42)
    assert len(a.y) == half
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.images, b.images)


def test_different_seeds_give_different_subsamples():
    ds = load_jet_images(str(FIXTURE_DIR))
    n_total = len(ds.y)
    half = max(n_total // 2, 1)
    if half >= n_total:
        pytest.skip("fixture too small to demonstrate seed difference")
    a = load_jet_images(str(FIXTURE_DIR), max_jets=half, seed=1)
    b = load_jet_images(str(FIXTURE_DIR), max_jets=half, seed=2)
    assert not np.array_equal(a.y, b.y) or not np.array_equal(a.images, b.images)


def test_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jet_images(str(tmp_path / "does_not_exist"))


def test_schema_mismatch_raises_on_bad_feature_names(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("jetImage", data=np.zeros((2, 100, 100), dtype=np.float64))
        f.create_dataset("jets", data=np.zeros((2, 59), dtype=np.float64))
        f.create_dataset("jetFeatureNames", data=np.array([f"col_{i}".encode() for i in range(59)]))
    with pytest.raises(SchemaMismatchError):
        load_jet_images(str(tmp_path))


def test_undef_label_present_raises(tmp_path):
    from hep_quant.data.jet_images import EXPECTED_LABEL_NAMES

    bad = tmp_path / "undef.h5"
    names = [f"col_{i}" for i in range(53)] + EXPECTED_LABEL_NAMES
    jets = np.zeros((1, 59), dtype=np.float64)
    jets[0, 58] = 1.0  # undef = 1
    with h5py.File(bad, "w") as f:
        f.create_dataset("jetImage", data=np.zeros((1, 100, 100), dtype=np.float64))
        f.create_dataset("jets", data=jets)
        f.create_dataset("jetFeatureNames", data=np.array([n.encode() for n in names]))
    with pytest.raises(SchemaMismatchError):
        load_jet_images(str(tmp_path))


def test_sum_pool_conserves_total_energy():
    rng = np.random.default_rng(0)
    images = rng.random((5, 100, 100)).astype(np.float32)
    pooled = sum_pool(images, factor=4)
    assert pooled.shape == (5, 25, 25)
    np.testing.assert_allclose(pooled.sum(axis=(1, 2)), images.sum(axis=(1, 2)), rtol=1e-5)


def test_sum_pool_rejects_non_divisible_factor():
    images = np.zeros((2, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        sum_pool(images, factor=7)


def test_load_and_downsample_produces_25x25_and_conserves_energy():
    raw = load_jet_images(str(FIXTURE_DIR))
    downsampled = load_and_downsample_jet_images(str(FIXTURE_DIR), target_size=25)
    assert downsampled.images.shape[1:] == (25, 25)
    np.testing.assert_allclose(
        downsampled.images.sum(axis=(1, 2)), raw.images.sum(axis=(1, 2)), rtol=1e-4
    )


def test_load_and_downsample_rejects_non_divisor_target_size():
    with pytest.raises(ValueError):
        load_and_downsample_jet_images(str(FIXTURE_DIR), target_size=32)


@pytest.mark.skipif(not os.environ.get("RUN_NETWORK_TESTS"), reason="requires the full downloaded dataset")
def test_real_val_split_is_balanced_and_undef_free():
    full_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "hls4ml_jet_images_30p" / "val"
    if not full_dir.exists():
        pytest.skip("full val dataset not downloaded in this environment")
    ds = load_jet_images(str(full_dir))
    counts = np.bincount(ds.y, minlength=5)
    fractions = counts / counts.sum()
    assert all(0.15 < f < 0.25 for f in fractions)
