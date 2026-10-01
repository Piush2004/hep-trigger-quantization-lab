"""Phase 2: real-dataset (OpenML) loader tests.

These download the real hls4ml jet-tagging dataset (~hundreds of MB on first
fetch, cached afterward by sklearn under data/raw/openml_cache). That's too
slow/network-dependent to run on every `pytest -q`, so these are opt-in via
RUN_NETWORK_TESTS=1 rather than skipped-when-unavailable like the hls4ml
tests -- the dependency here is "slow and uses bandwidth", not "may not be
installed".
"""
import os

import numpy as np
import pytest

from hep_quant.data.loader import load_openml

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="network/bandwidth-heavy OpenML fetch; set RUN_NETWORK_TESTS=1 to run",
)


def test_load_openml_jet_tagging_schema():
    ds = load_openml(subset_size=500, seed=42)
    assert ds.X.shape == (500, 16)
    assert ds.y.shape == (500,)
    assert ds.class_names is not None
    assert set(ds.class_names) == {"g", "q", "t", "w", "z"}
    assert ds.source.startswith("openml:hls4ml_lhc_jets_hlf")


def test_load_openml_stratified_subsample_preserves_balance():
    full = load_openml(subset_size=20000, seed=42)
    small = load_openml(subset_size=2000, seed=42)

    full_balance = np.bincount(full.y) / len(full.y)
    small_balance = np.bincount(small.y) / len(small.y)
    assert np.allclose(full_balance, small_balance, atol=0.05)


def test_load_openml_passes_validation():
    ds = load_openml(subset_size=500, seed=42)
    report = ds.validate()
    assert report.n_nan == 0
    assert report.n_inf == 0
    assert report.n_features == 16
