"""hls4ml C-simulation accuracy helper (real conversion + compile; skipped if hls4ml/g++ missing)."""
import shutil

import numpy as np
import pytest
import torch

from hep_quant.models.mlp import MLPClassifier


def _usable() -> bool:
    try:
        import hls4ml  # noqa: F401
    except ImportError:
        return False
    return shutil.which("g++") is not None


@pytest.mark.skipif(not _usable(), reason="hls4ml and/or g++ unavailable")
def test_csim_wide_fixed_point_matches_float_predictions():
    from hep_quant.hardware.hls_csim import csim_accuracy_sweep

    torch.manual_seed(0)
    model = MLPClassifier(input_dim=16, hidden_dims=[32, 16], n_classes=5, dropout=0.0).eval()
    X = np.random.default_rng(0).normal(size=(400, 16)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(X)).argmax(dim=1).numpy()

    rows = csim_accuracy_sweep(model, X, ref, widths=[24], int_bits=8, reference_pred=ref)
    assert rows[0]["finite_outputs"]
    assert rows[0]["measurement_tier"] == "simulated"
    # 16 fractional bits is far finer than this tiny model needs -> predictions should (almost) all agree
    assert rows[0]["agreement_with_fp32_predictions"] >= 0.98
