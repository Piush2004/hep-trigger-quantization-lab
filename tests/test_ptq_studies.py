"""The sensitivity-study PTQ builder must reproduce the project's own PTQ exactly when units are uniform."""
import numpy as np
import pytest
import torch

from hep_quant.analysis.ptq_studies import build_ptq_variant, theoretical_weight_storage_bytes, uniform_units, unit_names
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.ptq import quantize_model_ptq


def _model_and_data():
    torch.manual_seed(0)
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8, 8], n_classes=5, dropout=0.0).eval()
    X = np.random.default_rng(0).normal(size=(300, 10)).astype(np.float32)
    return model, X


@pytest.mark.parametrize("bits", [8, 6, 4])
def test_uniform_variant_matches_quantize_model_ptq(bits):
    model, X = _model_and_data()
    ref = quantize_model_ptq(model, f"int{bits}", X[:200])
    wb, ab = uniform_units(model, bits)
    var = build_ptq_variant(model, X[:200], wb, ab)
    x = torch.from_numpy(X[200:])
    with torch.no_grad():
        assert torch.equal(ref(x), var(x))


def test_none_everywhere_is_identity():
    model, X = _model_and_data()
    var = build_ptq_variant(model, X[:200], {}, {})
    x = torch.from_numpy(X[200:])
    with torch.no_grad():
        assert torch.allclose(model(x), var(x))


def test_single_unit_changes_only_that_unit():
    model, X = _model_and_data()
    linears, acts = unit_names(model)
    assert len(linears) == 4 and len(acts) == 3
    var = build_ptq_variant(model, X[:200], {1: 4}, {})
    changed = [i for i, n in enumerate(linears)
               if not torch.equal(model.net.get_submodule(n).weight, var.net.get_submodule(n).weight)]
    assert changed == [1]


def test_mixed_precision_storage_between_extremes():
    model, _ = _model_and_data()
    lo = theoretical_weight_storage_bytes(model, uniform_units(model, 4)[0])
    hi = theoretical_weight_storage_bytes(model, uniform_units(model, 8)[0])
    mixed = theoretical_weight_storage_bytes(model, {0: 8, 1: 4, 2: 4, 3: 8})
    assert lo < mixed < hi
