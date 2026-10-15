"""Phase 12 QAT extension: prepare_qat_variant must wire the right module type and bit-width
per unit. No training loop here -- structural correctness only, via a single forward pass.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hep_quant.analysis.qat_studies import prepare_qat_variant
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.qat import QATActivation, QATLinear


def _tiny_model():
    return MLPClassifier(input_dim=16, hidden_dims=[8, 4, 4], n_classes=5)


def test_prepare_qat_variant_wires_per_unit_types_and_bits():
    model = _tiny_model()
    w = {0: 8, 1: 4, 2: 6, 3: 8}
    a = {0: 4, 1: 8, 2: 6}
    qat_model = prepare_qat_variant(model, w, a)

    linears = [m for m in qat_model.net if isinstance(m, (nn.Linear, QATLinear))]
    acts = [m for m in qat_model.net if isinstance(m, (nn.ReLU, QATActivation))]
    assert len(linears) == 4
    assert len(acts) == 3
    for i, lin in enumerate(linears):
        assert isinstance(lin, QATLinear)
        assert lin.bits == w[i]
    for i, act in enumerate(acts):
        assert isinstance(act, QATActivation)
        assert act.bits == a[i]


def test_prepare_qat_variant_leaves_none_units_unquantized():
    model = _tiny_model()
    w = {0: None, 1: 4, 2: 4, 3: 4}
    a = {0: None, 1: 4, 2: 4}
    qat_model = prepare_qat_variant(model, w, a)

    linears = [m for m in qat_model.net if isinstance(m, (nn.Linear, QATLinear))]
    acts = [m for m in qat_model.net if isinstance(m, (nn.ReLU, QATActivation))]
    assert isinstance(linears[0], nn.Linear) and not isinstance(linears[0], QATLinear)
    assert isinstance(acts[0], nn.ReLU) and not isinstance(acts[0], QATActivation)
    assert isinstance(linears[1], QATLinear)
    assert isinstance(acts[1], QATActivation)


def test_prepare_qat_variant_forward_pass_runs():
    model = _tiny_model()
    w = {0: 8, 1: 4, 2: 6, 3: 8}
    a = {0: 4, 1: 8, 2: 6}
    qat_model = prepare_qat_variant(model, w, a)
    x = torch.randn(5, 16)
    out = qat_model(x)
    assert out.shape == (5, 5)


def test_prepare_qat_variant_does_not_mutate_original_model():
    model = _tiny_model()
    original_linears = [m for m in model.net if isinstance(m, nn.Linear)]
    prepare_qat_variant(model, {0: 8, 1: 4, 2: 6, 3: 8}, {0: 4, 1: 8, 2: 6})
    assert all(isinstance(m, nn.Linear) and not isinstance(m, QATLinear) for m in original_linears)
