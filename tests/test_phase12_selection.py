"""Phase 12 gate: layout enumeration must be exhaustive/deterministic, selection must stay
validation-only, and the QAT per-unit builder must wire units correctly. No dataset or
checkpoint is touched here.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import torch

GATE = Path(__file__).resolve().parents[1] / "scripts" / "phase12_exhaustive_mixed_precision_pareto.py"


def _gate_module():
    spec = importlib.util.spec_from_file_location("phase12_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_layouts_is_exhaustive_and_unique():
    gate = _gate_module()
    seen = set()
    for w, a in gate.all_layouts():
        assert set(w) == {0, 1, 2, 3}
        assert set(a) == {0, 1, 2}
        assert all(b in gate.BIT_CHOICES for b in w.values())
        assert all(b in gate.BIT_CHOICES for b in a.values())
        seen.add(gate.layout_name(w, a))
    assert len(seen) == 3 ** 7 == 2187


def test_layout_name_is_deterministic_and_unique():
    gate = _gate_module()
    names = [gate.layout_name(w, a) for w, a in gate.all_layouts()]
    assert len(names) == len(set(names)) == 2187


def _confirm_rows(storage_by_layout: dict[str, float], auc_by_layout: dict[str, float], seeds=(42, 123, 2026, 7, 99)):
    rows = []
    for s in seeds:
        for name, storage in storage_by_layout.items():
            rows.append({"seed": s, "layout": name, "roc_auc": auc_by_layout[name],
                         "theoretical_weight_storage_bytes": storage})
    return pd.DataFrame(rows)


def test_select_picks_highest_auc_within_size_limit():
    gate = _gate_module()
    confirm = _confirm_rows(
        {"big": 5000.0, "small_good": 3000.0, "small_bad": 2900.0},
        {"big": 0.95, "small_good": 0.92, "small_bad": 0.85},
    )
    sel = gate.select(confirm, limit=3724.0)
    assert sel["selected_layout"] == "small_good"
    assert sel["selection_used_test_set"] is False


def test_select_tie_break_prefers_smaller_storage_then_name():
    gate = _gate_module()
    confirm = _confirm_rows(
        {"B_layout": 3000.0, "A_layout": 2800.0, "C_layout": 2800.0},
        {"B_layout": 0.90, "A_layout": 0.90, "C_layout": 0.90},
    )
    sel = gate.select(confirm, limit=3724.0)
    assert sel["selected_layout"] == "A_layout"  # same AUC+storage as C_layout -> alphabetical


def test_select_reports_no_eligible_configuration():
    gate = _gate_module()
    confirm = _confirm_rows({"too_big": 9000.0}, {"too_big": 0.99})
    sel = gate.select(confirm, limit=3724.0)
    assert sel["selected_layout"] is None
    assert sel["status"] == "NO ELIGIBLE CONFIGURATION"


def test_size_limit_bytes_matches_hand_computed_value():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hep_quant.models.mlp import MLPClassifier

    gate = _gate_module()

    class _P5Stub:
        @staticmethod
        def uniform_units(model, bits):
            from hep_quant.analysis.ptq_studies import uniform_units
            return uniform_units(model, bits)

        @staticmethod
        def theoretical_weight_storage_bytes(model, weight_bits):
            from hep_quant.analysis.ptq_studies import theoretical_weight_storage_bytes
            return theoretical_weight_storage_bytes(model, weight_bits)

    model = MLPClassifier(input_dim=16, hidden_dims=[8, 4, 4], n_classes=5)
    got = gate.size_limit_bytes(model, _P5Stub())

    expected = 0.0
    for lin in [m for m in model.net if isinstance(m, torch.nn.Linear)]:
        expected += lin.weight.numel() * 6 / 8 + lin.bias.numel() * 4
    assert got == expected
