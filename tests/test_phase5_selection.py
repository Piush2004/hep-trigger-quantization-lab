"""Phase 5 gate: the selection rule must use validation scores only and exclude non-mixed rows.

Uses synthetic validation scores; never touches any dataset or checkpoint.
"""
import importlib.util
from pathlib import Path

import pandas as pd

GATE = Path(__file__).resolve().parents[1] / "scripts" / "phase5_mixed_precision_gate.py"


def _gate_module():
    spec = importlib.util.spec_from_file_location("phase5_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(scores: dict[str, float], int8_units: dict[str, int]) -> pd.DataFrame:
    rows = []
    for seed in (42, 123, 2026, 7, 99):
        rows.append({"seed": seed, "candidate": "FP32", "int8_units": 0, "mixed": False, "roc_auc": 0.99})
        rows.append({"seed": seed, "candidate": "INT4_base", "int8_units": 0, "mixed": False, "roc_auc": 0.80})
        for name, auc in scores.items():
            rows.append({"seed": seed, "candidate": name, "int8_units": int8_units[name], "mixed": True, "roc_auc": auc})
    return pd.DataFrame(rows)


def test_selects_highest_mean_validation_auc_among_mixed_only():
    gate = _gate_module()
    val = _rows({"A0_int8": 0.90, "L0A0_int8": 0.92, "L3_int8": 0.85}, {"A0_int8": 1, "L0A0_int8": 2, "L3_int8": 1})
    sel = gate.select(val)
    assert sel["selected"] == "L0A0_int8"
    # FP32 (0.99) and the INT4 base (0.80) must never be candidates for selection
    assert {r["candidate"] for r in sel["ranking"]} == {"A0_int8", "L0A0_int8", "L3_int8"}


def test_tie_break_prefers_fewer_int8_units_then_name():
    gate = _gate_module()
    val = _rows({"A0_int8": 0.90, "L0_int8": 0.90, "L0A0_int8": 0.90}, {"A0_int8": 1, "L0_int8": 1, "L0A0_int8": 2})
    sel = gate.select(val)
    assert sel["selected"] == "A0_int8"  # same AUC, same #INT8 units as L0_int8 -> alphabetical


def test_selection_record_declares_validation_inputs():
    gate = _gate_module()
    sel = gate.select(_rows({"A0_int8": 0.9}, {"A0_int8": 1}))
    assert sel["selection_inputs"].startswith("validation split only")
    assert "include_test=False" in sel["selection_inputs"]
