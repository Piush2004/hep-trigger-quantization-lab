"""Controlled QAT variants for mixed-precision fine-tuning (the QAT analogue of ptq_studies.py).

`prepare_qat_variant` wires per-unit QATLinear/QATActivation (quantization/qat.py) instead of the
single uniform precision `prepare_qat_model` supports. `build_qat_variant` fine-tunes from a
trained FP32 base using the exact same epoch/LR formula the uniform QAT path in
hep_quant.experiments.runner uses (qat_epochs = max(5, epochs // 4), qat_lr = lr * 0.1) -- it
never trains from scratch.

Units for an MLP with hidden layers h1,h2,h3: L0..L3 = Linear weight tensors, A0..A2 = hidden
ReLU outputs -- same numbering as ptq_studies.build_ptq_variant.
"""
from __future__ import annotations

import copy

import numpy as np
import torch.nn as nn

from hep_quant.quantization.qat import QATActivation, QATLinear, _ACTIVATION_TYPES
from hep_quant.training.trainer import Trainer
from hep_quant.utils.seeds import set_seed


def prepare_qat_variant(model: nn.Module, weight_bits: dict[int, int | None], act_bits: dict[int, int | None]) -> nn.Module:
    """Return a copy of `model` with per-unit QATLinear/QATActivation.

    Missing/None entries stay the original (unquantized) module -- interface parity with
    build_ptq_variant, even though Phase 12's search grid has no None/FP32 option.
    """
    qat_model = copy.deepcopy(model)
    new_layers = []
    lin_idx = act_idx = 0
    for module in qat_model.net:
        if isinstance(module, nn.Linear):
            bits = weight_bits.get(lin_idx)
            new_layers.append(QATLinear(module, bits) if bits is not None else module)
            lin_idx += 1
        elif isinstance(module, _ACTIVATION_TYPES):
            bits = act_bits.get(act_idx)
            new_layers.append(QATActivation(module, bits) if bits is not None else module)
            act_idx += 1
        else:
            new_layers.append(module)
    qat_model.net = nn.Sequential(*new_layers)
    return qat_model


def build_qat_variant(
    fp32_model: nn.Module,
    weight_bits: dict[int, int | None],
    act_bits: dict[int, int | None],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_cfg: dict,
    checkpoint_dir: str,
    device: str = "cpu",
    seed: int = 42,
) -> nn.Module:
    """Fine-tune a per-unit QAT variant from a trained FP32 base. Never trains from scratch.

    Pins `seed` (Python/NumPy/PyTorch) before fine-tuning, matching the project-wide convention
    in hep_quant.experiments.runner -- without this, DataLoader shuffling and Dropout make
    repeated QAT fine-tunes of the same layout non-byte-identical (observed ~3e-4 ROC-AUC drift).
    """
    set_seed(seed)
    qat_model = prepare_qat_variant(fp32_model, weight_bits, act_bits)
    qat_epochs = max(5, training_cfg.get("epochs", 40) // 4)
    qat_cfg = {**training_cfg, "epochs": qat_epochs,
               "learning_rate": training_cfg.get("learning_rate", 0.001) * 0.1, "device": device}
    trainer = Trainer(qat_model, qat_cfg, checkpoint_dir)
    trainer.fit(X_train, y_train, X_val, y_val)
    return qat_model.eval()
