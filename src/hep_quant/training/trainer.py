"""Training loop with early stopping, checkpointing, and history logging."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hep_quant.training.evaluation import evaluate_model
from hep_quant.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_duration_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


class Trainer:
    def __init__(self, model: nn.Module, training_cfg: dict, checkpoint_dir: str | Path):
        self.model = model
        self.cfg = training_cfg
        self.device = resolve_device(training_cfg.get("device", "auto"))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = TrainingHistory()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> TrainingHistory:
        import time

        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.get("learning_rate", 0.001),
            weight_decay=self.cfg.get("weight_decay", 0.0),
        )
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.get("batch_size", 512), shuffle=True
        )

        patience = self.cfg.get("early_stopping_patience", 8)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(self.cfg.get("epochs", 40)):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            train_loss = running_loss / max(n_batches, 1)

            model.eval()
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).to(self.device)
                y_val_t = torch.from_numpy(y_val).to(self.device)
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()

            eval_result = evaluate_model(model, X_val, y_val, device=self.device)
            duration = time.perf_counter() - epoch_start

            self.history.epochs.append(epoch)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_accuracy.append(eval_result.metrics.accuracy)
            self.history.val_f1.append(eval_result.metrics.f1)
            self.history.val_roc_auc.append(eval_result.metrics.roc_auc or float("nan"))
            self.history.learning_rate.append(optimizer.param_groups[0]["lr"])
            self.history.epoch_duration_s.append(duration)

            logger.info(
                f"epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={eval_result.metrics.accuracy:.4f} val_f1={eval_result.metrics.f1:.4f}"
            )

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return self.history

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        # Writes model.get_config()'s keys at the top level (not nested under a
        # "model_config" key) so the on-disk format is unchanged for MLPClassifier --
        # every existing frozen checkpoint stays loadable by this same code path.
        checkpoint = {"model_state_dict": self.model.state_dict(), **self.model.get_config()}
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path: str | Path, model_cls, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = {k: checkpoint[k] for k in model_cls.CONFIG_KEYS if k in checkpoint}
        model = model_cls.from_config(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
