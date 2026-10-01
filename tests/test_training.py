"""Gate 3: training loop, checkpointing, and reproducibility tests."""
import numpy as np
import torch

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import split_dataset
from hep_quant.models.mlp import MLPClassifier
from hep_quant.training.evaluation import evaluate_model
from hep_quant.training.trainer import Trainer
from hep_quant.utils.seeds import set_seed


def _make_splits():
    ds = load_synthetic(n_samples=1500, n_features=10, seed=42)
    splits = split_dataset(ds.X, ds.y, seed=42)
    pre = Preprocessor.fit(splits.X_train)
    splits.X_train = pre.transform(splits.X_train)
    splits.X_val = pre.transform(splits.X_val)
    splits.X_test = pre.transform(splits.X_test)
    return splits


def test_trainer_trains_without_crashing(tmp_path):
    set_seed(42)
    splits = _make_splits()
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    trainer = Trainer(model, {"epochs": 3, "batch_size": 128, "learning_rate": 0.01, "device": "cpu"}, tmp_path)
    history = trainer.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
    assert len(history.epochs) == 3
    assert all(np.isfinite(loss) for loss in history.train_loss)


def test_trainer_checkpoint_roundtrip(tmp_path):
    set_seed(42)
    splits = _make_splits()
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    trainer = Trainer(model, {"epochs": 2, "batch_size": 128, "learning_rate": 0.01, "device": "cpu"}, tmp_path)
    trainer.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)

    ckpt_path = tmp_path / "model.pt"
    trainer.save_checkpoint(ckpt_path)

    reloaded = Trainer.load_checkpoint(ckpt_path, MLPClassifier, device="cpu")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(splits.X_test[:16])
        original_out = model(x)
        reloaded_out = reloaded(x)

    assert torch.allclose(original_out, reloaded_out, atol=1e-6)


def test_evaluate_model_produces_finite_metrics():
    set_seed(42)
    splits = _make_splits()
    model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
    result = evaluate_model(model, splits.X_test, splits.y_test, device="cpu")
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert np.isfinite(result.inference_time_s)
    assert result.y_pred.shape == splits.y_test.shape


def test_training_reproducible_with_same_seed(tmp_path):
    """Two independent training runs with the same seed should reach the same val accuracy."""
    accuracies = []
    for _ in range(2):
        set_seed(42)
        splits = _make_splits()
        model = MLPClassifier(input_dim=10, hidden_dims=[16, 8], n_classes=2)
        trainer = Trainer(model, {"epochs": 3, "batch_size": 128, "learning_rate": 0.01, "device": "cpu"}, tmp_path)
        trainer.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        result = evaluate_model(model, splits.X_test, splits.y_test, device="cpu")
        accuracies.append(result.metrics.accuracy)

    assert abs(accuracies[0] - accuracies[1]) < 1e-6
