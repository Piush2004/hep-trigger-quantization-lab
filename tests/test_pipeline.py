"""Full synthetic pipeline: data -> preprocessing -> train -> eval -> quantize -> eval.

Run twice in the same test to satisfy the two-test reproducibility rule for
this milestone (Gate 2 + Gate 12).
"""
import numpy as np

from hep_quant.data.loader import load_synthetic
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import check_no_leakage, split_dataset
from hep_quant.models.mlp import MLPClassifier
from hep_quant.quantization.ptq import quantize_model_ptq
from hep_quant.training.evaluation import evaluate_model
from hep_quant.training.trainer import Trainer
from hep_quant.utils.seeds import set_seed


def _run_full_pipeline(tmp_path):
    set_seed(42)
    ds = load_synthetic(n_samples=1500, n_features=12, seed=42)
    ds.validate()

    splits = split_dataset(ds.X, ds.y, seed=42)
    assert check_no_leakage(splits) == 0

    pre = Preprocessor.fit(splits.X_train)
    X_train, X_val, X_test = (pre.transform(x) for x in (splits.X_train, splits.X_val, splits.X_test))

    model = MLPClassifier(input_dim=12, hidden_dims=[32, 16], n_classes=2)
    trainer = Trainer(model, {"epochs": 4, "batch_size": 128, "learning_rate": 0.01, "device": "cpu"}, tmp_path)
    trainer.fit(X_train, splits.y_train, X_val, splits.y_val)

    fp32_result = evaluate_model(model, X_test, splits.y_test, device="cpu")
    assert np.isfinite(fp32_result.metrics.accuracy)

    quantized = quantize_model_ptq(model, "int8", X_train[:200])
    quant_result = evaluate_model(quantized, X_test, splits.y_test, device="cpu")
    assert np.isfinite(quant_result.metrics.accuracy)

    return fp32_result.metrics.accuracy, quant_result.metrics.accuracy


def test_full_pipeline_runs_twice_consistently(tmp_path):
    acc1, qacc1 = _run_full_pipeline(tmp_path / "run1")
    acc2, qacc2 = _run_full_pipeline(tmp_path / "run2")
    assert abs(acc1 - acc2) < 1e-6
    assert abs(qacc1 - qacc2) < 1e-6


def test_fp32_finetune_control_runs_and_records_method(tmp_path):
    """The compute-matched QAT control must train+fine-tune in FP32 and record method='fp32_finetune'."""
    import yaml

    from hep_quant.experiments.runner import run_experiment

    cfg = {
        "seed": 1, "experiment": {"id": "EXP_TEST_FT"},
        "dataset": {"name": "synthetic", "n_samples": 800, "n_features": 8, "n_classes": 2, "val_fraction": 0.15, "test_fraction": 0.15},
        "model": {"type": "mlp", "hidden_dims": [8], "dropout": 0.0, "activation": "relu"},
        "training": {"epochs": 4, "batch_size": 128, "learning_rate": 0.01, "device": "cpu", "early_stopping_patience": 3},
        "quantization": {"method": "fp32_finetune", "precision": "fp32"},
        "hardware": {"backend": "software_estimate"},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rec = run_experiment(str(path), results_dir=str(tmp_path / "r"), checkpoint_dir=str(tmp_path / "c"), figures_dir=str(tmp_path / "f"))
    assert rec["status"] == "completed"
    assert rec["model_info"]["method"] == "fp32_finetune" and rec["model_info"]["precision"] == "fp32"
