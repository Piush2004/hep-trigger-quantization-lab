"""End-to-end experiment orchestration: data -> model -> train -> (quantize) -> evaluate -> record."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hep_quant.config import apply_overrides, load_config
from hep_quant.data.hls4ml_jets import load_hls4ml_jets
from hep_quant.data.loader import load_dataset, load_openml
from hep_quant.data.preprocessing import Preprocessor
from hep_quant.data.splits import check_no_leakage, split_dataset
from hep_quant.experiments.registry import ExperimentRegistry, create_experiment_record
from hep_quant.hardware.latency import benchmark_inference_latency
from hep_quant.hardware.resource_estimator import get_hardware_backend
from hep_quant.models.factory import build_model
from hep_quant.quantization.ptq import quantize_model_ptq
from hep_quant.quantization.analysis import estimate_model_size
from hep_quant.quantization.qat import prepare_qat_model
from hep_quant.training.evaluation import evaluate_model
from hep_quant.training.metrics import confusion_matrix, pr_curve_points, roc_curve_points
from hep_quant.training.trainer import Trainer, resolve_device
from hep_quant.utils.logging import get_logger
from hep_quant.utils.seeds import set_seed
from hep_quant.visualization.plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_training_curves,
)

logger = get_logger(__name__)


def run_experiment(
    config_path: str,
    overrides: list[str] | None = None,
    results_dir: str = "experiments/results",
    checkpoint_dir: str = "experiments/checkpoints",
    figures_dir: str = "reports/figures",
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    experiment_id = cfg.get("experiment", {}).get("id", Path(config_path).stem)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    logger.info(f"=== Running experiment {experiment_id} ===")

    try:
        dataset_cfg = cfg["dataset"]
        source = dataset_cfg.get("source", "synthetic")
        if source == "hls4ml_jets":
            ds = load_hls4ml_jets(subset_size=dataset_cfg.get("subset_size"), seed=seed)
        elif source == "openml":
            ds = load_openml(
                name=dataset_cfg.get("openml_name", "hls4ml_lhc_jets_hlf"),
                version=dataset_cfg.get("openml_version", 1),
                subset_size=dataset_cfg.get("subset_size"),
                seed=seed,
            )
        else:
            ds = load_dataset(
                dataset_cfg.get("path"),
                n_samples=dataset_cfg.get("n_samples", 20000),
                n_features=dataset_cfg.get("n_features", 16),
                n_classes=dataset_cfg.get("n_classes", 2),
                seed=seed,
            )
        ds.validate()  # raises loudly on NaN/Inf/shape issues

        splits = split_dataset(
            ds.X, ds.y,
            val_fraction=dataset_cfg.get("val_fraction", 0.15),
            test_fraction=dataset_cfg.get("test_fraction", 0.15),
            seed=seed,
        )
        leakage = check_no_leakage(splits)
        if leakage > 0:
            raise RuntimeError(f"Train/val/test leakage detected: {leakage} overlapping rows")

        pre = Preprocessor.fit(splits.X_train)
        X_train = pre.transform(splits.X_train)
        X_val = pre.transform(splits.X_val)
        X_test = pre.transform(splits.X_test)

        n_classes = int(len(np.unique(ds.y)))
        model = build_model(cfg["model"], input_dim=X_train.shape[1], n_classes=n_classes)

        training_cfg = cfg["training"]
        device = resolve_device(training_cfg.get("device", "auto"))
        ckpt_dir = Path(checkpoint_dir) / experiment_id
        trainer = Trainer(model, training_cfg, ckpt_dir)

        quant_cfg = cfg.get("quantization", {"method": "none", "precision": "fp32"})
        method = quant_cfg.get("method", "none")
        precision = quant_cfg.get("precision", "fp32")

        if method == "qat" and precision != "fp32":
            fp32_model = build_model(cfg["model"], input_dim=X_train.shape[1], n_classes=n_classes)
            fp32_trainer = Trainer(fp32_model, training_cfg, ckpt_dir / "fp32_base")
            fp32_trainer.fit(X_train, splits.y_train, X_val, splits.y_val)
            model = prepare_qat_model(fp32_model, precision)
            qat_epochs = max(5, training_cfg.get("epochs", 40) // 4)
            qat_cfg = {**training_cfg, "epochs": qat_epochs, "learning_rate": training_cfg.get("learning_rate", 0.001) * 0.1}
            trainer = Trainer(model, qat_cfg, ckpt_dir)
            history = trainer.fit(X_train, splits.y_train, X_val, splits.y_val)
        else:
            history = trainer.fit(X_train, splits.y_train, X_val, splits.y_val)
            if method == "ptq" and precision != "fp32":
                calib_n = quant_cfg.get("calibration_batches", 20) * training_cfg.get("batch_size", 512)
                model = quantize_model_ptq(model, precision, X_train[: min(calib_n, len(X_train))], device=device)

        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        history.save(Path(checkpoint_dir) / experiment_id / "training_history.json")
        plot_training_curves(history.to_dict(), Path(figures_dir) / f"{experiment_id}_training_curves.png")

        eval_result = evaluate_model(model, X_test, splits.y_test, device=device)
        # ROC/PR *curve* plots are binary-only (roc_curve/precision_recall_curve
        # need a single positive-class probability vector); multiclass still
        # gets its macro ROC-AUC/PR-AUC scalars in metrics_out, just not these
        # two curve figures. See hep_quant.training.metrics docstring.
        if eval_result.metrics.roc_auc is not None and n_classes == 2:
            fpr, tpr, _ = roc_curve_points(eval_result.y_true, eval_result.y_prob)
            plot_roc_curve(fpr, tpr, eval_result.metrics.roc_auc, Path(figures_dir) / f"{experiment_id}_roc.png")
            precision_c, recall_c, _ = pr_curve_points(eval_result.y_true, eval_result.y_prob)
            plot_pr_curve(precision_c, recall_c, eval_result.metrics.pr_auc, Path(figures_dir) / f"{experiment_id}_pr.png")
        cm = confusion_matrix(eval_result.y_true, eval_result.y_pred)
        class_labels = ds.class_names if ds.class_names else [str(c) for c in range(n_classes)]
        plot_confusion_matrix(cm, class_labels, Path(figures_dir) / f"{experiment_id}_confusion.png")

        size_report = estimate_model_size(model, precision)
        latency_report = benchmark_inference_latency(model, input_dim=X_train.shape[1], device=device)

        hardware_backend = get_hardware_backend(cfg.get("hardware", {"backend": "unavailable"}))
        hardware_report = hardware_backend.analyze(model, precision, cfg.get("hardware", {}))

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": X_train.shape[1],
                "hidden_dims": getattr(model, "hidden_dims", cfg["model"].get("hidden_dims")),
                "n_classes": n_classes,
            },
            ckpt_dir / "best_model.pt",
        )

        metrics_out = {
            "test": eval_result.metrics.to_dict(),
            "model_size": {
                "estimated_size_mb": size_report.estimated_size_mb,
                "total_params": size_report.total_params,
                "is_estimated": size_report.is_estimated,
            },
            "latency": {
                "device": latency_report.device,
                "mean_ms_per_sample": latency_report.mean_ms_per_sample,
                "p95_ms": latency_report.p95_ms,
            },
        }
        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        record = create_experiment_record(
            experiment_id=experiment_id,
            config=cfg,
            metrics=metrics_out,
            model_info={
                "architecture": cfg["model"].get("type", "mlp"),
                "hidden_dims": cfg["model"].get("hidden_dims"),
                "n_parameters": size_report.total_params,
                "method": method,
                "precision": precision,
            },
            hardware=hardware_report.to_dict(),
            status="completed",
        )
    except Exception as exc:
        logger.error(f"Experiment {experiment_id} failed: {exc}")
        record = create_experiment_record(
            experiment_id=experiment_id, config=cfg, metrics={}, model_info={}, status="failed", error=str(exc)
        )
        registry = ExperimentRegistry(results_dir)
        registry.save(record)
        raise

    registry = ExperimentRegistry(results_dir)
    saved_path = registry.save(record)
    logger.info(f"Experiment {experiment_id} completed. Record saved to {saved_path}")
    return record.to_dict()
