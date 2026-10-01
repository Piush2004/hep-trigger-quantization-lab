# Reproducibility Report

## One-command reproduction

```bash
# Inside the project's WSL2/Linux Python 3.11 environment (see environment_report.md):
python -m hep_quant train --config configs/fp32.yaml
python -m hep_quant train --config configs/int8.yaml
python -m hep_quant train --config configs/int6.yaml
python -m hep_quant train --config configs/int4.yaml
python -m hep_quant train --config configs/int8_qat.yaml
python -m hep_quant train --config configs/int6_qat.yaml
python -m hep_quant train --config configs/int4_qat.yaml
python -m hep_quant report
```

Each `train` invocation is fully determined by its config file plus the fixed seed
(42, set in `configs/base.yaml`); no manual steps or random choices happen in between.

## Captured per experiment

Every record in `experiments/results/<experiment_id>.json` includes a
`reproducibility` block with:

- `seed` — the seed used for Python/NumPy/PyTorch/CUDA and for the data split.
- `system.python_version`, `system.torch_version`, `system.cuda_version`,
  `system.gpu_name`, `system.platform` — captured live via
  `hep_quant.utils.system.get_system_info()` at the time the experiment ran.
- `system.git_commit` — the exact commit the code was at when the experiment ran.

## Environment used to produce the results in this repository

See [environment_report.md](../environment_report.md) for the full write-up. Summary:

| | |
|---|---|
| OS | Windows 11 Home, executed inside WSL2 Ubuntu 26.04.1 LTS |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8, NVIDIA GeForce RTX 5060 Laptop GPU |
| Seed | 42 (all experiments) |

## Verified reproducibility (automated tests)

- `tests/test_training.py::test_training_reproducible_with_same_seed` — two
  independent training runs with the same seed produce identical test accuracy.
- `tests/test_pipeline.py::test_full_pipeline_runs_twice_consistently` — the full
  synthetic pipeline (data → train → eval → quantize → eval) run twice end-to-end
  produces identical FP32 and quantized accuracy.
- `tests/test_quantization.py::test_quantization_reproducible` — PTQ run twice with
  the same calibration data produces bit-identical quantized weights.
- `tests/test_data.py::test_split_dataset_deterministic` — the train/val/test split
  itself is deterministic given the seed.

## Known sources of nondeterminism (documented, not hidden)

- `torch.backends.cudnn.deterministic = True` is requested in
  `hep_quant.utils.seeds.set_seed`, but cuDNN algorithm selection can still introduce
  small (sub-1e-6-level) nondeterminism on some GPU/driver combinations for certain
  op/shape combinations not covered by a deterministic kernel. This project's own
  reproducibility tests did not observe this in practice (see above), but it is not
  something `set_seed` can mathematically guarantee away on every GPU.
- QAT's activation-range EMA observer updates its running min/max incrementally
  during training, so results depend on batch order, which depends on the DataLoader's
  `shuffle=True` RNG — seeded, and therefore reproducible given the same seed, but
  worth naming explicitly since it's a different mechanism from PTQ's one-shot
  calibration.
