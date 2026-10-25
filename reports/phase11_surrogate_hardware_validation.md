# Phase 11: Surrogate Hardware-Resource Validation

## 1. Objective

`SoftwareEstimateBackend` (the project's only always-available hardware-resource estimator)
is an analytical heuristic (`dsp = ceil(weight_params/reuse_factor)`, `lut = dsp * bits * 1.5`,
`ff = total_params * bits`, `bram = 0`), explicitly documented as `validated_against_synthesis:
False`. No Vivado/Vitis HLS is installed in this environment (confirmed `docs/REPRODUCIBILITY.md`
section 8, checked 2026-10-05), so real synthesis-grade numbers cannot be produced directly.

This phase asks: can a hardware-resource estimate be grounded in *real* hls4ml synthesis data —
not this project's own (unavailable), but a large public corpus of real synthesis records —
rather than a hand-derived arithmetic rule? And if so, how good is it, honestly measured?

## 2. Data source and provenance

[`fastmachinelearning/wa-hls4ml`](https://huggingface.co/datasets/fastmachinelearning/wa-hls4ml)
(Hugging Face Hub dataset, arXiv:2511.05615). License: **CC-BY-NC-4.0** (non-commercial) —
disclosed here; the trained surrogate inherits this restriction and is not presented as freely
redistributable.

Raw format confirmed by direct download and inspection (not assumed): flat JSON lists, one
record per synthesized network, grouped by model-type (`2layer`, `3layer`, `conv1d`, `conv2d`,
`resource`, `latency`, `2_20`, `exemplar`). This project used only the **`resource`** and
**`latency`** fully-connected splits — never `2layer`/`3layer`/`conv1d`/`conv2d`/`2_20`, which
the dataset card itself flags as mostly tiny, unactivated, non-representative networks — plus
the small `exemplar` split for an architecture-matched validation check (124 of its 887 records
are a "Jet"-family LHC-jet-tagging MLP, the same task family as this project's own model).

Confirmed record schema (8 top-level keys; no `hls_resource_report` field, correcting an earlier,
unverified assumption): `meta_data`, `model_config` (per-layer list), `hls_config` (per-layer
precision/reuse-factor/strategy), `resource_report` (`lut`/`dsp`/`bram`/`ff`/`uram`),
`latency_report` (`cycles_min`/`cycles_max`/`target_clock`/`interval_min`/`interval_max`),
`target_part`, `vivado_version`, `hls4ml_version`.

Downloaded volume: `exemplar` (6 MB, 887 records), `train`+`val` resource+latency (43,042
records after streaming/schema-validating), `test` resource+latency (7,598 held-out records,
wa-hls4ml's own split, not re-split by this project).

## 3. Feature design and model (pre-declared before training)

Features are restricted to what **both** a wa-hls4ml record and this project's own live
`MLPClassifier` can supply (board/Vivado-version/hls4ml-version are recorded as training-data
provenance, not model features, since `HLSBackend`/`SoftwareEstimateBackend` give no way to
specify them): `n_dense_layers`, `total_params`, `reuse_factor`, `strategy_is_resource`
(0/1), `target_clock_ns`, and per-Dense-layer `width_slot_0..6` / `bits_slot_0..6` (zero-padded
to 7 slots). One regressor per target (`lut`, `dsp`, `bram`, `ff`, `latency_ns`, `ii`) —
`RandomForestRegressor(n_estimators=200)`, declared before training.

## 4. Held-out results

**Bulk held-out (wa-hls4ml's own test split, n=5,364 after schema filtering):**

| target | R² | MAE |
|---|---|---|
| lut | 0.559 | 63,485 |
| dsp | 0.189 | 861 |
| bram | 0.842 | 1.1 |
| ff | 0.255 | 54,715 |
| latency_ns | 0.944 | 1,647 |
| ii | 0.996 | 15.4 |

**Jet-family exemplar held-out (architecture-matched, but small-n = 124 — reported separately,
never pooled with the bulk metric):**

| target | R² | MAE |
|---|---|---|
| lut | **-2.211** | 24,310 |
| dsp | 0.234 | 329 |
| bram | **-1.491** | 1.8 |
| ff | **-3.163** | 17,618 |
| latency_ns | 0.427 | 3,595 |
| ii | 0.450 | 187 |

Negative R² means worse than predicting the mean. **Latency and II transfer reasonably to the
Jet family; raw resource counts (LUT/BRAM/FF) do not.**

## 5. A material limitation, found and verified, not assumed

Inspecting feature importances (for both the RandomForestRegressor used in the saved model,
and a GradientBoostingRegressor trained as a robustness check) shows every target's importance
split **entirely** between `total_params` and `reuse_factor` — `bits_slot_*`, `width_slot_*`,
`n_dense_layers`, `strategy_is_resource`, and `target_clock_ns` all have **exactly zero**
importance, for every target, under both model types. This was verified directly (not
inferred): two feature vectors for the same architecture at FP32 vs. INT8 vs. INT4 differ only
in `bits_slot_*` (confirmed by printing them), yet the trained surrogate returns the **same**
LUT/DSP/FF/BRAM prediction for all three, applied to Phase 6's real checkpoints (see section 6).

Why: `total_params` (a literal parameter count, confirmed not bit-width-scaled) and
`reuse_factor` already explain nearly all variance in resource/latency across the *extremely*
heterogeneous set of architectures in wa-hls4ml's bulk splits (networks of wildly different
sizes and depths). Whatever residual variance bit-width contributes *within* a fixed
architecture is small relative to the between-architecture variance the trees already capture,
so neither a bagged nor a boosted tree ensemble, under plain global-MSE-reduction splitting,
ever finds it worthwhile to split on bit-width. This is a known failure mode of training one
global tree ensemble across a dataset dominated by a much larger-effect confound — not a bug in
feature extraction (verified above) and not specific to the RandomForest choice.

**Practical consequence:** the v1 surrogate's LUT/DSP/FF/BRAM numbers should **not** be used to
compare precisions of the same architecture — exactly the comparison this project's whole
Pareto analysis is built around. `SoftwareEstimateBackend`'s cruder arithmetic heuristic, for
all its lack of real-synthesis grounding, is the only one of the two that scales with bit-width
at all. The surrogate's latency/II numbers, which transfer well on the bulk held-out set, remain
the more useful part of this phase's output. **Future work** (not pursued here, out of this
phase's scope): normalize targets by `total_params` before fitting, or add an explicit
bits×total_params interaction feature, so the tree ensembles are not free to ignore precision.

## 6. Applied to Phase 6's real configurations

Applied (read-only, via `load_checkpoint_model`, no retraining) to Phase 6's two architectures
— **A** (`hidden=[64,32,32]`, 4,389 params) and **B** (`hidden=[128,64,64,32]`, 16,837 params) —
both re-run on CPU specifically to avoid the documented CPU/CUDA INT4 divergence
(`docs/REPRODUCIBILITY.md` section 6), rather than Phase 2's CUDA-era checkpoints. **Correction
to an earlier planning assumption:** Phase 6 produced 5 configurations per architecture (FP32,
INT8/INT6/INT4 PTQ, INT4 QAT — no INT6/INT8 QAT), not 7 — so this is 2 architectures × 5 seeds
× 5 configurations = 50 rows (`reports/tables/phase11_surrogate_hardware.csv`), not the
originally planned "14 configs."

Mean LUT/DSP, heuristic vs. surrogate, by architecture and precision:

| architecture | precision | heuristic LUT | surrogate LUT | heuristic DSP | surrogate DSP |
|---|---|---|---|---|---|
| A | FP32 | 204,288 | 75,652 | 4,256 | 2,088 |
| A | INT8 PTQ | 51,072 | 75,652 | 4,256 | 2,088 |
| A | INT6 PTQ | 38,304 | 75,652 | 4,256 | 2,088 |
| A | INT4 PTQ/QAT | 25,536 | 75,652 | 4,256 | 2,088 |
| B | FP32 | 794,112 | 281,467 | 16,544 | 1,645 |
| B | INT8 PTQ | 198,528 | 281,467 | 16,544 | 1,645 |
| B | INT6 PTQ | 148,896 | 281,467 | 16,544 | 1,645 |
| B | INT4 PTQ/QAT | 99,264 | 281,467 | 16,544 | 1,645 |

As section 5 predicts: the surrogate's LUT/DSP columns are constant within an architecture
regardless of precision, while the heuristic's scale down with bit-width as designed. Neither
is validated against this project's own real synthesis (not available); the heuristic is shown
here as the complementary estimate it actually is, not a worse alternative.

## 7. Limitations

- No Vivado ground truth for *this project's own* architectures exists (unchanged from before
  this phase) — the surrogate is validated only against wa-hls4ml's held-out data and its
  small Jet-exemplar slice, not against a real synthesis of Phase 6's own networks.
- The surrogate does not currently use bit-width or width/depth information at all (section 5)
  — its LUT/DSP/FF/BRAM output should not be read as precision-sensitive.
- Jet-family exemplar held-out is small-n (124) — the negative R² values are informative but
  carry wide uncertainty; not a precise error bound.
- CC-BY-NC-4.0 training-data license — non-commercial research use only.

## 8. Reproducibility

- `scripts/phase11_train_surrogate.py` trains and saves `experiments/surrogate_models/v1/`
  (deterministic: fixed `random_state=42` for both model types; same train/test records every
  run since they come from wa-hls4ml's own fixed splits, not a random re-split).
- `scripts/phase11_apply_surrogate_hardware.py` applies the trained surrogate (read-only,
  `load_checkpoint_model`, no retraining) to Phase 6's 50 real configurations.
- `tests/test_surrogate_backend.py` (8 passed, 1 skipped, run twice — pytest's pass/skip counts
  matched exactly both times) covers schema validation, the Jet-exemplar filter, the
  feature-key-identity contract between the two feature-extraction entry points, and the
  backend's honest-fallback behavior when no trained model is present.

## 9. Reproduction

```bash
# WSL2 Ubuntu, .venv-wsl
pip install -e ".[surrogate]"
python scripts/phase11_train_surrogate.py
python scripts/phase11_apply_surrogate_hardware.py
python -m pytest -q tests/test_surrogate_backend.py
```
