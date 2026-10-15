# Phase 14: Baseline Pre-Declaration (Image Data)

Written before any model is trained, following the same discipline as every prior gate
(Phase 5/5B/12's selection rules, Phase 13's baseline choice) — declared here, not tuned after
seeing results.

## Dataset and splits

- Official `train/` (hls4ml_LHCjet_30p, Zenodo 10.5281/zenodo.3601436) split internally 85/15
  into train/val via `hep_quant.data.splits.split_dataset` (seed=42, stratified).
- Official `val/` (270,000 jets) held out entirely as the test set, touched once, at the end.
- Images downsampled 100x100 → 25x25 via `load_and_downsample_jet_images` (sum-pooling,
  energy-conserving).

## Baselines (both pre-declared, both default-hyperparameter — no tuning, matching Phase 13's
"out-of-the-box, not best-case" discipline)

1. **Hand-engineered image features + `HistGradientBoostingClassifier`** (default hyperparameters).
   Features extracted per image (`hep_quant.data.image_features`): total energy, nonzero-pixel
   count, energy-weighted centroid (row, col), second-moment spread (row, col), max-pixel energy.
   This is the physics-honest baseline — analogous to how the existing tabular dataset's own 16
   features are hand-engineered substructure observables, not raw pixels.
2. **Flattened-pixel + `LogisticRegression`** (default hyperparameters, `max_iter=1000`). A pure
   "no physics knowledge" lower bound, directly analogous to Phase 13's own LogisticRegression
   baseline on raw features.

## Metric suite (identical to Phase 13, reused unmodified)

ROC-AUC (macro, one-vs-rest), F1 (macro), accuracy, background rejection at 70% signal efficiency
(macro one-vs-rest) via `hep_quant.training.metrics.macro_background_rejection_at_efficiency`.

## What counts as "the CNN earning its keep"

Phase 13's own precedent (a GBM baseline beat the FP32 MLP outright on the tabular task) means
there is no assumption the CNN wins here either. The CNN's result is reported against both
baselines above on the identical held-out `val/` split; whichever model actually scores highest
is reported as such, with no post-hoc reweighting of this rule.

## Status

Declared 2026-10-15, before any baseline or CNN training has run on this dataset.
