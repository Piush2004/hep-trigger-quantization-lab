# Phase 14: Jet-Image CNN — A Generality Check, Not a Second Headline Result

Framing note (confirmed with the user before writing this): the quantization/mixed-precision
work in Phases 1-13 is this project's core contribution. This phase asks a narrower, secondary
question — does the same "real data, baseline-before-model, honest limitations" methodology
hold up on a genuinely different architecture (CNN, not MLP) and dataset (pixelated calorimeter
images, not hand-engineered high-level features) — and is reported as exactly that, not as a
second co-equal result.

## 1. Objective

The README's actual "Future work" text (quoted precisely, not paraphrased) says a second
architecture is already done (Phase 6: two MLP shapes) and names a second **dataset** as the
remaining optional item, "only if a generality claim is wanted." This phase fulfills that
dataset item with a real particle-image companion to the existing tabular dataset — the same
underlying Pythia8 simulation, the same 5-class (g/q/w/z/t) label, confirmed by directly
inspecting the downloaded files, not assumed.

## 2. Dataset

Source: Zenodo 10.5281/zenodo.3601436, "HLS4ML LHC Jet dataset (30 particles)" (CC BY 4.0),
supplement to Moreno et al. 2020 ("JEDI-net", arXiv:1908.05318) and Duarte et al. 2018
(arXiv:1804.06913 — the same paper the existing tabular `hls4ml_lhc_jets_hlf` dataset itself
supplements). Confirmed HDF5 schema by direct inspection: each shard has a pre-rendered 100×100
`jetImage` (single combined calorimeter-energy channel — `jetImageECAL`/`jetImageHCAL` exist
separately, but ECAL is ~4e-6 nonzero for these hadronic jets, confirmed directly, so using the
single combined channel matches the single-channel convention in the jet-image literature:
Cogan et al. 2015, arXiv:1407.5675; de Oliveira et al. 2016, arXiv:1511.05190). Class balance
confirmed matching the tabular dataset's own (19.4-20.2% per class).

**Splits**: official `train/` (61 shards, 610,000 jets after a full, MD5-verified download —
two earlier download attempts were silently truncated mid-transfer despite `curl` reporting
exit code 0; caught by checking file size against the Zenodo API's authoritative byte count,
not by trusting the reported exit code) subsampled to 100,000 jets, split 85/15 internally into
train/val. Official `val/` (full 270,000 jets) held out entirely as the test set, touched once.
Images downsampled 100×100 → 25×25 via energy-conserving sum-pooling (25 is a clean divisor of
100 and the literature-standard size; 32, mentioned in an earlier planning pass, is **not** a
divisor of 100 and was corrected before any training ran).

**A real memory bug found and fixed before it caused a failure**: the first implementation of
the downsampling wrapper loaded every subsampled jet's full 100×100 image before pooling — for
the full 270,000-jet test split, that would be ~10.8GB against ~10GB available RAM in this
environment (checked directly with `free -h`). Fixed by pooling each shard immediately after
reading it, before accumulating — verified on the real full val split with `/usr/bin/time -v`:
peak RSS 1.9GB, not the ~10.8GB the original implementation would have hit.

## 3. Baseline-before-CNN (pre-declared in `reports/phase14_baseline_predeclaration.md`, before
any training)

Two baselines, same 3 seeds as the CNN (`[42, 123, 2026]`), same fixed 100,000-jet subsample
pool as the CNN, varying only the internal train/val split per seed:

1. **Hand-engineered image features + `HistGradientBoostingClassifier`** (default
   hyperparameters): total energy, nonzero-pixel count, energy-weighted centroid/spread,
   max-pixel energy (`hep_quant.data.image_features`).
2. **Flattened 25×25 pixels + `LogisticRegression`** (default hyperparameters): a pure
   "no physics knowledge" lower bound.

## 4. CNN

`JetImageCNN`: 3 conv blocks (16→32→64 channels, 3×3 kernels, ReLU, 2×2 max-pool) →
`Linear(flatten_dim, 64)` → `Linear(64, 5)`, ~90k parameters. Trained with the project's existing
`Trainer` class — generalized this phase to support any model exposing a `get_config()`/
`from_config()`/`CONFIG_KEYS` contract (previously hardcoded to `MLPClassifier`'s exact
attributes; verified unchanged behavior for the MLP by loading an actual frozen Phase 2
checkpoint, not just a fresh test model).

**Seed count — measured, not guessed.** An isolated single-seed timing probe measured 485.6s
(8.1 min) for seed 42 alone, giving a full-5-seed estimate of ~40 min — under this project's
own 5-minute-per-run threshold for using the full `[42,123,2026,7,99]` set. However, the
*actual* full run (started concurrently with a baseline-script rerun, a scheduling choice made
to save wall-clock time) took 1884.9s (31.4 min) for seed 42 under that CPU contention — a
~4× slowdown from contention, not a change in the underlying computational cost. The **seed
count decision itself** was made correctly from the isolated, uncontended measurement (485.6s
→ full-5-seed estimate ~40 min, which is what should have driven the choice); the script's
own internal re-measurement, taken under contention, triggered the fallback to the project's
existing 3-seed precedent (`[42,123,2026]`) instead. Both the decision logic and the actual
run are disclosed here exactly as they happened, not retroactively cleaned up.

## 5. Results (mean ± SD over 3 seeds, held-out `val/` test, full 270,000 jets)

| model | ROC-AUC | F1 | accuracy | bkg. rejection @ 70% eff. |
|---|---|---|---|---|
| **ImageFeatures+GBM** | **0.8879 ± 0.0002** | 0.6403 | 0.6419 | 10.91 |
| CNN (JetImageCNN) | 0.8835 ± 0.0003 | 0.6260 | 0.6299 | 11.29 |
| FlattenedPixels+LogReg | 0.7581 ± 0.0003 | 0.4663 | 0.4809 | 6.55 |

**Paired comparison, CNN minus each baseline** (95% CI, `scipy.stats.t.ppf(0.975, n-1)`,
descriptive given n=3):

| comparator | mean diff | 95% CI | seeds positive |
|---|---|---|---|
| GBM | **-0.0044** | **[-0.0052, -0.0035]** | **0/3** |
| LogReg | +0.1255 | [0.1242, 0.1267] | 3/3 |

## 6. Interpretation

**The CNN reliably beats the "no physics knowledge" pixel baseline, and is reliably — if only
slightly — below the hand-engineered-features GBM baseline.** This is the same qualitative
finding as Phase 13 (a GBM baseline beat the FP32 MLP on the tabular task), now replicated on
an independent architecture (CNN vs. MLP) and an independent dataset (images vs. high-level
features). Taken together, the two results support a broader, more defensible claim than
either alone: **across two different data representations and two different neural
architectures, a simple gradient-boosted-tree baseline on hand-engineered features matches or
exceeds the neural network.** Neither phase claims the relevant neural network is the best
available model for its task — the case for each remains downstream deployability (FPGA
trigger latency via hls4ml for the MLP), not raw accuracy.

**Both image-based approaches are substantially below the tabular approach's ~0.93-0.94
ROC-AUC** (FP32 MLP, GBM-on-HLF-features) — a genuine, expected finding: the tabular dataset's
16 features are sophisticated, physics-motivated substructure observables (energy correlation
functions, N-subjettiness ratios) specifically engineered for jet tagging over decades of HEP
research; a 25×25 pixelized calorimeter image, even processed by a CNN capable of learning its
own features, does not recover that same density of discriminating information at this
resolution and training-data scale.

## 7. Limitations

- Train subsampled to 100,000 of the available 610,000 jets (matching the tabular MLP's own
  subset-size convention) — not the full dataset.
- 3 seeds, not the project's full 5-seed convention, for the measured-cost reason in §4.
- CNN architecture, resolution (25×25), and training recipe were not tuned — a materially
  different architecture or more training data could plausibly close some or all of the gap to
  GBM; this result characterizes the specific configuration run, not an upper bound on CNN
  performance for this task.
- The baseline comparison uses 3 seeds varying only the internal train/val split (not the
  100,000-jet subsample itself, which is drawn once with a fixed seed) — a narrower source of
  variance than resampling the subsample per seed.

## 8. Reproducibility

- `tests/test_jet_images.py` (10 passed, 1 skipped), `tests/test_image_features.py` (5 passed),
  the new CNN checkpoint round-trip test in `tests/test_training.py` — all run twice, identical
  pass counts both times.
- The memory-safety fix (streaming per-shard pooling) verified directly on the real full val
  split with `/usr/bin/time -v`, not estimated.
- Baseline script run twice end-to-end on the real dataset: `reports/tables/phase14_baselines.csv`
  byte-identical between runs.
- CNN training determinism spot-checked (not re-run at full scale, given the ~30+ min per-seed
  cost): two in-process runs of the same seed on a 5,000-jet subset gave `diff = 0.0` exactly
  for validation ROC-AUC, after `hep_quant.utils.seeds.set_seed` is called at the start of each
  run (the same fix pattern Phase 12 needed for `build_qat_variant`).

## 9. Reproduction

```bash
# WSL2 Ubuntu, .venv-wsl -- requires the two Zenodo archives downloaded and extracted to
# data/raw/hls4ml_jet_images_30p/{train,val}/ (see Section 2 for the DOI and known download
# truncation issue -- verify the final file size/MD5 against the Zenodo API, not just exit code)
python scripts/phase14_baselines.py
python scripts/phase14_jet_image_cnn.py
python -m pytest -q tests/test_jet_images.py tests/test_image_features.py tests/test_training.py
```
