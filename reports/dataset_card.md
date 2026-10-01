# Dataset Card

## Status: development subset (synthetic) — real HEP dataset experiment pending

Per the project's data-fabrication policy, this card documents exactly what data was
used to produce the results in this repository: a **synthetic classification
dataset**, not a real ATLAS/CMS dataset. This section will be updated once a real HEP
dataset is integrated (see "Real dataset integration plan" below); until then, every
result in `results_summary.csv` and `reports/research_report.md` is explicitly labeled
as a development-validation result, not a physics result.

## Why synthetic, not ATLAS JetSet / CMS HiggsToBB directly

Downloading and validating a multi-GB public HEP dataset (e.g. the ATLAS Open Data
jet-tagging sets, or CMS Higgs-to-bb benchmark data used in HLS4ML-style trigger
studies) was out of scope for the time available in this build session, and doing it
hastily risked exactly the kind of silent correctness problem this project is
structured to avoid (Gate 2 explicitly requires validating the engineering pipeline on
a synthetic dataset *before* touching real data). The loader (`hep_quant.data.loader`)
already supports HDF5/CSV/Parquet, which covers the common distribution formats for
these datasets, so switching to a real dataset is a configuration change
(`dataset.path` in a YAML config), not a code change.

## Synthetic dataset specification

| Field | Value |
|---|---|
| Generator | `sklearn.datasets.make_classification`, wrapped in `hep_quant.data.synthetic.generate_synthetic_dataset` |
| Source | Procedurally generated (not downloaded); deterministic given a seed |
| License | N/A (synthetic, no license needed) |
| File format | In-memory NumPy arrays (no on-disk artifact by default) |
| Default size | 20,000 samples × 16 features (configurable via `configs/base.yaml`) |
| Target variable | Binary class label `{0, 1}` (configurable to multi-class) |
| Classes | 2 (binary), with `n_clusters_per_class=2` and `class_sep=1.2` for moderate separability — intended to loosely resemble the separability profile of a HEP signal/background classification task, NOT a physics-accurate simulation |
| Class balance | Balanced by construction (`make_classification` default), verified by `validate_dataset` |
| Missing values | None (synthetic, numerically clean by construction); the loader's NaN/Inf checks are tested against this guarantee in `tests/test_data.py` |
| Duplicate rows | Checked at validation time; expected count is 0 for the sample sizes used here |
| Train/val/test split | 70/15/15, stratified, seed=42 (see `hep_quant.data.splits.split_dataset`) |
| Preprocessing | Per-feature standardization (zero mean, unit variance), fit on the training split only |
| Known leakage risks | None by construction — `check_no_leakage` verifies zero row-level overlap between splits on every run |

## Real dataset integration plan

To integrate a real HEP dataset (e.g. ATLAS Open Data or CMS Open Data jet-tagging
sets):

1. Download the dataset (HDF5/CSV/Parquet) into `data/raw/` (gitignored — not committed).
2. Point `configs/base.yaml`'s `dataset.path` at the file; `hep_quant.data.loader.load_dataset`
   dispatches on file extension automatically.
3. Re-run `python -m hep_quant validate-data --config configs/base.yaml` and confirm it
   passes before training anything.
4. Update this dataset card's specification table with the real dataset's provenance,
   citation, license, and feature schema.
5. Re-run the full experiment matrix; results produced from real data must be clearly
   distinguished from the synthetic-validation results already in this repository
   (e.g. a separate `results_summary_real.csv`), per the project's scientific-honesty
   policy — they are never merged into one table silently.

## Potential bias (synthetic data, explicitly)

`make_classification`'s informative/redundant feature structure is a convenient,
well-understood testbed for validating a classification pipeline, but it has no
physical meaning: there is no detector response model, no realistic feature
correlation structure from kinematics, and no class-conditional noise model
resembling real trigger backgrounds. Any accuracy/F1/ROC-AUC numbers reported on this
dataset describe how well the *pipeline* works, not how well a HEP trigger classifier
would perform on real collision data.
