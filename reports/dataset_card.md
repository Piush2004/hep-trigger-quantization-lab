# Dataset Card

## Status: two datasets used, for two different purposes — read this before citing any number

This project uses **two** datasets, and conflating them would be a scientific-honesty
failure, so this card is explicit about which is which:

1. **Synthetic development dataset** (`hep_quant.data.synthetic`) — used only to
   validate the engineering pipeline (Gate 2) before touching real data. Every result
   produced from it is a pipeline-validation result, never a physics claim.
2. **hls4ml LHC jet-tagging dataset** (OpenML `hls4ml_lhc_jets_hlf`) — a real,
   citable, physics-motivated benchmark used by the actual HEP/FPGA-trigger research
   community. This is the dataset this project's physics-relevant conclusions should
   be drawn from. **It is particle-level Monte Carlo jet data (MadGraph5 + Pythia8 +
   FastJet; no detector simulation is described in the dataset paper), not
   recorded ATLAS/CMS detector data** — see "What 'real' means here" below; this
   distinction matters and is not the same thing as "synthetic" in sense #1 above.

---

## Dataset 2: hls4ml LHC jet-tagging dataset (primary, Phase 2 onward)

| Field | Value |
|---|---|
| Name | `hls4ml_lhc_jets_hlf` |
| Source | OpenML (data_id 42468), mirrored from Zenodo |
| DOI | [10.5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260) |
| Creators | Maurizio Pierini, Javier Duarte, Nhan Tran, Marat Freytsis, Vladimir Loncar |
| Associated publication | ["Fast inference of deep neural networks in FPGAs for particle physics"](https://iopscience.iop.org/article/10.1088/1748-0221/13/07/P07027), JINST 13 (2018) P07027 — the original hls4ml paper |
| License | CC BY (Creative Commons Attribution) — this card + `CITATION.cff` constitute the attribution |
| Download mechanism | `sklearn.datasets.fetch_openml("hls4ml_lhc_jets_hlf", version=1)`, wrapped in `hep_quant.data.loader.load_openml`; cached locally under `data/raw/openml_cache/` (gitignored) |
| Total instances | **830,000 jets** in the OpenML copy (measured by this project; see `reports/real_data_validation.md`). An earlier version of this card said ~880,000, taken from a search snippet, which was wrong |
| Features | 16 physics-motivated jet-substructure observables: `zlogz`, `c1_b0_mmdt`, `c1_b1_mmdt`, `c1_b2_mmdt`, `c2_b1_mmdt`, `c2_b2_mmdt`, `d2_b1_mmdt`, `d2_b2_mmdt`, `d2_a1_b1_mmdt`, `d2_a1_b2_mmdt`, `m2_b1_mmdt`, `m2_b2_mmdt`, `n2_b1_mmdt`, `n2_b2_mmdt`, `mass_mmdt`, `multiplicity` |
| Target | `class`, 5-way: gluon (`g`), light quark (`q`), top quark (`t`), W boson (`w`), Z boson (`z`) |
| Class balance | Approximately balanced across the 5 classes per the dataset's documentation; verified empirically for each subsample via `validate_dataset` (see `tests/test_openml_loader.py::test_load_openml_stratified_subsample_preserves_balance`) |
| Missing values | None expected (verified via `validate_dataset`'s NaN/Inf check on every load) |
| Train/val/test split | 70/15/15, **stratified** (5-way), seed=42 — stratification matters more here than for the synthetic binary case, since macro-averaged multiclass metrics are sensitive to per-class representation in each split |
| Preprocessing | Same as the synthetic path: per-feature standardization fit on the training split only |
| Subsampling | Stratified (class-proportional), not a plain truncation — see `load_openml`'s `subset_size` parameter. Used for the staged validation sequence below |

### What "real" means here

This is **particle-level Monte Carlo** jet data: parton-level events from MadGraph5 (LO, v2.3.1),
showered with Pythia8 (v8.212, Monash 2013 tune, with underlying event), jets clustered with FastJet
anti-kT R=0.8, jet pT window 0.8–1.6 TeV (arXiv:1804.06913, "Input generation and features"). The paper
describes no detector simulation. It is not data recorded by a physical ATLAS/CMS detector. (An earlier
version of this card said "Pythia8 + Delphes"; Delphes is not mentioned in the source.)
This is the correct and standard thing for this kind of study: jet-flavor ground truth
(is this jet "really" from a top quark vs. a gluon?) is not directly observable in
recorded collision data at all — it can only be established via simulation truth-
matching. The real HEP/FPGA-trigger literature (including the hls4ml papers)
uses exactly this kind of simulated-with-truth-labels dataset for this reason. Calling
it "real" (as opposed to this project's own `make_classification`-based synthetic
dataset) means: produced by the actual particle-physics simulation community, with
physically meaningful features and labels, published and cited in the field — not that
it was recorded by a detector during a live LHC run.

### Staged validation sequence (Phase 2)

Per the project's risk-management plan, this dataset is validated in three stages
before committing to a full experiment matrix, so a dataset-side problem is caught
cheaply rather than after hours of compute:

| Stage | Config | Subset size | Purpose |
|---|---|---:|---|
| 1 | `configs/jet_validate_10k.yaml` | 10,000 | Pipeline validation: fetch → validate → split → train → evaluate runs end-to-end without error |
| 2 | `configs/jet_validate_50k.yaml` | 50,000 | Training validation: confirms convergence behavior and per-epoch timing at a more realistic scale |
| 3 | `configs/jet_fp32.yaml` + `configs/jet_int{8,6,4}{,_qat}.yaml` | 100,000 | The actual experiment matrix (FP32 baseline, PTQ×3, QAT×3) |

Results from each stage are recorded separately (`EXP_010`/`EXP_011` for stages 1–2,
`EXP_012`–`EXP_018` for stage 3) and never silently merged with the synthetic-dataset
results (`EXP_001`–`EXP_009`) in a single "this is the result" table — see
`results_summary.csv`'s `dataset` column, which always says which dataset produced
each row.

---

## Dataset 1: synthetic development dataset (historical / pipeline-validation only)

Retained from Phase 1 for the pipeline tests that still validate against it
(`tests/test_data.py`, `tests/test_pipeline.py`, etc.) — fast, deterministic, no
network dependency, so it stays the right tool for engineering tests even now that a
real dataset is integrated.

| Field | Value |
|---|---|
| Generator | `sklearn.datasets.make_classification`, wrapped in `hep_quant.data.synthetic.generate_synthetic_dataset` |
| Source | Procedurally generated (not downloaded); deterministic given a seed |
| License | N/A (synthetic, no license needed) |
| Default size | 20,000 samples × 16 features (configurable via `configs/base.yaml`) |
| Target variable | Binary class label `{0, 1}` |
| Classes | 2, `n_clusters_per_class=2`, `class_sep=1.2` — loosely resembles a signal/background separability profile, NOT a physics-accurate simulation |
| Train/val/test split | 70/15/15, stratified, seed=42 |
| Known leakage risks | None by construction — `check_no_leakage` verifies zero row-level overlap between splits on every run |

### Potential bias (synthetic data, explicitly)

`make_classification`'s informative/redundant feature structure is a convenient,
well-understood testbed for validating a classification pipeline, but it has no
physical meaning: no detector response model, no realistic feature correlation
structure from kinematics, no class-conditional noise model resembling real trigger
backgrounds. Accuracy/F1/ROC-AUC numbers from this dataset (`EXP_001`–`EXP_009`)
describe how well the *pipeline* works, not how well a HEP trigger classifier performs
on physically meaningful data — that question is what Dataset 2 answers.

---

## Known limitations of the real-dataset integration (be precise about these)

- **No real detector data.** See "What 'real' means here" above — this is simulation,
  not recorded collisions. A genuinely detector-level validation (e.g. actual ATLAS/CMS
  Open Data with real reconstructed events) is a further step beyond this project's
  current scope.
- **High-level features only.** The 16 features are already-reconstructed jet
  substructure observables, not raw detector hits or low-level kinematics — this
  dataset does not test a model's ability to learn directly from raw data.
- **One dataset, one task.** Generalization across different HEP classification tasks
  (e.g. a genuinely different physics process) is not established by this one dataset,
  however standard it is for this specific trigger/FPGA use case.
