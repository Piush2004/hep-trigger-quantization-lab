# Dataset Selection

> **Chronology note (read first).** The hls4ml jet-tagging dataset was chosen and used in the
> previous phase, *before* this decision record existed. This document records the evidence for
> that choice, re-verified from primary sources in this phase, and states plainly which earlier
> claims turned out to be wrong (see "Corrections to earlier documentation").

## Research Requirement

The benchmark studies how post-training quantization (PTQ) and quantization-aware training (QAT)
affect accuracy, model size, inference characteristics and hardware-oriented deployment
characteristics of **small neural networks of the kind proposed for FPGA trigger-level event
classification**. The dataset therefore needs:

1. a jet/event classification task from HEP, with truth labels,
2. a fixed-length feature vector that a small MLP can consume without a bespoke representation
   learning stage (the existing pipeline and the hls4ml conversion path are MLP-based),
3. an openly licensed, citable source with documented schema,
4. a size that is feasible on a laptop (Ryzen 7, 24 GB RAM, 8 GB GPU) without chunked streaming,
5. a published reference result for a comparable model, so the pipeline can be sanity-checked.

## Candidate Datasets

### A. ATLAS JetSet — "ATLAS ttbar simulation for ML-based jet flavour tagging"
Verified from the CERN Open Data portal record and its JSON API (record 93940):

| Item | Verified value |
|---|---|
| DOI | 10.7483/OPENDATA.ATLAS.QG8W.TO8P |
| License | CC0-1.0 |
| Content | simulated ttbar, 13.6 TeV; HDF5 with event-, jet-, track-level and truth-hadron information |
| Files | `mc-flavtag-ttbar-small.h5` 3,055,139,850 B (~1.36 M events, ~5.6 M jets); `-medium` 13.9 GB; `-large` 90.8 GB |
| Documented use | training/evaluating jet flavour taggers; the accompanying ATLAS paper (arXiv:2505.19689, GN2) uses a **transformer on low-level track inputs** |
| Smaller official subset | none below 3.06 GB |
| Variable/label documentation | linked from the record to a GitLab repository, which returned HTTP 403 to automated fetch; **the HDF5 variable names and flavour-label definition could not be verified** |
| Partial download | the portal server ignored HTTP range requests (a ranged request streamed the whole file), so the schema cannot be inspected without downloading the full 3.06 GB. The EOS host serves a CERN-CA certificate that is not trusted here; TLS verification was deliberately **not** disabled. |

### B. hls4ml LHC jet-tagging benchmark (OpenML `hls4ml_lhc_jets_hlf`, Zenodo 10.5281/zenodo.3602260)
Verified from the Zenodo record, OpenML's JSON API (data_id 42468) and the dataset paper:

| Item | Verified value |
|---|---|
| License | CC BY 4.0 (Zenodo), "CC_BY" (OpenML) |
| Source documents | Zenodo record; JINST 13 (2018) P07027 = arXiv:1804.06913 (generation + features, Table 1) |
| Generation (arXiv:1804.06913, "Input generation and features") | 13 TeV; MadGraph5 v2.3.1 LO parton-level W+W-, ZZ, ttbar, qqbar, gg; Pythia8 v8.212 (Monash 2013 tune, with underlying event); 200,000 events per final state; FastJet anti-kT R=0.8; jet pT window 0.8–1.6 TeV (parton pT centred at 1 TeV). **No detector simulation is described.** |
| Task | 5-class jet tagging: g, q, W, Z, top |
| Features | 16 high-level jet-substructure observables (paper Table 1) |
| Size | OpenML copy: **830,000 jets** (measured by this project, see `reports/real_data_validation.md`); local cache 150 MB. Zenodo additionally offers jet images and particle lists (3.9 GB total), which this project does not use. |
| Reference ML result | arXiv:2003.06308 (Ngadiuba et al.), Table 5: 16×64×32×32×5 Keras MLP, floating point, **accuracy 75%**, per-class AUC 0.904–0.958; split 60/20/20 |
| Purpose | introduced explicitly as the benchmark for FPGA-trigger studies with hls4ml |

### C. CMS "HiggsToBB"
A search of the CERN Open Data portal returned no record matching this name (it returned unrelated
CMS trigger/AOD records and the ATLAS datasets above). **No evidence could be gathered, so this
candidate was not evaluated.** It is not rejected on merit.

### Other
The ATLAS Top Tagging Open Data Set (portal records 15013/80030) appeared in search results only;
it was not fetched or evaluated.

## Evidence

Requirement-by-requirement (factual, not scored):

| Requirement | JetSet | hls4ml jets |
|---|---|---|
| Truth-labelled HEP classification | yes (flavour tagging) | yes (5-class jet tagging) |
| Fixed-length vector for an MLP | not provided: documented inputs are track sets for a transformer; an MLP input would have to be engineered by this project | yes: 16 features, ready-made |
| Open license / citation | CC0 / citation specified | CC BY 4.0 / citation specified |
| Schema verified | no (docs inaccessible; needs 3 GB) | yes (names, classes, generation, counts all checked) |
| Laptop-feasible | 3.06 GB minimum download, variable-length arrays | 150 MB, 830k x 16 floats (~53 MB) |
| Published comparable reference | GN2 transformer results (not comparable to an MLP) | 16-64-32-32-5 MLP, 75% (comparable architecture) |
| Designed for FPGA-trigger studies | no | yes |

## Selected Dataset

**hls4ml LHC jet-tagging benchmark (OpenML `hls4ml_lhc_jets_hlf`, v1)**, accessed through the
dedicated adapter `src/hep_quant/data/hls4ml_jets.py`.

## Why It Fits

It is the dataset introduced for the very question this project asks (small quantized MLPs for
FPGA-trigger jet classification); its input is a fixed 16-feature vector that matches the MLP
pipeline and the hls4ml conversion path; its generation, features and labels are documented in a
peer-reviewed paper; its license is permissive with a clear citation; and a published reference
exists for an identical architecture.

## Known Limitations

- Particle-level Monte Carlo (MadGraph5 + Pythia8 + FastJet) with **no detector simulation described**;
  not recorded collision data and not detector-level simulation.
- Only the 16 high-level features are used; jet images and particle lists from the Zenodo record are
  not in the OpenML copy, so constituent-level structure is unavailable to this benchmark.
- A narrow kinematic window (jet pT 0.8–1.6 TeV); results do not transfer to other pT regimes.
- JetSet was **not** adopted and remains unevaluated at schema level; it is a candidate for a future,
  separate study of track-based models (see Decision).

## License

CC BY 4.0 — attribution required; satisfied by the citations below and `CITATION.cff`.

## Citation

- Pierini, M., Duarte, J. M., Tran, N., Freytsis, M. (2020). *HLS4ML LHC Jet dataset*. Zenodo.
  DOI 10.5281/zenodo.3602260 (OpenML metadata additionally lists V. Loncar).
- Duarte, J. et al. (2018). *Fast inference of deep neural networks in FPGAs for particle physics*.
  JINST 13 P07027, arXiv:1804.06913.
- Reference baseline: Ngadiuba, J. et al. *Compressing deep neural networks on FPGAs to binary and
  ternary precision with hls4ml*, arXiv:2003.06308.

## Download Method

`sklearn.datasets.fetch_openml("hls4ml_lhc_jets_hlf", version=1)` inside `load_openml`, cached under
`data/raw/openml_cache/` (git-ignored) plus a parsed `.npz`. No smaller official subset exists; the
whole file is ~150 MB, which is itself below the size where a staged download would help. Development
subsets are drawn by stratified sampling after loading (`dataset.subset_size`), staged 10k → 50k → 100k.

## Expected Computational Cost

Measured: cold download + ARFF parse + full validation 84 s wall clock, 434 MB peak RAM (WSL2).
Training cost per experiment is reported in `reports/REAL_HEP_RESULTS.md`.

## Corrections to earlier documentation

Verified against the primary sources in this phase, these earlier statements were wrong or imprecise
and have been corrected in the affected files:

1. "Pythia8 + Delphes" — the dataset paper describes MadGraph5 + Pythia8 + FastJet and **no** detector simulation.
2. "~880,000 jets" — the OpenML copy contains **830,000** jets.
3. "within 1 pp of the original hls4ml paper's own baseline" — the reference is arXiv:2003.06308 Table 5
   (75%, rounded to an integer percent) with a different split (60/20/20), optimizer settings and
   training budget; our 74.95% is **consistent with**, not a reproduction of, that number.

## Decision

**Proceed** with the hls4ml jet-tagging benchmark.
**Reconsider JetSet** in a later phase only if the project moves to track-level models; that would
need the full 3.06 GB download before the schema can even be confirmed.
