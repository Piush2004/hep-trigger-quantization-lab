# Phase 4 evidence lock

Written before any Phase 4 methodology change. Records the state the Phase 4 analyses start from.

## Repository state
- Branch `master`, HEAD `b413636` (Phase 3 documentation closure, on top of Phase 3 code `8377844`).
- Working tree: one untracked file, `docs/CERN_Study_Guide.pdf`, not created by this project and deliberately left alone.
- Phase 4 additions so far (uncommitted at the time of writing): `reports/related_work_review.md`, `reports/research_novelty_analysis.md`, `reports/int4_ptq_stability_analysis.md`, `scripts/phase4_novelty_check.py`, `scripts/phase4_int4_stability.py`, `reports/tables/phase4_int4_*.csv`, this file.

## Environment
- OS: Windows 11 host; analysis runs in WSL2 Ubuntu.
- Python 3.11.16; PyTorch 2.11.0+cu128; `torch.cuda.is_available()` = **False** (CUDA unavailable in WSL this session; cause not established, same as the earlier intermittent loss documented in `environment_report.md`).
- All Phase 4 computation so far is therefore CPU-only.

## Test results (two-test rule)
| Run | Command | Result |
|---|---|---|
| Phase 0, TEST #1 | `pytest -q` | 91 passed, 5 skipped (179 s) |
| Phase 0, TEST #2 | `pytest -q` (independent) | 91 passed, 5 skipped |

Earlier Phase 3 closure reported 92 passed, 4 skipped. The difference is one extra skip: `tests/test_hardware.py:120` (GPU-model conversion path) is skipped because CUDA is unavailable. The four network tests in `tests/test_openml_loader.py` remain opt-in skips. **The CUDA skip is not hidden.**

## Frozen data integrity
- `python scripts/freeze_phase2.py verify`: 84 files checked, 0 changed. Verified before and after Phase 4 work.
- `experiments/results/frozen_phase2/` was not written by any Phase 4 script (the INT4 diagnosis reads frozen files and writes only to `reports/tables/` and `reports/`).
- Seeds 7 and 99 are not in the frozen directory; their checkpoints are in `experiments/checkpoints/`. They are read-only inputs to Phase 4.

## Current claims (carried from Phase 3, with Phase 4 status)
| Claim | Status after Phase 4 so far |
|---|---|
| INT8 and INT6 PTQ are practically lossless | Supported (INT8 and INT6 flat across calibration sizes, `phase4_int4_D_calib_size.csv`) |
| INT4 PTQ is unstable (ROC-AUC 0.8819 ± 0.0308) | **Refined.** The spread is mostly between model seeds (SD of seed means 0.029 vs within-seed calibration SD 0.008). See `int4_ptq_stability_analysis.md`. |
| INT4 PTQ is device-dependent | **Not established.** Not tested this session (no CUDA). Withdrawn as an explanation. |
| INT4 QAT is stable and substantially better than INT4 PTQ | Carried from Phase 3; not re-examined in Phase 4 yet. |
| QAT's INT8/INT6 edge over PTQ is not separable from extra training | Carried from Phase 3 compute-matched control; not re-examined. |
| Mixed precision (L0+A0 INT8, rest INT4) reaches ROC-AUC 0.9278 ± 0.0037 (n = 3) | **Under audit, see unresolved item 1.** |
| hls4ml default arithmetic collapses at narrow widths; rounding and saturation fix most of it | Carried from Phase 3 C-simulation; not yet re-checked in Phase 4. |
| No FPGA synthesis performed; hardware resources are estimates | Still true. Toolchain check on 2026-10-05 found no Vivado, Vitis HLS or Quartus. |

## Unsupported or withdrawn claims found in this lock
1. **"INT4 is device-dependent" as an explanation** for the INT4 spread. No evidence in Phase 4 (CPU only). The single seed-42 CPU/CUDA observation (0.651 vs 0.572) remains an unexplained observation.
2. **"INT4 instability is caused by saturation."** The clip rate at INT4 is essentially 0 under min/max calibration. Withdrawn.
3. **Dataset size.** The Hawks et al. summary (arXiv:2102.11289) states 870,000 jets. The project's verified OpenML `hls4ml_lhc_jets_hlf` v1 has 830,000. This is unresolved; the paper's text must be checked before any citation comparing sizes.
4. **"First" or "novel" HEP PTQ-vs-QAT claims.** Not supportable on the current search (`research_novelty_analysis.md`). Classification: C (benchmark), with B only for analyses that survive the Phase 4 checks.

## Unresolved questions (open, ordered by impact)
1. **Mixed-precision selection leakage.** The per-unit sensitivity study that chose "L0 + A0 INT8, rest INT4" (`scripts/phase3_ptq_studies.py`, S3) computed ΔROC-AUC on the **test** split (`Xte`). The mixed configuration was therefore selected using test data, and its reported 0.9278 is optimistically biased. Required before any further mixed-precision claim: re-select on the validation split and evaluate on test. Not yet done. Status: **open, must be resolved in Phase 5.**
2. **CUDA cause and device dependence.** Not reproducible this session. Device claims stay unresolved.
3. **Zero-point not rounded to an integer** in `fake_quantize_tensor`. Affects the realism of "uniform affine" INT4; not changed (would alter frozen-method semantics); recorded as a limitation.
4. **hls4ml default rounding/saturation (TRN/WRAP).** The search snippet agrees with the project's C-simulation finding, but primary AMD/hls4ml documentation was not read in full.
5. **Prior-work numbers** from automated PDF summaries are unverified at the level needed to compare numbers in a table (`related_work_review.md`, §5).
6. **Architecture generalization, calibration-method comparison, hls4ml fixed-point fidelity, final statistics, hostile review, and the final report** are not yet done (see the Phase 4 status in the final response).

## Figures and report references
- `reports/figures/` contains the Phase 3 figures and the numbered copies `01_…`–`08_…`. The Phase 4 INT4 figures are **not yet produced**.
- `reports/research_report.md` has the Phase 3 section map. It has **not yet been updated** for Phase 4 results.
