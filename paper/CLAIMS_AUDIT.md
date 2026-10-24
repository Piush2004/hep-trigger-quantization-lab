# Claims audit — every numeric claim in `main.tex`, traced to its source

Format: claim (as it appears in the paper) → exact source file/table. Every number below was
copied from the cited report, not recalled from memory or re-derived; the report filename is a
real, checked-out path in this repository at the time of writing.

## Section 4 — Experimental Design

| Claim | Source |
|---|---|
| Exploratory mixed-precision result ROC-AUC 0.9278, n=3, excluded from confirmatory claims | `reports/final_results.md` §"Mixed precision (no longer a headline result)" |
| Training: Adam, lr 1e-3, weight decay 1e-4, batch 1024, ≤30 epochs, patience 6 | `reports/phase6_architecture_generalization.md` §5 |
| QAT: 7 fine-tuning epochs at 0.1× lr | `reports/phase6_architecture_generalization.md` §5 |

## Section 5 — Uniform PTQ and QAT Results (Table 1)

| Claim | Source |
|---|---|
| FP32 0.9343 ± 0.0011, 17,556 B | `reports/final_results.md` §"Primary architecture A" |
| INT8 PTQ 0.9342 ± 0.0011, Δ −0.0001, 4,788 B | ditto |
| INT6 PTQ 0.9329 ± 0.0014, Δ −0.0014, 3,724 B | ditto |
| INT4 PTQ 0.8819 ± 0.0308, Δ −0.0524, 2,660 B | ditto |
| INT4 QAT 0.9246 ± 0.0018, Δ −0.0097, 2,660 B | ditto |

## Section 6 — Architecture Comparison (Table 2)

| Claim | Source |
|---|---|
| Architecture A: 16→64→32→32→5, 4,389 params; Architecture B: 16→128→64→64→32→5, 16,837 params | `reports/phase6_architecture_generalization.md` §2 |
| A: FP32 0.9343±0.0015; INT8 Δ −0.0001; INT6 Δ −0.0010; INT4 PTQ Δ −0.0438 (SD 0.0122); QAT recovery 78.5% | `reports/phase6_architecture_generalization.md` §7–8 |
| B: FP32 0.9376±0.0009; INT8 Δ 0.0000; INT6 Δ −0.0010; INT4 PTQ Δ −0.0264 (SD 0.0045); QAT recovery 73.6% | ditto |
| Paired diff, INT4 PTQ B−A: +0.0175, 95% CI [−0.0009, +0.0358], 4/5 seeds agreeing | `reports/phase6_architecture_generalization.md` §8 |
| CPU re-run INT4 PTQ 0.8904 ± 0.0122 vs. CUDA-era 0.8819 ± 0.0308 | `docs/REPRODUCIBILITY.md` §6; `reports/final_results.md` |
| Two independent runs produced identical per-seed metrics and byte-identical checkpoints | `reports/phase6_architecture_generalization.md` §3, §13 |

## Section 7 — Calibration Sensitivity (Table 3)

| Claim | Source |
|---|---|
| Calibration-size sweep (100/500/1,000/5,000/10,000 jets): ROC-AUC, A0 hi, step, clip rate | `reports/int4_ptq_stability_analysis.md` §"D. Calibration-size sweep" |
| Determinism: 5 rebuilds + 3 repeated inferences, bit-identical (SHA-256) | `reports/int4_ptq_stability_analysis.md` §"C. Determinism" |
| Paired effect of size, mean +0.021, 95% CI [0.010, 0.033], all 5 seeds same direction | `reports/int4_ptq_stability_analysis.md` §"Paired effect of calibration size" |
| Between-seed SD 0.029 vs. within-seed SD 0.008 | `reports/int4_ptq_stability_analysis.md` §"B. Model seed vs calibration draw" |

## Section 8 — Mixed-Precision Analysis (Table 4)

| Claim | Source |
|---|---|
| 3^7 = 2,187 layouts; 23 non-dominated Pareto-frontier layouts | `reports/phase12_exhaustive_mixed_precision_pareto.md` §1–2 |
| Mixed QAT selected: 2,916 B, 0.9342 ROC-AUC; beats INT6 PTQ +0.0013, 95% CI [0.0011,0.0015], 5/5 seeds | `reports/phase12_exhaustive_mixed_precision_pareto.md` (headline table, §"PTQ selected"/"QAT selected") |
| Mixed PTQ: 2,996 B, 0.9321 | ditto |
| 22% storage reduction vs. uniform INT6 (2,916/3,724 = 78.3%, i.e. 21.7% less, rounded to 22%) | computed directly from the two storage figures above |
| QAT fine-tuning seed bug found and fixed, ~3e-4 ROC-AUC drift | prior-session fix, documented in this project's commit history for the Phase 12 quantization-variant builder |

## Section 9 — hls4ml Fixed-Point C-Simulation (Table 5)

| Claim | Source |
|---|---|
| Seed-42 FP32 reference: accuracy 0.7495, ROC-AUC 0.9334 | `reports/phase7_hls_numerical_validation.md` §1 |
| C-simulation accuracy table (W=8..16, 4 arithmetic modes) | `reports/phase7_hls_numerical_validation.md` §3 |
| Surrogate R² up to 0.996 on held-out synthesis data | `reports/phase11_surrogate_hardware_validation.md` |
| Surrogate's near-zero bit-width feature importance | `reports/phase11_surrogate_hardware_validation.md` |
| No Vivado/Vitis HLS/Quartus in the experimental environment | `reports/phase7_hls_numerical_validation.md` §0 (header); `docs/REPRODUCIBILITY.md` §8 |

## Section 10 — Discussion

All bullet points are restatements of disclosures already sourced above (device observational-only,
rounding/saturation not isolated, no synthesis, no detector data, no production claim) — no new
numeric claims are introduced in this section.

## Reproducibility Statement

| Claim | Source |
|---|---|
| 141 tests, two independent runs, identical results | run directly in this session; see `docs/REPRODUCIBILITY.md` for the project's two-test-rule methodology this follows |

## Numbers deliberately NOT included in the paper

- The biased, test-set-selected ROC-AUC 0.9278 — mentioned only as an example of disclosed bias
  (Section 4), never presented as a result.
- Any Phase 14 (CNN / jet-image) result — that work exists on a separate, not-yet-merged branch
  (`phase14-cnn-jet-images`, PR open against `master`) and is out of scope for this paper's
  pre-registered 14-section structure. If a generality-check section is wanted, it is available to
  add from `reports/phase14_cnn_jet_images.md` and `reports/phase14_cnn_quantization.md` on that
  branch — not folded in here without an explicit decision to do so.
