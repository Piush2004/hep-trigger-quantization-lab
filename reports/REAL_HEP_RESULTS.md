# Real HEP Quantization Results (hls4ml jet-tagging benchmark)

All numbers are generated from experiment artifacts (`reports/tables/*.csv`, produced by `scripts/audit_phase2.py`, `scripts/phase3_*.py`). Frozen primary evidence:
`experiments/results/frozen_phase2/` (seeds 42/123/2026, SHA-256 manifest). Later runs (seeds 7/99 and FP32 fine-tune controls) have new IDs and are never mixed into the frozen set.

## Dataset

OpenML `hls4ml_lhc_jets_hlf` v1 (Zenodo 10.5281/zenodo.3602260, CC BY 4.0): 830,000 jets, 16 high-level substructure features, 5 classes (g, q, t, w, z, ≈20% each). Particle-level
**Monte Carlo** (MadGraph5 + Pythia8 + FastJet, jet pT 0.8–1.6 TeV; the dataset paper describes no detector simulation) — **not detector-level and not recorded collision data**. Provenance, schema and
validation: `reports/dataset_selection.md`, `reports/dataset_schema.md`, `reports/real_data_validation.md` (two from-scratch downloads gave byte-identical reports; 0 NaN/Inf/duplicates; no single feature predicts the label above 0.61 accuracy).
This is **not** the ATLAS JetSet dataset.

## Experimental setup

MLP 16-64-32-32-5 (4,389 parameters); 100,000-jet stratified subset per seed; 70/15/15 split; Adam, lr 1e-3, batch 1024, ≤30 epochs, early stopping; all runs on one device (CUDA).
Quantization is **simulated** uniform-affine fake-quantization (per-tensor min/max) of Linear weights and hidden ReLU outputs; inputs, logits, biases (FP32) and accumulators are not quantized; nothing is bit-packed.
PTQ: calibrate on 20,480 training jets. QAT: FP32 training + 7 fine-tuning epochs (lr × 0.1) with straight-through fake-quantization. **Control:** FP32 trained and then given the *same* 7 extra epochs with no quantization.
Seeds control the 100k subsample, the split, initialization and shuffling. Statistics: sample SD (ddof = 1); 95% t-interval; paired effect size d_z. **n = 5 is small: intervals and d_z are descriptive aids, not significance tests.**

## FP32 baseline

Macro one-vs-rest ROC-AUC 0.9343 ± 0.0011 (n = 5; range 0.9330–0.9356); accuracy 0.753 ± 0.004 (n = 3 frozen). Consistent with the 75% (accuracy) reference for this architecture in arXiv:2003.06308 Table 5; **not** a reproduction (different split, lr, budget, ~1M jets).

## PTQ and QAT results (n = 5 seeds: 42, 123, 2026, 7, 99)

| Configuration | ROC-AUC mean ± SD | min – max | ΔAUC vs same-seed FP32 (mean ± 95% t-CI) | relative | d_z (paired) |
|---|---:|---|---:|---:|---:|
| FP32 | 0.9343 ± 0.0011 | 0.9330 – 0.9356 | — | — | — |
| INT8 PTQ | 0.9342 ± 0.0011 | 0.9330 – 0.9355 | −0.0001 ± 0.0001 | −0.009% | −1.2 |
| INT6 PTQ | 0.9329 ± 0.0014 | 0.9313 – 0.9345 | −0.0014 ± 0.0004 | −0.15% | −4.2 |
| INT4 PTQ | **0.8819 ± 0.0308** | 0.8353 – 0.9090 | −0.0524 ± 0.0370 | −5.6% | −1.8 |
| INT8 QAT | 0.9348 ± 0.0017 | 0.9324 – 0.9367 | +0.0005 ± 0.0008 | +0.06% | +0.8 |
| INT6 QAT | 0.9342 ± 0.0015 | 0.9320 – 0.9358 | −0.0001 ± 0.0007 | −0.006% | −0.1 |
| INT4 QAT | 0.9246 ± 0.0018 | 0.9217 – 0.9260 | −0.0097 ± 0.0022 | −1.03% | −5.5 |

(`reports/tables/seed_statistics.csv`, `quantization_comparison.csv`; the n = 3 frozen-set versions are in the same files and in `phase2_results_audit.md`; they agree: the two added seeds fell inside the n = 3 range.)

**QAT vs PTQ and the compute-matched control** (paired, n = 5; `reports/tables/qat_vs_ptq.csv`):

| Comparison | ΔAUC mean ± 95% CI | seeds in which the first term is higher |
|---|---:|---:|
| QAT − PTQ, INT8 | +0.0006 ± 0.0009 | 4 / 5 |
| QAT − PTQ, INT6 | +0.0013 ± 0.0005 | 5 / 5 |
| QAT − PTQ, INT4 | +0.0427 ± 0.0366 | 5 / 5 |
| FP32 fine-tune control − FP32 baseline (extra epochs alone) | +0.0004 ± 0.0002 | 5 / 5 |
| QAT INT8 − control | +0.0001 ± 0.0007 | 4 / 5 |
| QAT INT6 − control | −0.0005 ± 0.0006 | 0 / 5 |
| QAT INT4 − control | −0.0101 ± 0.0023 | 0 / 5 |

Reading: extra training alone raises AUC by +0.0004. QAT's apparent edge over FP32 at INT8 (+0.0005) is no larger than that; **against a compute-matched control, QAT at INT8 shows no measurable advantage and QAT at INT6/INT4 still sits below the control** — i.e. the real quantization cost under QAT is ≈ 0.0000 (INT8), −0.0005 (INT6), −0.0101 (INT4). QAT's large benefit appears only where PTQ fails (INT4).
Even-handed note: QAT−PTQ at INT6 (+0.0013, 5/5 seeds) is real but small; roughly a third is explained by the extra epochs.

## Why does INT4 PTQ fail? Checkpoint studies (frozen FP32 weights, seeds 42/123/2026, n = 3; `reports/tables/phase3_ptq_*.csv`)

- **Reproduction check:** uniform INT-N variants rebuilt from the frozen FP32 checkpoints reproduce the recorded PTQ experiments (max |Δaccuracy| = 6.7e-5, one test jet), so the study tooling matches the project's PTQ.
- **Calibration-set size (INT4):** median ROC-AUC *decreases* as calibration data increases (seed 42: 0.895 with 32 jets → 0.832 with all 70k; seed 123: 0.925 → 0.906; seed 2026: 0.898 → 0.866); INT8/INT6 are insensitive (< 0.0015 across all sizes). Even the best INT4 setting (32 jets) stays below FP32 (gap −0.039 / −0.010 / −0.035). This is consistent with min/max ranges widening as more data (and outliers) are seen; the ranges were not inspected directly, so this explanation is a hypothesis, not a finding.
- **Weights vs activations (INT4, ΔAUC for seeds 42/123/2026):** weights only −0.026/−0.007/−0.026; activations only −0.037/−0.083/−0.044; both −0.098/−0.027/−0.067. Effects are **not additive** and the ranking varies by seed.
- **Per unit (INT4, one unit alone):** the first Linear (L0) and first hidden activation (A0) dominate (L0 −0.028/−0.005/−0.028; A0 −0.019/−0.080/−0.021); later units alone cost ≤ 0.005. "All INT4 except one unit" is non-monotonic across seeds (protecting L0 alone does not reliably help), again indicating interactions.
- **Sensitivity-derived mixed precision (PTQ, n = 3, derived from the unit study, not arbitrary):**

| Strategy | theoretical weight storage | ROC-AUC mean ± SD | ΔAUC |
|---|---:|---:|---:|
| uniform INT8 | 4,788 B | 0.9339 ± 0.0014 | −0.0001 |
| uniform INT4 | 2,660 B | 0.8702 ± 0.0370 | −0.0638 |
| L0 at INT8, rest INT4 | 3,172 B | 0.8798 ± 0.0180 | −0.0542 |
| A0 at INT8, rest INT4 | 2,660 B | 0.9039 ± 0.0180 | −0.0301 |
| **L0 and A0 at INT8, rest INT4** *(exploratory — TEST_SET_SELECTED / BIASED; excluded from confirmatory analysis)* | 3,172 B | ~~0.9278 ± 0.0037~~ (biased; see `exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md`) | — |

Keeping only the first layer's weights *and* first activation at INT8 recovers most of the INT4-PTQ loss with +19% storage over uniform INT4 (activation bit-width does not enter the weight-storage figure). For reference INT4 QAT is 0.9246 at 2,660 B (n = 5) and INT6 PTQ 0.9329 at 3,724 B. Not tested: mixed precision under QAT, other mixed assignments, other seeds.

## Per-class, calibration and error analysis (frozen checkpoints, n = 3; observations only)

- INT8/INT6 (PTQ and QAT): per-class ΔAUC within ±0.0023, per-class recall changes ≤ 0.012.
- **INT4 PTQ concentrates damage in W and Z jets:** Δrecall W −0.252 ± 0.092, Z −0.291 ± 0.234; ΔAUC(OvR) W −0.077, Z −0.144 ± 0.110; top-jet precision −0.217 ± 0.151. **INT4 QAT** is uniform: ΔAUC −0.008 to −0.013 for every class, Δrecall within ±0.03.
- **Calibration:** top-label ECE (15 bins) FP32 0.0085; INT8 PTQ 0.0092; INT6 PTQ 0.0093; INT4 PTQ **0.0888 ± 0.0372**; INT4 QAT 0.0149 ± 0.0023. Brier score FP32 0.348; INT4 PTQ 0.525 ± 0.087; INT4 QAT 0.374.
- **Agreement with FP32 predictions (of 15,000 test jets, mean of 3 seeds):** INT8 PTQ 0.994; INT6 PTQ 0.973; INT4 PTQ 0.791; INT8 QAT 0.972; INT6 QAT 0.966; INT4 QAT 0.924. INT4 PTQ turns ≈ 2,264 FP32-correct jets wrong (and fixes ≈ 411); INT4 QAT ≈ 540 (fixes ≈ 301). QAT models agree with FP32 *less* than PTQ at INT8/INT6 although their accuracy is as good — they are different (further-trained) models, not perturbed copies.
- **Where flips occur:** for INT4 PTQ the fraction of FP32-correct jets flipped to wrong is ≈ 6× higher in the third quartile of `mass_mmdt`/`c1_*` than in the outer quartiles (≈ 0.39 vs 0.07–0.09); for INT4 QAT the largest spread is across `multiplicity` quartiles (0.012–0.080). No causal explanation is claimed.

## Storage, latency, hardware-oriented results

- **Theoretical weight storage** (analytical, bit-packed equivalent; no packed artifact exists; checkpoints are float32, 21–24 kB): FP32 17,556 B; INT8 4,788 B; INT6 3,724 B; INT4 2,660 B.
- **Latency (software, simulated quantization; not hardware).** Controlled study of the seed-42 checkpoints (`scripts/phase3_latency_study.py`; CPU 1 thread, `perf_counter`; CUDA events + sync per repetition; batch 1: 1,000 reps; batch 4096: 100 reps; devices never compared with each other). Median ms/forward at batch 1 — CPU: FP32 0.055, INT8/6/4 PTQ 0.139/0.258/0.145, INT8/6/4 QAT 0.508/0.561/0.477; CUDA: FP32 0.418, PTQ 1.65/1.73/1.67, QAT 4.70/4.47/4.25. Batch 4096 throughput (CPU): FP32 3.65 M jets/s vs PTQ ≈ 0.55 M/s and QAT ≈ 0.47–0.51 M/s. Batch-1 timings are noisy (SD ≈ median). The ordering FP32 < PTQ < QAT is the cost of emulating quantization with extra elementwise operations (it persists at batch 4096 on CPU); it says nothing about quantized hardware. The 21-run in-experiment latencies (CUDA, batch 1, shared machine) are indicative only.
- **Resources (LUT/DSP/BRAM/FF):** all records are **`measurement_tier = estimated`** (software heuristic `dsp = ⌈weights/reuse⌉`, `lut = ⌈1.5·dsp·bits⌉`, `ff = params·bits`, `bram = 0`; linear in bit-width by construction; never compared with a synthesis result). **Physical FPGA synthesis not performed; Vivado/Vitis HLS not installed.** Each record now carries source, tool, version and configuration (`reports/tables/hardware_summary.csv`; the 21 frozen records predate these fields and are filled with N/A).
- **hls4ml C-simulation (tier simulated; FP32 seed-42 weights, 15,000 test jets; `reports/numerical_consistency.md`).** Under the hls4ml *default* uniform type (truncation, wrap-around, accumulator = same width) accuracy is 0.20 at `ap_fixed<8,6>`, 0.35 at 10, 0.73 at 12 bits, 0.75 at ≥ 14. With **rounding + saturation** the same widths give ΔAUC −0.096 (8 bits), −0.007 (10), −0.0005 (12), 0.0000 (≥14), class agreement with FP32 0.596/0.940/0.987/0.996; adding a wide accumulator `ap_fixed<28,12>` improves 8 bits to ΔAUC −0.039. So the default-mode collapse is largely a truncation/overflow-mode artifact; what a tuned per-layer fixed-point design would need cannot be inferred here. C-simulation yields no latency, initiation interval or resources. The C-sim "INT" widths are a different scheme from the project's calibrated fake-quantization and the two are not interchangeable.

## Pareto view (no ranking; seed means; tiers kept separate)

Frontier on (ROC-AUC ↑, theoretical weight storage ↓): INT8 QAT, INT6 QAT, INT4 QAT are non-dominated among the seven n = 3 configurations (`reports/figures/phase2_fig8_pareto.png`); with n = 3–5 and SDs of 0.001–0.002 the AUC differences among FP32, INT8 and INT6 are within noise, so only the *storage-reducing* direction is informative: INT8 and INT6 occupy "≈ FP32 accuracy at 3.7× / 4.7× less theoretical weight storage"; INT4 PTQ occupies "6.6× less storage at a large, unstable accuracy cost"; INT4 QAT occupies "6.6× less storage at −0.010 AUC". Software latency is not used as a Pareto axis (not comparable to hardware latency); the estimated-LUT axis is shown in a separate panel labelled `estimated`.

## Reproducibility

- Two reproductions of `EXP_S42_JET_INT4_QAT` from its recorded config, in isolation, against the frozen reference: all metrics, weights, predictions (class agreement 1.000000, max |Δprob| 0) and metadata identical; `python -m hep_quant reproduce --experiment <ID>` now performs this check.
- **Cross-device:** same seed on CPU vs CUDA differs by ≤ 1e-3 accuracy for FP32/INT8 but by **8 accuracy points for INT4 PTQ** (0.651 vs 0.572). INT4 PTQ is sensitive to tiny numerical differences; conclusions about it must come from multi-seed, single-device statistics (as here), never one run.
- Frozen results verified unchanged (84 files). Environment: Python 3.11.16, PyTorch 2.11.0+cu128, NumPy 2.4.6, scikit-learn 1.9.1, hls4ml 1.3.0 (`requirements-lock.txt`; newer records store package versions and an environment hash).

## Limitations

Simulated, particle-level MC data with one pT window and 16 high-level features; one 4,389-parameter MLP; per-tensor min/max fake-quantization (not per-channel/learned, not bit-packed, inputs/logits/biases/accumulators unquantized); n = 5 seeds; one device for pooled statistics; no FPGA synthesis (all hardware numbers are estimated or C-simulated); latency is software overhead; QAT uses a fixed 7-epoch recipe; checkpoint studies use n = 3 seeds; PTQ calibration is min/max only (no percentile/MSE calibration was tested).

## Observations

INT8 and INT6 are not a differentiator on this task. The one large effect — INT4 — is dominated by how ranges are set and by first-layer sensitivity; QAT removes the instability at a stable ≈ 1% AUC cost. Quantization-aware training's gain over a compute-matched control is not detectable at INT8 and is negative-but-small at INT6/INT4.

## Conclusions (what the evidence supports)

*In simulation, on this simulated dataset and this MLP:* (1) INT8 and INT6 PTQ are practically lossless (≤ 0.15% relative AUC); (2) INT4 min/max PTQ is unreliable (−5.6% mean AUC, range 0.835–0.909 across seeds, device-sensitive, damaging W/Z jets and calibration most), partly attributable to calibration range and first-layer sensitivity; (3) INT4 QAT is stable at −1.0% AUC; (4) selective INT8 on the first layer/activation recovers most of the INT4-PTQ loss; (5) QAT's benefit over PTQ at INT8/INT6 is not distinguishable from extra training. **Not supported:** any statement about FPGA latency/resources, generality to other architectures/datasets, or detector-level data.

## Future experiments

Real synthesis on a Vivado/Vitis machine; per-layer-profiled hls4ml precision; percentile/MSE calibration for INT4 PTQ; mixed precision under QAT; a second architecture and dataset only if a generality claim is wanted; more seeds if the INT4-PTQ distribution itself is of interest.
