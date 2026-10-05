# Final Project Status (after Phase 3)

## Research readiness

**Reproducible Benchmark** (unchanged classification, now better evidenced). Not "Preliminary Research Study" or "Publication Candidate": there is no FPGA synthesis, one small architecture, one simulated dataset, n = 5 seeds, and no novelty claim is supportable
from the (abstract-level) related-work survey. What justifies "Reproducible Benchmark": frozen hash-verified evidence, a documented dataset, paired multi-seed statistics, a compute-matched control, bit-identical re-runs, and strict labeling of what is measured, simulated or estimated.

## Research question

How do post-training quantization and quantization-aware training affect accuracy, model size, inference characteristics and hardware-oriented characteristics of a small neural network for HEP jet classification? Answered here **in simulation only** (fake-quantization) on one simulated dataset and one MLP.

## Dataset

OpenML `hls4ml_lhc_jets_hlf` v1 (Zenodo 10.5281/zenodo.3602260, CC BY 4.0): **830,000 jets, 16 features, 5 classes**; particle-level Monte Carlo (MadGraph5 + Pythia8 + FastJet; the dataset paper describes no detector simulation). **Not detector-level data, not recorded collisions, not JetSet.**
Validated twice from scratch (byte-identical reports; 0 NaN/Inf/duplicates; no single-feature leakage). Documents: `reports/dataset_selection.md`, `dataset_schema.md`, `real_data_validation.md`. A synthetic dataset remains only for pipeline tests.

## Experiments

- **Frozen primary set** (`experiments/results/frozen_phase2/`, 84 files, SHA-256 manifest, verified unchanged): 3 seeds (42, 123, 2026) × 7 configurations (FP32; PTQ and QAT at INT8/INT6/INT4) = 21 runs, all CUDA.
- **Added in Phase 3 (new IDs):** seeds 7 and 99 × 7 configurations (14 runs; n = 5 total) and 5 compute-matched FP32 fine-tune controls (`EXP_C*`). Checkpoint-level studies on the frozen FP32 weights (calibration, weights-vs-activations, per-unit, mixed precision, per-class/calibration/error analysis); numerical-consistency and fixed-point-mode ablation of hls4ml C-simulation; controlled latency study.
- Earlier work retained, kept separate: synthetic-dataset matrix (EXP_001–009, pipeline validation only) and the CPU single-seed real-data matrix (EXP_010–018; superseded by the multi-seed results).
- Deliberately **not** run: a second architecture, a second dataset, ParticleNet/Transformer, power/energy measurement, anything claiming measured hardware numbers (see `reports/research_gap_analysis.md` for the reasoning).

## Statistical reliability

n = 5 seeds (n = 3 for checkpoint studies); sample SD, 95% t-intervals and paired effect sizes are descriptive only — no significance tests. FP32 ROC-AUC 0.9343 ± 0.0011; INT4 PTQ 0.8819 ± 0.0308 (0.835–0.909): its spread is the dominant uncertainty. Seeds control subsample, split, init and shuffling.

## Quantization findings (simulation; this dataset; this MLP)

INT8 PTQ ΔAUC −0.0001; INT6 PTQ −0.0014 (−0.15%): practically lossless. **INT4 PTQ −0.052 (−5.6%) and unreliable**: seed-dependent, device-dependent (same seed: accuracy 0.651 CPU vs 0.572 CUDA), calibration-size-dependent (more calibration data → worse), concentrated in the first layer (L0 weights, A0 activation), damaging W/Z jets (Δrecall −0.25/−0.29) and probability calibration (ECE 0.009 → 0.089). A sensitivity-derived PTQ variant (L0 + A0 at INT8, rest INT4) reaches AUC 0.9278 ± 0.0037 (n = 3) at 3,172 B theoretical storage.

## QAT findings

QAT is stable at INT4 (ΔAUC −0.0097, −1.0%). QAT − PTQ: +0.0006 (INT8, 4/5 seeds), +0.0013 (INT6, 5/5), +0.043 (INT4, 5/5). **Against a compute-matched FP32 fine-tune control** (extra epochs alone: +0.0004): QAT INT8 +0.0001 ± 0.0007, INT6 −0.0005 ± 0.0006, INT4 −0.0101 ± 0.0023. QAT's INT8/INT6 edge over PTQ is therefore not attributable to quantization-awareness; its value is rescuing INT4. The earlier single-seed claim "QAT recovers most of a 9.9-point collapse" is superseded.

## Hardware findings

Theoretical weight storage (analytical, not file size): 17,556 / 4,788 / 3,724 / 2,660 B (FP32/INT8/INT6/INT4). LUT/DSP/BRAM/FF: **estimated** by a linear software heuristic, never validated against synthesis. Software latency (controlled CPU and CUDA-event study) rises FP32 → PTQ → QAT on both devices and at batch 4096: emulation overhead, not hardware latency. **Physical FPGA synthesis not performed. A toolchain check on 2026-10-05 (WSL PATH, standard Xilinx/Intel install dirs in WSL and Windows) found no Vivado, Vitis HLS, Quartus or similar; only g++ (used for hls4ml C-simulation). No measured hardware number exists.**

## hls4ml findings

hls4ml 1.3.0 conversion and C-simulation work (tier **simulated**). Default uniform `ap_fixed<W,6>` (truncation, wrap-around, accumulator = W): accuracy 0.20 / 0.35 / 0.73 / 0.75 at W = 8 / 10 / 12 / ≥ 14. With rounding + saturation: ΔAUC −0.096 / −0.007 / −0.0005 / 0.000; class agreement with FP32 0.596 / 0.940 / 0.987 / ≥ 0.996 — the default-mode collapse is largely an arithmetic-mode artifact. C-simulation gives no latency/II/resources. Single model, single run.

## Related work and contribution

`reports/related_work_gap.md` (abstract-level survey): fixed-point/low-precision MLP jet tagging on this dataset, QAT, automatic mixed precision (QKeras/AutoQKeras, HGQ) and FPGA synthesis are established. `reports/contribution_statement.md`: controlled multi-seed/device analysis, compute-matched QAT control, INT4-PTQ attribution studies, measurement-tier discipline and frozen reproducible artifacts. **No novelty claim.**

## Reproducibility

Two isolated re-runs of `EXP_S42_JET_INT4_QAT` were bit-identical (metrics, weights, predictions); `python -m hep_quant reproduce --experiment <ID>` automates it. Frozen snapshot verified (84 files unchanged). Pinned environment (`requirements-lock.txt`, `environment.yml`; Python 3.11.16, PyTorch 2.11.0+cu128, NumPy 2.4.6, scikit-learn 1.9.1, hls4ml 1.3.0). Not reproducible across devices for INT4 PTQ.
Provenance gap: the 21 frozen runs recorded commit `4e9b673` while the working tree was dirty (executed code = commit `b44311d`); records now carry `git_dirty`, package versions, an environment hash, dataset fingerprint and calibration size.
**CUDA:** worked early, was unavailable from between EXP_008 and EXP_009 until 2026-10-03 (cause not established; the earlier attribution to a `wsl --shutdown` was wrong), and was lost once more mid-run on 2026-10-03 (recovered by a WSL restart). Timeline in `environment_report.md`; pooled statistics never mix devices.

## Tests

`pytest -q`: **92 passed, 4 skipped** on run 1 and **92 passed, 4 skipped** on run 2 (the 4 skips are opt-in network tests). Report-generation pipeline executed afterwards; frozen snapshot re-verified (84 files, 0 changed). `python -m hep_quant reproduce` exercised on `EXP_S42_JET_INT8_PTQ` (PASS, bit-identical, 2 repeats).

Re-check on 2026-10-05 (docs-only changes since): **91 passed, 5 skipped** on two consecutive runs. The extra skip versus 92/4 is `tests/test_hardware.py:120` (GPU-model conversion), skipped because CUDA was again unavailable in WSL (`torch.cuda.is_available()` = False; cause still not established). Frozen snapshot re-verified: 84 files, 0 changed.

## Limitations

Simulated particle-level MC, one pT window, 16 high-level features; one 4,389-parameter MLP; simple per-tensor min/max fake-quantization (inputs/logits/biases/accumulators unquantized; not bit-packed); untuned QAT recipe and naive PTQ calibration; n = 5 seeds; one device for pooled results; no FPGA synthesis; software latency only; INT4 PTQ point values not portable across devices.

## Remaining gaps (ranked)

1. FPGA synthesis on a machine with Vivado/Vitis (the only way to obtain measured hardware numbers).
2. Percentile/MSE calibration and per-channel/learned quantizers, to separate "INT4 is hard" from "min/max calibration is weak".
3. Mixed precision under QAT and with more seeds.
4. A second architecture and dataset, only if a generality claim is desired.
5. A full-text related-work review before any novelty claim.
