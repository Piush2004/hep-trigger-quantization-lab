# Contribution Statement

Written **after** `reports/related_work_gap.md` and limited to what that survey does not contradict and what the repository actually demonstrates.
It is not a claim of novelty: the survey was abstract-level, so it cannot show that any item below is new.

## This project contributes

1. **A frozen, hash-verified multi-seed benchmark of simulated PTQ and QAT at INT8/INT6/INT4** on the standard hls4ml jet-tagging dataset: 21 frozen runs (seeds 42/123/2026, SHA-256 manifest, 84 files) plus a 14-run extension (seeds 7/99) and 5 controls, with paired statistics
   and every number regenerated from artifacts. Evidence: `experiments/results/frozen_phase2/`, `reports/phase2_results_audit.md`, `reports/tables/{seed_statistics,quantization_comparison}.csv`.
2. **A controlled characterization of why simple INT4 post-training quantization is unreliable on this benchmark**: seed-dependent (ROC-AUC 0.835–0.909 over five seeds), device-dependent (8 accuracy points CPU vs CUDA at one seed), calibration-set-size dependent
   (more calibration data → worse INT4), concentrated in the first layer's weights and first activation, concentrated in W/Z-jet classes, and accompanied by a ~10× rise in expected calibration error. Evidence: `scripts/phase3_ptq_studies.py`, `scripts/phase3_error_analysis.py`, `reports/REAL_HEP_RESULTS.md`.
   (The mechanism "min/max range widening" is a hypothesis; ranges were not inspected.)
3. **A compute-matched control for QAT**, showing that QAT's small advantage over PTQ at INT8/INT6 is not separable from the extra fine-tuning epochs (control alone: +0.0004 AUC) and quantifying QAT's residual cost against that control (≈ 0 / −0.0005 / −0.0101 at INT8/INT6/INT4). Evidence: `EXP_C*_JET_FP32_FT`, `reports/tables/qat_vs_ptq.csv`.
4. ~~**A sensitivity-derived mixed-precision PTQ check**~~ **WITHDRAWN as a result.** Exploratory mixed-precision result (L0 + A0 at INT8; AUC 0.9278 ± 0.0037, n = 3) — excluded from confirmatory analysis due to test-set selection leakage (TEST_SET_SELECTED / BIASED; see `reports/exploratory_mixed_precision_TEST_SET_SELECTED_BIASED.md`). The validation-selected replacement is pending `reports/phase5_mixed_precision_gate.md`.
5. **A numerical-consistency study of hls4ml C-simulation** (single model): the collapse seen under hls4ml's default uniform fixed-point type is largely a truncation/wrap-around artifact; rounding + saturation gives ΔAUC −0.0005 at 12 bits. Evidence: `reports/numerical_consistency.md`. Simulated tier only.
6. **A measurement-honesty and reproducibility discipline**: every hardware number carries `measurement_tier`, source, tool, version and configuration; `measured/simulated/estimated/unavailable` are never mixed in a figure; results are frozen and verifiable; `python -m hep_quant reproduce` re-runs a recorded experiment in isolation and compares metrics, weights and predictions; environment versions and a hash are recorded.
   Evidence: `src/hep_quant/hardware/resource_estimator.py`, `src/hep_quant/experiments/reproduce.py`, `reports/reproducibility_real_EXP_S42_JET_INT4_QAT.md`.

## This project does NOT contribute

A new quantization method; hardware (FPGA) latency, resource or power results (none exist; Vivado/Vitis not available); state-of-the-art accuracy; results on detector-level data; any generality claim across architectures or datasets; a demonstrated advantage of QAT over compute-matched fine-tuning at INT8/INT6.
