# Hostile peer review

Three reviewers each tried to invalidate the work. For each criticism: status, evidence, response, and any remaining limitation. Criticisms fixed by documentation were fixed in documentation; no new experiment was run unless a criticism would otherwise have made a conclusion scientifically invalid.

Status key: **RESOLVED** (fixed and checked), **DOCUMENTED** (stated as a limitation, conclusion scoped accordingly), **OPEN** (not fixable in this project; conclusion withdrawn or not made).

---

## Reviewer 1 — HEP researcher

**1.1 Is the dataset appropriate for a trigger-physics claim?**
Strongest criticism: it is particle-level Monte Carlo with no detector simulation, pT 0.8–1.6 TeV, and 16 high-level features; a trigger sees raw detector data.
Evidence: `reports/dataset_selection.md`, `dataset_schema.md`; provenance verified against arXiv:1804.06913 (no detector simulation described).
Response: the study makes no claim about trigger performance on detector data. It is a benchmark on a public simulated jet dataset used by the hls4ml community.
Status: **DOCUMENTED.** Remaining limitation: results do not transfer to detector-level or trigger-level inputs.

**1.2 Is the dataset provenance described correctly?**
Strongest criticism: earlier documents said "Pythia8 + Delphes", "~880,000 jets" and "JetSet". All three were wrong.
Evidence: the correction history is in `reports/phase2_results_audit.md` and `FINAL_STATUS.md`; the OpenML record gives 830,000 jets; the dataset paper gives MadGraph5 + Pythia8 + FastJet.
Response: corrected in the repository. One residual inconsistency remains: a fetched summary of a later paper reports 870,000 jets; the verified source is the 830,000-jet OpenML/Zenodo record.
Status: **RESOLVED** for this project's documents; **DOCUMENTED** for the residual citation discrepancy, which is in a secondary summary and was not verified against the PDF.

**1.3 Is the FP32 baseline credible relative to the literature?**
Strongest criticism: the baseline might be weak.
Evidence: FP32 ROC-AUC ~0.934 macro OvR; accuracy ~0.75 on the same 5-class task. The published 75% for a 16-64-32-32-5 MLP (arXiv:2003.06308, Table 5, as summarised) is consistent.
Response: consistency, not reproduction. Different split and training budget.
Status: **DOCUMENTED.**

**1.4 Physics interpretation: does "INT8 is lossless" mean anything for physics?**
Strongest criticism: ROC-AUC on a 5-class jet task is not a physics figure of merit (no signal efficiency vs background rejection at a working point).
Response: the study reports ROC-AUC, PR-AUC, F1 and accuracy as ML metrics only. It makes no trigger-efficiency claim.
Status: **DOCUMENTED.** Remaining limitation: no physics operating-point analysis.

**1.5 Generalization beyond this task.**
Strongest criticism: one dataset, one task.
Response: Phase 6 adds a second MLP. No second dataset was added.
Status: **DOCUMENTED.** Remaining limitation: generalization is established across two MLPs on one simulated dataset only.

---

## Reviewer 2 — ML researcher

**2.1 Is the quantization methodology correct?**
Strongest criticism: fake quantization with a per-tensor min/max range, zero-point not rounded to an integer, inputs/logits/biases unquantized. That is not what an integer deployment does.
Evidence: `src/hep_quant/quantization/ptq.py` (`fake_quantize_tensor`); `docs/quantization.md`.
Response: the implementation is internally consistent and documented as simulation. Integer-only deployment is not claimed. The unrounded zero-point is a real deviation from integer hardware and is recorded as a limitation.
Status: **DOCUMENTED.** Open item for a real deployment: integer zero-points and quantized biases/accumulators.

**2.2 Is the PTQ vs QAT comparison fair?**
Strongest criticism: QAT gets extra training epochs that PTQ does not.
Evidence: Phase 3 compute-matched control (`EXP_C*`, FP32 with the same extra epochs). QAT − control: INT8 +0.0001, INT6 −0.0005, INT4 −0.0101 ± 0.0023.
Response: the control shows QAT's INT8/INT6 advantage over PTQ is not separable from extra training. The INT4 advantage remains after the control. Architecture B's control was not run.
Status: **RESOLVED** for architecture A; **DOCUMENTED** for architecture B (no control run; whether QAT's INT4 gain is quantization-awareness or extra training is not established for B).

**2.3 Calibration.**
Strongest criticism: min/max calibration on a training subset is naive; percentile or MSE calibration might remove the INT4 instability.
Evidence: Phase 4 calibration-size study (INT4 AUC falls with calibration size through a widening activation range; clip rate ≈ 0 under min/max). `reports/int4_ptq_stability_analysis.md`.
Response: calibration sensitivity is characterised for min/max only. Other calibrators were not tested.
Status: **DOCUMENTED.** Open: percentile/MSE calibration not run; the claim is about min/max PTQ only.

**2.4 Seeds.**
Strongest criticism: five seeds is small.
Evidence: the seed controls subsample, split, initialisation and shuffling. Phase 4 showed the spread is mostly model-seed (between-seed SD 0.029 vs within-seed calibration 0.008).
Response: intervals are descriptive over five seeds and are reported as such; no significance is claimed on a single pooled test.
Status: **DOCUMENTED.** Remaining limitation: n = 5 is small for estimating the spread itself.

**2.5 Statistics.**
Strongest criticism: earlier documents used "significant" loosely, and mixed descriptive intervals with tests.
Response: `reports/final_statistical_audit.md` separates statistical from practical significance and uses paired intervals only where stated. No p-values are reported as results.
Status: **RESOLVED.**

**2.6 Model-size definitions.**
Strongest criticism: "model size" was once reported as file size (MB), which is misleading for float32 checkpoints.
Evidence: the earlier column was renamed to `theoretical_weight_storage_mb`; the audit table `phase2_model_size_audit.csv` reports serialized checkpoint bytes and theoretical storage separately.
Response: all size claims in the final table are theoretical weight storage, labelled as analytical. Checkpoints are float32.
Status: **RESOLVED.**

**2.7 Test-set leakage.**
Strongest criticism: the Phase 3 mixed-precision layout was chosen using test-split ROC-AUC.
Evidence: the original S3 per-unit sensitivity scored on the test split. Confirmed and labelled `TEST_SET_SELECTED — BIASED — EXCLUDED`. Corrected by the validation-only gate (Phase 5) and the size-constrained gate (Phase 5B), which were verified byte-identical across two runs and scored the test set once for the selected layout.
Status: **RESOLVED** for the mixed-precision branch. The biased 0.9278 figure is kept, labelled, and excluded from the confirmatory analysis.

**2.8 Mixed-precision search.**
Strongest criticism: nine layouts is a small search; a better layout might exist.
Response: the claim is limited to "no evaluated mixed layout beat INT6 under 3,724 B within the pre-declared space". Optimality is not claimed.
Status: **DOCUMENTED.**

---

## Reviewer 3 — FPGA researcher

**3.1 Are there hardware synthesis claims?**
Strongest criticism: any LUT/DSP/BRAM number is an unsupported estimate.
Evidence: no Vivado/Vitis/Quartus available (checked 2026-10-05). The resource numbers are a linear software heuristic, labelled `estimated` throughout.
Response: all resource figures are labelled ESTIMATED. The text states "HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT".
Status: **RESOLVED** (no synthesis claimed); the limitation remains that no synthesis exists.

**3.2 Is the hls4ml validation sufficient for its conclusion?**
Strongest criticism: C-simulation confirms arithmetic, not hardware behaviour; the collapse at narrow widths might be a conversion artefact.
Evidence: `reports/numerical_consistency.json` and `reports/hls_csim_accuracy.md`; C-simulation is labelled SIMULATED. The arithmetic ablation separates default truncation/wrap from rounding+saturation and from a wide accumulator. Rounding+saturation removes most of the collapse at W = 8 (accuracy 0.20 → 0.50) and at W = 10 (0.35 → 0.73).
Response: the claim is that fixed-point arithmetic configuration materially changes numerical fidelity in C-simulation. It is not a claim about hardware.
Status: **DOCUMENTED.** Open: rounding and saturation were not separated into individual factors; a full factorial was not run.

**3.3 Latency methodology.**
Strongest criticism: software timings are emulation overhead and were presented as latency.
Evidence: `reports/phase3_latency_study.json`; the README and docs state that software timings are not hardware latency.
Response: latency is labelled software emulation and is not used as a hardware claim. No hardware latency or initiation interval exists.
Status: **DOCUMENTED.** Open: no hardware latency.

**3.4 Evidence tiers.**
Strongest criticism: estimated, simulated and measured numbers could be mixed in one figure or table.
Response: each table and figure carries the tier (`measured`, `simulated`, `estimated`, `unavailable`). The Phase 8 table contains no hardware column.
Status: **RESOLVED.**

**3.5 Reproducibility of hls4ml results.**
Strongest criticism: C-simulation might not be deterministic.
Evidence: `scripts/phase3_numerical_consistency.py` was run twice in this sprint. Both outputs are byte-identical to each other and to the JSON that existed before the sprint (`reports/phase7_hls_numerical_validation.md`, §7).
Status: **RESOLVED** for C-simulation determinism on this machine. Remaining limitation: C-simulation is not hardware, and the single model and single test subset are the only inputs tested.

---

## Summary
- Resolved: test-set leakage in mixed precision; storage definitions; statements about synthesis; provenance errors that were in earlier documents.
- Documented as limitations: simulation vs integer deployment; min/max calibration only; n = 5 seeds; no physics operating point; no hardware measurements; no second dataset.
- Open and not fixable here: hardware synthesis and physical measurement; integer zero-point deployment; detector-level data.
- No new experiment was run in this review beyond the Phase 7 re-execution, because no criticism required one to avoid an invalid conclusion.
