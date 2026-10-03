# Research Gap Analysis (Phase 3 decision gate)

Basis: `reports/phase2_results_audit.md` and the frozen snapshot `experiments/results/frozen_phase2/` (21 runs, 3 seeds), plus the repository state at
commit `e6fc7c2`. Classification: **CRITICAL** (a stated conclusion is unsupported without it), **IMPORTANT** (materially strengthens or bounds a
conclusion), **OPTIONAL**, **NOT NEEDED**. Priority order follows: scientific validity > reproducibility > experimental depth > hardware relevance > novelty > feature count.

One framing fact: the audit found that the project's only large effect — INT4 post-training quantization — is also its least reliable one
(per-seed ROC-AUC 0.835 / 0.909 / 0.866; 8 accuracy points of CPU-vs-CUDA difference at a fixed seed). Most gaps below trace to that.

| Dimension | Current status | Evidence | Gap | Class |
|---|---|---|---|---|
| Dataset validity | OpenML `hls4ml_lhc_jets_hlf`, provenance traced to arXiv:1804.06913; validation PASS twice (byte-identical) | `reports/dataset_selection.md`, `real_data_validation.json`, audit §10 | Particle-level MC only; one dataset | IMPORTANT (limit, not fixable here; claims must stay scoped) |
| Baseline validity | FP32 ROC-AUC 0.934 ± 0.001, acc 0.753 ± 0.004; consistent with the 75% reference (arXiv:2003.06308 Table 5) | audit §3, §10 | Not a reproduction (different split, lr, budget, 100k vs ~1M jets). Not needed for the quantization question | OPTIONAL |
| Quantization methodology | Simulated per-tensor uniform-affine fake-quant of Linear weights + hidden ReLU outputs; inputs, logits, biases, accumulators unquantized; no per-channel; storage never bit-packed | audit §6 | Component-level effect unknown (is INT4 failure in weights or activations? which layer?); scope of "INT-N" under-documented | IMPORTANT |
| PTQ | Min/max calibration on 20,480 rows; INT8/INT6 near-lossless; INT4 unstable | audit §3 | Sensitivity to calibration set size/draw untested; weights-vs-activations attribution untested | IMPORTANT |
| QAT | Stable at INT4 (−1.0% AUC); ≥ PTQ in 3/3 seeds at every precision | audit §3 | **Not compute-matched**: QAT gets +7 epochs at 0.1× lr that PTQ does not; no FP32-fine-tune control | **CRITICAL** |
| Multi-seed reliability | n = 3 | audit §3 | SD of INT4 PTQ (0.037) is itself poorly estimated with n = 3 | IMPORTANT |
| Statistical analysis | Mean / sample SD / min / max; paired deltas; no tests | audit §3–5 | No confidence intervals or effect sizes; none are meaningful at n = 3 | IMPORTANT (needs n = 5) |
| Latency methodology | Batch 1, CUDA, 100 reps, synchronized; mean/p95 stored | audit §7 | Overhead-dominated; `.item()` syncs inflate fake-quant; no median/SD stored; some runs shared the machine; no CPU measurement for the 21 runs; only mean/p95 recorded | IMPORTANT |
| Model-size methodology | Analytical "theoretical weight storage"; renamed in reporting; verified against checkpoints (grid levels ≤ 2^bits) | audit §6 | No physically bit-packed artifact | NOT NEEDED (labelled correctly; packing would not change any conclusion) |
| hls4ml validation | One model, uniform `ap_fixed<W,6>` for everything incl. accumulator, TRN/WRAP defaults; accuracy 20% at W=8 → 75% at W≥14 | audit §8 | No numerical consistency vs quantized PyTorch; low-width collapse not attributed (rounding? wrap? accumulator?) | IMPORTANT |
| FPGA resource evidence | None. Resources are a linear software heuristic (`estimated`) | audit §9 | No synthesis; Vivado/Vitis not installed. Cannot be remedied in this environment | **CRITICAL for any hardware claim → remedy = restrict claims**, not new work |
| Architecture diversity | One MLP (4,389 params) | — | "Does sensitivity depend on architecture?" unanswerable | OPTIONAL (only needed for a generality claim, which is not being made) |
| Dataset diversity | One dataset | — | No external validation | OPTIONAL (same reasoning) |
| Per-class robustness | Aggregate macro metrics only | — | Unknown whether INT4 hurts one class disproportionately | IMPORTANT (cheap; uses existing checkpoints) |
| Probability calibration | Not measured | — | Unknown whether quantization changes confidence | OPTIONAL (cheap; included with the per-class analysis) |
| Reproducibility | Two bit-identical reproductions on the same device; frozen manifest; recorded commit ≠ executed code for the 21 runs (dirty tree); no environment lock; no one-command reproduce | audit §1, §12 | Cross-device INT4 PTQ differs; provenance gap; no `reproduce` command; numpy/sklearn versions not recorded | IMPORTANT |
| Scientific novelty | None claimed | — | Related work never surveyed | IMPORTANT (must precede any contribution statement) |
| External validity | Simulated, pT 0.8–1.6 TeV, 16 high-level features | audit §10–11 | Detector effects, pileup, other pT regimes untested | IMPORTANT (documented limitation; not fixable here) |

## Decision gate

**To implement (cheap, no new infrastructure, each tied to a gap above):**

1. *Compute-matched FP32 control* (QAT row, CRITICAL): FP32 trained then fine-tuned with the identical QAT schedule but no quantization. New config + one small runner branch; 5 runs of ~45 s.
2. *Two more seeds* (7, 99) for the full 7-configuration matrix → n = 5 (multi-seed + statistics rows). New IDs; the frozen n = 3 snapshot is untouched and stays the primary record. t-based 95% CIs and paired effect sizes reported with explicit small-n warnings; no p-values.
3. *PTQ studies on the existing frozen FP32 checkpoints* (no retraining): calibration-set size and draw sensitivity; weights-only vs activations-only; per-layer sensitivity; and — only because the sensitivity result will exist — a small mixed-precision check derived from it (not arbitrary combinations).
4. *Per-class, calibration and error analysis* on all frozen checkpoints (existing probabilities; no new runs).
5. *Numerical consistency* FP32 PyTorch ↔ quantized PyTorch ↔ hls4ml fixed point, plus a rounding/overflow/accumulator ablation of the C-simulation to attribute the low-width collapse.
6. *Standalone latency study* on the frozen seed-42 checkpoints (CPU and CUDA-events, batch 1 and large batch, mean/median/SD/p50/p95), reported separately per device; the 21-run latency stays labelled indicative.
7. *Reproducibility*: `python -m hep_quant reproduce --experiment <ID>`; registry/provenance fields (environment versions, calibration size, hardware tier source); pinned environment lock; CUDA timeline documented.
8. *Related-work survey* then a contribution statement limited to what survives it.

**Deliberately NOT implemented:** a second architecture; a second dataset (and therefore no `secondary_dataset_selection.md`); physical or "measured" hardware numbers (no tool);
power/energy measurement; a ParticleNet/Transformer; per-channel or learned-step quantizers; real bit-packing. Each is either only needed for a generality claim this project will not make,
or cannot be done honestly here. Revisit only if the studies above expose a concrete problem that requires one.

**Stop rule:** stop when the INT4-PTQ instability has been attributed (calibration vs weights vs activations vs layer), the QAT confound has been controlled, n = 5 is in, hardware claims are scoped, and the limitations are written down.
