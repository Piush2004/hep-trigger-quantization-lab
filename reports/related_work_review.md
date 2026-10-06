# Related-Work Review (Phase 4, Phase 1)

Scope: closely related primary literature for the question *PTQ vs QAT, INT8/INT6/INT4, on HEP jet classification, with hardware-oriented deployment*.

**Verification levels used below**
- **F** = arXiv/publisher page fetched and the stated fields extracted (still an automated summary; re-check the PDF before citing a specific number).
- **S** = only a search-result snippet was available; the field is not verified.
- **N** = not established from any source read in this review.

Where a field is not established, it is written as `NOT ESTABLISHED`. Nothing here is taken from a blog.

---

## 1. Closely related papers

### 1.1 Duarte et al. (2018), *Fast inference of deep neural networks in FPGAs for particle physics*
- Citation: J. Duarte et al., *Journal of Instrumentation* (2018), arXiv:1804.06913. Journal volume/page **NOT VERIFIED**. **S** (search snippet: jet-substructure classifier, hls4ml companion compiler, ~100 ns latency scale).
- Dataset: jet substructure (the hls4ml jet task). Details **NOT ESTABLISHED** from a read of the full text.
- Architecture: NOT ESTABLISHED.
- Precision: floating point / fixed point (details NOT ESTABLISHED).
- PTQ/QAT: NOT ESTABLISHED.
- Hardware target / synthesis: FPGA; the snippet says the design fits the resources of modern FPGAs. Tool and version NOT ESTABLISHED.
- Similarity: same application domain and the hls4ml toolchain used in this project.
- Difference: a general FPGA-inference case study, not a PTQ-vs-QAT comparison.

### 1.2 Coelho et al. (2021), *Automatic heterogeneous quantization of deep neural networks for low-latency inference on the edge for particle detectors*, Nature Machine Intelligence 3 (2021); arXiv:2006.10159. **F** (summary page; the PDF fetch could not be parsed reliably, so the numbers below are from the abstract-level page and one secondary summary).
- Dataset: the public LHC jet dataset (16 high-level features, five classes). Size NOT ESTABLISHED.
- Architecture: 16-input MLP with three hidden layers. The fetched page does not give widths in the abstract, so **NOT ESTABLISHED** beyond that.
- Precision: heterogeneous per-layer, per-parameter bit-widths, searched by AutoQKeras over quantizers including binary, ternary and quantized_bits. Baseline bit-width NOT ESTABLISHED from the fetched text.
- PTQ/QAT: **QAT** (QKeras). A PTQ baseline is also described, per a secondary summary.
- Hardware target: Xilinx Virtex UltraScale+ VU9P, 200 MHz, Vivado HLS (version reported as 2019.2 in one secondary source; **not** verified from the primary page).
- FPGA synthesis: **yes**, post-synthesis. The summary reports ~98% DSP and ~80% LUT reduction for the energy-optimized model, and accuracy of about 72.3% against a 74.8% baseline. Those figures are from a secondary summary, so **treat as S, not F**.
- Metrics: accuracy, latency, resource and energy estimates. QTools energy estimates are reported as validated against post-place-and-route power reports (a claim from the paper, **S**).
- Major conclusion: automatic heterogeneous QAT finds accuracy/resource trade-offs well beyond uniform quantization.
- Similarity: same task family, the same kind of MLP, QAT, and FPGA-resource framing.
- Difference: heterogeneous search with QAT and real synthesis. This project has uniform per-tensor fake-quantization, no synthesis, and a controlled FP32-finetune control.

### 1.3 Hawks et al. (2021), *Ps and Qs: Quantization-Aware Pruning for Efficient Low Latency Neural Network Inference*, Frontiers in AI (2021); arXiv:2102.11289. **F** (fetched; the summary is an automated extraction).
- Dataset: the hls4ml LHC jet dataset. The fetched summary states **870,000 jets**. **Conflict:** this project's verified OpenML `hls4ml_lhc_jets_hlf` v1 has **830,000 jets**. This discrepancy is unresolved and is raised again in §5.
- Architecture: 16 → 64 → 32 → 32 → 5 MLP with ReLU and softmax, optional batch normalisation. This is the **same architecture** as the present project.
- Precision: 32-bit float, 12-bit, 6-bit and 4-bit scaled-integer. The fetched summary says the 6-bit model "maintained comparable performance to full precision while 4-bit showed degradation".
- PTQ/QAT: **QAT** with Brevitas (uniform affine quantization, straight-through estimator, learned activation scales), combined with pruning (fine-tuning and lottery-ticket variants).
- Hardware target: none synthesised. The paper says this is future work using hls4ml. **FPGA synthesis: not performed.**
- Metrics: background efficiency at 50% signal efficiency, bit operations (BOPs), neural efficiency (Shannon entropy).
- Major conclusion: quantization-aware pruning yields more efficient models than pruning or quantization alone; the best configuration reaches a 25× BOPs reduction at 80% pruning with negligible accuracy loss.
- Similarity: **high.** Same dataset family, same architecture, same bit-width range (6-bit and 4-bit), QAT, and the same HEP motivation.
- Difference: this project uses **PTQ** as well as QAT, a **compute-matched FP32 fine-tune control** (the prior work does not describe one, as far as the fetched text shows), **seed-level variance** (n = 5 with a paired analysis), **device-dependence analysis**, **calibration-sensitivity** work, and an **explicit measurement-tier discipline**. The prior work uses no pruning; this project uses none.

### 1.4 QKeras / AutoQKeras library
- github.com/google/qkeras (library documentation; the AutoQKeras method is the one in §1.2). **S**.

### 1.5 *HGQ: High Granularity Quantization for Real-time Neural Networks on FPGAs*, arXiv:2405.00645 (2024). **F** (fetched; the PDF text extraction was partial). Author list **NOT VERIFIED**; check before citing.
- Dataset: the jet substructure classification (JSC) benchmark in two variants: **HLF** (16 features, five classes) and **PLF** (particle-level features, up to 150 features). Also SVHN and a muon-tracking regression.
- Architecture: HLF uses a fully connected network with datalanes **16-64-32-32-5**, the same as this project. PLF uses a linearised GNN (JEDI-linear).
- Precision: gradient-based **per-weight and per-activation bit-width** optimisation, including zero-bit (pruned) weights. The fetched text says that at 75.3% accuracy weights use up to 8 bits, and at 72.4% the maximum is 4 bits, with most weights pruned to zero.
- PTQ/QAT: **QAT** (gradient-based, hardware-aware, with an EBOPs-type resource term).
- Hardware: **Vivado 2025.1**, Xilinx Virtex UltraScale+ **VU13P**, **place-and-route out-of-context** with retiming. Latency and LUT/DSP numbers: **S** (not extracted reliably).
- Major conclusion: HGQ outperforms previous quantized-NN methods on the HLF JSC by a large margin, with large resource and latency reductions.
- Similarity: **high** on task, architecture and the mixed-precision/low-bit theme.
- Difference: HGQ is a per-parameter *trainable* bit-width method with a resource penalty and **real post-P&R** numbers. This project has no trained per-weight bit-width and no synthesis.

### 1.6 *Compressing deep neural networks on FPGAs to binary and ternary precision with hls4ml*, arXiv:2003.06308, Machine Learning: Science and Technology (2020). **F** (fetched). Author list **NOT VERIFIED**; check before citing.
- Dataset: MNIST and simulated LHC jet data (the hls4ml jet dataset).
- Architecture: MLP 16-64-32-32-5 for jets (**this is the reference architecture used by the present project**).
- Precision: binary and ternary, plus a **⟨16,6⟩** fixed-point baseline. The fetched text says ⟨16,6⟩ "is sufficient to reproduce the full-precision accuracy".
- PTQ/QAT: training with binary/ternary weights in the forward pass and full-precision gradients (a QAT-like scheme).
- Hardware: Xilinx **Virtex UltraScale+ VU9P (xcvu9p-flga2104-2L-e)**, **Vivado HLS 2018.2**, 200 MHz, reported latency about 60 ns (II = 1) and 57% DSP utilisation, **post-synthesis**.
- Accuracy: full precision ~75%, consistent with this project's FP32 baseline. The project's reference (arXiv:2003.06308 Table 5) is the baseline cited in the README; **this review re-confirmed the 75% figure but did not re-read Table 5 directly.**
- Major conclusion: binary and ternary networks achieve similar performance to higher precision with far fewer FPGA resources.
- Similarity: same dataset, same architecture, same toolchain.
- Difference: this project does not use binary/ternary, and does not synthesise. Its fixed-point study is C-simulation only.

### 1.7 Jet tagging and Level-1 applications
- **AXOL1TL** (CMS Level-1 anomaly detection, arXiv:2411.19506; arXiv:2511.01908). **S.** A VAE trained with QKeras QAT and deployed with hls4ml. Synthesised for the µGT 50 ns constraint. Different task (anomaly detection, not jet classification), so it is **context** only.
- **Knowledge distillation of a normalising flow at the LHC Level-1 trigger** (arXiv:2609.15295). **S.** Not read; not used for claims.
- **Sub-microsecond Transformers for jet tagging on FPGAs** (arXiv:2510.24784). **S.** Not read; not used for claims.
- **RQP: Resource-Oriented Quantiser Pruning** (arXiv:2606.30382). **S.** Not read; not used for claims.
- **Hessian-aware QNN codesign** (arXiv:2304.06745, Hawks/Duarte et al.). **S.** Layer-sensitivity-guided mixed precision via HAWQ-style Hessian spectra. Relevant to the mixed-precision discussion, but not read in full.
- **JEDI-linear** (arXiv:2508.15468). **S.** Not used.

### 1.8 Generic PTQ calibration literature (non-HEP, supporting context only)
- Percentile, MSE and entropy calibration are standard PTQ range-setting methods. A 2026 preprint on evolution-strategy calibration (arXiv:2603.08173) reports Max calibration degrading at INT4 while percentile/MSE do better in its experiments. **S** (snippet only). These are **not HEP** results and are used only to motivate a calibration-method comparison, which this project has **not** run.
- Coverage-based PTQ calibration (arXiv:2604.24008). **S.** Not used.

---

## 2. Status of each requested topic

| # | Topic | Status | Primary papers in this review |
|---|---|---|---|
| 1 | Quantization of NNs for HEP | Established, active | 1.2, 1.3, 1.5, 1.6 |
| 2 | PTQ for HEP classification | **Thin.** Few papers report a PTQ-only analysis; §1.6 (fixed-point) and 1.2 (PTQ baseline) touch it | 1.2, 1.6 |
| 3 | QAT for HEP classification | Established | 1.2, 1.3, 1.5 |
| 4 | hls4ml quantization | Established | 1.1, 1.6 |
| 5 | QKeras | Established | 1.2, 1.5 (library) |
| 6 | FPGA NN inference for HEP triggers | Established | 1.1, 1.6, 1.7 |
| 7 | Low-bit NNs for Level-1 triggers | Established but mostly anomaly detection | 1.7 (AXOL1TL) |
| 8 | Jet-classification quantization | **Well covered** (same dataset, same MLP) | 1.3, 1.5, 1.6 |
| 9 | Fixed-point numerical effects in hls4ml | **Not established from primary sources in this review.** Default-rounding facts are from a search snippet of AMD/mintlify docs (S). | §5 |
| 10 | Calibration sensitivity of PTQ | Generic literature only (S); **no HEP-specific study found** | 1.8 |
| 11 | Mixed-precision HEP inference | Covered (HGQ, HAWQ-style) | 1.5, 1.7 |
| 12 | Existing PTQ-vs-QAT benchmarks in HEP | **No head-to-head PTQ-vs-QAT benchmark on the jet dataset was found.** Searches returned only general statements. Not conclusive: the search was not exhaustive. | §5 |

---

## 3. Points the literature already settles (to be carried into the novelty analysis)
- QAT outperforms naive PTQ at aggressive bit-widths on this kind of MLP (1.2, 1.3).
- A 16-64-32-32-5 MLP on this dataset tolerates ~6-bit quantization (1.3, 1.6).
- Low-bit and heterogeneous QAT with hls4ml/QKeras and synthesis is established (1.2, 1.5).
- Hardware-aware and per-parameter bit-width optimisation is established (1.5).

## 4. Points the literature does **not** clearly settle (candidate contributions, to be tested, not claimed)
- Whether the **INT4 PTQ instability** on this task is calibration-, seed- or numerically driven. No source found studies it.
- A **compute-matched control** distinguishing QAT's benefit from extra training epochs (not seen in the fetched HEP papers).
- Device-dependence of INT4 PTQ point estimates.
- Measurement-tier discipline when reporting FPGA resource numbers (estimated vs synthesised).

## 5. Unresolved items in this review
1. **Dataset size conflict**: 870,000 (Hawks et al., fetched summary) vs 830,000 (OpenML/Zenodo, verified in Phase 2). The OpenML record is the authority for this project; the paper discrepancy must be checked against the original PDF before any citation.
2. **Primary PDFs could not be parsed** for several papers. Their numeric fields remain **S** until checked by hand.
3. **Search was not exhaustive.** The absence of a PTQ-vs-QAT benchmark is a finding *of this search*, not proof of absence.
4. **hls4ml default rounding/saturation** (TRN/WRAP): the search snippet agrees with the project's C-simulation finding but the primary hls4ml/AMD documentation was not read in full (the fetched hls4ml page did not contain the detail).
5. **Duarte et al. 2018** full text not read.

## 6. Source list
- Duarte et al., arXiv:1804.06913 (S)
- Coelho et al., arXiv:2006.10159 / Nature Machine Intelligence 2021 (F, summary)
- Hawks et al., arXiv:2102.11289 / Frontiers in AI 2021 (F, summary)
- Sun/Ngadiuba et al., arXiv:2405.00645 (F, partial)
- Ngadiuba et al. / Hawks et al., arXiv:2003.06308 / MLST 2020 (F)
- AXOL1TL, arXiv:2411.19506, arXiv:2511.01908 (S)
- Hessian-aware codesign, arXiv:2304.06745 (S)
- hls4ml default rounding: AMD UG1399 "Arbitrary Precision Fixed Point Data Types" (S)
