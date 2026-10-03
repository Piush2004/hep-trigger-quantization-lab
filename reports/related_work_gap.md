# Related-Work Gap Analysis

**Method and its limits.** Web searches plus reading of abstracts/landing pages (and, for arXiv:1804.06913 and arXiv:2003.06308, the relevant body text).
This is **not a systematic literature review**. For most entries below only the abstract was read, so statements about what a paper does *not* do are
statements about what its abstract does not mention. Conclusions about novelty are accordingly provisional and one-directional: this survey can show that
something *has* been done, but not that it *hasn't*.

## What has already been done (by source)

| Work | Relevance | What it covers (as far as read) |
|---|---|---|
| Duarte et al., JINST 13 P07027 (2018), arXiv:1804.06913 | Origin of hls4ml and of this dataset | Introduces the 16-feature, 5-class jet-substructure benchmark; MLP 16-64-32-32-5; fixed-point scans and FPGA latency/resource results (body text read for dataset generation/features) |
| Ngadiuba et al., arXiv:2003.06308 | Same dataset and architecture | Binary/ternary and fixed-point models; Table 5 reports the 75% FP baseline and per-class AUC; synthesis-level resources/latency reported (Table 6 not read in detail) |
| Coelho et al., Nature Mach. Intell. 2021, arXiv:2006.10159 | Quantization of HEP trigger networks | QKeras/AutoQKeras: automatic per-layer, per-parameter heterogeneous quantization, energy/area estimation, applied to LHC event selection with <1 µs latency (abstract only) |
| Sun et al., arXiv:2405.00645 (HGQ) | Mixed precision | Gradient-based per-weight/per-activation bit-width optimization with QAT, bit-accurate hls4ml conversion, jet tagging, reported resource/latency gains (abstract/landing pages only) |
| Odagiu et al., Mach. Learn.: Sci. Technol. (2024), arXiv:2402.01876 | QAT for FPGA jet tagging | Deep Sets and Interaction Networks with QAT, ~100 ns inference, FPGA synthesis (abstract only) |
| Coppi et al., arXiv:2601.10801 | PTQ on FPGA jet taggers | Tensor-network jet classifiers; PTQ explored; sub-µs targets (abstract only) |
| Niemi et al., arXiv:2603.26595 (PQuantML) | Compression tooling | Unified pruning + quantization incl. HGQ, evaluated on jet substructure, compared with QKeras/HGQ (abstract only) |
| Titles seen, not read | Context | MLP-Mixers for jet tagging on FPGAs (arXiv:2503.03103); JetFormer (arXiv:2601.17215); hls4ml platform overview (arXiv:2512.01463) |

Searches also covered ParticleNet-on-FPGA and Level-1-trigger low-precision work; no source was found or read that this project's claims depend on beyond the list above.

## Consequences for this project

- **Not novel in method.** Fixed-point/low-precision MLP jet tagging on exactly this dataset, QAT, automatic mixed precision, and real FPGA synthesis are all established. The project's quantizer (per-tensor min/max fake-quantization) is simpler than QKeras/HGQ, and it has **no synthesis** where the literature does.
- **Not a hardware study.** It must not be compared to those papers on latency/resources.
- **Where it could still add something (unverified as unique):**
  1. A *controlled, multi-seed, paired* comparison of INT8/6/4 PTQ vs QAT on the standard benchmark, including a compute-matched FP32 control and device-dependence of results. Abstract-level reading does not show whether the papers above report seed variance for PTQ; that is not established either way.
  2. A *mechanistic attribution* of simple-PTQ failure at INT4 (calibration range effect; first-layer concentration; per-class and calibration damage) using checkpoint-level ablations, plus a sensitivity-derived mixed-precision check.
  3. A *measurement-tier discipline* (measured / simulated / estimated / unavailable) and frozen, hash-verified, reproducible artifacts. This is methodology, not physics.

## What remains unknown

Whether the specific finding (INT4 min/max PTQ is seed- and device-chaotic on this benchmark; calibration-set size matters at INT4) appears in prior work. A full-text read of the sources above, and of the broader post-training-quantization literature outside HEP (where range-clipping/percentile calibration is standard), would be required before any claim of novelty. Outside HEP the instability of naive min/max INT4 PTQ is widely expected, so the finding should be presented as a *confirmation on a HEP benchmark with controls*, not a discovery.
