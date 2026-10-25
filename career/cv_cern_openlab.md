# Piush Kole

Scientific Computing · ML for High-Energy Physics · Reproducible Research Software

piushkole2004@gmail.com · Bengaluru, India · github.com/Piush2004

---

## Summary

MSc Data Science candidate with an independent, publicly documented research project on
**hardware-aware neural-network compression** (quantization and mixed-precision search) for
real-time inference, applied to a simulated high-energy-physics jet-classification task and
validated through `hls4ml` fixed-point simulation. This is the same technical direction as CERN's
own Next Generation Triggers programme (Task 1.3, "Hardware-Aware AI Optimization" — quantization,
pruning and NAS for FPGA-deployable trigger models) and openlab/NGT's existing `hls4ml`-based
student projects, though this work is entirely independent and not CERN-affiliated. Experience
spans experimental design, statistical rigor (pre-registered selection rules, multi-seed
evaluation, leakage prevention), and reproducible software engineering (automated test suites,
documented environments, byte-identical reproduction runs). Seeking a CERN openlab summer-student
placement to apply this experience to a real experiment's software stack — with an explicit
interest in learning the HLS-synthesis step this project could not access.

## Relevant technical skills

- **HEP-adjacent scientific computing:** `hls4ml` (PyTorch → fixed-point `ap_fixed` C-simulation),
  working familiarity with the measured/simulated/estimated/unavailable evidence-tier discipline used
  in FPGA-trigger ML literature
- **ML & quantization:** PyTorch, post-training and quantization-aware training, mixed-precision
  search, scikit-learn
- **Reproducibility & statistics:** pre-registered analysis plans, paired confidence intervals,
  multi-seed experimental design, dataset/environment fingerprinting
- **Software engineering:** Python, pytest, Git, Docker, CI (GitHub Actions), FastAPI
- **Data & visualization:** NumPy, pandas, Plotly, SQL

## Independent research — HEP-Trigger Quantization Lab

*github.com/Piush2004/hep-trigger-quantization-lab — public repository, reports, and reproduction
instructions*

A reproducible benchmark studying how aggressively a neural network trigger-style classifier can be
quantized before physics performance (ROC-AUC on a 5-class jet-tagging task) degrades, using the
public `hls4ml_lhc_jets_hlf` dataset (particle-level Monte Carlo; explicitly not real detector data,
and this distinction is preserved throughout the project rather than blurred).

- Evaluated FP32 baseline against INT8/INT6/INT4 PTQ and QAT across two MLP architectures, 5 seeds
  each, with a pre-registered selection protocol specifically designed to prevent test-set leakage —
  one earlier exploratory result that *was* selected on test data was caught, disclosed as biased,
  and excluded from every confirmatory claim.
- Implemented the PyTorch → `hls4ml` fixed-point C-simulation conversion path and validated numerical
  agreement against the floating-point model; extended it to a second architecture family (a CNN on
  pixelated jet-image data) and confirmed the conversion path generalizes beyond a single model type.
- Trained an ML surrogate for FPGA resource estimation on the public `fastmachinelearning/wa-hls4ml`
  synthesis dataset, since no local HLS/Vivado toolchain or physical FPGA was available — and reported
  the surrogate's own limitation (bit-width insensitivity) rather than presenting it as validated.
- Ran a near-exhaustive per-unit mixed-precision search (2,187 configurations) and found a layout that
  reliably beats uniform INT6 PTQ at a smaller storage footprint, all 5 seeds, with reported
  confidence intervals rather than a single point estimate.
- Maintained a strict evidence-tier distinction (MEASURED / SIMULATED / ESTIMATED / NOT AVAILABLE)
  across every reported number, the research dashboard, and the written report — no HLS synthesis or
  physical FPGA result is claimed anywhere, by design.
- 141 automated tests (two independent runs, identical results), full reproduction instructions and
  dataset fingerprints documented, environment and git state recorded with every experiment.

## Education

**M.Sc. Data Science (Computer Science)** — CHRIST (Deemed to be University), Bengaluru · *Jul 2025 – May 2027 (in progress)*

**B.Sc. (Hons.) Data Science** — Techno India University, Kolkata · *May 2022 – Apr 2025* · 69.67%

## Experience

**Data Executive Intern — Cybertoss Pvt. Ltd.**, Kolkata (onsite) · *Jun 2024 – May 2025*

- Owned the end-to-end data lifecycle (collection, cleaning, validation, reporting) for large-scale
  datasets, automating Python workflows that reduced manual effort for cross-functional analytics.

**Data Science Trainee — RemarkSkill**, via KSHITIJ (IIT Kharagpur's techno-management festival),
remote · *Jun 2024*

- Applied Python (pandas, NumPy, Matplotlib) and Power BI to real-world dataset projects.

## Certifications & leadership

AWS Cloud Practitioner Essentials (AWS Academy, 2025) · Python for Data Science & Machine Learning
(Udemy, 2023) · IEEE Committee Member, CHRIST University (Dec 2025) — coordinated outreach to 10+
industry companies for a 20-member student-conference committee.

