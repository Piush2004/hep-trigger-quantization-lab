# CERN Internship Proposal — Reproducible Quantization Benchmarking for Real-Time Triggers

*Prepared 2026-10-13. Research synthesized from two independent passes: a literature/research-gap scan of the quantized-ML-for-HEP-trigger field, and a review of CERN's current internship programs and eligibility rules. Both converged on the same project framing independently — a good sign it's the right scope.*

## 1. Executive summary

- **Target program:** CERN openlab Summer Student Programme (not the plain Summer Student Programme, and not the Technical Studentship — see §3).
- **Why it fits:** open to all nationalities, explicitly welcomes "computer science, mathematics, engineering, physics, or any relevant computing-related field," and its 2026 cohort ran projects under CERN's Next Generation Triggers (NGT) collaboration — the exact domain of your existing [hep-trigger-quantization-lab](https://github.com/Piush2004/hep-trigger-quantization-lab) repo.
- **Proposed topic:** harden your existing PTQ/QAT quantization-benchmarking pipeline into a **resource-aware, reproducible Pareto-frontier benchmark** for real-time L1 trigger classifiers — by (a) replacing the heuristic hardware-resource estimates with validated numbers, and (b) replacing the hand-picked 9-candidate mixed-precision search with a near-exhaustive, pre-registered one. Both are **data-analytics contributions** (benchmarking methodology, statistical rigor, Pareto analysis), not particle-physics ones — which matches your domain.
- **Deadline reality check:** the 2026 cycle (deadline ~26 Jan 2026) has already passed. You're targeting the **2027 call** — watch [careers.cern](https://careers.cern) and [openlab.cern](https://openlab.cern) for when it opens (historically a few months before a similar January deadline, but confirm, don't assume).

---

## 2. Research gaps in the field — and the detailed solution draft

### 2.0 What's already been checked (don't re-propose this)

Your own repo already runs a rigorous internal gap-finding process (`reports/related_work_gap.md`, `research_gap_analysis.md`, `research_novelty_analysis.md`) and has already closed several obvious gaps: 5 seeds, a compute-matched FP32/QAT control, two MLP sizes (Phase 6), a 9-candidate mixed-precision search (Phase 5/5B — verdict: *no advantage over INT6, gate explicitly closed*), per-class/calibration error analysis, and hls4ml C-sim consistency checks. The research agent deliberately searched for what's changed in the literature **since** those internal reports were written (they cite papers only through ~Jan–Mar 2026) rather than re-treading old ground.

### 2.1 Landscape (with sources)

- Your dataset is the standard FastML/hls4ml LHC jet-tagging benchmark (Duarte et al. 2018, [arXiv:1804.06913](https://arxiv.org/pdf/1804.06913)).
- Current SOTA for automatic mixed-precision in this niche is **HGQ/HGQ2** (gradient-based, per-parameter bit-width learning via a differentiable resource proxy; [arXiv:2405.00645](https://arxiv.org/pdf/2405.00645), FPGA'26 best paper) — more principled than hand-picked layouts, but no brute-force baseline exists to benchmark it against.
- The field has moved toward GNNs/Transformers/MLP-Mixers for jet tagging under hls4ml in just the last 12 months (JEDI-linear [arXiv:2508.15468](https://arxiv.org/pdf/2508.15468); CMS's WOMBAT trigger system [arXiv:2505.05532](https://arxiv.org/pdf/2505.05532)) — real gap, but multi-month scope, not internship-sized. Flagged, not pursued.
- **The single most relevant recent finding:** [**wa-hls4ml**](https://arxiv.org/abs/2511.05615) (Nov 2025) — a public dataset of ~680,000 real hls4ml-synthesized networks plus pretrained surrogate models that predict LUT/FF/DSP/BRAM/latency "within a few percent" of real synthesis **without running synthesis at all**. This directly targets your project's own repeatedly-flagged weakness: hardware numbers are currently `estimated` (a linear heuristic), never validated.
- CERN openlab's 2026 cohort ran NGT-affiliated projects (e.g. converting CNN tau-ID models to Binary Neural Networks on ATLAS FPGAs) — confirming this niche is an active, funded, internship-shaped track at CERN right now.

### 2.2 The two solution drafts

**Solution 1 — "Surrogate" hardware-validation tier.** Add a fourth measurement tier (`surrogate`, between your existing `estimated` and `unavailable`) using wa-hls4ml's pretrained models to get validated LUT/FF/DSP/BRAM/latency estimates for your existing 14 frozen configs (7 precisions × 2 architectures from Phases 2+6), with zero retraining. If Vivado/Vitis becomes available at CERN, add a small (~14-point) real-synthesis validation subset and produce the project's first actual **measurement-tier accuracy table**: heuristic vs. surrogate vs. ground truth. Deliverable: `reports/phase8_surrogate_hardware_validation.md`.

**Solution 2 — Near-exhaustive, pre-registered mixed-precision Pareto search.** Your Phase 5/5B search covered only 9 hand-picked layouts and said so explicitly ("mixed precision was not searched exhaustively"). With 7 quantizable tensors × 3 candidate bit-widths, the true space is only 3^7 ≈ 2,187 configs — small enough to search near-exhaustively on existing frozen checkpoints (PTQ reuses them with no retraining; QAT fine-tuning is only run on the configs that land on the PTQ Pareto frontier). This either **strengthens** your existing "no advantage over INT6" claim with real evidence, or finds a genuinely new result. Deliverable: `reports/phase9_exhaustive_mixed_precision_pareto.md`.

**Combined (stretch goal):** running Solution 2's search using Solution 1's validated resource numbers (instead of storage-only) produces your first *resource-aware* mixed-precision Pareto frontier — the same kind of result HGQ/AutoQKeras report, reached independently via brute-force search instead of gradient-based learning. This is the strongest single deliverable.

**Why it's internship-scoped:** no new dataset, no new architecture, no new training infra — both solutions reuse existing frozen checkpoints, the existing quantizer code, and the existing statistical/reproducibility conventions (5 seeds, pre-registered selection rule, two-test-rule). This is methodology and analysis work on top of infrastructure that already exists, which is exactly the right shape for a few months, not a multi-year program.

---

## 3. CERN internship programs — eligibility comparison

| Program | Nationality rule | Academic level | Field of study | Deadline pattern | Duration | Pay |
|---|---|---|---|---|---|---|
| **openlab Summer Student (recommended)** | All nationalities welcome (explicit exception) | Bachelor's/Master's, not PhD; ≥6 semesters completed | CS, maths, engineering, physics, "or any relevant computing-related field" | ~26 Jan, for ~22 June start | 9 weeks | 93 CHF/day tax-free + health insurance |
| Plain Summer Student Programme | All nationalities welcome | Same as above | Physics, Engineering, CS, Maths (more physics-leaning) | ~26 Jan, mid-June start | 8–13 weeks | 93 CHF/day + health insurance |
| Technical Studentship (IT/Maths/Robotics) | **Member/Associate-State nationals only**; current 2026 quota exclusions: Pakistan, Lithuania, Cyprus, Latvia, Croatia, Turkey | ≥18 months of study; **PhD not eligible**; explicitly excludes particle-physics specialists | Applied physics, IT, maths, engineering | Rolling, ~2 panels/year | 4–12 months | Monthly allowance + health insurance |
| Short-Term Internship (lists "Data Science & Data Analytics" directly) | **Member/Associate-State nationals only**, same quota issue | Enrolled diploma/Bachelor's/Master's | Open list includes Data Science & Data Analytics | Rolling | 1–6 months | ~1,593 CHF/month (unverified, check current) |

**Why openlab over the others:** no nationality restriction (unlike Technical Studentship/Short-Term Internship, which currently exclude several nationalities under a filled Associate-Member quota), its field-of-study language is written for computing/data profiles rather than physics majors, and its 2026 cohort explicitly ran NGT (trigger-system) projects — direct topical overlap with your existing work.

**Before applying, verify directly on [careers.cern/eligibility-criteria](https://careers.cern/eligibility-criteria/):**
- Your own nationality's quota status if you consider the Technical Studentship/Short-Term Internship as a backup (quotas are cycle-dependent).
- The 2027 call dates and whether NGT-labeled projects are offered again — mention NGT/trigger relevance in your motivation statement regardless.
- Current stipend figures (some sourced figures above range from 2018 to 2026 and should be re-verified).

---

## 4. Proposed application topic

**Working title:** *A Reproducible, Resource-Aware Benchmarking Framework for Pareto-Optimal Quantization of Jet-Tagging Classifiers in Real-Time L1 Triggers*

**Description:** Real-time Level-1 trigger systems increasingly deploy ML classifiers on FPGAs, where every bit of precision trades directly against latency and chip resources. Most published quantization studies report a single operating point rather than systematically characterizing the trade-off space. This project extends an existing open-source benchmarking pipeline (PTQ/QAT × FP32/INT8/INT6/INT4, hls4ml-based hardware estimation, Pareto analysis, github.com/Piush2004/hep-trigger-quantization-lab) in two concrete ways: (1) replacing heuristic hardware estimates with validated numbers using the public wa-hls4ml surrogate-model dataset, and (2) replacing a 9-candidate hand-picked mixed-precision search with a near-exhaustive, pre-registered one over the full per-tensor bit-width space. The result is a reusable, reproducible methodology — not a new physics claim — that lets trigger-firmware teams reliably answer "which quantization configuration gives the best accuracy per hardware resource at a given latency budget." This is squarely a data-analytics contribution: experiment-matrix design, benchmarking rigor, statistical reporting, and trade-off visualization, built on a working prototype rather than starting from zero.

---

## 5. Open items / what to verify before applying

1. 2027 openlab application dates (not yet published as of Oct 2026).
2. Your nationality's current quota status for the restricted programs, if used as backup.
3. Whether wa-hls4ml's surrogate accuracy actually holds on your specific tiny MLPs (the validation step *is* the contribution — don't assume the authors' aggregate accuracy transfers).
4. Current stipend/allowance figures directly from careers.cern (sourced figures above mix years).
