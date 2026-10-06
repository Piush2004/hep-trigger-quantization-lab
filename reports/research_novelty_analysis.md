# Research Novelty Analysis (Phase 4, Phase 1)

Source base: `reports/related_work_review.md`. Every claim below cites a source ID from that review (`§1.x`). Verification levels (F/S/N) are inherited from there.

## Classification

**Primary: C. Reproducibility / benchmark contribution.**
**Partial: B (incremental but useful)**, for the specific analyses listed under "Candidate contributions" that survive the consistency check below.
**Not A.** Clearly-novel is not supportable.
**Not E as a whole**, because the specific combination below is not covered by any source read. But the *setup* (dataset, architecture, precision range, QAT) is substantially covered by §1.3, so E applies to the setup.

## Why not A (clearly novel)
- The dataset, the 16-64-32-32-5 architecture and the 6/4-bit QAT range are the same as Hawks et al. (§1.3, **F**). A reviewer will say "already done".
- Per-parameter / mixed-precision low-bit HEP jet tagging with QAT is covered by HGQ (§1.5, **F**) and AutoQKeras (§1.2, **F**), with real synthesis.
- Choosing A on the grounds that "this exact combination of words is uncommon" would be the error the phase rules forbid.

## Setup coverage (what is **not** new)
| Element | Covered by | Source status |
|---|---|---|
| hls4ml jet dataset, 16-64-32-32-5 MLP | §1.3, §1.5, §1.6 | F |
| QAT at 6 and 4 bits on this MLP | §1.3 | F (summary) |
| hls4ml fixed-point simulation of this MLP | §1.6 (⟨16,6⟩) | F |
| Mixed-precision bit allocation for jet tagging | §1.2, §1.5 | F |
| QAT with QKeras/Brevitas + FPGA synthesis | §1.2, §1.5 | F |

## Candidate contributions (each must survive the consistency check)
1. **Compute-matched QAT control.** Prior work (§1.3, §1.2) does not describe a control that gives FP32 the same extra epochs. *Our data:* QAT's INT8/INT6 advantage over PTQ is not separable from the extra epochs; the INT4 advantage remains. **Status: candidate (B).** Caveat: only the fetched summaries were read, so "not described" means not in the text we could read.
2. **Seed- and device-level variance of INT4 PTQ.** Prior work reports single-run numbers in the sources read. *Our data:* 5 seeds, ROC-AUC 0.835–0.909; device-dependent point values. **Status: candidate (B/C).** Needs the INT4 stability study (Phase 2 of this phase) before it can be stated.
3. **Explicit measurement-tier labelling** (measured / simulated / estimated / unavailable) for FPGA-oriented numbers. Prior work (§1.2) validates estimates against post-P&R; this project has no synthesis at all. **Status: C (benchmark discipline), not a technical novelty.**
4. **Reproducible frozen artifacts** with hash verification and bit-identical re-runs. **Status: C.**

## Candidate claims that are **not** supportable
- "First PTQ-vs-QAT comparison for HEP jet tagging." **Not supportable**: the review found no such study, but the search was not exhaustive (§5.3 of the review), and §1.3 already compares 6- and 4-bit QAT with float.
- "First INT4 instability finding for hls4ml jet models." **Not supportable** until the search is exhaustive and the causal study is run.
- "Better than HGQ / AutoQKeras." **Not supportable.** This project does no per-parameter optimisation and no synthesis; comparison is out of scope.
- Any FPGA resource or latency advantage. **Not supportable.** No synthesis (see `docs/hardware.md`).
- Any claim that the dataset is CERN detector data. **False**, per the dataset provenance (`reports/dataset_selection.md`).

---

## TEST #1 — structured consistency check (novelty statements vs cited sources)

Run as `scripts/phase4_novelty_check.py` (see below). The check asserts that every novelty-bearing statement in this file carries a `§1.x` citation that exists in `related_work_review.md`, and that no sentence containing "first", "novel", or "only" appears without a citation or a "not supportable" marker.

Result: recorded in `reports/phase4_evidence_lock.md`, section "Consistency check".

## TEST #2 — skeptical-reviewer challenges (independent re-read)

Three ways a skeptical reviewer would attack the contribution claim (not a priority claim, so no citation is needed):

1. **"This is Hawks et al. (2021) with a different baseline."** The architecture, dataset, and 6/4-bit QAT are the same (§1.3). The response must be the *control* (candidate 1) and *variance* (candidate 2), not the setup. If the control is weak (the prior paper may have run a comparable baseline that the fetched summary omits), the claim collapses to C. **Mitigation required:** re-read the full text of arXiv:2102.11289 to confirm whether a same-epoch baseline exists. Not done in this pass.
2. **"Seeds 5 is too few to talk about variance."** Five seeds give a 95% t-interval of about ±2.8 SD/√5 on the mean, and INT4 PTQ's range is wide enough that its variance is real. But the claim "INT4 is unstable" rests on one configuration with 5 draws, and the causal explanation is not yet established. The write-up must say "variable" not "unstable" until the causal study is done.
3. **"The novelty is a benchmark, and benchmarks are not research contributions."** Accepted as the classification (C); the response is that the benchmark has specific, falsifiable findings (candidates 1–2), and that the classification is stated up front rather than inflated.

Additional challenge recorded for the review: the prior-work numbers come from automated summaries (**F** with caveats), and two dataset sizes disagree (870,000 vs 830,000). Until the primary PDFs are checked by hand, no quantitative comparison with prior work may be made in a figure or table.

## Summary statement (conservative)
This project is a **reproducibility and benchmark study** of PTQ vs QAT at INT8/INT6/INT4 on the public hls4ml LHC jet dataset with the 16-64-32-32-5 MLP used in prior work. Its candidate incremental contributions are (1) a compute-matched QAT control and (2) a seed-level characterisation of INT4 PTQ variability. Neither is yet established as novel; both are to be confirmed or refuted by the Phase 2 and Phase 4 analyses.
