# Phase 12: Near-Exhaustive Mixed-Precision Pareto Search

## 1. Scientific question

Phase 5/5B's mixed-precision search covered only 9 hand-picked layouts and said so explicitly
(`reports/phase5b_size_constrained_gate.md` section 14: "the search space is nine layouts...
it says nothing about the best possible mixed layout"), concluding "no advantage over INT6
PTQ." **Does a near-exhaustive search over the full per-unit bit-width space change that
conclusion?**

## 2. Experimental design

Search space: 7 quantizable units (L0-L3 weight tensors, A0-A2 activations) × {INT4, INT6,
INT8} = 3⁷ = **2,187 layouts**, on Phase 2's frozen FP32 base architecture (`hidden=[128,64,32]`,
7 units) — the same architecture Phase 5/5B searched, reusing its exact frozen checkpoints
(`experiments/results/frozen_phase2/`), `load_splits`, `score`, and `build_ptq_variant` plumbing
unchanged, via Python import (not copied).

**Protocol decided by measurement, not assumption:** a timing micro-benchmark (20 randomly
sampled layouts, seed 42) measured 0.107 s/evaluation, giving an estimated full 2,187×5-seed
grid cost of **0.32 hours** — well under a declared 2-hour threshold for running the full grid
directly. The staged screen→confirm shortcut (seed-42-only screening, then confirming only the
Pareto frontier at all 5 seeds) was therefore **not needed and not used** — all 2,187 layouts
were evaluated at all 5 seeds directly, avoiding any single-seed screening approximation.
Actual measured run time: ~20 minutes (confirmed by two independent full runs, both producing
`reports/tables/phase12_timing_benchmark.json`).

## 3. Pre-declared selection rule

Mirrors Phase 5B exactly: among the 2,187 confirmed layouts with
`theoretical_weight_storage_bytes <= size_limit_bytes` (computed **dynamically** as the uniform
INT6 storage for the loaded architecture — 3,724.0 bytes here, not hardcoded), select the
highest mean validation macro ROC-AUC over the 5 seeds; ties within `TIE_EPS=0.001` broken by
smaller storage, then layout name. Written to `reports/tables/phase12_selection.json` **before**
any test-split evaluation (`selection_used_test_set: false`).

## 4. Results

**The true Pareto frontier** (storage vs. mean validation ROC-AUC, computed via the general
`find_pareto_frontier` in `hep_quant.visualization.pareto`, not Phase 5B's inline 2-point
dominance loop) contains **23 of the 2,187 layouts** — `reports/tables/phase12_pareto_frontier.csv`
— ranging from storage 2,660 B (all-INT4 except activations INT8, AUC 0.918) up to 4,788 B
(all-INT8, AUC 0.933).

**PTQ selected layout:** `L0-6_L1-4_L2-4_L3-8_A0-8_A1-8_A2-8` — storage **2,996 B** (80% of
uniform INT6's 3,724 B), mean validation ROC-AUC 0.9318. **94 layouts were statistically
indistinguishable from the best** within `TIE_EPS` — the frontier near the selection boundary
is a wide plateau, not a single sharp winner.

**QAT stage** (only on the 23 true-Pareto-frontier layouts, never all 2,187 — fine-tuned via
the new `hep_quant.analysis.qat_studies.build_qat_variant`, same epoch/LR formula as the
project's existing uniform QAT path: `qat_epochs = max(5, 30 // 4) = 7`, `qat_lr = 0.001 * 0.1`):
**selected layout `L0-6_L1-4_L2-4_L3-4_A0-8_A1-8_A2-8`, storage 2,916 B (78% of INT6's), mean
validation ROC-AUC 0.9333.**

**Final test results (one evaluation each, 5 seeds):**

| configuration | storage (B) | mean test ROC-AUC |
|---|---|---|
| FP32 (reference) | 17,556 | 0.9343 |
| INT8 PTQ (reference) | 4,788 | 0.9342 |
| INT6 PTQ (reference) | 3,724 | 0.9329 |
| INT4 PTQ (reference) | 2,660 | 0.8813 |
| INT4 QAT (reference) | 2,660 | 0.9246 |
| **PTQ selected (this phase)** | **2,996** | **0.9321** |
| **QAT selected (this phase)** | **2,916** | **0.9342** |

## 5. Statistical analysis (paired, 95% CI, `scipy.stats.t.ppf(0.975, n-1)`, descriptive given n=5)

`reports/tables/phase12_paired_comparisons.csv`:

| selected | vs. | mean diff | 95% CI | seeds positive |
|---|---|---|---|---|
| PTQ (2,996 B) | INT6 PTQ | -0.0001 | [-0.0007, 0.0005] | 3/5 (indistinguishable) |
| PTQ (2,996 B) | INT4 PTQ | +0.0509 | [0.0139, 0.0878] | 5/5 |
| PTQ (2,996 B) | INT4 QAT | +0.0082 | [0.0061, 0.0102] | 5/5 |
| **QAT (2,916 B)** | **INT6 PTQ** | **+0.0013** | **[0.0011, 0.0015]** | **5/5** |
| QAT (2,916 B) | FP32 | -0.0001 | [-0.0005, 0.0003] | 2/5 (indistinguishable) |
| QAT (2,916 B) | INT8 PTQ | -0.00004 | [-0.0004, 0.0004] | 3/5 (indistinguishable) |
| QAT (2,916 B) | INT4 QAT | +0.0096 | [0.0073, 0.0118] | 5/5 |

## 6. Scientific interpretation

**This revises Phase 5B's conclusion.** With a true near-exhaustive (not hand-picked) search,
QAT fine-tuning finds a mixed-precision layout using **less storage than uniform INT6 PTQ**
(2,916 B vs. 3,724 B — 78%) that **reliably beats uniform INT6 PTQ** (all 5 seeds, 95% CI
entirely above zero) and is **statistically indistinguishable from FP32 and from uniform INT8
PTQ** — while using roughly a sixth of FP32's storage. Phase 5B's 9-candidate search could not
have found this layout (none of its hand-picked candidates matched this specific per-unit
pattern). The PTQ-only version of the search (no fine-tuning) lands within noise of uniform
INT6 PTQ at slightly smaller storage — consistent with, not contradicting, Phase 5B's original
PTQ-only finding; it is specifically the **QAT** stage, applied to a properly-identified true
Pareto frontier rather than 9 guesses, that produces the new result.

## 7. A reproducibility gap found and fixed mid-phase

`build_qat_variant`'s initial implementation did not pin a random seed before fine-tuning
(DataLoader shuffling and Dropout are stochastic), unlike every other part of this project's
pipeline (which calls `hep_quant.utils.seeds.set_seed` at the top of `run_experiment`). This was
caught by a spot-check: two in-process fine-tunes of the same layout/seed produced ROC-AUC
0.93493 vs. 0.93458 (diff ≈ 3×10⁻⁴) — small, but non-zero, and inconsistent with the project's
two-test-rule discipline. Fixed by adding `seed` as an explicit parameter to `build_qat_variant`
and calling `set_seed(seed)` before fine-tuning; re-verified the same two-run spot-check then
gave **diff = 0.0** exactly. All QAT results in this report were produced with the fix in place.

## 8. Limitations

- PTQ full-grid evaluation used CPU throughout (never CUDA), per the documented CPU/CUDA INT4
  divergence (`docs/REPRODUCIBILITY.md` section 6) — consistent with Phase 5/5B/6's protocol,
  but means these results are not directly comparable to any CUDA-evaluated numbers.
- QAT fine-tuning was restricted to the 23 true-Pareto-frontier layouts, not all 2,187 — a
  scope restriction necessary for feasibility (each fine-tune costs tens of seconds vs. a PTQ
  evaluation's ~0.1s), not a claim that no other layout would respond as well to QAT.
- `n=5` seeds throughout; the paired-comparison intervals are descriptive, not formal
  significance tests, matching this project's existing statistical convention.
- The selection rule's frontier-plateau (94 PTQ layouts within `TIE_EPS` of the best) means the
  *specific* selected layout is somewhat arbitrary among many similarly-good options — the
  finding that *some* layout near INT6's storage budget beats INT6 reliably is the robust part,
  not the exact bit-assignment chosen.

## 9. Reproducibility (two-test rule)

- **PTQ stage:** full pipeline run twice, independently. `phase12_confirm_validation.csv`,
  `phase12_final_test.csv`, and `phase12_selection.json` were **byte-identical** between runs
  (`cmp`, all three files). CPU-only, no CUDA involved.
- **QAT stage:** after the seeding fix (section 7), the full QAT frontier script (23 layouts ×
  5 seeds, fine-tuning) was run twice end-to-end. `phase12_qat_confirm_validation.csv`,
  `phase12_qat_final_test.csv`, and `phase12_qat_selection.json` were **byte-identical** between
  the two runs (`cmp`, all three files).
- `tests/test_phase12_selection.py` (6 tests: layout enumeration exhaustiveness/uniqueness,
  selection-rule correctness and tie-breaking, dynamic size-limit calculation) and
  `tests/test_qat_studies.py` (4 tests: per-unit QAT wiring, None-unit pass-through, forward
  pass, non-mutation of the original model) both passed identically on two independent runs.

## 10. Reproduction

```bash
# WSL2 Ubuntu, .venv-wsl
python scripts/phase12_exhaustive_mixed_precision_pareto.py
python scripts/phase12_qat_frontier.py
python -m pytest -q tests/test_phase12_selection.py tests/test_qat_studies.py
```
