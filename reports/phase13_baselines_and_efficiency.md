# Phase 13: Non-NN Baselines and Background Rejection at Fixed Efficiency

## 1. Objective

Two things a reviewer would reasonably ask for that nothing earlier in this project reports:

1. **Does the MLP's complexity earn its keep?** Everything so far compares *quantized variants
   of the same MLP* against each other. None of it asks whether a neural network is the right
   choice in the first place, against simple, standard baselines on identical data.
2. **A HEP-standard operating-point metric.** ROC-AUC/F1/accuracy are reported throughout, but
   not background rejection at a fixed signal efficiency — the way trigger-design discussions
   actually frame the accuracy/cost trade-off.

## 2. Method

Logistic regression and `HistGradientBoostingClassifier` (scikit-learn, both with default
hyperparameters — no tuning, to keep this an honest baseline comparison, not a cherry-picked
one) trained on the **identical preprocessed train/val/test splits** the frozen FP32 MLP record
uses (same `load_hls4ml_jets` config, same `split_dataset` seed, same `Preprocessor`), across
the same 5 seeds `[42, 123, 2026, 7, 99]`.

**Background rejection at fixed efficiency**: this is inherently a binary signal-vs-background
concept; this is a 5-class jet-tagging task, not a binary search, so it is computed one-vs-rest
per class (`hep_quant.training.metrics.macro_background_rejection_at_efficiency`) and
macro-averaged, at a 70% signal-efficiency working point (a common trigger-relevant choice, not
tuned to favor any model) — the adaptation is reported explicitly, with the per-class breakdown
available in `reports/tables/phase13_baselines_and_efficiency.csv`, not hidden behind one number.

Compared against: the frozen FP32 MLP, the frozen uniform INT6 PTQ MLP, and this project's own
Phase 12 mixed-precision PTQ-selected and QAT-selected layouts (rebuilt read-only from the
trained FP32 base — PTQ is a forward pass; QAT reuses the exact fine-tune recipe Phase 12
already validated, not a new training procedure).

## 3. Results (mean over 5 seeds)

| model | ROC-AUC | F1 | bkg. rejection @ 70% eff. (macro) | predict time (µs/sample) |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.9373 ± 0.0017** | 0.7632 | **52.59** | 9.71 |
| MLP FP32 (reference) | 0.9343 ± 0.0011 | 0.7569 | 49.63 | 2.58 |
| MLP mixed QAT selected (Phase 12) | 0.9342 ± 0.0014 | 0.7561 | 48.91 | 1.86 |
| MLP INT6 PTQ uniform (reference) | 0.9329 ± 0.0014 | 0.7547 | 44.99 | 2.09 |
| MLP mixed PTQ selected (Phase 12) | 0.9328 ± 0.0012 | 0.7533 | 47.59 | 0.64 |
| LogisticRegression | 0.8870 ± 0.0017 | 0.6908 | 14.36 | 0.67 |

**Paired comparison vs. MLP FP32** (95% CI, `scipy.stats.t.ppf(0.975, n-1)`, descriptive given n=5):

| comparator | mean ROC-AUC diff | 95% CI | seeds positive |
|---|---|---|---|
| HistGradientBoosting | **+0.0030** | **[0.0018, 0.0041]** | **5/5** |
| LogisticRegression | -0.0473 | [-0.0484, -0.0462] | 0/5 |

## 4. Interpretation — reported plainly, not spun

**The gradient-boosted-tree baseline reliably beats the MLP** on this tabular, 16-high-level-
feature dataset — a statistically clean result (95% CI entirely positive, all 5 seeds), and
consistent with a well-known pattern in tabular ML more broadly (tree ensembles often beat small
MLPs on structured feature sets like this one, as opposed to raw/low-level or image-like inputs
where NNs typically win). Logistic regression, as expected, is clearly worse than either
nonlinear model — confirming the classification boundary genuinely needs nonlinearity, just not
necessarily a neural network's specific form of it.

**This does not undermine the rest of this project.** The reason to study a quantized MLP
specifically is not "the MLP is the most accurate model available" — it measurably is not — but
that `hls4ml`'s PyTorch-to-FPGA pipeline, and this project's entire quantization/mixed-precision
analysis, target a neural-network-shaped computation that maps onto a real-time, microsecond-
budget Level-1 trigger FPGA. A `HistGradientBoostingClassifier` is not the kind of model this
project's deployment target (or the broader hls4ml ecosystem) is built around, and deploying a
tree ensemble within a comparable latency/resource budget is a different (and not out-of-the-box
solved) engineering problem, not something this phase attempted or can claim. The honest
conclusion is: **for raw predictive accuracy alone, use the GBM; the case for the quantized MLP
specifically is deployability on this project's target hardware path, not accuracy.**

The **mixed-precision QAT result from Phase 12** (0.9342 ROC-AUC, within noise of the FP32
baseline, at roughly a sixth of its storage) remains the most relevant comparison *within* the
deployable-MLP family — it is not being compared against the GBM as if they were interchangeable
deployment options.

## 5. Limitations

- No hyperparameter tuning was performed for either baseline — this is deliberately an
  out-of-the-box comparison, not a best-case one for either side. A tuned GBM or MLP could move
  either number; neither was tuned, so neither is being quietly favored.
- Background rejection at 70% efficiency is one working point; the ranking between models could
  differ at a different efficiency choice (e.g. 90%, where rare high-purity regions behave
  differently) — not checked here.
- Inference-time comparisons were measured on the same CPU, same process, same session — not a
  controlled benchmark against production inference serving infrastructure for any of the model
  types; treat the relative ordering as indicative, not a deployment-grade latency claim.
- The dataset's 5 classes are close to balanced (19.4%-20.2% per class, confirmed directly, not
  assumed) — class-imbalance handling was not needed and was not added, since it would not be a
  response to anything actually present in this dataset.

## 6. Reproducibility (two-test rule)

Full script run twice independently. Every scientific metric column
(`roc_auc`, `f1`, `accuracy`, `bkg_rejection_at_70pct_eff_macro`, `n_params_or_equivalent`) was
**byte-identical** between runs (verified via `DataFrame.equals`, not just visual inspection).
Only the wall-clock timing columns (`fit_time_s`, `predict_time_s_per_sample_us`) differed
between runs, as expected — these measure real elapsed time, not a deterministic computation,
and are reported as such rather than claimed reproducible.

## 7. Reproduction

```bash
# WSL2 Ubuntu, .venv-wsl
python scripts/phase13_baselines_and_efficiency.py
python -m pytest -q tests/test_background_rejection.py
```
