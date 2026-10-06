# Phase 7 — hls4ml numerical validation

Tiers used throughout: **MEASURED** = PyTorch software measurement; **SIMULATED** = hls4ml C-simulation (fixed-point arithmetic, no hardware); **ESTIMATED** = software heuristic; **SYNTHESIZED** = HLS synthesis, **not available**; **PHYSICAL** = FPGA measurement, **not available**.

HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT. No Vivado, Vitis HLS or Quartus is installed. No resource, latency or initiation-interval number from synthesis exists. No physical FPGA measurement exists.

## 1. Scope and reference
- Model: the seed-42 primary architecture (FP32 weights, 16-64-32-32-5, ReLU).
- Data: the 15,000 held-out test jets of seed 42, the same rows for every row below.
- Reference: PyTorch FP32 logits (MEASURED). Accuracy 0.7495, ROC-AUC 0.9334.
- hls4ml: version 1.3.0, `csim_predict`, all hidden and output layers with `ap_fixed<W,6>` (6 integer bits, W − 6 fractional bits). Widths W = 8, 10, 12, 14, 16. Four arithmetic configurations (below).
- Script: `scripts/phase3_numerical_consistency.py`, outputs `reports/numerical_consistency.{json,md}`.

## 2. PyTorch fixed-point variants (MEASURED)
| Variant | Accuracy | Agreement with FP32 | Logit MAE | Logit max abs | Prob MAE | Prob max abs | ROC-AUC | Δ ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 PTQ | 0.7503 | 0.9931 | 0.0333 | 0.480 | 0.0030 | 0.150 | 0.9333 | −0.0001 |
| INT6 PTQ | 0.7485 | 0.9735 | 0.1258 | 2.594 | 0.0123 | 0.755 | 0.9317 | −0.0017 |
| INT4 PTQ | 0.5716 | 0.7193 | 0.7988 | 7.747 | 0.1060 | 0.995 | 0.8353 | −0.0981 |
| INT4 QAT | 0.7322 | 0.9169 | 0.4925 | 11.058 | 0.0359 | 0.972 | 0.9217 | −0.0118 |

(The row list follows the JSON, which also includes INT8 QAT and INT6 QAT; they are in `numerical_consistency.json`.)

## 3. hls4ml C-simulation (SIMULATED), accuracy and agreement with FP32

| Width W | Default: TRN, WRAP, accum = W | Rounding + saturation, accum = W | Default rounding, wide accum ap_fixed<28,12> | Rounding + saturation, wide accum |
|---:|---:|---:|---:|---:|
| 8 | acc 0.2021, agree 0.1624, ΔAUC −0.4257 | acc 0.5034, agree 0.5959, ΔAUC −0.0963 | acc 0.2017, agree 0.2152, ΔAUC −0.4060 | acc 0.6508, agree 0.7961, ΔAUC −0.0392 |
| 10 | acc 0.3541, agree 0.3433, ΔAUC −0.2540 | acc 0.7343, agree 0.9403, ΔAUC −0.0072 | acc 0.5486, agree 0.6145, ΔAUC −0.1037 | acc 0.7471, agree 0.9696, ΔAUC −0.0025 |
| 12 | acc 0.7255, agree 0.8958, ΔAUC −0.0249 | acc 0.7489, agree 0.9871, ΔAUC −0.0005 | acc 0.7462, agree 0.9631, ΔAUC −0.0032 | acc 0.7500, agree 0.9913, ΔAUC −0.0002 |
| 14 | acc 0.7481, agree 0.9705, ΔAUC −0.0009 | acc 0.7505, agree 0.9963, ΔAUC −0.0000 | acc 0.7507, agree 0.9896, ΔAUC −0.0003 | acc 0.7502, agree 0.9973, ΔAUC −0.0000 |
| 16 | acc 0.7511, agree 0.9929, ΔAUC −0.0001 | acc 0.7497, agree 0.9989, ΔAUC 0.0000 | acc 0.7500, agree 0.9975, ΔAUC −0.0000 | acc 0.7498, agree 0.9993, ΔAUC 0.0000 |

Full rows (logit and probability errors, ROC-AUC) are in `reports/numerical_consistency.json`. Cross-check: hls4ml W = 16 default versus PyTorch INT8 PTQ class agreement = 0.9918.

## 4. Quantities requested, and what was measured
| Quantity | Status |
|---|---|
| ROC-AUC (PyTorch and C-sim) | MEASURED / SIMULATED — reported above |
| Classification agreement | MEASURED / SIMULATED — reported above |
| Mean absolute output difference (logits and probabilities) | MEASURED / SIMULATED — reported above |
| Maximum absolute output difference | MEASURED / SIMULATED — reported above |
| Prediction correlation (Pearson) | **NOT MEASURED.** The per-jet C-simulation probabilities are not persisted by the script, so correlation was not computed. Agreement and error metrics above are the measured substitutes. |
| Relative output error | Not reported, because outputs near zero make it ill-defined; absolute errors are given instead. |

## 5. Mechanism: why the narrow-width collapse happens

**What the data shows.**
1. At W = 8 the default configuration collapses (accuracy 0.20, agreement 0.16). At W = 16 it matches FP32 (agreement 0.993). The collapse is a width effect, not a conversion failure: the same network is faithful at W ≥ 14.
2. Changing the arithmetic mode changes the outcome a lot at fixed width. Rounding with saturation lifts W = 8 from 0.20 to 0.50 and W = 10 from 0.35 to 0.73, while the default truncation-and-wrap leaves the model collapsed.
3. A wide accumulator alone lifts W = 10 from 0.35 to 0.55 and helps less at W = 8 (0.20 → 0.20). So the accumulator is a secondary contributor. Most of the gain comes from the representation of values, not the accumulation.
4. Once rounding and saturation are both on, the wide accumulator adds a further improvement at W = 8 (0.50 → 0.65), so the two effects partly stack.

**Mechanism, as far as the data supports.**
- **Resolution.** With 6 integer bits the fractional resolution at W = 8 is 2 bits (step 0.25). The hidden activations and weights are represented too coarsely for the 5-class decision boundaries, which explains the loss of agreement even with the most favourable arithmetic.
- **Truncation bias.** Default `AP_TRN` rounds toward minus infinity. Each operation adds a systematic negative offset of up to one step, and these offsets accumulate across layers. The rounding mode is therefore a systematic-error source, not just noise. This is the most likely reason rounding changes the outcome so much at W = 8.
- **Overflow under wrap (hypothesis, not tested).** The default saturation is `AP_WRAP`: an out-of-range value wraps to the opposite sign. The largest C-simulation logit errors (about 20 at W = 8) are consistent with some values exceeding the 6-integer-bit range, but overflow was not instrumented, so this is a hypothesis. If it holds, it would explain why saturation helps as much as it does at narrow widths.
- **Scale.** All layers share one format, `ap_fixed<W,6>`, with no per-layer scale. Layers whose activations need a different integer range are therefore either coarser or overflow. A per-layer precision choice was the subject of Phase 5 mixed precision, but it was not applied to the hls4ml run here.

**What this study did NOT separate.**
- Rounding and saturation were changed together in one mode, so their individual contributions are not isolated. A full 2×2 ablation (rounding × saturation) was not run; it would be the next step to attribute the effect.
- Per-layer scale and integer-bit choices were not varied.
- Pearson correlation was not computed (§4).

## 6. Conclusion for this phase
- Fixed-point arithmetic configuration changes numerical fidelity materially at narrow widths in hls4ml C-simulation (SIMULATED). The default configuration (truncation, wrap) is the worst, and rounding with saturation removes most of the collapse at W = 8 and W = 10.
- At W ≥ 14 the default configuration already matches FP32 closely (agreement ≥ 0.97; ΔAUC ≤ 0.001).
- These statements are about arithmetic in simulation. They are not about hardware resources, latency or physical behaviour, none of which was measured.

## 7. Verification (TEST #1 and TEST #2)
- **TEST #1:** `scripts/phase3_numerical_consistency.py` run from the repository, output `reports/numerical_consistency.json` (copied to a temporary location).
- **TEST #2:** the same script run again from the same code, output copied again.
- **Result:** the two JSON outputs are **byte-identical** (`cmp`), and both are identical to the JSON that existed before this phase. The C-simulation is therefore deterministic for this configuration on this machine.
- No C-simulation project was edited between runs.

## 8. Limitations
- Single model (seed 42), single test subset (15,000 jets), one fixed integer width of 6 bits.
- Simulation only. C-simulation does not give latency, initiation interval, LUT, DSP, BRAM or FF.
- The hardware-oriented analysis elsewhere in the repository (LUT/DSP/BRAM/FF) is ESTIMATED by a linear heuristic and was never checked against synthesis.
