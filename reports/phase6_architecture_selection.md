# Phase 6 — architecture selection (pre-experiment, fixed before any run)

## Selected second architecture (Architecture B)
A deeper, wider MLP with the same input, output, activation and training protocol as the primary architecture (Architecture A). Only the hidden layer dimensions change.

| | Architecture A (primary) | Architecture B (second) |
|---|---|---|
| Input | 16 high-level features | 16 (same) |
| Hidden dims | [64, 32, 32] | **[128, 64, 64, 32]** |
| Hidden layers | 3 | **4** |
| Hidden activation | ReLU | ReLU (same) |
| Dropout | 0.1 | 0.1 (same) |
| Output | 5 logits, softmax | 5 logits, softmax (same) |
| Parameters | 4,389 | **16,837** |
| FP32 theoretical weight storage | 17,556 B | 67,348 B |
| Linear weight tensors (quantized units) | 4 (L0–L3) | 5 (L0–L4) |

Parameter count for B: 16·128+128 = 2,176; 128·64+64 = 8,256; 64·64+64 = 4,160; 64·32+32 = 2,080; 32·5+5 = 165; total 16,837.

## Rationale
- **Scientifically appropriate for the feature representation.** The 16 high-level features and 5-class output are unchanged, so the dataset and task are identical. Only capacity and depth change.
- **Materially different depth and width.** One more hidden layer and about 3.8× the parameters. This tests whether the precision boundary depends on the specific topology.
- **Lightweight.** Still under 17k parameters, consistent with the hardware-motivated scope of the project.
- **Not a Transformer or a graph network.** Those are outside the scope of this project and would change the task representation.

## Rejected alternatives
- A narrower MLP: smaller than A; less informative about scaling.
- Adding a large network or a convolutional or attention model: would change the inductive bias and is not justified by the feature representation.
- Multiple extra architectures: explicitly excluded by the phase rules.

## Fairness and training procedure
Architecture B uses **exactly the primary training protocol** with only `model.hidden_dims` overridden:
- Optimizer: Adam; learning rate 1e-3; weight decay 1e-4; batch size 1024.
- Epochs: at most 30, early stopping patience 6 on validation loss.
- Dataset: `hls4ml_lhc_jets_hlf`, 100,000-jet stratified subset (seed-controlled), 70/15/15 train/validation/test split, seed-controlled.
- Preprocessing: fitted on the training split only.
- QAT: same recipe as the primary: 7 fine-tuning epochs (`max(5, E//4)`) at 0.1× learning rate, straight-through estimator, starting from the FP32 model of the same seed.
- PTQ calibration: first 20,480 training jets, same procedure as Phase 5.

## Device control (added before running)
The frozen primary results (EXP_S*) were produced on **CUDA**. This session has **no CUDA**. Phase 4 showed INT4 PTQ point estimates change with the device (seed 42: 0.651 on CPU vs 0.572 on CUDA). Comparing a CPU-trained Architecture B against CUDA-trained Architecture A would confound architecture with device. Therefore **Architecture A is re-run on CPU under the Phase 6 protocol** as a control. The frozen CUDA results are kept as they are and are reported separately for reference only.

## Experimental grid
2 architectures × 5 configurations × 5 seeds (42, 123, 2026, 7, 99), all on CPU:
FP32, INT8 PTQ, INT6 PTQ, INT4 PTQ, INT4 QAT.

## No test-set selection
- Architecture B and its hyperparameters are fixed here, before any run.
- Checkpoints are chosen by validation loss (early stopping), as in the primary protocol.
- Test data is used only for final metrics.
- No hyperparameter is changed after seeing any result.
