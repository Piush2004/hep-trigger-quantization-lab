# Numerical consistency: FP32 PyTorch ↔ quantized PyTorch ↔ hls4ml fixed point

Model family: MLP 16-64-32-32-5, seed-42 models, 15,000 held-out test jets (hls4ml_lhc_jets_hlf). Reference: FP32 PyTorch (accuracy 0.7495, macro ROC-AUC 0.9334). Logit/probability errors are against the FP32 reference on the same jets.

## Quantized PyTorch (fake-quantization; measured)

| model | class agreement | accuracy | logit MAE | logit max abs | prob MAE | prob max abs | ROC-AUC | ΔROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8_PTQ | 0.9931 | 0.7503 | 0.0333 | 0.480 | 0.0030 | 0.150 | 0.9333 | -0.0001 |
| INT6_PTQ | 0.9735 | 0.7485 | 0.1258 | 2.594 | 0.0123 | 0.755 | 0.9317 | -0.0017 |
| INT4_PTQ | 0.7193 | 0.5716 | 0.7988 | 7.747 | 0.1060 | 0.995 | 0.8353 | -0.0981 |
| INT8_QAT | 0.9735 | 0.7533 | 0.2818 | 9.777 | 0.0130 | 0.479 | 0.9342 | +0.0007 |
| INT6_QAT | 0.9636 | 0.7509 | 0.3000 | 10.510 | 0.0160 | 0.691 | 0.9336 | +0.0002 |
| INT4_QAT | 0.9169 | 0.7322 | 0.4925 | 11.058 | 0.0359 | 0.972 | 0.9217 | -0.0118 |

## hls4ml C-simulation of the FP32 weights, ap_fixed<W,6> (simulated; physical FPGA synthesis not performed)

Input, weights, biases, layer outputs and result share ap_fixed<W,6> (plus the modes below). hls4ml 1.3.0, backend Vivado, io_parallel, Strategy Latency, ReuseFactor 1.

| mode | W | class agreement | accuracy | logit MAE | logit max abs | prob MAE | ROC-AUC | ΔROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default (TRN, WRAP, accum = W bits) | 8 | 0.1624 | 0.2021 | 1.635 | 20.29 | 0.2295 | 0.5077 | -0.4257 |
| default (TRN, WRAP, accum = W bits) | 10 | 0.3433 | 0.3541 | 1.508 | 16.91 | 0.2161 | 0.6794 | -0.2540 |
| default (TRN, WRAP, accum = W bits) | 12 | 0.8958 | 0.7255 | 0.972 | 6.77 | 0.1312 | 0.9086 | -0.0249 |
| default (TRN, WRAP, accum = W bits) | 14 | 0.9705 | 0.7481 | 0.295 | 1.74 | 0.0292 | 0.9325 | -0.0009 |
| default (TRN, WRAP, accum = W bits) | 16 | 0.9929 | 0.7511 | 0.076 | 0.44 | 0.0070 | 0.9333 | -0.0001 |
| RND + SAT, accum = W bits | 8 | 0.5959 | 0.5034 | 1.805 | 11.00 | 0.1662 | 0.8372 | -0.0963 |
| RND + SAT, accum = W bits | 10 | 0.9403 | 0.7343 | 0.396 | 5.35 | 0.0353 | 0.9262 | -0.0072 |
| RND + SAT, accum = W bits | 12 | 0.9871 | 0.7489 | 0.069 | 1.21 | 0.0068 | 0.9330 | -0.0005 |
| RND + SAT, accum = W bits | 14 | 0.9963 | 0.7505 | 0.019 | 0.32 | 0.0017 | 0.9334 | -0.0000 |
| RND + SAT, accum = W bits | 16 | 0.9989 | 0.7497 | 0.004 | 0.07 | 0.0004 | 0.9334 | +0.0000 |
| TRN, WRAP, wide accum ap_fixed<28,12> | 8 | 0.2152 | 0.2017 | 1.626 | 30.96 | 0.2259 | 0.5274 | -0.4060 |
| TRN, WRAP, wide accum ap_fixed<28,12> | 10 | 0.6145 | 0.5486 | 1.337 | 14.19 | 0.1847 | 0.8297 | -0.1037 |
| TRN, WRAP, wide accum ap_fixed<28,12> | 12 | 0.9631 | 0.7462 | 0.456 | 5.29 | 0.0466 | 0.9302 | -0.0032 |
| TRN, WRAP, wide accum ap_fixed<28,12> | 14 | 0.9896 | 0.7507 | 0.120 | 1.29 | 0.0106 | 0.9331 | -0.0003 |
| TRN, WRAP, wide accum ap_fixed<28,12> | 16 | 0.9975 | 0.7500 | 0.031 | 0.32 | 0.0026 | 0.9334 | -0.0000 |
| RND + SAT, wide accum ap_fixed<28,12> | 8 | 0.7961 | 0.6508 | 0.671 | 5.58 | 0.0831 | 0.8942 | -0.0392 |
| RND + SAT, wide accum ap_fixed<28,12> | 10 | 0.9696 | 0.7471 | 0.158 | 2.60 | 0.0139 | 0.9309 | -0.0025 |
| RND + SAT, wide accum ap_fixed<28,12> | 12 | 0.9913 | 0.7500 | 0.047 | 0.94 | 0.0047 | 0.9332 | -0.0002 |
| RND + SAT, wide accum ap_fixed<28,12> | 14 | 0.9973 | 0.7502 | 0.014 | 0.23 | 0.0013 | 0.9334 | -0.0000 |
| RND + SAT, wide accum ap_fixed<28,12> | 16 | 0.9993 | 0.7498 | 0.003 | 0.04 | 0.0003 | 0.9334 | +0.0000 |

Cross-implementation check: class agreement between quantized-PyTorch INT8 PTQ and hls4ml `ap_fixed<16,6>` (default mode): 0.9918 (both are compared to FP32 above; they are different quantization schemes).
