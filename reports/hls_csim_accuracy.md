# hls4ml C-simulation accuracy vs fixed-point width

Model: `EXP_S42_JET_FP32` (FP32 weights), 15,000 held-out test jets, measurement tier **simulated**. FP32 (PyTorch) accuracy on the same rows: 0.7495. Physical FPGA synthesis not performed; no latency/resources.

| ap_fixed<W,I> | frac bits | accuracy | agreement with FP32 predictions |
|---|---:|---:|---:|
| <8,6> | 2 | 0.2021 | 0.1624 |
| <10,6> | 4 | 0.3541 | 0.3433 |
| <12,6> | 6 | 0.7255 | 0.8958 |
| <14,6> | 8 | 0.7481 | 0.9705 |
| <16,6> | 10 | 0.7511 | 0.9929 |
| <18,6> | 12 | 0.7499 | 0.9985 |
