# Reproducibility check — EXP_S42_JET_INT4_QAT

Re-ran from the recorded config, in isolation, 2x. Absolute tolerance per metric: 0.005. Devices: original `cuda`, re-run `cuda`. **Overall: PASS**

| repeat | |Δaccuracy| | |Δprecision| | |Δrecall| | |Δf1| | |Δroc_auc| | |Δpr_auc| | max |Δweight| | class agreement | max |Δprob| | metadata identical | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.000000 | 0.00e+00 | True | True |
| 2 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.000000 | 0.00e+00 | True | True |

Excluded from comparison (legitimately differ): timestamp, git_commit, latency (wall-clock), system.cuda_available if the device changed.
