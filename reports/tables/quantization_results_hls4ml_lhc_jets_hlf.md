# Quantization Results Summary — hls4ml_lhc_jets_hlf

| experiment_id        | precision   | method   |   accuracy |     f1 |   roc_auc |   theoretical_weight_storage_mb |   inference_latency_ms |
|:---------------------|:------------|:---------|-----------:|-------:|----------:|--------------------------------:|-----------------------:|
| EXP_012_JET_FP32     | fp32        | none     |     0.7495 | 0.7527 |    0.9335 |                          0.0167 |                 0.0691 |
| EXP_013_JET_INT8_PTQ | int8        | ptq      |     0.7503 | 0.7535 |    0.9333 |                          0.0046 |                 0.1765 |
| EXP_014_JET_INT6_PTQ | int6        | ptq      |     0.7501 | 0.7533 |    0.9323 |                          0.0036 |                 0.1725 |
| EXP_015_JET_INT4_PTQ | int4        | ptq      |     0.6507 | 0.6473 |    0.8832 |                          0.0025 |                 0.1801 |
| EXP_016_JET_INT8_QAT | int8        | qat      |     0.7523 | 0.7555 |    0.9344 |                          0.0046 |                 0.4721 |
| EXP_017_JET_INT6_QAT | int6        | qat      |     0.7517 | 0.7550 |    0.9336 |                          0.0036 |                 0.7171 |
| EXP_018_JET_INT4_QAT | int4        | qat      |     0.7291 | 0.7324 |    0.9232 |                          0.0025 |                 0.4902 |
