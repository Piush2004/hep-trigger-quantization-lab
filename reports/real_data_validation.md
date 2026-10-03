# Real-data validation

Dataset: `hls4ml_lhc_jets_hlf` (OpenML version 1), seed 42. **Status: PASS**

## Full dataset
- Samples: 830,000; features: 16; classes: g, q, t, w, z
- NaN: 0; Inf: 0; duplicate feature rows: 0
- Fingerprint (sha256 of float32 X): `b2863e6c92578c774dc4b677f88a8f054906226197e2ade5a938d3f2531f096c`

| Class | Count | Fraction |
|---|---:|---:|
| g | 167,125 | 0.2014 |
| q | 161,066 | 0.1941 |
| t | 167,851 | 0.2022 |
| w | 167,083 | 0.2013 |
| z | 166,875 | 0.2011 |

| Feature | min | max | mean | std | #(|z|>10) |
|---|---:|---:|---:|---:|---:|
| zlogz | -4.76 | -0.439 | -2.865 | 0.5804 | 0 |
| c1_b0_mmdt | 0.0911 | 0.4938 | 0.4333 | 0.05545 | 0 |
| c1_b1_mmdt | 7.343e-05 | 0.1652 | 0.03777 | 0.02915 | 0 |
| c1_b2_mmdt | 4.472e-08 | 0.07123 | 0.007995 | 0.009403 | 0 |
| c2_b1_mmdt | 2.5e-06 | 0.219 | 0.04561 | 0.03866 | 0 |
| c2_b2_mmdt | 1.473e-10 | 0.1079 | 0.007609 | 0.01217 | 0 |
| d2_b1_mmdt | 0.005866 | 3.968 | 1.296 | 0.458 | 0 |
| d2_b2_mmdt | 0.0001562 | 6.408 | 1.084 | 0.7301 | 0 |
| d2_a1_b1_mmdt | 0.005866 | 3.968 | 1.296 | 0.458 | 0 |
| d2_a1_b2_mmdt | 0.0002128 | 0.3666 | 0.1904 | 0.07542 | 0 |
| m2_b1_mmdt | 7.744e-05 | 0.1878 | 0.09002 | 0.03652 | 0 |
| m2_b2_mmdt | 2.187e-06 | 0.1377 | 0.04246 | 0.0264 | 0 |
| n2_b1_mmdt | 0.0006426 | 0.4495 | 0.2812 | 0.08456 | 0 |
| n2_b2_mmdt | 1.84e-05 | 0.3376 | 0.1439 | 0.08046 | 0 |
| mass_mmdt | 0.1134 | 573.6 | 75.15 | 55.61 | 0 |
| multiplicity | 6 | 212 | 51.89 | 21.68 | 0 |

## Configured subset and splits
- Requested subset size: 100000; actual: 100,000 (stratified)
- Train / validation / test: 69,999 / 15,001 / 15,000
- Duplicate feature rows shared across splits: 0
- Train fingerprint `37445332d190f8e27fb32424a73f4396e471bb43f9dee5b954718e6349d1aa9b`; test fingerprint `d4a0ceee7eab7998db8a58c34e9787abd2cec8baf95e373d85b2dff0990508f3`
- Preprocessed train: max |mean| = 1.83e-07, std range = [0.999993622303009, 1.0000075101852417]

## Leakage screen
Method: depth-3 decision tree on each single feature, fit on first half / scored on second half of up to 40000 subset rows; flag if accuracy > 0.9. Chance accuracy = 0.20. Flagged features: none.

| Feature | single-feature accuracy |
|---|---:|
| zlogz | 0.343 |
| c1_b0_mmdt | 0.329 |
| c1_b1_mmdt | 0.558 |
| c1_b2_mmdt | 0.606 |
| c2_b1_mmdt | 0.426 |
| c2_b2_mmdt | 0.428 |
| d2_b1_mmdt | 0.315 |
| d2_b2_mmdt | 0.328 |
| d2_a1_b1_mmdt | 0.315 |
| d2_a1_b2_mmdt | 0.343 |
| m2_b1_mmdt | 0.326 |
| m2_b2_mmdt | 0.300 |
| n2_b1_mmdt | 0.332 |
| n2_b2_mmdt | 0.315 |
| mass_mmdt | 0.611 |
| multiplicity | 0.361 |
