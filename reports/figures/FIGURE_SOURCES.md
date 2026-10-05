# Figure sources

Every figure is produced by a script from experiment artifacts; the table(s) it plots are saved next to it in `reports/tables/`.
The un-prefixed names below are the final figure set (identical copies of the `phase2_fig*` files: frozen n = 3 seeds, seed-42 ROC curves).
The numbered files `01_roc_curves.png` ... `08_pareto_frontier.png` are byte-identical copies of the un-prefixed names (same order: ROC, AUC by precision, degradation, QAT vs PTQ, storage, latency, hardware, Pareto). They show the frozen n = 3 seeds; the n = 5 aggregate is in `phase3_auc_by_config_all_seeds.png` and `reports/tables/seed_statistics.csv`.
Phase 3 figures (n = 5 seeds, studies) have their own names.

| Figure | Script | Source data |
|---|---|---|
| `roc_curves.png` (= `phase2_fig1_*`) | `scripts/audit_phase2.py` | frozen seed-42 checkpoints; AUCs in `phase2_checkpoint_consistency.csv` |
| `auc_by_precision.png` (= `phase2_fig2_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv` |
| `quantization_degradation.png` (= `phase2_fig3_*`) | `scripts/audit_phase2.py` | `phase2_degradation_mean.csv`, `phase2_degradation_per_seed.csv` |
| `qat_vs_ptq.png` (= `phase2_fig4_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv`, `phase2_qat_vs_ptq_*.csv` |
| `storage_vs_auc.png` (= `phase2_fig5_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv` (theoretical weight storage) |
| `latency_vs_auc.png` (= `phase2_fig6_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv` (in-run, indicative latency); controlled latency: `phase3_latency_study.png` / `phase3_latency_study.csv` |
| `hardware_vs_auc.png` (= `phase2_fig7_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv` (estimated LUT), `reports/hls_csim_accuracy.json` (simulated) |
| `pareto_frontier.png` (= `phase2_fig8_*`) | `scripts/audit_phase2.py` | `phase2_stats.csv` |
| `phase3_auc_by_config_all_seeds.png` | `scripts/phase3_tables.py` | `experiment_summary.csv`, `seed_statistics.csv` |
| `phase3_qat_vs_ptq_vs_control.png` | `scripts/phase3_tables.py` | `qat_vs_ptq.csv` |
| `phase3_ptq_calibration_sensitivity.png`, `phase3_ptq_unit_sensitivity_int4.png` | `scripts/phase3_ptq_studies.py` | `phase3_ptq_s1_calibration.csv`, `phase3_ptq_s3_units.csv` |
| `phase3_per_class_int4.png` | `scripts/phase3_error_analysis.py` | `phase3_per_class_delta.csv` |
| `phase3_hls_fixed_point_modes.png` | `scripts/phase3_numerical_consistency.py` | `reports/numerical_consistency.json` |
| `phase3_latency_study.png` | `scripts/phase3_latency_study.py` | `phase3_latency_study.csv` |
