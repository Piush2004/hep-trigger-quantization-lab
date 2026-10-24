# Paper evidence audit

Full claim-by-claim traceability for `paper/main.tex` lives in `paper/CLAIMS_AUDIT.md` (kept next
to the paper so it travels with it). This file is the top-level pointer the release checklist
references, plus a summary of the audit method.

## Method

Before drafting `paper/main.tex`, every section's source report was read directly (not recalled
from memory) and the exact numbers were copied from the report's own tables:

- Uniform PTQ/QAT → `reports/final_results.md`
- Architecture comparison → `reports/phase6_architecture_generalization.md`
- Calibration sensitivity → `reports/int4_ptq_stability_analysis.md`
- Mixed-precision search → `reports/phase12_exhaustive_mixed_precision_pareto.md`
- hls4ml C-simulation → `reports/phase7_hls_numerical_validation.md`
- Hardware surrogate → `reports/phase11_surrogate_hardware_validation.md`

No number in the paper was estimated, rounded beyond what the source report itself rounds to, or
derived by a calculation not shown in `paper/CLAIMS_AUDIT.md`.

## Result

Zero claims were found that could not be traced to a specific report. Zero numbers were adjusted
to make a comparison look stronger than the source data supports. One scope decision was made and
is disclosed in `paper/CLAIMS_AUDIT.md`'s final section: Phase 14 (CNN/jet-image) results exist on
a separate branch and were deliberately left out of this paper's pre-registered 14-section
structure rather than folded in without a decision to do so.

## Outstanding

- The paper has not been submitted, reviewed, or published anywhere. `paper/main.pdf` is a
  compiled manuscript draft, not a publication.
- Figures (`paper/figures/`) and standalone data tables (`paper/tables/`) are reserved directories,
  currently empty — the paper's tables are typeset directly in `main.tex` from the audited numbers
  above rather than generated as separate image/CSV assets. If plots are wanted (e.g. the
  mixed-precision Pareto frontier, the calibration-sensitivity curve), they can be generated from
  the same CSVs already compiled in `reports/tables/` and added without changing any reported
  number.
