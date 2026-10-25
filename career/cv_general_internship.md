# Piush Kole

Data Science & Machine Learning · Research · Applied AI · Analytics

piushkole2004@gmail.com · Bengaluru, India · github.com/Piush2004

---

## Summary

MSc Data Science candidate with two independent, publicly documented research projects spanning
applied machine learning, reproducible experimental design, and honest reporting of what each
result does and doesn't support. Comfortable across the full practical cycle: Python/PyTorch model
development, statistical evaluation, automated testing, and clear technical writing — plus
data-analytics tooling (SQL, Power BI) from prior internship work. Looking to contribute to a data
science, ML, or applied-AI internship where rigor and clear documentation matter as much as the
model itself.

## Technical skills

- **Programming:** Python, SQL
- **ML & data science:** PyTorch, scikit-learn, pandas, NumPy, statistical/experimental design
  (multi-seed evaluation, confidence intervals)
- **Data visualization & analytics:** Power BI, Matplotlib, Plotly
- **Research & engineering tools:** Git, FastAPI, pytest, Docker, CI (GitHub Actions), `hls4ml`

## Research & selected projects

### HEP-Trigger Quantization Lab (independent research)
*github.com/Piush2004/hep-trigger-quantization-lab — public repository, code, and reports*

A reproducible benchmark of neural-network quantization (compressing a model's numerical precision)
for a simulated physics classification task.

- Benchmarked PTQ and QAT at INT8/INT6/INT4 across two architectures, 5 seeds each, with the
  comparison rule fixed before the test results were examined — one earlier result that violated
  this was caught, disclosed as biased, and excluded from every confirmatory claim.
- Converted the trained model to fixed-point arithmetic for hardware deployment (`hls4ml`) and
  validated it against the full-precision model: strong agreement at sufficient bit-width, and a
  real narrow-width accuracy collapse found, documented, and substantially mitigated by rounding
  and saturation — reported as found, not smoothed over.
- Benchmarked against a non-neural baseline, which reliably beat the neural network on raw accuracy
  (+0.0030 ROC-AUC, 95% CI entirely positive, 5/5 seeds) — reported plainly rather than omitted.
  141 automated tests passing across two independent runs with identical results.

### Synthetic Silence: Mapping Information Voids & Epistemic Risk in Generative AI
*Independent research prototype — in progress*

A system estimating epistemic risk in LLM responses: how likely a claim is to be unsupported,
fabricated, or overconfident, without claiming to determine true or false.

- Built a pipeline extracting atomic claims from LLM output, retrieving supporting literature live
  (OpenAlex), and combining multiple real signals — evidence-density, autoencoder reconstruction
  error, linguistic specificity/uncertainty — into a continuous, explainable risk score; iterated
  through three architecture versions after live multi-LLM validation surfaced real bugs, documented
  and fixed rather than hidden.
- 100/100 backend tests and 15/15 extension tests passing; CI covers the test suite, an extension
  syntax check, and a Docker build check.

## Experience

**Data Executive Intern — Cybertoss Pvt. Ltd.**, Kolkata (onsite) · *Jun 2024 – May 2025*

- Owned the end-to-end data lifecycle (collection, cleaning, validation, reporting) for large-scale
  datasets, automating Python workflows that reduced manual effort for cross-functional analytics.

**Data Science Trainee — RemarkSkill**, via KSHITIJ (IIT Kharagpur's techno-management festival),
remote · *Jun 2024*

- Applied Python (pandas, NumPy, Matplotlib) and Power BI to real-world dataset projects.

## Education

**M.Sc. Data Science (Computer Science)** — CHRIST (Deemed to be University), Bengaluru
*Jul 2025 – May 2027 (in progress)*

**B.Sc. (Hons.) Data Science** — Techno India University, Kolkata · *May 2022 – Apr 2025* · 69.67%

## Certifications & leadership

**Certifications:** AWS Cloud Practitioner Essentials (AWS Academy, 2025) · Python for Data Science
& Machine Learning (Udemy, 2023)

**Leadership:** IEEE Committee Member, CHRIST University (Dec 2025) — coordinated outreach to 10+
industry companies for a 20-member student-conference committee.
