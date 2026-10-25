# Piush Kole

Data Science Student · Applying for CERN Short-Term Internship — Data Science, AI & Analytics

piushkole2004@gmail.com · Bengaluru, India · github.com/Piush2004

---

## Summary

MSc Data Science candidate (CHRIST University, Bengaluru), currently a full-time student, applying
for CERN's Short-Term Internship Programme under Data Science, AI & Analytics. Built and
publicly documented an independent machine-learning research project — a reproducible benchmark of
neural-network quantization — covering the full practical skill set a data-science placement draws
on: Python/PyTorch model development, statistical evaluation, automated testing, and clear written
documentation of methods and limitations. Comfortable picking up new tools and methods quickly
(this project required learning `hls4ml`, a HEP-specific ML-deployment toolchain, from scratch), and
used to working precisely and communicating findings clearly — including where a result went against
the simplest narrative. Eager to apply this foundation to a real project under supervision, and to
learn from CERN's own data-science and computing teams.

## Technical skills

- **ML & data science:** Python, PyTorch, scikit-learn, pandas, NumPy — model training, evaluation,
  and statistical analysis (multi-seed experimental design, confidence intervals)
- **Tools picked up for this project:** `hls4ml` (HEP-specific ML-deployment toolchain, learned from
  its own documentation), FastAPI, Docker, Git, pytest, CI (GitHub Actions)
- **Data & visualization:** SQL, Power BI, Plotly, Matplotlib
- **Working habits:** writing a selection rule before touching results, automated regression testing,
  documenting what a result does and doesn't support

## Project — HEP-Trigger Quantization Lab (independent)

*github.com/Piush2004/hep-trigger-quantization-lab — public repository, code, and reports*

An independent research project benchmarking how much a neural network's numerical precision can be
reduced (a compression technique called quantization) before its accuracy on a classification task
breaks down, using a public physics dataset.

- Built and evaluated multiple model variants (standard precision vs. three reduced-precision
  variants, two training strategies) across 5 random seeds each, with the comparison rule decided
  *before* looking at the held-out test results — to avoid picking whichever number looked best after
  the fact.
- Learned and applied `hls4ml`, a specialized toolchain used in HEP computing to convert a trained
  PyTorch model into fixed-point arithmetic for hardware deployment, and validated it against the
  original model: strong agreement at sufficient bit-width, and a real accuracy collapse found and
  documented at narrow width — substantially mitigated by rounding and saturation.
- Found and reported a result that didn't match the simplest expectation: a standard, non-neural
  baseline model reliably beat the neural network on raw accuracy (+0.0030 ROC-AUC, 95% CI entirely
  positive, 5/5 seeds) — reported plainly rather than left out.
- Wrote 141 automated tests (run twice independently to confirm identical results) and full
  reproduction documentation, so every number in the project's reports can be independently checked.

## Education

**M.Sc. Data Science (Computer Science)** — CHRIST (Deemed to be University), Bengaluru
*Jul 2025 – May 2027 (in progress, full-time)*

**B.Sc. (Hons.) Data Science** — Techno India University, Kolkata · *May 2022 – Apr 2025* · 69.67%

## Experience

**Data Executive Intern — Cybertoss Pvt. Ltd.**, Kolkata (onsite) · *Jun 2024 – May 2025*

- Owned the end-to-end data lifecycle (collection, cleaning, validation, reporting) for large-scale
  datasets, automating Python workflows that reduced manual effort for cross-functional analytics.

**Data Science Trainee — RemarkSkill**, via KSHITIJ (IIT Kharagpur's techno-management festival),
remote · *Jun 2024*

- Applied Python (pandas, NumPy, Matplotlib) and Power BI to real-world dataset projects.

## Certifications & leadership

AWS Cloud Practitioner Essentials (AWS Academy, 2025) · Python for Data Science & Machine Learning
(Udemy, 2023) · IEEE Committee Member, CHRIST University (Dec 2025) — coordinated outreach to 10+
industry companies for a 20-member student-conference committee, developing communication and
teamwork skills in a multicultural setting.
