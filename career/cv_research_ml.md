# Piush Kole

ML Research & Applied Machine Learning | Model Compression · Reproducible Benchmarking · Scientific Computing

piushkole2004@gmail.com · Bengaluru, India · github.com/Piush2004

---

## Summary

MSc Data Science candidate with hands-on research experience in neural network quantization and
reproducible experimental design, built through an independent benchmark evaluating post-training and
quantization-aware training for a high-energy-physics jet-classification task. Comfortable owning a
project end to end: experimental design, implementation, statistical evaluation, automated testing, and
public documentation. Seeking ML research, applied ML, or scientific-computing internship roles.

## Technical skills

- **ML / quantization:** PyTorch, post-training quantization (PTQ), quantization-aware training (QAT),
  mixed-precision search, scikit-learn, statistical evaluation (paired CIs, multi-seed design)
- **Scientific computing / HEP tooling:** hls4ml (PyTorch → fixed-point C-simulation), NumPy, pandas
- **Software engineering:** Python, pytest (automated regression + API test suites), Git, FastAPI,
  Docker, CI (GitHub Actions)
- **Data & visualization:** SQL, Power BI, Tableau, Plotly, Matplotlib
- **Other:** AWS (Cloud Practitioner), LangChain/LangGraph (agentic LLM systems)

## Research project — HEP-Trigger Quantization Lab (independent research)

*github.com/Piush2004/hep-trigger-quantization-lab — public repository, code + reports + reproduction
instructions*

An independent, reproducible benchmark of low-bit neural-network quantization for a simulated
high-energy-physics jet-classification task (not CERN-affiliated; built on a public, particle-level
Monte Carlo dataset, not real detector data).

- Designed and ran a multi-seed (5-seed) benchmark comparing FP32, INT8/INT6/INT4 PTQ, and QAT across
  two MLP architectures, with pre-registered selection rules to prevent test-set leakage; one
  exploratory result that *was* test-set-selected was identified, disclosed as biased, and excluded
  from every confirmatory claim rather than hidden.
- Ran a near-exhaustive mixed-precision search (2,187 layouts; every combination of INT4/INT6/INT8
  across 7 quantizable weight/activation units) and found a QAT-selected layout that reliably beats
  uniform INT6 PTQ at 78% of its storage (all 5 seeds, 95% CI [0.0011, 0.0015] ROC-AUC).
- Benchmarked the quantized networks against non-neural baselines (logistic regression, gradient-boosted
  trees) and reported, rather than omitted, that a default-hyperparameter GBM beats the FP32 MLP on raw
  accuracy — framing the neural network's actual case as FPGA deployability, not accuracy.
- Implemented and validated a PyTorch → fixed-point (`hls4ml`) C-simulation conversion path, and trained
  a machine-learning surrogate for FPGA resource estimation on the public `wa-hls4ml` dataset —
  explicitly labelling every number by evidence tier (measured / simulated / estimated / not available)
  so simulated and measured results are never conflated.
- Built a FastAPI + Plotly research dashboard exposing the frozen results, covered by 14 automated API
  and security tests (biased-result exclusion, CORS/secret checks, device-separation invariants);
  packaged with Docker and a PWA manifest for public deployment.
- Engineering discipline: 141 automated tests passing across the data pipeline, quantization code,
  hardware-estimation path, and API layer (two independent runs, byte-identical results); every
  reproduction command and dataset fingerprint documented in `docs/REPRODUCIBILITY.md`.
- Wrote up the full study as a research-style manuscript (methodology, statistics, limitations,
  reproducibility statement) — see `paper/` in the repository.

## Education

**M.Sc. Data Science (Computer Science)** — CHRIST (Deemed to be University), Bengaluru · *Jul 2025 – May 2027 (in progress)*

**B.Sc. (Hons.) Data Science** — Techno India University, Kolkata · *May 2022 – Apr 2025* · 69.67%

## Experience

**Data Executive Intern — Cybertoss Pvt. Ltd.**, Kolkata (onsite) · *Jun 2024 – May 2025*

- Owned the end-to-end data lifecycle (collection, cleaning, validation, reporting) for large-scale
  datasets feeding downstream analytics and model-ready pipelines.
- Automated data-processing workflows in Python, reducing manual effort and improving delivery speed
  for cross-functional analytical projects.

**Data Science Summer Intern — RemarkSkill (KSHITIJ, IIT Kharagpur)**, remote · *Jun 2024*

- Built hands-on proficiency in Python (pandas, NumPy, Matplotlib) and Power BI through applied
  dataset projects and data-storytelling deliverables.

## Certifications

Agentic AI in Action: RAG with LangChain (Skillsoft, 2025) · AWS Cloud Practitioner Essentials (AWS
Academy, 2025) · Introduction to Business Analysis Using Spreadsheets (Coursera, 2025) · Python for
Data Science & Machine Learning: Zero to Hero (Udemy, 2023)

## Leadership

**IEEE Committee Member, CHRIST University** · *Dec 2025* — coordinated outreach to 10+ industry
companies for the IEEE student conference as part of a 20-member organizing committee.

---
*Other experience (full-stack/agentic LLM application development, business-analytics projects) is
available on request; this CV is scoped to research and ML-engineering evidence most relevant to
scientific-computing and ML-research roles.*
