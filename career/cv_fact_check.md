# CV fact-check — sources and verification status

This file exists so every claim in `cv_research_ml.md` / `cv_cern_openlab.md` /
`cv_cern_short_term_internship.md` / `project_portfolio.md` traces to an actual source. Nothing
below is invented; anything not verifiable is listed as missing, not guessed.

## Which CV targets what

- `cv_cern_short_term_internship.md` — **the actual submission target**, tailored to CERN's real,
  currently-open posting: **Short-Term Internship (STAG 2026)**, field of work "Data Science, AI &
  Analytics," https://careers.cern/jobs/short-term-internship/, deadline 01/11/2026 23:59 Geneva
  time. Verified directly from CERN's own page (not a third-party repost) via browser, 2026-10-24.
  Confirmed this is the *only* Short-Term Internship posting currently open — every "Short-Term
  Internship" link across all field-of-work categories on careers.cern/jobs/ (Data Science AI &
  Analytics, Civil Engineering, HR, Logistics, etc.) resolves to this same single generic listing;
  it is not a data-science-specific role, the field-of-work tag is just how applicants indicate
  interest within one shared application.
- `cv_cern_openlab.md` — built earlier for a *different* CERN programme (the openlab/Summer Student
  track), not this one. Kept for reference since the underlying research-alignment content (Next
  Generation Triggers Task 1.3) may still be useful for that separate application, but it is **not**
  tailored to the Short-Term Internship posting above and should not be submitted for it.
- `cv_research_ml.md` — general ML-research CV, not CERN-specific.

**Eligibility check for the Short-Term Internship, done directly from CERN's own page**: requires
being a national of a CERN Member State or Associate Member State. India is listed as a CERN
Associate Member State (agreement signed 2016); the posting's only named exclusions (quota ceiling
reached) are Pakistan, Lithuania, Latvia, Cyprus, Croatia, and Turkey — India is not among them. Also
requires current full-time enrollment in a technical/engineering/administrative post-secondary
programme (Bachelor's/Master's/diploma; PhD students excluded) and good English or French — both
satisfied by the verified facts below. Required documents: CV (English/French, PDF) and "any
document you consider relevant" — no reference letter or transcript is required for this specific
posting (unlike CERN's separate openlab programme, which does require those).

## Primary sources used

1. **Existing resume** — `C:\Users\piush\Downloads\Piush_Kole_Resume (3).pdf` (most recently modified
   of 15 resume files/variants found in Downloads, 2026-07-14; used as the primary source per the
   request's own instruction to prefer an existing CV when one is found). This resume is currently
   angled toward a "Gen AI Application Developer / LangChain" role, not research/HEP — the new CVs
   below re-angle the same underlying facts toward ML-research/CERN-Openlab without changing any fact.
2. **This repository** (`hep-trigger-quantization-lab`, branches `master` / `demo/web-release` /
   `phase14-cnn-jet-images`) — reports under `reports/`, test output from this session, and commit
   history. Every project metric below cites the specific report/table it came from.
3. **A web-research pass on CERN openlab/NGT requirements and project catalogue** (public sources
   only), run via a research agent this session — findings and every source URL are in
   `career/cern_openlab_application_notes.md`. The CERN/NGT framing added to `cv_cern_openlab.md`'s
   summary (the "hardware-aware neural-network compression" phrasing, the Next Generation Triggers
   Task 1.3 mention) comes from this pass, not from assumption.

## Verified facts (safe to state as-is)

| Fact | Source |
|---|---|
| Name: Piush Kole | resume |
| Email: piushkole2004@gmail.com | resume; matches this session's own user-email context |
| Location: Bengaluru, Karnataka, India | resume |
| M.Sc. Data Science (Computer Science), CHRIST (Deemed to be University), Bengaluru — pursuing, Jul 2025 – May 2027 | resume |
| B.Sc. (Hons.) Data Science, Techno India University, Kolkata — 69.67%, May 2022 – Apr 2025 | resume |
| Data Executive Intern, Cybertoss Pvt. Ltd., Kolkata (onsite) — Jun 2024 – May 2025 | resume |
| Data Science Summer Intern, RemarkSkill (KSHITIJ, IIT Kharagpur), remote — Jun 2024 | resume |
| Certifications: "Agentic AI in Action: RAG with LangChain" (Skillsoft, 2025); "AWS Cloud Practitioner Essentials" (AWS Academy, 2025); "Introduction to Business Analysis Using Spreadsheets" (Coursera, 2025); "Python for Data Science & Machine Learning: Zero to Hero" (Udemy, 2023) | resume |
| IEEE Committee Member, CHRIST University, Dec 2025 — outreach to 10+ companies, 20+ member committee | resume |
| 2nd Prize, Science Olympiad | resume (no date given on the source resume) |
| GitHub repository is public and live: https://github.com/Piush2004/hep-trigger-quantization-lab | `reports/deployment_report.md` §3, verified via `gh repo view` this session |
| Test suite (this session, `release/v1` branch): 141 passed, 6 skipped, two independent runs | run directly in this session (see `reports/release_test_report.md` for the methodology this follows) |
| PTQ/QAT evaluated at INT8/INT6/INT4 on two MLP architectures, 5 seeds | `reports/final_results.md`, `reports/phase6_architecture_generalization.md` |
| Near-exhaustive mixed-precision search (2,187 layouts; 3^7 over 7 quantizable units) found a QAT-selected layout (2,916 B, 78% of uniform INT6's storage) that reliably beats uniform INT6 PTQ, all 5 seeds, 95% CI [0.0011, 0.0015] | `reports/phase12_exhaustive_mixed_precision_pareto.md` |
| A gradient-boosted-tree baseline (`HistGradientBoostingClassifier`, default hyperparameters) reliably beats the FP32 MLP on raw ROC-AUC (0.9373 vs the MLP; +0.0030, 95% CI [0.0018, 0.0041], 5/5 seeds) — reported as an honest limitation, not hidden | `reports/phase13_baselines_and_efficiency.md` |
| hls4ml PyTorch→fixed-point C-simulation conversion implemented and verified (SIMULATED tier only — no HLS synthesis, no physical FPGA) | `reports/phase7_hls_numerical_validation.md`, `docs/REPRODUCIBILITY.md` §8 |
| ML surrogate hardware-resource model trained on the public `fastmachinelearning/wa-hls4ml` dataset, with its own disclosed limitation (bit-width insensitivity) | `reports/phase11_surrogate_hardware_validation.md` |
| FastAPI + Plotly research dashboard; 14 passing API/security tests (biased-result exclusion, CORS absence, no secrets, device separation) | `reports/release_test_report.md` §B, §E |
| PWA implemented (manifest, icons, service worker) | `reports/final_release_checklist.md`; service-worker *registration* specifically is flagged there as unverified in the automated browser tool, not as broken |

## Facts needing your confirmation before publishing

- **Phone number** (+91 74394-06095, from the resume): present on your existing resume but I have
  **not** added it to the CV files below, and I am **not** committing `career/` to the public GitHub
  repository by default — a personal phone number in a public, indexable repo is permanent and
  scrapable. Tell me if you want it included in the CV documents (fine for a document you send
  directly to an employer) and/or whether `career/` should be pushed to the public repo at all.
- **LinkedIn URL**: your resume has a "LinkedIn" hyperlink but the PDF extraction did not surface the
  underlying URL as text. I need the actual URL to include it — left as a placeholder below.
- **Hugging Face Space URL**: does not exist yet (deployment is blocked on your HF credentials — see
  `reports/release_readiness_audit.md`). The CV's "Live demo" line is written to degrade gracefully
  (points to the GitHub repo) until/unless a Space is deployed; update it once one exists.

## Claims deliberately excluded (could not be verified, or would overclaim)

- No FPGA synthesis, no physical hardware measurement, no production-trigger deployment — the CVs
  below describe this explicitly as *simulated fixed-point evaluation*, never as validated FPGA work.
- No claim that the paper is published, submitted, or peer-reviewed.
- No claim of CERN affiliation, employment, or collaboration — "independent research project,
  publicly documented" is the only framing used.
- The non-HEP resume projects (Agentix, TT-GAN hallucination-detection research, Sales Performance
  Dashboard) are **not** included in the research-focused CV below, since they would dilute a
  one-page CERN/ML-research-targeted document; they remain on your existing general-purpose resume
  and are listed in `cv_research_ml.md`'s "other experience" line only as a one-line pointer, not
  expanded with bullets, so nothing about them is restated or re-verified here.
- No download counts, user counts, star counts, or "production" claims for the dashboard — it is a
  research/portfolio demo, described as such.
