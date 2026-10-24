# CERN openlab / Summer Student application notes

Compiled from a web-research pass (public sources only — see citations) on what CERN openlab /
the Summer Student Programme actually requires and recruits for, so the CV and application are
positioned against real criteria rather than guesswork.

## Eligibility and required documents (verify against the live posting before applying — programme
details change year to year)

- Bachelor's or Master's students only (no PhDs); at least 6 semesters (3 years) of full-time
  university study completed by the programme start. Students graduating that summer can still
  apply if enrolled through the programme's start.
  Source: https://openlab.cern/faqs-summer-student-programme/
- Priority fields: computer science, mathematics, engineering, physics — but other fields are
  explicitly welcomed "if their transcript shows the relevant technical knowledge... and a strong
  computing profile." Good English required; all nationalities welcome.
- No prior CERN summer-student stint, and no more than 3 months of prior CERN work.
- **Required documents**: CV, an academic reference letter (less than 12 months old, from someone
  who has actually worked with you — does not need to be a professor), and a transcript
  (English/French preferred).
  Source: https://openlab.cern/faqs-summer-student-programme/ ;
  https://careers.cern/?p=14772
- Applicants cannot choose their own project directly, but can state preferences/skills in a
  **"message to the hiring manager"** field on the application.

## The one directly-relevant, actively-recruiting CERN research line found

**Next Generation Triggers (NGT), Task 1.3 — "Hardware-Aware AI Optimization"** (lead: Lorenzo
Moneta) builds a unified interface for neural-network pruning, quantization, and neural
architecture search, producing the open-source tool **PQuantML**, benchmarked against QKeras and
HGQ on jet-classification tasks. Current students (Najla Sadek, Anastasiia Petrovych) work
specifically on quantization within this task.
Sources: https://nextgentriggers.web.cern.ch/t13/ ;
https://nextgentriggers.web.cern.ch/meet-the-students-najla-sadek/ ;
https://nextgentriggers.web.cern.ch/anastasiia-petrovych/

Separately, openlab/NGT lightning-talk listings show other student projects extending `hls4ml`
itself (FPGA firmware synthesis, Xilinx AI Engine support) and binary neural networks on FPGAs for
ATLAS — confirming `hls4ml` is a recognized, active ecosystem there, not a niche tool.
Source: https://indico.cern.ch/event/1311334/

**This project's quantization/mixed-precision/`hls4ml` work maps onto this task's own stated scope
almost exactly** — this is worth naming explicitly in an application, not left implicit.

## Draft "message to the hiring manager" paragraph

Use/adapt this for the application's free-text field (not a cover letter — keep it to this length):

> My independent research project (github.com/Piush2004/hep-trigger-quantization-lab) benchmarks
> post-training and quantization-aware-training quantization, mixed-precision search, and
> `hls4ml` fixed-point simulation for a jet-classification model — the same technical direction as
> the Next Generation Triggers programme's Task 1.3 (Hardware-Aware AI Optimization / PQuantML). I
> would welcome a project in this area, or in ML-for-trigger-systems more broadly. I have no
> FPGA/HLS-synthesis experience yet and would be glad to learn that step as part of the internship
> — my own work stopped at C-simulation precisely because I didn't have synthesis tooling
> available.

**Why this phrasing**: names the specific CERN task by its own terminology (per the research
agent's finding that this reads as domain fluency rather than keyword-stuffing), and states the
FPGA/HLS gap honestly up front rather than letting a reviewer discover it — consistent with this
whole project's own disclosure discipline.

## What was explicitly NOT found (gap, not filled with guesswork)

No current (2023–2025) public first-person account by a recent openlab/NGT alumnus describing what
specifically got them selected, and no official CERN-published "what makes an application stand
out" statement, were found. The only alumni-advice source located was a 2017 blog post (archived,
partially unreadable — image-scanned PDF) recommending applying early and building depth across
fewer, more complete projects. Treat the positioning advice above as inferred from the published
eligibility criteria and project catalogue, not as confirmed insider guidance.

## Honest caveat on the NGT framing above

This project is **not** affiliated with CERN, openlab, or the Next Generation Triggers programme in
any way — the similarity above is a genuine technical overlap found through research, not a claim
of contact, collaboration, or endorsement. State it exactly that way in any application; do not let
the phrasing drift toward implying a connection that doesn't exist.
