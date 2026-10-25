# Project descriptions — reusable blurbs

Short and long versions of both projects, for reuse across CVs, LinkedIn, application forms, and
cover letters without rewriting from scratch each time. Every claim here is sourced the same way as
the CVs — see `cv_fact_check.md` and `CV_CHANGELOG.md`.

---

## HEP-Trigger Quantization Lab

**One-liner (application-form length):**
> A reproducible benchmark of neural-network quantization (PTQ/QAT, INT8/INT6/INT4) for a simulated
> high-energy-physics classification task — 141 automated tests, full reproduction documentation,
> public GitHub repository.

**Short (~60 words):**
> Independent research project benchmarking how much a neural network's numerical precision can be
> reduced before accuracy breaks down, using a public physics dataset. Found and disclosed a biased
> exploratory result rather than hiding it, a real narrow-width hardware-conversion failure mode, and
> that a standard non-neural baseline reliably beats the neural network on raw accuracy. 141 automated
> tests, two independent runs, identical results.

**Long (CV/portfolio bullet form):**
- Benchmarked PTQ and QAT at INT8/INT6/INT4 across two architectures, 5 seeds each, with the
  comparison rule fixed before the test results were examined — one earlier result that violated
  this was caught, disclosed as biased, and excluded from every confirmatory claim.
- Converted the trained model to fixed-point arithmetic for hardware deployment (`hls4ml`) and
  validated it against the full-precision model: strong agreement at sufficient bit-width (≥97%
  class agreement, W≥14), and a real accuracy collapse found and documented at narrow width —
  substantially mitigated by rounding and saturation.
- Benchmarked against a non-neural baseline, which reliably beat the neural network on raw accuracy
  (+0.0030 ROC-AUC, 95% CI entirely positive, 5/5 seeds) — reported plainly rather than omitted.
- 141 automated tests passing across two independent runs with identical results; full reproduction
  instructions and dataset fingerprints documented.

**Link:** github.com/Piush2004/hep-trigger-quantization-lab

---

## Synthetic Silence: Mapping Information Voids & Epistemic Risk in Generative AI

**Status: independent research prototype, in progress.** No manuscript or paper exists for this
project — don't describe it as published, submitted, or peer-reviewed anywhere.

**One-liner (application-form length):**
> A research prototype estimating epistemic risk in LLM responses (unsupported, fabricated, or
> overconfident claims) via claim extraction and live evidence retrieval — not a truth classifier.

**Short (~60 words):**
> Builds a system that scores how likely an LLM's claims are to be unsupported or fabricated, by
> extracting atomic claims, retrieving supporting literature live, and combining multiple real
> signals into a continuous, explainable risk score. Explicitly does not claim to determine true or
> false. Iterated through three architecture versions after live validation found real bugs, each
> documented and fixed rather than hidden.

**Long (CV/portfolio bullet form):**
- Built a pipeline extracting atomic claims from LLM output, retrieving supporting literature live
  (OpenAlex), and combining multiple real signals — evidence-density/Mahalanobis distance, sparse-
  autoencoder reconstruction error, linguistic specificity/uncertainty — into a continuous,
  explainable risk score; explicitly documented as a heuristic, not a calibrated truth classifier.
- Iterated through three architecture versions (an early GAN-based framing that was audited and
  found to have no real working implementation, replaced by a multi-agent pipeline, then by the
  current continuous-scoring version) after live multi-LLM validation surfaced real bugs — false
  positives on true claims, a false negative on a fabricated one — each documented and fixed rather
  than hidden.
- 100/100 backend tests and 15/15 extension tests passing; CI covers the test suite, an extension
  syntax check, and a Docker build check. Includes a web dashboard and a Chrome extension sharing
  one backend and account system.

**What NOT to claim:** published paper, peer review, calibrated/validated risk scores (the system's
own health endpoint reports `calibration_status: "NOT_CALIBRATED"` by design — no labeled dataset
exists yet to calibrate against), live production deployment, or verified Gemini API usage in a real
environment (the live-LLM code path exists but has never been exercised with a real API key in any
environment this project has run in).

**Link:** github.com/Piush2004/synthetic-silence (private repository — make public first if you want
to link it in an application, or describe it without a link)
