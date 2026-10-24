# LinkedIn content — ready to paste

I don't have access to your LinkedIn account (no credentials, and I won't attempt to log in as
you), so this is drafted content for you to paste into your own profile — not something I can
publish directly. Everything below uses only verified facts from `cv_fact_check.md`.

## Headline (220 char limit)

```
MSc Data Science | ML Research: Neural Network Quantization & Hardware-Aware Compression | Python · PyTorch · hls4ml
```

Alternative, shorter:
```
MSc Data Science Candidate | Quantization & Reproducible ML Benchmarking | Python · PyTorch
```

## About section

```
MSc Data Science candidate (CHRIST University, Bengaluru) researching neural network quantization
and reproducible ML benchmarking, with an independent project applying hardware-aware compression
(PTQ, QAT, mixed-precision search) to a high-energy-physics jet-classification task — the same
technical direction as CERN's Next Generation Triggers hardware-aware AI research line.

Core project: HEP-Trigger Quantization Lab — a public, fully reproducible benchmark (5-seed
statistical design, pre-registered selection rules, 141 automated tests) comparing full-precision
and low-bit quantized neural networks, validated against hls4ml fixed-point C-simulation and an
independently-trained hardware-resource surrogate. Every result is labelled by how it was obtained
(measured, simulated, estimated, or unavailable) — no claim is made beyond what the evidence
actually supports.

Background: data pipeline automation and analytics (Cybertoss Pvt. Ltd.), applied Python/Power BI
work (IIT Kharagpur KSHITIJ programme), and hands-on PyTorch/scikit-learn experience from this
research project.

Open to ML research, applied ML, and scientific-computing internships.

Repository: github.com/Piush2004/hep-trigger-quantization-lab
```

## Featured post — project announcement (for a LinkedIn post, not the About section)

```
I spent the last several months building a reproducible benchmark of neural-network quantization
for high-energy-physics jet classification — PTQ and QAT at INT8/INT6/INT4, a near-exhaustive
mixed-precision search over 2,187 configurations, and a PyTorch-to-fixed-point (hls4ml)
C-simulation validation path.

A few honest findings worth sharing:
→ INT8 quantization is nearly free; INT4 degrades substantially unless you fine-tune with QAT.
→ A standard gradient-boosted-tree baseline actually beats the neural network on raw accuracy —
  the case for the quantized model is FPGA deployability, not accuracy, and I think that's worth
  saying plainly rather than burying it.
→ A near-exhaustive mixed-precision search found a layout that reliably beats the best uniform
  precision at a smaller storage footprint — something a smaller, hand-picked search had missed.

Everything is public, tested (141 automated tests, two independent runs), and documented with full
reproduction instructions: no claim beyond what's actually measured, simulated, or estimated.

Code + reports: github.com/Piush2004/hep-trigger-quantization-lab
#MachineLearning #Quantization #HEP #OpenSource #ReproducibleResearch
```

## Project entry (LinkedIn "Projects" section)

```
Title: HEP-Trigger Quantization Lab
Dates: [fill in your actual start month] – Present
Description:
Independent research project benchmarking low-bit neural-network quantization (PTQ/QAT,
INT8/INT6/INT4) for a simulated high-energy-physics jet-classification task. Built a near-exhaustive
mixed-precision search, a PyTorch→hls4ml fixed-point simulation path, and an ML surrogate for
FPGA resource estimation. 141 automated tests, full statistical reporting (5-seed, confidence
intervals), and complete reproduction documentation. Public repository with an interactive
FastAPI/Plotly results dashboard.
Skills: Python, PyTorch, scikit-learn, hls4ml, FastAPI, Docker, Statistical Analysis
URL: github.com/Piush2004/hep-trigger-quantization-lab
```

## Note on the CERN/NGT framing

The About section and post above mention CERN's Next Generation Triggers programme as a
*direction of overlap*, not an affiliation. Keep that distinction exact if you reuse this text —
see `cern_openlab_application_notes.md` for the sourcing and the explicit caveat.
