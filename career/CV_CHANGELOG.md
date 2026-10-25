# CV changelog — line-by-line audit and corrections

Triggered by an external review (pasted by the user, 2026-10-25) of `cv_cern_short_term_internship.pdf`.
Every flagged claim below was independently re-verified against this session's own primary sources
(repository reports, direct test runs, the original resume) before any change was made — not taken on
the external review's word alone. One flagged claim turned out to be **already correct**; it's recorded
as such rather than silently "fixed" to match the review.

## Verified correct as-is (external review flagged, re-checked, confirmed true)

| Claim | External review's concern | Verification | Outcome |
|---|---|---|---|
| "A standard, non-neural baseline model matched or beat the neural network on raw accuracy" | "Not established by the project status shared... should not remain without evidence" | Re-read `reports/phase13_baselines_and_efficiency.md` directly: *"The gradient-boosted-tree baseline reliably beats the MLP... a statistically clean result (95% CI entirely positive, all 5 seeds)."* Exact figures: GBM 0.9373 ± 0.0017 vs. FP32 MLP, **+0.0030 ROC-AUC, 95% CI [0.0018, 0.0041], 5/5 seeds**. | **KEPT**, tightened from "matched or beat" (hedged) to "reliably beat" with the exact figures, since that's what the source actually shows. |
| "141 automated tests... two independent runs, identical results" | "Verify actual current count... distinguish passed/skipped/collected" | Ran `pytest -q -rs` twice myself, directly, on `release/v1` this session: **141 passed, 6 skipped**, byte-identical both runs. | **KEPT** — this is a directly-measured fact from this session, not inherited from an older report. |
| GitHub repository exists and is public | "Verify... do not claim a public repository without verification" | Checked via `gh repo view`; confirmed public, 200 on GitHub Pages. (Separately: the repo was deleted and recreated during this session — see `reports/deployment_report.md` for that incident; it is live again as of 2026-10-25.) | **KEPT**, confirmed live. |
| Docker / GitHub Actions CI listed as tools used | "Verify... before listing as practical experience" | `demo/Dockerfile` is real, reviewed, hardened (non-root user, `HEALTHCHECK`). `.github/workflows/tests.yml` was verified this session by building a genuinely fresh scratch venv and running the suite against it — not just "it exists." | **KEPT**. |

## Corrected

| Section | Before | After | Reason |
|---|---|---|---|
| hls4ml bullet (all CVs) | "validated that its output numerically agreed with the original model" (unqualified) | "validated it against the original model: strong agreement at sufficient bit-width (≥97% class agreement, W≥14), and a real accuracy collapse found and documented at narrow width — substantially mitigated by rounding and saturation" | The unqualified version was imprecise. `reports/phase7_hls_numerical_validation.md` shows real width-dependent behavior: W=8 under default (truncate/wrap) arithmetic collapses to 0.16 class agreement; W≥14 reaches ≥0.97. The corrected wording is both more accurate *and* a stronger signal of rigor (found a real failure mode, not just "it worked"). |
| RemarkSkill role title (all CVs) | "Data Science Summer Intern — RemarkSkill (KSHITIJ, IIT Kharagpur)" | "Data Science Trainee — RemarkSkill, via KSHITIJ (IIT Kharagpur's techno-management festival)" | The original phrasing could read as employment located at IIT Kharagpur. KSHITIJ is IIT Kharagpur's techno-management festival (public, verifiable fact) that RemarkSkill ran a training programme through — the corrected phrasing states that structure explicitly instead of implying direct IIT Kharagpur employment. **I could not independently verify beyond this** whether RemarkSkill's programme was formally an internship, a training programme, or a workshop — this came from your own original resume, and I have no access to RemarkSkill's own records. See "Needs your confirmation" below. |
| Trailing footer note (`cv_cern_openlab.md`, earlier session) | Referenced an internal filename (`cv_research_ml.md`) | Removed | A real employer/reviewer has no reason to know what that filename means; it only reveals the document was templated. (Fixed in an earlier pass, noted here for the complete record.) |

## New files created (not corrections — net-new content)

- **`cv_general_internship.md/.docx/.pdf`** — a general-purpose internship CV (not CERN-specific), per your request, covering data science / ML / applied-AI / analytics roles broadly. Includes both research projects.
- Added **Synthetic Silence** as a second research project to `cv_general_internship.md` and `cv_research_ml.md` (not to the CERN-specific CVs, which stay tightly scoped to the HEP project most relevant to those specific applications — adding a second, unrelated project there would dilute a CERN-targeted pitch rather than strengthen it; happy to add it there too if you want).

## Needs your confirmation

- **RemarkSkill's exact nature** (internship vs. training programme vs. workshop): I used your own original resume as the source and have no independent way to verify further. If you know it was formally one of these, tell me and I'll make the title exact rather than the current structurally-honest-but-generic "Trainee."
- **AWS Cloud Practitioner Essentials**: your resume lists this as a credential from "AWS Academy, 2025." I have not independently verified whether this is a completion certificate from an AWS Academy course versus the official AWS Certified Cloud Practitioner exam credential — these are different things. If you have the certificate, confirm which it is so the CV states it precisely.
- **Exact start month for each project** (for a "Dates" field, if a specific application form asks for one) — not currently stated in any CV version since the projects don't have clean single start dates I can verify independently.
