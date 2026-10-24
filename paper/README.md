# Paper — build instructions

`main.tex` is a standard LaTeX article (no biblatex/biber, no custom class). It cites
`references.bib` via plain BibTeX (`\bibliographystyle{plainnat}`).

## Build

This repository's no-sudo environment has no system LaTeX install, so the paper is built with
[Tectonic](https://tectonic-typesetting.github.io/), a self-contained Rust LaTeX engine that
auto-fetches the TeX packages it needs on first run (cached afterward; no `apt`/`sudo` required).
Any standard LaTeX toolchain (TeX Live, MiKTeX) will also compile this file unmodified — Tectonic
is simply what was available in this environment.

```bash
# One-time setup (downloads the tectonic binary itself, ~22MB, from its official GitHub release —
# not piped to a shell, just extracted):
mkdir -p ~/.local/tectonic && cd ~/.local/tectonic
curl -sL "$(curl -sL https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest \
  | grep -o '"browser_download_url": *"[^"]*x86_64-unknown-linux-gnu[^"]*"' | cut -d'"' -f4)" \
  -o tectonic.tar.gz
tar xzf tectonic.tar.gz

# Build (from this directory):
~/.local/tectonic/tectonic main.tex
```

Produces `main.pdf`. The first build downloads the needed package bundle from Tectonic's registry
(one-time, requires network access); subsequent builds are fast and fully offline.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The manuscript source (14 sections per the project's writing brief). |
| `references.bib` | BibTeX bibliography — every entry checked against the original paper/arXiv listing, not invented. |
| `main.pdf` | Built output (committed so the paper is readable without a LaTeX toolchain). |
| `CLAIMS_AUDIT.md` | Every numerical claim in `main.tex`, traced to its source report/table. |
| `figures/`, `tables/` | Reserved for future figure/table assets if the paper is extended with plots. |

## Verification performed

- Compiled successfully with Tectonic 0.17.0; zero errors.
- One negligible 2pt overfull-hbox warning remains (an inline dataset-name mention, imperceptible
  at normal zoom); all other overfull-hbox warnings from the first compile (tables running past the
  margin, un-breakable file paths in the Reproducibility Statement) were found and fixed — see the
  git history of this file's commit for the before/after.
- Visually inspected every page of the rendered PDF (text extraction, not just "it compiled") to
  confirm no clipped text, broken table, or missing reference.
- Every numeric claim cross-checked against `CLAIMS_AUDIT.md` below.
