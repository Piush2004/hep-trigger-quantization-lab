#!/usr/bin/env bash
# Builds paper.pdf from paper.md + refs.bib.
#
# Two-stage (pandoc has no built-in PDF engine in this environment -- pdflatex needs a LaTeX
# install this environment doesn't have and installing one needs sudo, which times out
# non-interactively here): pandoc renders Markdown + citeproc to a standalone HTML file, then
# weasyprint (pip-installable, no sudo needed) converts that HTML to PDF. Verified end-to-end:
# the output PDF's extracted text matches the source Markdown's content and citations resolve
# to real author-year entries from refs.bib.
#
# Requires: pandoc on PATH (or set PANDOC=/path/to/pandoc), and `weasyprint` installed in the
# project's Python environment (pip install weasyprint).
set -euo pipefail
cd "$(dirname "$0")"

PANDOC="${PANDOC:-pandoc}"
PYTHON="${PYTHON:-../.venv-wsl/bin/python}"

"$PANDOC" paper.md -o paper.html --citeproc --bibliography=refs.bib --standalone \
  --metadata title="Benchmarking Quantization for Real-Time HEP Trigger Classifiers"
"$PYTHON" -m weasyprint paper.html paper.pdf
echo "built paper.pdf"
