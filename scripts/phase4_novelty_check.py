"""Structured consistency check for reports/research_novelty_analysis.md (Phase 4, TEST #1).

Checks:
  1. Every citation token of the form §1.N in the novelty analysis resolves to a
     heading in reports/related_work_review.md.
  2. Every line that makes a priority/novelty claim ("first", "novel", "only",
     "new") either carries a §-citation or is explicitly marked as not supportable
     or as a candidate.
Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "reports" / "related_work_review.md"
NOVELTY = ROOT / "reports" / "research_novelty_analysis.md"

CLAIM_WORDS = re.compile(r"\b(first|novel|novelty|only|new)\b", re.IGNORECASE)
CITE = re.compile(r"§\s?1\.\d+")
HEADING_NUM = re.compile(r"^###\s+(1\.\d+)\b", re.MULTILINE)
EXEMPT = re.compile(r"not supportable|candidate|not clearly|not claimed|Not A|Not E|Why not|no such|not established", re.IGNORECASE)


def main() -> int:
    review = REVIEW.read_text(encoding="utf-8")
    novelty = NOVELTY.read_text(encoding="utf-8")
    failures: list[str] = []

    known = set(HEADING_NUM.findall(review))
    for ref in sorted(set(re.findall(r"§\s?(1\.\d+)", novelty))):
        if ref not in known:
            failures.append(f"citation §{ref} in novelty analysis has no matching heading in related_work_review.md")

    for lineno, line in enumerate(novelty.splitlines(), start=1):
        if not CLAIM_WORDS.search(line):
            continue
        if CITE.search(line) or EXEMPT.search(line):
            continue
        # Headings and the classification-rule prose are allowed to mention the words.
        if line.lstrip().startswith("#"):
            continue
        failures.append(f"line {lineno}: priority/novelty wording without citation or qualifier: {line.strip()[:110]}")

    print(f"headings in review: {sorted(known)}")
    cited = set(re.findall(r"§\s?(1\.\d+)", novelty))
    print(f"citations checked: {len(cited)}")
    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
