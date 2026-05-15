"""Study-phase sweep de-dupe helper.

The outer-loop's /study-sweep cron fetches arXiv listings every 2-4 hours.
Because `sortBy=submittedDate` + `max_results=8` returns the same recent
window per query, sweep rotations re-fetch the same 5-8 papers and waste
scoring effort. This helper extracts already-ingested arXiv IDs from
``research-studying.md`` so the sweep workflow can post-filter WebFetch
results before scoring.

Usage::

    # Print all known IDs, one per line (version suffix stripped)
    python scripts/sweep_dedupe.py

    # Check a specific candidate
    python scripts/sweep_dedupe.py --check 2605.14449v1
    # → prints "KNOWN" with exit 0, or "NEW" with exit 1

    # Filter a list of IDs from stdin (one per line); print only NEW ones
    cat candidates.txt | python scripts/sweep_dedupe.py --filter

Why this lives in scripts/ and not memory:
  * Memory is for narrative/preference, not callable code.
  * The cron prompt is fixed wording; the workflow step that calls this
    is the sweep skill itself, not the prompt.
  * Version-bump exceptions stay explicit: ``2512.15605v3`` and
    ``2512.15605v4`` dedupe to the same base ID by default. The operator
    (or future-me) can override by deleting the line from
    ``research-studying.md`` to force a re-score.

Filed by: outer-loop 2026-05-15 (operator confirmed de-dupe as the
efficiency win after 4 saturation-pattern sweeps in 24h).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "research-studying.md"

# Matches arXiv citations in any reasonable format the source file uses:
#   arXiv:2605.14449v1
#   arxiv:2605.14449
#   arXiv: 2605.14449
#   ArXiv ID:** 2605.14694v1
# The ID itself is YYMM.NNNNN with optional version suffix vN.
ARXIV_PATTERN = re.compile(
    r"(?:arxiv)\s*[:#]?\s*\**\s*(\d{4}\.\d{4,5})(v\d+)?",
    re.IGNORECASE,
)


def known_ids(source: Path = SOURCE) -> set[str]:
    """Return base arXiv IDs (no version) cited in research-studying.md."""
    if not source.exists():
        return set()
    text = source.read_text()
    matches = ARXIV_PATTERN.findall(text)
    # matches is list of (base_id, version_or_empty) tuples
    return {base for base, _version in matches}


def normalize(arxiv_id: str) -> str:
    """Strip arXiv: prefix and version suffix; return base YYMM.NNNNN."""
    cleaned = re.sub(r"^arxiv\s*[:#]?\s*", "", arxiv_id.strip(), flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", cleaned)


def main() -> int:
    args = sys.argv[1:]
    known = known_ids()

    if not args:
        for arxiv_id in sorted(known):
            print(arxiv_id)
        return 0

    if args[0] == "--check":
        if len(args) < 2:
            print("usage: sweep_dedupe.py --check <arxiv_id>", file=sys.stderr)
            return 2
        target = normalize(args[1])
        if target in known:
            print("KNOWN")
            return 0
        print("NEW")
        return 1

    if args[0] == "--filter":
        kept = 0
        skipped = 0
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            base = normalize(stripped)
            if base in known:
                skipped += 1
            else:
                print(stripped)
                kept += 1
        print(
            f"# filter: {kept} new, {skipped} known-skipped",
            file=sys.stderr,
        )
        return 0

    print(f"unknown argument: {args[0]}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
