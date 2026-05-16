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
    # → prints "KNOWN" / "VERSION_BUMP" / "NEW" with exit 0/0/1

    # Filter a list of IDs from stdin (one per line); print only NEW
    cat candidates.txt | python scripts/sweep_dedupe.py --filter

    # Filter but ALSO output VERSION_BUMP candidates for re-scoring
    cat candidates.txt | python scripts/sweep_dedupe.py --filter --include-version-bumps

    # Print full known IDs WITH version suffix (so version-bumps surface)
    python scripts/sweep_dedupe.py --with-versions

Version-bump semantics (added 2026-05-16 operator-approved):

  When research-studying.md has ``arXiv:2512.15605v3`` and a candidate
  is ``2512.15605v4``:
    - Default mode: ``v4`` is KNOWN-skipped (same base ID).
    - --include-version-bumps: ``v4`` surfaces as VERSION_BUMP for
      explicit re-evaluation (abstract may have shifted with the
      new revision).
    - --check 2512.15605v4: prints ``VERSION_BUMP`` (exit 0 — not NEW,
      but worth attention).

  This catches the silent-skip footgun where a paper's substantive
  update gets filtered as "we've seen this ID" when actually the
  abstract has materially changed.

Filed by: outer-loop 2026-05-15 / extended 2026-05-16 (operator approved
version-aware mode after asking whether the filter catches things
that could help the project).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "research-studying.md"

# Matches arXiv citations in any reasonable format the source file uses.
# Capture base ID + version separately.
ARXIV_PATTERN = re.compile(
    r"(?:arxiv)\s*[:#]?\s*\**\s*(\d{4}\.\d{4,5})(v\d+)?",
    re.IGNORECASE,
)


def known_base_ids(source: Path = SOURCE) -> set[str]:
    """Return base arXiv IDs (no version) cited in research-studying.md."""
    if not source.exists():
        return set()
    text = source.read_text()
    matches = ARXIV_PATTERN.findall(text)
    return {base for base, _version in matches}


def known_versions(source: Path = SOURCE) -> dict[str, set[str]]:
    """Return {base_id: {versions_seen}} from research-studying.md.

    Empty-string in the version set means we've seen the unversioned form.
    """
    out: dict[str, set[str]] = {}
    if not source.exists():
        return out
    text = source.read_text()
    matches = ARXIV_PATTERN.findall(text)
    for base, version in matches:
        out.setdefault(base, set()).add(version or "")
    return out


def normalize(arxiv_id: str) -> tuple[str, str]:
    """Strip arXiv: prefix; return (base, version) tuple. Version may be ""."""
    cleaned = re.sub(r"^arxiv\s*[:#]?\s*", "", arxiv_id.strip(), flags=re.IGNORECASE)
    m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?", cleaned)
    if not m:
        return (cleaned, "")
    return (m.group(1), m.group(2) or "")


def classify(arxiv_id: str, versions: dict[str, set[str]]) -> str:
    """Return one of 'NEW', 'KNOWN', 'VERSION_BUMP'."""
    base, version = normalize(arxiv_id)
    if base not in versions:
        return "NEW"
    seen = versions[base]
    if not version:
        # Candidate has no version; KNOWN regardless of what we have
        return "KNOWN"
    if version in seen:
        return "KNOWN"
    # Same base, different version (or we have unversioned and candidate has version)
    # If the only thing we have is "" (unversioned), the candidate's specific
    # version is technically new info — treat as VERSION_BUMP.
    return "VERSION_BUMP"


def main() -> int:
    args = sys.argv[1:]
    versions = known_versions()

    # Default: print base IDs sorted (back-compat with original behavior)
    if not args:
        for base in sorted(versions):
            print(base)
        return 0

    if args[0] == "--with-versions":
        # Print "base + max version" pairs for each known paper
        for base in sorted(versions):
            seen = versions[base]
            # Pick the lexicographically latest version we've seen, "" if none
            latest = max(seen) if seen else ""
            print(f"{base}{latest}")
        return 0

    if args[0] == "--check":
        if len(args) < 2:
            print("usage: sweep_dedupe.py --check <arxiv_id>", file=sys.stderr)
            return 2
        result = classify(args[1], versions)
        print(result)
        return 0 if result in ("KNOWN", "VERSION_BUMP") else 1

    if args[0] == "--filter":
        include_version_bumps = "--include-version-bumps" in args[1:]
        kept = 0
        version_bumped = 0
        skipped_known = 0
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            result = classify(stripped, versions)
            if result == "NEW":
                print(stripped)
                kept += 1
            elif result == "VERSION_BUMP":
                if include_version_bumps:
                    # Prefix VERSION_BUMP to a comment line so downstream
                    # can split: NEW IDs are bare, version-bumps are tagged.
                    print(f"{stripped}  # VERSION_BUMP")
                    version_bumped += 1
                else:
                    skipped_known += 1
            else:  # KNOWN
                skipped_known += 1
        bump_msg = (
            f", {version_bumped} version-bumps surfaced"
            if include_version_bumps
            else ""
        )
        print(
            f"# filter: {kept} new, {skipped_known} known-skipped{bump_msg}",
            file=sys.stderr,
        )
        return 0

    print(f"unknown argument: {args[0]}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
