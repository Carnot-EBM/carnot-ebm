#!/usr/bin/env python3
"""Canonical-URL lint — fail when forbidden URL patterns leak into source.

Per CLAUDE.md "Canonical Repository URL Discipline" (2026-05-20):
the project's canonical GitHub URL is
`github.com/Carnot-EBM/carnot-ebm`. Source files, docs, configs, tests,
and specs MUST NOT contain `github.com/ianblenke/carnot` — that is the
operator's local-filesystem-mirror name, not the canonical URL.

This linter scans tracked files for the forbidden URL pattern and
exits non-zero on any hit. It SKIPS build-artifact / log / immutable-
historical paths where the pattern appears legitimately (Vivado bakes
local filesystem paths into outputs; `ops/status.md` and
`ops/changelog.md` quote the old URL in historical-sweep descriptions).

Usage (CLI):

    python scripts/canonical_url_lint.py [--verbose]

Exit codes:
    0 — clean
    1 — at least one forbidden URL found in actionable file

Designed for pre-commit hook and CI. Pairs with
`scripts/exclusion_manifest_lint.py` as the second URL-discipline gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Forbidden URL pattern. Local-filesystem paths like
# `/home/ianblenke/github.com/ianblenke/carnot/...` are filtered out
# in a post-match check: we match the bare URL substring, then reject
# lines that have a local-absolute-path prefix containing the URL as
# a path segment (preceded by `/home/`, `/Users/`, `/tmp/`, etc.).
URL_PATTERN = re.compile(r"github\.com/ianblenke/carnot")
# Local-path prefix pattern: when the URL appears as a path component
# of an absolute filesystem path, skip. E.g. `/home/ianblenke/github.com/
# ianblenke/carnot/results/...` is a local dir, not a URL.
LOCAL_PATH_CONTEXT = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:/home/|/Users/|/tmp/|/opt/|/var/|/mnt/|/srv/|/root/|"
    r"~/|\$HOME/|file://)"
    r"[^\s\"'`]*github\.com/ianblenke/carnot"
)

# Paths to skip (build artifacts, logs, immutable history). The lint
# treats these as out-of-scope: they may contain the forbidden URL
# because Vivado/CI/build tools bake local paths into outputs, or
# because the file is a frozen historical record.
SKIP_PATH_PATTERNS = [
    r"^\.Xil/",
    r"sim_work/",
    r"^output/",
    r"^logs/",
    r"^results/",  # immutable experiment artifacts
    r"^models/.*\.log$",
    r"^ops/lineage-retirements/",
    r"^hardware/kv260/.*\.(log|jou|wdb|rlx|dbg|pb)$",
    r"^hardware/kv260/sim_work/",
    r"^\.preflight_test_cache\.json$",
    r"^\.session_context\.tmp$",
    r"^cov_output\.txt$",
    r".*\.backup\.(log|jou)$",
    r"^vivado\.(log|jou)$",
]
_SKIP_RE = re.compile("|".join(SKIP_PATH_PATTERNS))

# Files where the OLD URL appears as quoted prose describing the sweep
# itself — historical context, not canonical-URL refs.
SKIP_FILES = {
    "ops/changelog.md",
    "ops/status.md",
    # canonical_url_lint.py itself contains the forbidden pattern as
    # the regex it's checking for; exempt to avoid self-trigger.
    "scripts/canonical_url_lint.py",
    # CLAUDE.md describes the rule including a forbidden-example block.
    "CLAUDE.md",
    # .pre-commit-config.yaml comments describe the forbidden pattern.
    ".pre-commit-config.yaml",
}


def list_tracked_files() -> list[str]:
    """Return git ls-files for the repo, one path per line."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [f for f in result.stdout.splitlines() if f]


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, line_text), ...] for URL-form hits.

    Lines where the bad pattern appears ONLY as part of a local
    filesystem path (preceded by /home/, /Users/, /tmp/, etc.) are
    filtered out: those are local-disk paths, not canonical URL refs.
    """
    hits: list[tuple[int, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not URL_PATTERN.search(line):
                    continue
                # If every match on the line is local-path-prefixed, skip.
                # If even one match is URL-form, flag.
                line_no_local = LOCAL_PATH_CONTEXT.sub("", line)
                if URL_PATTERN.search(line_no_local):
                    hits.append((lineno, line.rstrip("\n")[:200]))
    except Exception:
        pass
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    files = list_tracked_files()
    total_hits = 0
    flagged_files: list[tuple[str, list[tuple[int, str]]]] = []

    for relpath in files:
        if _SKIP_RE.search(relpath):
            continue
        if relpath in SKIP_FILES:
            continue
        full = PROJECT_ROOT / relpath
        if not full.is_file():
            continue
        hits = scan_file(full)
        if hits:
            flagged_files.append((relpath, hits))
            total_hits += len(hits)

    if not flagged_files:
        if args.verbose:
            print(f"Canonical-URL lint clean ({len(files)} files scanned).")
        return 0

    print(
        f"Canonical-URL lint FAIL: {total_hits} hit(s) in "
        f"{len(flagged_files)} file(s).\n"
        f"Per CLAUDE.md 'Canonical Repository URL Discipline':\n"
        f"  WRONG: github.com/ianblenke/carnot\n"
        f"  RIGHT: github.com/Carnot-EBM/carnot-ebm\n"
    )
    for relpath, hits in flagged_files:
        print(f"  {relpath}:")
        for lineno, text in hits[:5]:
            print(f"    {lineno}: {text}")
        if len(hits) > 5:
            print(f"    ... and {len(hits) - 5} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
