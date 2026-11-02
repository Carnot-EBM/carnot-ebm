#!/usr/bin/env python3
"""commit-msg hook: refuse [conductor] commits that touch operator-curated docs.

CLAUDE.md "Public Documentation Discipline" defines a set of files as
operator-curated; the autonomous loop is forbidden from editing them.
That rule lives in CLAUDE.md as design-time discipline only — there is
no mechanical enforcement, and exp3166 (2026-05-26 21:21Z, commit
4a65df9312b8, codex agent) demonstrated the gap by reverting README.md
from the project-level intro back to a HuggingFace model card. This
hook is the Layer 1 mechanical backstop for that rule.

How it works:
  1. Receives the commit message file path as sys.argv[1] (commit-msg
     hook contract).
  2. Reads the message subject (first non-empty line).
  3. If the subject starts with `[conductor]` (case-insensitive),
     inspects `git diff --cached --name-only` for any staged file
     whose path matches the operator-curated set.
  4. Exits non-zero with a structured explanation if both conditions
     are true; otherwise exits 0.

How it interacts with the conductor:
  - A failed commit-msg hook leaves the conductor's staged changes in
    the index (the commit didn't happen). On next iteration the
    conductor's `git add -A` will re-stage them and try again — the
    hook will keep refusing. The operator (or outer-loop) sees the
    repeated FAIL pattern in the conductor log and can either revert
    the offending edit or override the hook via `--no-verify` (with
    explicit operator authorization).

Outer-loop commits (`[outer-loop]` prefix) are allowed because they
represent operator-authorized work. Operator commits (no marker) are
also allowed. Only `[conductor]` is restricted.

Design rationale: the cheaper alternative — a positive-list of "loop
agents may edit only these paths" — was rejected because new code
paths are added every milestone and a positive-list becomes a
maintenance burden. The forbid-list is small (~14 paths), stable, and
the consequence of missing one is "an extra autonomous edit happens",
which gets caught at outer-loop review.

Spec coverage: CLAUDE.md "Public Documentation Discipline"
+ feedback_no_pruning_docs.md memory.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path


# Operator-curated paths per CLAUDE.md "Public Documentation Discipline".
# Glob patterns are supported via fnmatch (used for docs/blog/**/*.html).
# Update this list when CLAUDE.md's table changes.
OPERATOR_CURATED_PATHS: tuple[str, ...] = (
    "README.md",
    "NOTICE",
    "LICENSE",
    "docs/index.html",
    "docs/roadmap.md",
    "docs/research-log.md",
    "docs/blog/*.html",
    "docs/blog/**/*.html",
    "docs/getting-started.md",
    "docs/cli-usage.md",
    "docs/mcp-server.md",
    "docs/tutorial.md",
    "docs/concepts.md",
    "docs/api-reference.md",
    "docs/CNAME",
    "docs/arxiv-paper/main.tex",
)

# Subject prefixes that mark a commit as conductor-originated. Match is
# case-insensitive, and only the first non-empty subject line is checked.
CONDUCTOR_SUBJECT_PREFIXES: tuple[str, ...] = ("[conductor]",)


def _read_subject(path: Path) -> str:
    """Return the first non-empty, non-comment line of the message file."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return ""


def _is_conductor_commit(subject: str) -> bool:
    """True when the subject starts with any conductor-marker prefix."""
    s = subject.lower()
    return any(s.startswith(p) for p in CONDUCTOR_SUBJECT_PREFIXES)


def _paths_from_name_status(text: str) -> list[str]:
    """Every path a --name-status diff touches, BOTH sides of renames/copies.

    The first ship used --name-only, which reports only the DESTINATION of a
    rename -- so `R100 README.md docs/archive/project-intro.md` moved a
    protected doc out from under the guard without naming it (QA-layer
    SILENT_NON_FIRING finding, 2026-08-23). Moving a protected doc away IS an
    edit to it; the source side must be checked too.
    """
    paths: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split("\t")
        # A/M/D entries: status + one path. R###/C### entries: status +
        # source + destination. Everything after the status cell is a path.
        paths.extend(c.strip() for c in cells[1:] if c.strip())
    return paths


def _staged_files() -> list[str]:
    """Return every path the pending commit touches (rename sources included)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _paths_from_name_status(result.stdout)


def _matches_operator_curated(path: str) -> bool:
    """True iff path matches any operator-curated glob pattern."""
    for pat in OPERATOR_CURATED_PATHS:
        if fnmatch.fnmatchcase(path, pat):
            return True
    return False


def main() -> int:
    if len(sys.argv) < 2:
        # Hook contract requires the message file path. If something
        # invokes us without it, fail open rather than blocking
        # everything: the operator-side intent is to catch the
        # `[conductor]` + protected-path combination, not to block
        # malformed invocations.
        print("warning: operator_curated_docs_lint called without message path", file=sys.stderr)
        return 0

    msg_path = Path(sys.argv[1])
    if not msg_path.exists():
        print(f"warning: commit message file {msg_path} not found", file=sys.stderr)
        return 0

    subject = _read_subject(msg_path)
    if not _is_conductor_commit(subject):
        return 0  # not a conductor commit — irrelevant

    staged = _staged_files()
    if not staged:
        return 0  # empty commit — nothing to block

    violations = [p for p in staged if _matches_operator_curated(p)]
    if not violations:
        return 0

    print("=" * 72, file=sys.stderr)
    print("operator_curated_docs_lint: refusing [conductor] commit", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(file=sys.stderr)
    print(
        'CLAUDE.md "Public Documentation Discipline" forbids the',
        file=sys.stderr,
    )
    print(
        "autonomous loop from editing operator-curated files. The commit",
        file=sys.stderr,
    )
    print(
        f'subject ("{subject[:60]}...") marks this as a conductor commit,',
        file=sys.stderr,
    )
    print("but it touches the following protected paths:", file=sys.stderr)
    print(file=sys.stderr)
    for v in violations:
        print(f"  - {v}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "If the operator authorized this edit, the commit must be made by",
        file=sys.stderr,
    )
    print(
        "the outer-loop session (subject prefix [outer-loop]) or",
        file=sys.stderr,
    )
    print(
        "directly by the operator (no prefix). If the conductor produced",
        file=sys.stderr,
    )
    print(
        "this edit inadvertently, revert the changes to those paths and",
        file=sys.stderr,
    )
    print("re-stage without them.", file=sys.stderr)
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
