#!/usr/bin/env python3
"""PreToolUse deny-hook: forbidden Bash commands become a harness boundary.

REQ-CONDUCTOR-DENYHOOK-1 (Conversion 3 of
docs/research-notes/cumulative-coherence-rule-to-check-2026-08-21.md).
Two prose rules become boundaries that cannot be forgotten:

  - CLAUDE.md "Never Stash — Always Commit-First": `git stash` reverts
    the working tree while conductor subprocesses may hold open file
    handles. The sanctioned path is `scripts/safe-pull.sh` (commit-first).
  - CLAUDE.md "Operator-Only External Publication": `arxiv submit`,
    OpenReview calls, `gh release create`, and `twine upload` create
    publicly-citable artifacts. Capability does not imply authorization.
  - Standing feedback: `--no-verify` on git commands skips the guard
    stack; fix the failing hook and retry instead.

Wired in `.claude/settings.json` as a PreToolUse hook with a `Bash`
matcher. Protocol: tool-call JSON on stdin; exit 2 denies the call and
feeds stderr back to the agent; exit 0 allows.

Known false-positive mode, accepted deliberately: a command ABOUT a
forbidden command (`grep 'git stash' CLAUDE.md`) is denied too. The
denial message names the rule, so an honest agent rephrases the search
(`grep 'git.stash'`). Simplicity beats pattern cleverness here — this
is a boundary for honest agents, not an adversarial sandbox.

Fail direction: OPEN, deliberately. Unparseable stdin allows the call
(exit 0) with a stderr note. This hook is a targeted boundary on six
command shapes; failing closed would block every Bash call on any
harness glitch, which trains bypassing the hook.
"""

from __future__ import annotations

import json
import re
import sys

# Ordered (pattern, message) rules. First match wins.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        # `git stash list` / `git stash show` are read-only and stay allowed.
        re.compile(r"\bgit\s+stash\b(?!\s+(?:list|show)\b)"),
        "git stash is forbidden (CLAUDE.md 'Never Stash — Always Commit-First'): "
        "a stash reverts the working tree under live conductor subprocesses. "
        "Use scripts/safe-pull.sh, or commit-first: "
        "git add -A && git commit -m '[outer-loop] preserve transient state'.",
    ),
    (
        # --no-verify on a git command (commit/push). The [^|;&]* guard keeps
        # the match inside one shell command, so `git log | tool --no-verify`
        # does not trip on the git half.
        re.compile(r"\bgit\s+[^|;&]*--no-verify\b"),
        "--no-verify is forbidden without explicit operator authorization "
        "(standing feedback: never skip pre-commit hooks). Fix the failing "
        "hook and retry the commit instead.",
    ),
    (
        re.compile(r"\barxiv\s+(?:submit|upload)\b", re.IGNORECASE),
        "arXiv submission is OPERATOR-ONLY (CLAUDE.md 'Operator-Only External "
        "Publication'). Prepare the package and checklist; the operator submits.",
    ),
    (
        re.compile(r"\bopenreview\b", re.IGNORECASE),
        "OpenReview calls are OPERATOR-ONLY (CLAUDE.md 'Operator-Only External "
        "Publication'). Prepare the package and checklist; the operator submits.",
    ),
    (
        re.compile(r"\bgh\s+release\s+create\b"),
        "gh release create is OPERATOR-ONLY (CLAUDE.md 'Operator-Only External "
        "Publication'): a release is a publicly-citable artifact.",
    ),
    (
        re.compile(r"\btwine\s+upload\b"),
        "twine upload is OPERATOR-ONLY. PyPI publishing ships via CI trusted "
        "publishing on a tagged release (see feedback_pypi_publish_via_ci), "
        "never from a local twine call.",
    ),
)


def forbidden(command: str) -> str | None:
    """The denial message for a forbidden command, or None to allow."""
    for pattern, message in _RULES:
        if pattern.search(command):
            return message
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        command = str((payload.get("tool_input") or {}).get("command", ""))
    except Exception as exc:
        # Fail-open on a malformed payload — see the module docstring.
        print(f"deny-hook: could not parse hook input ({exc}); allowing", file=sys.stderr)
        return 0
    message = forbidden(command)
    if message is not None:
        print(f"DENIED: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
