"""Shared helpers for the hostile-reviewer audits that call an external model CLI.

REQ-OPS-AUDIT-REVIEWER-1. Why this file exists: six audits launched plain `codex`. Inside the
conductor service, plain `codex` is the old system copy in /usr/bin, which rejects newer model
names. The conductor itself avoids that copy through the `CODEX_BIN` variable. The audits did
not read it, so their fallback failed on every call. The failure text was also cut to the first
200 characters of stderr, which is only the startup banner, so the real error was never seen.
See docs/research-notes/claim-audit-reviewer-failure-diagnosis-2026-10-08.md.
"""

from __future__ import annotations

import os
import shutil

#: How much of the END of stderr a failure message keeps. CLIs print the real error last,
#: after a banner and (for codex) an echo of the whole prompt.
FAILURE_TAIL_CHARS = 300

#: stderr up to this many characters over the limit is kept whole instead of being trimmed.
TAIL_SLACK_CHARS = 100

#: In a chain, the first (primary) reason is cut to this many characters so the fallback's
#: error stays near the front of the combined message.
CHAIN_FIRST_REASON_CHARS = 160


def codex_bin() -> str:
    """Return the codex command to run: `$CODEX_BIN` if set, else plain `codex`.

    This is the same override the conductor reads (scripts/research_conductor.py), so the audits
    and the conductor always run the same binary.
    """
    return os.environ.get("CODEX_BIN") or "codex"


def failure_text(
    label: str,
    returncode: int | None,
    stderr: str | None,
    *,
    binary: str | None = None,
    limit: int = FAILURE_TAIL_CHARS,
) -> str:
    """Explain a failed reviewer call: the end of stderr first, then who failed.

    Keeps the last `limit` characters of stderr, because CLIs print the real error at the end.
    After it, in brackets, goes the first stderr line (for codex, its version) when that line is
    not already in the tail, and the binary path, so a stale binary shows up in the report.
    A very long final line, or output printed after the error, can still push the error out of a
    short prefix: callers should keep at least 600 characters of this text.
    """
    err = (stderr or "").strip()
    first = next((ln.strip() for ln in err.splitlines() if ln.strip()), "")
    if len(err) <= limit + TAIL_SLACK_CHARS:
        tail = err  # near the limit: keep it whole rather than cut a word in half
    else:
        tail = err[-limit:]
        if "\n" in tail:
            tail = tail.split("\n", 1)[1]  # start on a line boundary, not mid-word
    tail = tail.strip() or "<no stderr>"
    parts = []
    if first and first not in tail:
        parts.append(first[:60])
    if binary:
        parts.append(f"at {shutil.which(binary) or binary}")
    ident = f" [{' '.join(parts)}]" if parts else ""
    return f"{label} exit {returncode}: {tail}{ident}"


def chain_failure(first: str, second: str) -> str:
    """Join two failure reasons when a primary reviewer and its fallback both fail.

    The primary's reason is cut short (it is usually one stable line), so the fallback's error,
    which is the part that varies, stays near the front.
    """
    return f"{first[:CHAIN_FIRST_REASON_CHARS]} | fallback: {second}"
