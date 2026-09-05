#!/usr/bin/env python3
"""Detect a memory file whose body grew while its summary stayed the same.

WHAT THIS GUARDS (REQ-INFRA-6975, 2026-09-05). The project memory directory holds one fact
per file. Each file has a `description:` line, and `MEMORY.md` holds a one-line pointer per
file. The pointer is what loads into a session. The body is what nobody reads until something
goes wrong. So an append that leaves the summary describing the OLD body makes the summary
lie, and the lie is the part that gets read. It cost real time twice on 2026-09-05; see
commit 9975deb7bd and `feedback_write_it_down_or_lose_it.md`.

WHY A BASELINE AND NOT A HEURISTIC. "Does this sentence summarise this file" has no ground
truth. What CAN be measured is change-coupling: the body changed and the summary did not. The
memory directory is outside git, so there is no history to diff. This script keeps its own
baseline in a hidden sidecar next to the memory files, and maintains it by OBSERVATION: when
the summary moves, the baseline follows; when only the body moves, the file is flagged and the
baseline is held until the summary moves. No one has to remember to update the sidecar. That
is the difference from the discipline that failed.

THE RULE. A file drifts when, since its baseline, its body gained at least MIN_GROWTH_LINES
non-blank lines AND its `description:` is unchanged AND its `MEMORY.md` line is unchanged.
Both halves of the summary must move to clear a flag, because a description fixed without the
index line is exactly the state the first incident was found in. Small edits (typo fixes,
one-line touch-ups) re-baseline silently; a new fact is never one line.

TWO CALLERS. `scripts/outer_loop_dashboard.py` calls `memory_lines()` hourly; that call
updates the baseline. The `--hook` mode is wired as a Claude Code PostToolUse hook on Edit and
Write; it is read-only and reminds the editor at the moment of the append, from the edit
payload alone. The dashboard catches what the reminder did not prevent.

FAIL DIRECTION: CLOSED AND LOUD. An unreadable memory directory prints UNREADABLE. A corrupt
baseline prints RESET. A first run prints that it created the baseline. The check never
returns an empty result that could be read as "clean".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

MIN_GROWTH_LINES = 2
BASELINE_NAME = ".memory_index_drift_baseline.json"
PREFIX = "memory      "

_DESC_RE = re.compile(r"^description:\s*(.*)$", re.M)
_INDEX_RE = re.compile(r"^- \[[^\]]*\]\(([^)]+)\)")


def memory_dir(repo: Path | None = None) -> Path:
    """Where Claude Code keeps this project's memory files.

    Claude Code names the project directory by replacing every non-alphanumeric character of
    the working directory with `-`. `CLAUDE_MEMORY_DIR` overrides the derivation for tests and
    for a relocated directory. Derived, never hardcoded, per the write-target rule.
    """

    env = os.environ.get("CLAUDE_MEMORY_DIR")
    if env:
        return Path(env)
    root = repo or Path(__file__).resolve().parents[1]
    encoded = re.sub(r"[^A-Za-z0-9]", "-", str(root))
    return Path.home() / ".claude" / "projects" / encoded / "memory"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _nonblank(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def split_memory_file(text: str) -> tuple[str, str]:
    """Return (description, body). Body is everything after the closing `---`."""

    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                front = "".join(lines[1:i])
                m = _DESC_RE.search(front)
                desc = m.group(1).strip() if m else ""
                return desc, "".join(lines[i + 1 :])
    return "", text


def index_lines(mem: Path) -> dict[str, str]:
    """File name -> its full `MEMORY.md` line. Any edit to the line counts as touching it."""

    path = mem / "MEMORY.md"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = _INDEX_RE.match(line)
        if m:
            out[Path(m.group(1)).name] = line
    return out


def snapshot(mem: Path) -> dict[str, dict]:
    """The current state of every memory file, reduced to what drift needs."""

    idx = index_lines(mem)
    out: dict[str, dict] = {}
    for path in sorted(mem.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        desc, body = split_memory_file(path.read_text(errors="replace"))
        line = idx.get(path.name)
        out[path.name] = {
            "body_sha": _sha(body),
            "body_lines": _nonblank(body),
            "desc_sha": _sha(desc),
            "index_sha": _sha(line) if line is not None else None,
        }
    return out


def compare(
    baseline: dict[str, dict], current: dict[str, dict], min_growth: int = MIN_GROWTH_LINES
) -> tuple[list[tuple[str, int]], dict[str, dict]]:
    """Return (drifted, next_baseline).

    `drifted` lists (file, lines added since the summary last moved). `next_baseline` keeps the
    old entry for a drifted file, so the flag persists and the growth count accumulates until
    BOTH the description and the index line change.
    """

    drifted: list[tuple[str, int]] = []
    nxt: dict[str, dict] = {}
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None or base["body_sha"] == cur["body_sha"]:
            nxt[name] = cur
            continue
        desc_moved = base["desc_sha"] != cur["desc_sha"]
        index_moved = base["index_sha"] != cur["index_sha"]
        unindexed_both = base["index_sha"] is None and cur["index_sha"] is None
        summary_moved = desc_moved and (index_moved or unindexed_both)
        growth = cur["body_lines"] - base["body_lines"]
        if summary_moved or growth < min_growth:
            nxt[name] = cur
            continue
        drifted.append((name, growth))
        nxt[name] = base
    return drifted, nxt


def load_baseline(mem: Path) -> tuple[dict[str, dict] | None, str]:
    """Return (baseline, state) where state is 'ok', 'missing', or 'corrupt'."""

    path = mem / BASELINE_NAME
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
        files = data["files"]
        if not isinstance(files, dict):
            raise TypeError("files is not a dict")
        return files, "ok"
    except (OSError, ValueError, KeyError, TypeError):
        return None, "corrupt"


def save_baseline(mem: Path, files: dict[str, dict]) -> None:
    path = mem / BASELINE_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": 1, "files": files}, indent=1, sort_keys=True))
    os.replace(tmp, path)


def memory_lines(mem: Path | None = None, write: bool = True) -> list[str]:
    """Dashboard lines. Always at least one line, so a silent check is impossible."""

    mem = mem or memory_dir()
    if not mem.is_dir() or not os.access(mem, os.R_OK):
        return [f"{PREFIX}UNREADABLE {mem}"]
    current = snapshot(mem)
    baseline, state = load_baseline(mem)
    if baseline is None:
        if write:
            save_baseline(mem, current)
        why = "created" if state == "missing" else "RESET (baseline json unreadable)"
        return [f"{PREFIX}baseline {why} for {len(current)} files; drift detectable from next run"]
    drifted, nxt = compare(baseline, current)
    if write:
        save_baseline(mem, nxt)
    if not drifted:
        return [f"{PREFIX}{len(current)} files, 0 drifted"]
    items = "  ".join(f"{n}(+{g})" for n, g in sorted(drifted, key=lambda t: -t[1]))
    return [
        f"{PREFIX}{len(current)} files, {len(drifted)} DRIFTED (body grew, summary untouched): "
        f"{items}"
    ]


def hook_reminder(payload: dict, mem: Path | None = None) -> str:
    """The reminder for one Edit/Write payload, or '' when nothing needs saying.

    Stateless for Edit: growth and description-touch are read from the payload itself, so a
    stale or absent baseline cannot confuse it. Write has no old content in the payload, so it
    uses the baseline when one exists, and otherwise only checks the index line.
    """

    mem = mem or memory_dir()
    tool = payload.get("tool_name", "")
    inp = payload.get("tool_input") or {}
    fp = inp.get("file_path", "")
    if tool not in ("Edit", "Write") or not fp:
        return ""
    path = Path(fp)
    try:
        path.resolve().relative_to(mem.resolve())
    except ValueError:
        return ""
    if path.suffix != ".md" or path.name == "MEMORY.md":
        return ""
    name = path.name
    indexed = name in index_lines(mem)
    growth = 0
    touched_desc = False
    if tool == "Edit":
        old = str(inp.get("old_string", ""))
        new = str(inp.get("new_string", ""))
        growth = _nonblank(new) - _nonblank(old)
        touched_desc = bool(_DESC_RE.search(old) or _DESC_RE.search(new))
        # Quiet once the description has moved since the last hourly baseline: the author is
        # already attending to the summary, and 16 appends to one file in five hours (2026-07-24)
        # is the one measured pattern that would otherwise repeat the same reminder.
        if not touched_desc and path.exists():
            baseline, _ = load_baseline(mem)
            base = (baseline or {}).get(name)
            if base is not None:
                desc, _body = split_memory_file(path.read_text(errors="replace"))
                touched_desc = _sha(desc) != base["desc_sha"]
    else:
        baseline, _ = load_baseline(mem)
        base = (baseline or {}).get(name)
        if base is not None and path.exists():
            desc, body = split_memory_file(path.read_text(errors="replace"))
            growth = _nonblank(body) - base["body_lines"]
            touched_desc = _sha(desc) != base["desc_sha"]
    parts: list[str] = []
    if growth >= MIN_GROWTH_LINES and not touched_desc:
        parts.append(
            f"body grew by {growth} non-blank lines and its `description:` has not moved"
        )
    if not indexed:
        parts.append("`MEMORY.md` has no line for it")
    if not parts:
        return ""
    return (
        f"memory_index_drift: `{name}` -- " + "; ".join(parts) + ". The `MEMORY.md` line is what "
        "a future session reads. Update `description:` and the `MEMORY.md` line for this file "
        "in this same edit series, or the hourly dashboard flags it as DRIFTED."
    )


def main(argv: list[str]) -> int:
    mem: Path | None = None
    if "--memory-dir" in argv:
        mem = Path(argv[argv.index("--memory-dir") + 1])
    if "--hook" in argv:
        try:
            payload = json.load(sys.stdin)
        except (ValueError, OSError):
            return 0
        text = hook_reminder(payload, mem)
        if text:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PostToolUse",
                            "additionalContext": text,
                        }
                    }
                )
            )
        return 0
    lines = memory_lines(mem, write="--dry-run" not in argv)
    print("\n".join(lines))
    return 1 if "DRIFTED" in lines[0] or "UNREADABLE" in lines[0] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
