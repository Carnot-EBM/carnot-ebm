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

FAIL DIRECTION: CLOSED AND LOUD for the check, OPEN for the hook. An unreadable directory or
file prints UNREADABLE. A missing `MEMORY.md` prints MISSING. A corrupt baseline prints RESET.
A first run prints that it created the baseline. The check never returns an empty result that
could be read as "clean". The hook exits 0 and prints nothing on any error, because a
reminder must never break an edit.

A TEST MUST NOT WRITE THE LIVE BASELINE. Found by adversarial review 2026-09-05: the dashboard's
own pre-existing tests call `render()` and would have rewritten the real sidecar on every pytest
run, shrinking the observation window from an hour to minutes. Under pytest, the implicit real
directory is read but never written unless `CLAUDE_MEMORY_DIR` opts in.
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
_ENTRY_KEYS = ("body_sha", "body_lines", "desc_sha", "index_sha")


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


def normalize_description(desc: str) -> str:
    """Strip YAML quoting so a re-serialized frontmatter is not a description change.

    Found 2026-09-05: the Edit tool's memory tooling rewrote an unquoted description as
    `"... \\"worth noting\\" ..."` on the first append. Hashed raw, that read as the author
    moving the description, which silenced the reminder on exactly the append that caused it.
    """

    d = desc.strip()
    if len(d) >= 2 and d[0] == d[-1] == '"':
        return d[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(d) >= 2 and d[0] == d[-1] == "'":
        return d[1:-1].replace("''", "'")
    return d


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


def snapshot(mem: Path) -> tuple[dict[str, dict], list[str]]:
    """The current state of every memory file, reduced to what drift needs, plus the names
    of any file that could not be read."""

    idx = index_lines(mem)
    out: dict[str, dict] = {}
    unreadable: list[str] = []
    for path in sorted(mem.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        try:
            desc, body = split_memory_file(path.read_text(errors="replace"))
        except OSError:
            unreadable.append(path.name)
            continue
        line = idx.get(path.name)
        out[path.name] = {
            "body_sha": _sha(body),
            "body_lines": _nonblank(body),
            "desc_sha": _sha(normalize_description(desc)),
            "index_sha": _sha(line) if line is not None else None,
        }
    return out, unreadable


def compare(
    baseline: dict[str, dict], current: dict[str, dict], min_growth: int = MIN_GROWTH_LINES
) -> tuple[list[tuple[str, int]], dict[str, dict]]:
    """Return (drifted, next_baseline).

    `drifted` lists (file, lines added since the summary last moved). `next_baseline` keeps the
    old entry for a drifted file, so the flag persists and the growth count accumulates until
    BOTH the description and the index line change. A file no longer on disk is dropped.
    """

    drifted: list[tuple[str, int]] = []
    nxt: dict[str, dict] = {}
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
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


def _entry_ok(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and all(k in entry for k in _ENTRY_KEYS)
        and isinstance(entry["body_sha"], str)
        and isinstance(entry["body_lines"], int)
        and isinstance(entry["desc_sha"], str)
        and (entry["index_sha"] is None or isinstance(entry["index_sha"], str))
    )


def load_baseline(mem: Path) -> tuple[dict[str, dict] | None, str]:
    """Return (baseline, state) where state is 'ok', 'missing', or 'corrupt'.

    Every entry is validated. A half-formed entry must read as corrupt (RESET), not crash the
    dashboard and not pass as an empty baseline that would silently re-baseline as clean.
    """

    path = mem / BASELINE_NAME
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
        files = data["files"]
        if not isinstance(files, dict) or not all(_entry_ok(v) for v in files.values()):
            raise TypeError("malformed baseline entry")
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

    implicit = mem is None
    mem = mem or memory_dir()
    if (
        implicit
        and write
        and os.environ.get("PYTEST_CURRENT_TEST")
        and not os.environ.get("CLAUDE_MEMORY_DIR")
    ):
        write = False  # a test that did not opt in must never touch the live baseline
    if not mem.is_dir() or not os.access(mem, os.R_OK):
        return [f"{PREFIX}UNREADABLE {mem}"]
    if not (mem / "MEMORY.md").exists():
        return [f"{PREFIX}MEMORY.md MISSING in {mem}; the index this check protects is gone"]
    current, unreadable = snapshot(mem)
    if unreadable:
        return [f"{PREFIX}UNREADABLE {len(unreadable)} file(s): " + " ".join(unreadable)]
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


def _hook_memory_dir(path: Path, mem: Path | None) -> Path | None:
    """The memory directory an edited file belongs to, or None when it is not a memory file.

    Derived from the EDITED FILE's own path when it sits under `~/.claude/projects/*/memory/`,
    so a session running in a git worktree (whose derived project name differs) still gets
    the reminder for the real memory files. Falls back to the given or derived directory.
    """

    parents = path.parents
    if (
        len(parents) >= 4
        and parents[0].name == "memory"
        and parents[2].name == "projects"
        and parents[3].name == ".claude"
    ):
        return parents[0]
    mem = mem or memory_dir()
    try:
        path.resolve().relative_to(mem.resolve())
    except ValueError:
        return None
    return mem


def hook_reminder(payload: dict, mem: Path | None = None) -> str:
    """The reminder for one Edit/Write payload, or '' when nothing needs saying.

    For Edit, growth and the description change are read from the payload itself, so a stale
    or absent baseline cannot confuse it. When a baseline exists it is also consulted, read-only,
    so the reminder stays quiet once BOTH halves of the summary have moved since the last
    hourly run, and names the half that has not.
    """

    tool = payload.get("tool_name", "")
    inp = payload.get("tool_input") or {}
    fp = inp.get("file_path", "")
    if tool not in ("Edit", "Write") or not isinstance(fp, str) or not fp:
        return ""
    path = Path(fp)
    if path.suffix != ".md" or path.name == "MEMORY.md":
        return ""
    mem_dir = _hook_memory_dir(path, mem)
    if mem_dir is None:
        return ""
    name = path.name
    idx = index_lines(mem_dir)
    indexed = name in idx
    growth = 0
    desc_moved = False
    if tool == "Edit":
        old = str(inp.get("old_string", ""))
        new = str(inp.get("new_string", ""))
        growth = _nonblank(new) - _nonblank(old)
        old_d, new_d = _DESC_RE.search(old), _DESC_RE.search(new)
        # A description line that merely appears as unchanged anchor context is not a touch.
        desc_moved = bool(
            old_d
            and new_d
            and normalize_description(old_d.group(1)) != normalize_description(new_d.group(1))
        )
    baseline, _ = load_baseline(mem_dir)
    base = (baseline or {}).get(name)
    index_moved = False
    if base is not None and path.exists():
        desc, body = split_memory_file(path.read_text(errors="replace"))
        if tool == "Write":
            growth = _nonblank(body) - base["body_lines"]
        pass
        idx_sha = _sha(idx[name]) if indexed else None
        index_moved = idx_sha != base["index_sha"] or (
            idx_sha is None and base["index_sha"] is None
        )
    parts: list[str] = []
    if growth >= MIN_GROWTH_LINES:
        if not desc_moved:
            parts.append(
                f"body grew by {growth} non-blank lines and its `description:` has not moved"
            )
        elif indexed and not index_moved:
            parts.append(
                f"body grew by {growth} non-blank lines; its `description:` moved but its "
                f"`MEMORY.md` line has not"
            )
    if not indexed and (tool == "Write" or growth >= MIN_GROWTH_LINES):
        parts.append("`MEMORY.md` has no line for it")
    if not parts:
        return ""
    return (
        f"memory_index_drift: `{name}` -- " + "; ".join(parts) + ". The `MEMORY.md` line is what "
        "a future session reads. Update `description:` and the `MEMORY.md` line for this file "
        "in this same edit series, or the hourly dashboard flags it as DRIFTED."
    )


def main(argv: list[str]) -> int:
    if "--hook" in argv:
        # Fail OPEN here, deliberately: a reminder must never break or block an edit. The
        # hourly dashboard is the fail-closed layer. Everything, argv parsing and the print
        # included, sits inside the guard.
        try:
            mem = Path(argv[argv.index("--memory-dir") + 1]) if "--memory-dir" in argv else None
            text = hook_reminder(json.load(sys.stdin), mem)
            if text:
                out = {"hookEventName": "PostToolUse", "additionalContext": text}
                print(json.dumps({"hookSpecificOutput": out}))
        except Exception:  # noqa: BLE001
            return 0
        return 0
    mem = Path(argv[argv.index("--memory-dir") + 1]) if "--memory-dir" in argv else None
    lines = memory_lines(mem, write="--dry-run" not in argv)
    print("\n".join(lines))
    bad = ("DRIFTED", "UNREADABLE", "MISSING")
    return 1 if any(b in lines[0] for b in bad) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
