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

THE LOAD ENVELOPE (REQ-INFRA-6976, 2026-09-05). The harness loads `MEMORY.md` through one
function: trim, keep at most 200 lines, cut at the last newline at or before UTF-16 unit 25,000,
append a `> WARNING` line. Whole lines fall off the tail. Read from the installed binary and
confirmed with two synthetic indexes through the real harness (135 of 180 lines loaded at 33,119
units; 200 of 220 at the line cap). The live index crossed the unit cap on 2026-09-05 and its
newest entry was invisible to every new session. A flat index therefore has a ceiling.

THE TWO TIERS. `MEMORY.md` is tier 1, the loaded surface. `_index_<group>.md` is tier 2, read on
demand, same pointer-line form, no frontmatter. `<group>` is a memory file's name prefix.
`TIER2_GROUPS` names the groups that live in tier 2 (`reference` first: paper pointers are what a
session needs least at startup). `--demote` moves a pointer line verbatim from tier 1 to tier 2
and keeps a group line in tier 1 that says where it went. Nothing is deleted. `index_capacity`
replicates the harness cut exactly, so the check names the invisible entries before the harness
drops them, and the hook says so to the editor of an index file at the moment of the write.
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
INDEX_FILE = "MEMORY.md"

# The harness caps, read from the installed `claude` binary (2.1.261; same in 2.1.247/2.1.251)
# and confirmed end to end. Units are UTF-16 code units, the JavaScript string length, not bytes.
HARNESS_MAX_UNITS = 25000
HARNESS_MAX_LINES = 200
# The project budget: the harness cap minus about six entries, so the check fires first.
BUDGET_UNITS = 24000
BUDGET_LINES = 190
# Name-prefix groups whose pointers live in a tier-2 index file, not in MEMORY.md.
TIER2_GROUPS = ("reference",)
# Demotion candidates are named in this group order, oldest file first inside a group.
# Operator-confirmed 2026-09-05 (relayed by the team lead); not provisional. Do not re-litigate.
_DEMOTE_ORDER = ("reference", "project", "incident", "feedback", "user")
_DEMOTE_CMD = "python3 scripts/memory_index_drift.py --demote"

_DESC_RE = re.compile(r"^description:\s*(.*)$", re.M)
_INDEX_RE = re.compile(r"^- \[[^\]]*\]\(([^)]+)\)")
_TIER2_RE = re.compile(r"^_index_([A-Za-z0-9]+)\.md$")
_GROUP_COUNT_RE = re.compile(r"\((\d+) entr(?:y|ies)\)")
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


def is_index_file(name: str) -> bool:
    """`MEMORY.md` or a tier-2 `_index_<group>.md`. Neither is a memory file."""

    return name == INDEX_FILE or _TIER2_RE.match(name) is not None


def group_of(name: str) -> str:
    """The name prefix a memory file belongs to: `feedback_x.md` -> `feedback`."""

    stem = name[:-3] if name.endswith(".md") else name
    head, sep, _ = stem.partition("_")
    return head.lower() if sep and head else "misc"


def index_files(mem: Path) -> list[Path]:
    """`MEMORY.md` first, then every tier-2 file, sorted. Only files that exist."""

    out = [mem / INDEX_FILE] if (mem / INDEX_FILE).exists() else []
    out.extend(sorted(p for p in mem.glob("_index_*.md") if _TIER2_RE.match(p.name)))
    return out


def index_entries(mem: Path) -> dict[str, list[tuple[str, str]]]:
    """Index file name -> [(target file name, full line)] for every pointer line in it.

    Group lines (a pointer whose target is itself an index file) are included, so a caller can
    see them; `index_lines` drops them because they index no memory.
    """

    out: dict[str, list[tuple[str, str]]] = {}
    for path in index_files(mem):
        rows: list[tuple[str, str]] = []
        for line in path.read_text(errors="replace").splitlines():
            m = _INDEX_RE.match(line)
            if m:
                rows.append((Path(m.group(1)).name, line))
        out[path.name] = rows
    return out


def index_lines(mem: Path) -> dict[str, str]:
    """File name -> its full index line, from `MEMORY.md` AND every tier-2 file.

    Any edit to the line counts as touching it. A demoted file keeps its line and its hash, so a
    demotion is not a summary move. When a target appears twice the first index file wins here;
    `index_capacity` reports the DUPLICATE.
    """

    out: dict[str, str] = {}
    for rows in index_entries(mem).values():
        for name, line in rows:
            if not is_index_file(name) and name not in out:
                out[name] = line
    return out


def index_locations(mem: Path) -> dict[str, str]:
    """File name -> the index file that holds its line. A demoted file maps to its tier-2 file.

    The hook uses this to name the RIGHT file to update. Telling the editor of a demoted file to
    fix its `MEMORY.md` line, when that line lives in `_index_reference.md`, is worse than saying
    nothing.
    """

    out: dict[str, str] = {}
    for fname, rows in index_entries(mem).items():
        for name, _ in rows:
            if not is_index_file(name) and name not in out:
                out[name] = fname
    return out


def expected_index_file(name: str) -> str:
    """Where a memory file's line belongs: its tier-2 file when its group is tier 2."""

    group = group_of(name)
    return f"_index_{group}.md" if group in TIER2_GROUPS else INDEX_FILE


def resolved_index_file(mem: Path, name: str) -> str:
    """Return the file holding a pointer, or its policy destination when absent."""

    actual = index_locations(mem).get(name)
    if actual is not None:
        return actual
    return expected_index_file(name)


def utf16_units(text: str) -> int:
    """The harness measures `.length` of a JavaScript string: UTF-16 code units."""

    return len(text.encode("utf-16-le")) // 2


def harness_cut(text: str) -> tuple[list[str], list[str], bool]:
    """Replicate the harness load of an index: (kept lines, dropped lines, first line cut).

    Trim; keep at most HARNESS_MAX_LINES lines; if the result is longer than HARNESS_MAX_UNITS
    units, cut at the last newline at or before unit HARNESS_MAX_UNITS. When no newline sits
    there the harness cuts mid-line at the cap; that case is reported as `first_line_cut`.
    """

    lines = text.strip().split("\n")
    dropped = lines[HARNESS_MAX_LINES:]
    kept = lines[:HARNESS_MAX_LINES]
    joined = "\n".join(kept)
    if utf16_units(joined) <= HARNESS_MAX_UNITS:
        return kept, dropped, False
    pos = -1
    fit = 0
    for i, line in enumerate(kept):
        pos += utf16_units(line) + 1  # the index of the newline that follows this line
        if i == len(kept) - 1:
            break
        if pos <= HARNESS_MAX_UNITS:
            fit = i + 1
    if fit == 0:
        return [], kept + dropped, True
    return kept[:fit], kept[fit:] + dropped, False


def index_capacity(mem: Path) -> dict:
    """Everything the envelope check needs about `MEMORY.md` and the tier-2 files."""

    path = mem / INDEX_FILE
    text = path.read_text(errors="replace") if path.exists() else ""
    kept, dropped, first_cut = harness_cut(text)
    entries = index_entries(mem)
    tier1 = [(n, ln) for n, ln in entries.get(INDEX_FILE, []) if not is_index_file(n)]
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    missing: list[str] = []
    for fname, rows in entries.items():
        for name, _ in rows:
            if is_index_file(name):
                continue
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.setdefault(name, fname)
            if not (mem / name).exists() and name not in missing:
                missing.append(name)
    tier2 = {f: sum(1 for n, _ in rows if not is_index_file(n)) for f, rows in entries.items()}
    tier2.pop(INDEX_FILE, None)
    stale: list[tuple[str, int, int]] = []
    for name, line in entries.get(INDEX_FILE, []):
        if not _TIER2_RE.match(name):
            continue
        m = _GROUP_COUNT_RE.search(line)
        actual = tier2.get(name)
        if m and actual is not None and int(m.group(1)) != actual:
            stale.append((name, int(m.group(1)), actual))

    def _mtime(name: str) -> float:
        try:
            return (mem / name).stat().st_mtime
        except OSError:
            return 0.0

    order = {g: i for i, g in enumerate(_DEMOTE_ORDER)}
    candidates = sorted(
        (n for n, _ in tier1),
        key=lambda n: (order.get(group_of(n), len(order)), _mtime(n), n),
    )
    invisible = [Path(m.group(1)).name for ln in dropped if (m := _INDEX_RE.match(ln))]
    units = utf16_units(text.strip())
    lines = text.strip().count("\n") + 1 if text.strip() else 0
    return {
        "units": units,
        "lines": lines,
        "invisible": invisible,
        "first_line_cut": first_cut,
        "over_budget": units > BUDGET_UNITS or lines > BUDGET_LINES,
        "tier1_entries": len(tier1),
        "tier2": tier2,
        "misplaced": [n for n, _ in tier1 if group_of(n) in TIER2_GROUPS],
        "duplicates": duplicates,
        "missing_targets": missing,
        "stale_group_counts": stale,
        "reachable": len(seen),
        "candidates": candidates[:5],
    }


def capacity_problems(cap: dict) -> list[str]:
    """The bad states, worst first, each with its fix where one applies."""

    out: list[str] = []
    fix = f"fix: {_DEMOTE_CMD} " + " ".join(cap["candidates"]) if cap["candidates"] else ""
    if cap["invisible"] or cap["first_line_cut"]:
        names = ("cut mid-line: " if cap["first_line_cut"] else "") + " ".join(cap["invisible"])
        out.append(
            f"OVER_CAP {len(cap['invisible'])} past the harness cut: {names}; "
            f"{cap['units']:,}/{HARNESS_MAX_UNITS:,} units, {cap['lines']}/{HARNESS_MAX_LINES} "
            f"lines; {fix}"
        )
    elif cap["over_budget"]:
        out.append(
            f"OVER_BUDGET {cap['units']:,}/{BUDGET_UNITS:,} units, {cap['lines']}/{BUDGET_LINES} "
            f"lines (harness cap {HARNESS_MAX_UNITS:,}/{HARNESS_MAX_LINES}); {fix}"
        )
    if cap["misplaced"]:
        out.append(
            f"MISPLACED tier-2 group in {INDEX_FILE}: {' '.join(cap['misplaced'])}; "
            f"fix: {_DEMOTE_CMD} {' '.join(cap['misplaced'])}"
        )
    if cap["duplicates"]:
        out.append(f"DUPLICATE indexed twice: {' '.join(cap['duplicates'])}")
    if cap["missing_targets"]:
        out.append(f"MISSING_TARGET no such file: {' '.join(cap['missing_targets'])}")
    for name, declared, actual in cap["stale_group_counts"]:
        out.append(f"GROUP_COUNT_STALE {name} says {declared} entries, holds {actual}")
    return out


def capacity_lines(mem: Path) -> list[str]:
    """Dashboard lines for the envelope. Always one line; bad states carry their token."""

    cap = index_capacity(mem)
    problems = capacity_problems(cap)
    if problems:
        return [f"{PREFIX}index {p}" for p in problems]
    tier2 = "; ".join(f"tier-2 {f} {n}" for f, n in sorted(cap["tier2"].items()))
    return [
        f"{PREFIX}index {INDEX_FILE} {cap['units']:,}/{HARNESS_MAX_UNITS:,} units, "
        f"{cap['lines']}/{HARNESS_MAX_LINES} lines (budget {BUDGET_UNITS:,}/{BUDGET_LINES}); "
        + (tier2 + "; " if tier2 else "")
        + f"{cap['reachable']} pointers reachable"
    ]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _group_line(group: str, count: int) -> str:
    return (
        f"- [Index: {group} ({count} entries)](_index_{group}.md) \u2014 tier-2 pointers for "
        f"`{group}_*` memories, not loaded at startup; open it when a {group} fact you expect "
        f"is not in this list"
    )


def demote(mem: Path, names: list[str]) -> list[str]:
    """Move each named file's pointer line, verbatim, from `MEMORY.md` to `_index_<group>.md`.

    Creates the tier-2 file when absent, inserts or re-counts the group line in `MEMORY.md`, and
    keeps the group lines together at the top. Never deletes a line. Returns one message per
    name. Both files are written atomically, tier-2 first, so a crash between the two writes
    leaves a DUPLICATE (reported), never a lost pointer.
    """

    idx = mem / INDEX_FILE
    text = idx.read_text(errors="replace")
    trailing = text.endswith("\n")
    lines = text.rstrip("\n").split("\n") if text.strip() else []
    msgs: list[str] = []
    moves: dict[str, list[str]] = {}
    for name in names:
        if is_index_file(name):
            msgs.append(f"{name}: refused, it is an index file")
            continue
        pos = next(
            (
                i
                for i, ln in enumerate(lines)
                if (m := _INDEX_RE.match(ln)) and Path(m.group(1)).name == name
            ),
            None,
        )
        if pos is None:
            where = next(
                (f for f, rows in index_entries(mem).items() if any(n == name for n, _ in rows)),
                None,
            )
            msgs.append(
                f"{name}: no {INDEX_FILE} line"
                + (f", already in {where}" if where else " anywhere")
            )
            continue
        moves.setdefault(group_of(name), []).append(lines.pop(pos))
        msgs.append(f"{name}: demoted to _index_{group_of(name)}.md")
    _place_in_tier2(mem, lines, moves)
    if moves:
        _atomic_write(idx, "\n".join(lines) + ("\n" if trailing or lines else ""))
    return msgs


def _place_in_tier2(mem: Path, lines: list[str], moves: dict[str, list[str]]) -> None:
    """Append each group's lines to its tier-2 file and keep the tier-1 group line current.

    `lines` is the caller's in-memory `MEMORY.md`; the caller writes it. Shared by `demote`
    (a line moving out of tier 1) and `adopt` (a line that never had a home).
    """

    for group, moved in moves.items():
        gpath = mem / f"_index_{group}.md"
        existing = (
            gpath.read_text(errors="replace")
            if gpath.exists()
            else (
                f"# Index: {group}\n\nTier-2 pointers for `{group}_*` memories. Same line form as "
                f"{INDEX_FILE}; not loaded at startup, read on demand.\n\n"
            )
        )
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_text = existing + "\n".join(moved) + "\n"
        count = sum(
            1
            for ln in new_text.splitlines()
            if (m := _INDEX_RE.match(ln)) and not is_index_file(Path(m.group(1)).name)
        )
        _atomic_write(gpath, new_text)
        gpos = next(
            (
                i
                for i, ln in enumerate(lines)
                if (m := _INDEX_RE.match(ln)) and Path(m.group(1)).name == gpath.name
            ),
            None,
        )
        if gpos is None:
            last_group = max(
                (
                    i
                    for i, ln in enumerate(lines)
                    if (m := _INDEX_RE.match(ln)) and _TIER2_RE.match(Path(m.group(1)).name)
                ),
                default=-1,
            )
            lines.insert(last_group + 1, _group_line(group, count))
        else:
            lines[gpos] = _GROUP_COUNT_RE.sub(f"({count} entries)", lines[gpos], count=1)


_NAME_RE = re.compile(r"^name:\s*(.*)$", re.M)


def pointer_line_from_file(mem: Path, name: str) -> str:
    """Build `- [Title](name) — hook` from the file's own frontmatter.

    Title is the `name:` field with `_`/`-` turned into spaces (a name that is already a
    sentence is kept as it is); the hook is the unquoted `description:`. Both are the file
    author's summary, which is the half of the record REQ-INFRA-6975 holds the index line to.
    """

    text = (mem / name).read_text(errors="replace")
    desc, _ = split_memory_file(text)
    head = ""
    fm = text.splitlines()
    if fm and fm[0].strip() == "---":
        for i in range(1, len(fm)):
            if fm[i].strip() == "---":
                head = "\n".join(fm[1:i])
                break
    m = _NAME_RE.search(head)
    title = normalize_description(m.group(1)) if m else name[:-3]
    if " " not in title:
        title = title.replace("_", " ").replace("-", " ")
    hook = normalize_description(desc) or "(no description)"
    return f"- [{title}]({name}) \u2014 {hook}"


def adopt(mem: Path, names: list[str]) -> list[str]:
    """Give an unindexed memory file a pointer line in its tier-2 `_index_<group>.md`.

    Never writes into `MEMORY.md` beyond the group line: the loaded surface is budgeted, and a
    person promotes a line to tier 1 by hand. Refuses a name that already has a line anywhere
    (that would be a DUPLICATE), an index file, or a file that does not exist. One message per
    name. Idempotent: a second run reports and changes nothing.
    """

    idx = mem / INDEX_FILE
    text = idx.read_text(errors="replace") if idx.exists() else ""
    trailing = text.endswith("\n")
    lines = text.rstrip("\n").split("\n") if text.strip() else []
    where = index_locations(mem)
    msgs: list[str] = []
    moves: dict[str, list[str]] = {}
    for name in names:
        if is_index_file(name):
            msgs.append(f"{name}: refused, it is an index file")
            continue
        if not (mem / name).is_file():
            msgs.append(f"{name}: refused, no such memory file")
            continue
        if name in where:
            msgs.append(f"{name}: already indexed in {where[name]}")
            continue
        moves.setdefault(group_of(name), []).append(pointer_line_from_file(mem, name))
        where[name] = f"_index_{group_of(name)}.md"
        msgs.append(f"{name}: indexed in _index_{group_of(name)}.md")
    _place_in_tier2(mem, lines, moves)
    if moves:
        _atomic_write(idx, "\n".join(lines) + ("\n" if trailing or lines else ""))
    return msgs


def snapshot(mem: Path) -> tuple[dict[str, dict], list[str]]:
    """The current state of every memory file, reduced to what drift needs, plus the names
    of any file that could not be read."""

    idx = index_lines(mem)
    out: dict[str, dict] = {}
    unreadable: list[str] = []
    for path in sorted(mem.glob("*.md")):
        if is_index_file(path.name):
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
        return [f"{PREFIX}{len(current)} files, 0 drifted", *capacity_lines(mem)]
    items = "  ".join(f"{n}(+{g})" for n, g in sorted(drifted, key=lambda t: -t[1]))
    return [
        f"{PREFIX}{len(current)} files, {len(drifted)} DRIFTED (body grew, summary untouched): "
        f"{items}",
        *capacity_lines(mem),
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
    if path.suffix != ".md":
        return ""
    mem_dir = _hook_memory_dir(path, mem)
    if mem_dir is None:
        return ""
    name = path.name
    where = resolved_index_file(mem_dir, name)
    if is_index_file(name):
        # REQ-INFRA-6976: the file as written is on disk (PostToolUse), so measure it.
        problems = [
            p
            for p in capacity_problems(index_capacity(mem_dir))
            if p.split(" ", 1)[0] in ("OVER_CAP", "OVER_BUDGET", "MISPLACED", "DUPLICATE")
        ]
        if not problems:
            return ""
        return (
            f"memory_index_drift: `{name}` -- " + "; ".join(problems) + ". The harness loads at "
            f"most {HARNESS_MAX_LINES} lines and {HARNESS_MAX_UNITS:,} UTF-16 units of "
            f"`{INDEX_FILE}` and drops whole lines from the tail."
        )
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
        desc_moved = desc_moved or _sha(normalize_description(desc)) != base["desc_sha"]
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
                f"`{where}` line has not"
            )
    if not indexed and (tool == "Write" or growth >= MIN_GROWTH_LINES):
        parts.append(f"`{where}` has no line for it")
    if not parts:
        return ""
    return (
        f"memory_index_drift: `{name}` -- " + "; ".join(parts) + f". The `{where}` line is what "
        f"a future session reads. Update `description:` and the `{where}` line for this file "
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
    if "--demote" in argv:
        names = [a for a in argv[argv.index("--demote") + 1 :] if not a.startswith("--")]
        for msg in demote(mem or memory_dir(), names):
            print(msg)
        return 0
    if "--adopt" in argv:
        names = [a for a in argv[argv.index("--adopt") + 1 :] if not a.startswith("--")]
        for msg in adopt(mem or memory_dir(), names):
            print(msg)
        return 0
    lines = memory_lines(mem, write="--dry-run" not in argv)
    print("\n".join(lines))
    bad = (
        "DRIFTED",
        "UNREADABLE",
        "MISSING",
        "OVER_CAP",
        "OVER_BUDGET",
        "MISPLACED",
        "DUPLICATE",
        "MISSING_TARGET",
        "GROUP_COUNT_STALE",
    )
    return 1 if any(b in line for line in lines for b in bad) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
