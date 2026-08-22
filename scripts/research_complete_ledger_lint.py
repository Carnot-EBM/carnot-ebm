#!/usr/bin/env python3
"""Ledger-invariant lint for research-complete.yaml (REQ-CONDUCTOR-ARCHIVE-2).

This is the regression lock for the truthful-archival fix (commit
87f209cdcf). That fix made the conductor derive each archived task's
`result` from evidence and refuse duplicate milestone appends. This lint
refuses any commit that would show either behavior returning.

Why a lint on top of the fix: the fix is one revert away from regressing,
and research-complete.yaml is the planner's failure record. Measured on
2026-08-21, before the fix: 1,892 milestone entries for 51 distinct ids
(worst: 684 copies of 2026.07.510) and 57 task rows stamped
"OK (conductor)" whose deliverables never existed.

FORWARD-ONLY: only entries whose `completed` date is on or after
2026-08-22 are checked, so the uncorrected historical corpus never blocks
a commit. Repair of history is an operator decision (never-prune).
Run with --report-historical to count violations over ALL entries
(evidence mode; always exits 0).

Fail-closed choices, stated per the QA-Layer Authenticity Discipline:
  - an unparseable ledger FAILS the lint (a corrupted record is worse
    than a blocked commit);
  - an entry with no `completed` date IS checked (the archiver always
    writes the date; its absence marks an anomalous new entry).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    _LOADER = yaml.CSafeLoader  # ~10x faster on the 80k-line real ledger
except AttributeError:  # pragma: no cover - depends on libyaml presence
    _LOADER = yaml.SafeLoader

# Entries completed on/after this date are checked. Everything in the
# ledger on 2026-08-21 (the fix date) predates it, so the historical
# corpus passes untouched while every NEW append is fully checked.
CUTOFF = "2026-08-22"

# The retired hardcoded stamp. derive_task_result() emits only the derived
# vocabulary (OK, FLAGGED, OK_NO_DELIVERABLE, ...), so this exact literal
# can only reappear if the archiver regresses to the fixed string.
RETIRED_LITERAL = "OK (conductor)"


def _entry_is_checked(entry: dict) -> bool:
    """True when the entry falls under the forward-only window.

    Missing `completed` counts as checked (fail-closed): the archiver
    always writes the date, so absence marks an anomalous new entry.
    """
    completed = entry.get("completed")
    if completed is None or str(completed).strip() == "":
        return True
    return str(completed) >= CUTOFF


def _iter_task_violations(entry: dict, root: Path) -> list[str]:
    """Rule 2 (no retired literal) and rule 3 (OK names an existing deliverable)."""
    out: list[str] = []
    mid = str(entry.get("id"))
    for task in entry.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id", "?"))
        result = str(task.get("result", ""))
        deliverable = str(task.get("deliverable", "") or "").strip()
        if result == RETIRED_LITERAL:
            # Note deliverable absence too, so the historical report can
            # surface the 57 phantom rows the literal was stamping over.
            also = (
                "; deliverable also does not exist"
                if deliverable and not (root / deliverable).exists()
                else ""
            )
            out.append(
                f"{mid}/{tid}: result is the retired literal '{RETIRED_LITERAL}' — "
                "the archiver derives results from evidence now; this stamp "
                f"returning means the hardcoded literal regressed{also} "
                "(REQ-CONDUCTOR-ARCHIVE-2)"
            )
            continue
        if result.startswith("OK") and result != "OK_NO_DELIVERABLE":
            if deliverable and not (root / deliverable).exists():
                out.append(
                    f"{mid}/{tid}: result {result!r} but deliverable "
                    f"{deliverable!r} does not exist — an OK row must name a "
                    "deliverable that is on disk (REQ-CONDUCTOR-ARCHIVE-2)"
                )
    return out


def _collect(milestones: list[dict], root: Path, forward_only: bool) -> list[str]:
    checked = [m for m in milestones if isinstance(m, dict)]
    if forward_only:
        scope = [m for m in checked if _entry_is_checked(m)]
    else:
        scope = checked

    violations: list[str] = []

    # Rule 1: an id in scope must appear exactly once in the WHOLE file.
    all_ids: dict[str, int] = {}
    for m in checked:
        key = str(m.get("id"))
        all_ids[key] = all_ids.get(key, 0) + 1
    flagged_ids = set()
    for m in scope:
        key = str(m.get("id"))
        if all_ids.get(key, 0) > 1 and key not in flagged_ids:
            flagged_ids.add(key)
            violations.append(
                f"{key}: duplicate milestone entry ({all_ids[key]} copies) — the "
                "archiver refuses duplicate appends; a second entry means the "
                "retry-append regressed (REQ-CONDUCTOR-ARCHIVE-2)"
            )

    for m in scope:
        violations.extend(_iter_task_violations(m, root))
    return violations


def check_ledger(ledger_path: Path) -> list[str]:
    """Forward-only violations for the given ledger file.

    Deliverable paths resolve against the ledger's own directory, so the
    lint works identically on the real repo root and on a test tmp_path.
    Raises on an unparseable file — main() maps that to a failure.
    """
    data = yaml.load(ledger_path.read_text(), Loader=_LOADER) or {}
    milestones = data.get("milestones") or []
    return _collect(milestones, ledger_path.resolve().parent, forward_only=True)


def report_historical(ledger_path: Path) -> str:
    """Evidence mode: count violations over ALL entries, ignore the cutoff."""
    data = yaml.load(ledger_path.read_text(), Loader=_LOADER) or {}
    milestones = [m for m in (data.get("milestones") or []) if isinstance(m, dict)]
    violations = _collect(milestones, ledger_path.resolve().parent, forward_only=False)
    dup = sum(1 for v in violations if "duplicate milestone entry" in v)
    literal = sum(1 for v in violations if f"retired literal '{RETIRED_LITERAL}'" in v)
    missing = sum(1 for v in violations if "does not exist" in v)
    lines = [
        f"research-complete.yaml historical report ({len(milestones)} entries):",
        f"  duplicate milestone ids: {dup}",
        f"  rows with the retired literal '{RETIRED_LITERAL}': {literal}",
        f"  OK rows whose deliverable does not exist: {missing}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["research-complete.yaml"],
        help="ledger path(s); pre-commit passes the staged filename",
    )
    parser.add_argument(
        "--report-historical",
        action="store_true",
        help="count violations over ALL entries (ignores the cutoff); exits 0",
    )
    args = parser.parse_args(argv)

    rc = 0
    for raw in args.paths:
        path = Path(raw)
        if args.report_historical:
            try:
                print(report_historical(path))
            except Exception as exc:  # fail-closed even in report mode
                print(f"LEDGER LINT: cannot read {path}: {exc}", file=sys.stderr)
                return 1
            continue
        try:
            violations = check_ledger(path)
        except Exception as exc:
            # Fail-closed: an unreadable/unparseable ledger blocks the
            # commit. A guard that answers "clean" when it could not look
            # is the trusted-and-silent failure mode.
            print(f"LEDGER LINT: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        for v in violations:
            print(f"LEDGER LINT VIOLATION: {v}", file=sys.stderr)
        if violations:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
