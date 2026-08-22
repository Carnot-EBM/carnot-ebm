#!/usr/bin/env python3
"""Audit-findings ledger: flagged audit verdicts someone must answer.

WHY THIS EXISTS (REQ-OPS-AUDIT-LEDGER-1). On 2026-08-22 the claim audit
found two CLAIM_OVERSTATED verdicts and nobody decided anything. An audit
whose findings accumulate unread is the next silent-but-trusted layer — the
same shape as the QA-layer audit that wrote no report for four weeks while
its caller reported success. A report can be ignored; a ledger row with a
visible, growing age cannot stay ignored quietly.

MECHANISM. This tool parses flagged verdicts (the claim audit's own
FLAGGED_VERDICTS tuple, imported — one list, one home) out of
ops/experiment_claim_audit_report.md and maintains
ops/audit-findings-ledger.md:

  * A new flagged finding appends one row with disposition OPEN.
  * Existing rows are NEVER rewritten or removed (never-prune). A human
    closes a row by editing its Disposition cell to ACCEPTED, FIXED, or
    WONTFIX, plus an optional note.
  * An OPEN row older than AGING_DAYS escalates through the run sentinel's
    durable writer (REQ-CONDUCTOR-SENTINEL-3), and re-escalates on a weekly
    age bucket until the disposition changes. The question is forced; the
    ANSWER stays human — this tool cannot judge whether a verdict warrants
    a corrigendum.

FAIL DIRECTION: a missing report is a no-op (the audit may not have run
this cycle — its own receipt check covers that); a MALFORMED ledger row is
a finding, never a silent skip (a row this tool cannot read is a row whose
age it cannot track).

v1 scope, stated: the claim audit only. The other five milestone-close
audits keep their existing operator flow until this disposition shape
proves out on one.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "ops" / "experiment_claim_audit_report.md"
DEFAULT_LEDGER = REPO / "ops" / "audit-findings-ledger.md"
AUDIT_NAME = "experiment_claim_audit"
AGING_DAYS = 7

_LEDGER_HEADER = """# Audit findings ledger

Flagged audit verdicts someone must answer (REQ-OPS-AUDIT-LEDGER-1).
Rows are append-only: never rewrite or remove one. To close a finding,
edit its Disposition cell to ACCEPTED, FIXED, or WONTFIX and add a note.
OPEN rows older than 7 days escalate to ops/conductor-log.md weekly.

| First seen | Audit | Artifact | Verdict | Disposition | Note |
|---|---|---|---|---|---|
"""

_OPEN = "OPEN"
_CLOSED_DISPOSITIONS = ("ACCEPTED", "FIXED", "WONTFIX")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flagged_verdicts() -> tuple[str, ...]:
    """The claim audit's own FLAGGED_VERDICTS — imported so the two tools
    cannot drift apart on what counts as flagged."""
    audit = _load_module("experiment_claim_audit", REPO / "scripts" / "experiment_claim_audit.py")
    return tuple(audit.FLAGGED_VERDICTS)


def parse_report(text: str, flagged: tuple[str, ...]) -> list[dict]:
    """Flagged (artifact, verdict) pairs from a claim-audit report.

    An artifact section starts at `## <name>.json`; its verdict is the
    first bold `**VERDICT**` line after the header. The reviewer's raw
    text inside a section also contains `## VERDICT`-style headings, so
    only `.json`-suffixed headings delimit sections.
    """
    findings = []
    current: str | None = None
    for line in text.splitlines():
        header = re.fullmatch(r"##\s+(\S+\.json)\s*", line)
        if header:
            current = header.group(1)
            continue
        if current:
            bold = re.fullmatch(r"\*\*([A-Z_]+)\*\*\s*", line)
            if bold:
                if bold.group(1) in flagged:
                    findings.append({"artifact": current, "verdict": bold.group(1)})
                current = None  # first bold line decides; ignore the rest
    return findings


def parse_ledger(text: str) -> tuple[list[dict], list[str]]:
    """(entries, malformed lines). A data row has exactly 6 cells and an
    ISO first-seen date; header/separator rows are structural, not data."""
    entries: list[dict] = []
    malformed: list[str] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] in ("First seen", "---"):
            continue
        if all(set(c) <= {"-"} for c in cells):
            continue
        if len(cells) != 6 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
            malformed.append(line)
            continue
        entries.append(
            {
                "first_seen": cells[0],
                "audit": cells[1],
                "artifact": cells[2],
                "verdict": cells[3],
                "disposition": cells[4],
                "note": cells[5],
            }
        )
    return entries, malformed


def _identity(entry: dict) -> tuple[str, str, str]:
    return (entry["audit"], entry["artifact"], entry["verdict"])


def append_new_rows(
    ledger_path: Path, report_findings: list[dict], today: str, dry_run: bool = False
) -> int:
    """Append rows for findings the ledger has never seen. Append-only:
    an existing row is never touched, whatever its disposition."""
    if ledger_path.exists():
        entries, _ = parse_ledger(ledger_path.read_text(encoding="utf-8"))
        known = {_identity(e) for e in entries}
        prefix = ""
    else:
        known = set()
        prefix = _LEDGER_HEADER
    fresh = [f for f in report_findings if (AUDIT_NAME, f["artifact"], f["verdict"]) not in known]
    if dry_run or not fresh:
        return len(fresh)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        if prefix:
            fh.write(prefix)
        for finding in fresh:
            fh.write(
                f"| {today} | {AUDIT_NAME} | {finding['artifact']} | "
                f"{finding['verdict']} | {_OPEN} | |\n"
            )
    return len(fresh)


def aging_escalations(entries: list[dict], today: datetime) -> list[tuple[str, dict]]:
    """(scope, finding) pairs for OPEN rows past AGING_DAYS.

    The scope carries the age-week bucket, so the sentinel's dedupe lets
    the same finding re-escalate each week it stays OPEN — silence never
    resumes until a human writes a disposition.
    """
    escalations = []
    for entry in entries:
        if entry["disposition"] in _CLOSED_DISPOSITIONS:
            continue
        if entry["disposition"] != _OPEN:
            # Fail closed: a typo'd disposition ("FIXEDD") must not silence
            # the finding — only a recognized closed disposition does.
            escalations.append(
                (
                    f"{entry['artifact']} disposition",
                    {
                        "code": "LEDGER_DISPOSITION_UNRECOGNIZED",
                        "severity": "WARN",
                        "detail": (
                            f"disposition {entry['disposition']!r} on "
                            f"{entry['artifact']} is not OPEN/ACCEPTED/FIXED/"
                            "WONTFIX; treating the row as untriaged"
                        ),
                    },
                )
            )
            continue
        try:
            seen = datetime.strptime(entry["first_seen"], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue  # unparseable date rows surface via the malformed path
        age_days = (today - seen).days
        if age_days < AGING_DAYS:
            continue
        week_bucket = age_days // 7
        escalations.append(
            (
                f"{entry['artifact']} age-week {week_bucket}",
                {
                    "code": "AUDIT_FINDING_UNTRIAGED",
                    "severity": "WARN",
                    # Age leads: the conductor-log row truncates details at
                    # 80 chars and the age is the load-bearing number.
                    "detail": (
                        f"OPEN {age_days} days: {entry['verdict']} on "
                        f"{entry['artifact']} — triage in ops/audit-findings-ledger.md"
                    ),
                },
            )
        )
    return escalations


def run(
    *,
    report_path: Path = DEFAULT_REPORT,
    ledger_path: Path = DEFAULT_LEDGER,
    conductor_log: Path | None = None,
    known_issues: Path | None = None,
    state_path: Path | None = None,
    today: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    today = today or datetime.now(UTC)
    today_str = today.strftime("%Y-%m-%d")
    sentinel = _load_module(
        "conductor_run_sentinel", REPO / "scripts" / "conductor_run_sentinel.py"
    )
    conductor_log = conductor_log or REPO / "ops" / "conductor-log.md"
    known_issues = known_issues or REPO / "ops" / "known-issues.md"
    state_path = state_path or REPO / "ops" / ".run_sentinel_state.json"

    appended = 0
    if report_path.exists():
        findings = parse_report(report_path.read_text(encoding="utf-8"), flagged_verdicts())
        appended = append_new_rows(ledger_path, findings, today_str, dry_run=dry_run)
    # else: no report -> no-op ingest; the audit's own receipt check owns
    # "the audit did not run", this tool owns "the findings sat unanswered".

    escalations: list[tuple[str, dict]] = []
    malformed: list[str] = []
    if ledger_path.exists():
        entries, malformed = parse_ledger(ledger_path.read_text(encoding="utf-8"))
        escalations = aging_escalations(entries, today)
        for line in malformed:
            escalations.append(
                (
                    f"ledger row {hash(line) & 0xFFFFFF:06x}",
                    {
                        "code": "LEDGER_ROW_MALFORMED",
                        "severity": "WARN",
                        "detail": f"unparseable ledger row (age untrackable): {line[:60]}",
                    },
                )
            )
    summary = sentinel.escalate(
        escalations,
        conductor_log=conductor_log,
        known_issues=known_issues,
        state_path=state_path,
        dry_run=dry_run,
    )
    return {
        "appended": appended,
        "aging": len(escalations) - len(malformed),
        "malformed": len(malformed),
        "escalated": summary["written"],
        "deduplicated": summary["deduplicated"],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        report_path=Path(args.report), ledger_path=Path(args.ledger), dry_run=args.dry_run
    )
    print(
        f"[audit-ledger] appended={summary['appended']} aging={summary['aging']} "
        f"malformed={summary['malformed']} escalated={summary['escalated']} "
        f"deduplicated={summary['deduplicated']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
