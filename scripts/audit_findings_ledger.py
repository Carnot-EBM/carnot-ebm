#!/usr/bin/env python3
"""Audit-findings ledger: flagged audit verdicts someone must answer.

WHY THIS EXISTS (REQ-OPS-AUDIT-LEDGER-1). On 2026-08-22 the claim audit
found two CLAIM_OVERSTATED verdicts and nobody decided anything. An audit
whose findings accumulate unread is the next silent-but-trusted layer — the
same shape as the QA-layer audit that wrote no report for four weeks while
its caller reported success. A report can be ignored; a ledger row with a
visible, growing age cannot stay ignored quietly.

MECHANISM. This tool parses flagged verdicts out of each audit report in
SOURCES (each audit's own FLAGGED_VERDICTS constant, imported — one list,
one home) and maintains ops/audit-findings-ledger.md:

  * A new flagged finding appends one row with disposition OPEN.
  * Existing rows are NEVER rewritten or removed (never-prune). A human
    closes a row by editing its Disposition cell to ACCEPTED, FIXED, or
    WONTFIX, plus an optional note.
  * An OPEN row older than AGING_DAYS (1 day since the 2026-08-23 spec
    amendment -- the loop closes several milestones per day, so a week of
    silence was structurally slow) escalates through the run sentinel's
    durable writer (REQ-CONDUCTOR-SENTINEL-3), and re-escalates on a weekly
    age bucket until the disposition changes. The question is forced; the
    ANSWER stays human — this tool cannot judge whether a verdict warrants
    a corrigendum.

FAIL DIRECTION: a missing report is a no-op (the audit may not have run
this cycle — its own receipt check covers that); a MALFORMED ledger row is
a finding, never a silent skip (a row this tool cannot read is a row whose
age it cannot track).

SCOPE, and the 2026-08-25 widening. v1 read ONE report, the claim audit's.
That was a pattern narrower than this module's own stated concept
("flagged audit verdicts someone must answer") — the exact class the
QA-layer discipline hunts, committed by the guard written to fix it. The
irony is precise: the docstring above cites the QA-layer audit's silent
failure as the reason this tool exists, and the tool did not read the
QA-layer audit. Measured cost: two milestone closes (.572, .573) produced
7 SILENT_NON_FIRING verdicts and ingested none. All 7 were hand-entered by
the outer loop. The report is REGENERATED at every close, so an
un-ingested finding is not merely unread — it is overwritten.

SOURCES now carries the claim audit plus the QA-layer audit. Two
milestone-close audits stay out, and the reason is mechanical rather than
editorial: neither exports a module-level flagged-verdict constant this
tool could import, and re-declaring a copy here is the drift this module
refuses on principle.

  * verifier_authenticity_audit.py — its flagged set is a local variable
    inside the run function (`flagged_verdicts = {...}`), unreachable.
  * artifact_convention_audit.py — flagged is a positional slice of
    VERDICTS, so there is no name to import.

Promoting either to a module constant makes it a one-line addition here.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "ops" / "experiment_claim_audit_report.md"
DEFAULT_QA_REPORT = REPO / "ops" / "qa_layer_authenticity_audit_report.md"
DEFAULT_LEDGER = REPO / "ops" / "audit-findings-ledger.md"
AUDIT_NAME = "experiment_claim_audit"
QA_AUDIT_NAME = "qa_layer_authenticity_audit"
# First escalation age. Was 7 days; lowered to 1 on 2026-08-23 (spec
# AMENDMENT to REQ-OPS-AUDIT-LEDGER-1 rule 3): the loop closes several
# milestones per day, and the three real 2026-08-22 CLAIM_OVERSTATED
# findings sat OPEN until an operator prompt. Weekly re-bucket unchanged.
AGING_DAYS = 1

_LEDGER_HEADER = """# Audit findings ledger

Flagged audit verdicts someone must answer (REQ-OPS-AUDIT-LEDGER-1).
Rows are append-only: never rewrite or remove one. To close a finding,
edit its Disposition cell to ACCEPTED, FIXED, or WONTFIX and add a note.
OPEN rows older than 1 day escalate to ops/conductor-log.md, then weekly.

| First seen | Audit | Artifact | Verdict | Disposition | Note |
|---|---|---|---|---|---|
"""

_OPEN = "OPEN"
_CLOSED_DISPOSITIONS = ("ACCEPTED", "FIXED", "WONTFIX")


def _load_module(name: str, path: Path):
    """Load an audit module by path so its own constants can be imported.

    The sys.modules entry is REQUIRED, not tidiness: `@dataclass` resolves
    its annotations through `sys.modules[cls.__module__]`, so a module
    loaded without one raises AttributeError at class-creation time.
    `qa_layer_authenticity_audit.py` has one, and this is what stopped its
    FLAGGED_VERDICTS being importable. A failed load restores whatever was
    registered before, so a broken audit cannot leave a half-built module
    behind for the next caller.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def flagged_verdicts() -> tuple[str, ...]:
    """The claim audit's own FLAGGED_VERDICTS — imported so the two tools
    cannot drift apart on what counts as flagged."""
    audit = _load_module("experiment_claim_audit", REPO / "scripts" / "experiment_claim_audit.py")
    return tuple(audit.FLAGGED_VERDICTS)


def source_flagged_verdicts(source: Source) -> tuple[str, ...]:
    """One audit's flagged set, read off that audit's own module.

    Imported, never copied. A second copy here is the drift that makes a
    ledger silently stop ingesting a verdict the audit still emits.
    """
    audit = _load_module(source.module, REPO / "scripts" / f"{source.module}.py")
    return tuple(sorted(getattr(audit, source.verdict_attr)))


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


def parse_qa_report(text: str, flagged: tuple[str, ...]) -> list[dict]:
    """Flagged (unit, verdict) pairs from a QA-layer audit report.

    Two independent readings of the same fact, unioned and deduped, so a
    format change on either side degrades to partial ingest rather than
    silent zero:

      * the per-unit section — a `## <unit>` heading whose next non-blank
        line is the writer's `**Verdict:** \\`X\\`` line;
      * the `### FLAGGED` summary list — `- \\`unit\\` — **VERDICT**`.

    Only `##` headings followed by that exact verdict line count as a
    unit. The reviewer's own prose carries `## VERDICT` / `## FINDINGS`
    headings, and they are followed by a bare token, so they never match.

    A verdict the Layer-1.5 integrity guard voided is already rewritten to
    CANNOT_DETERMINE before the report is written, so a hallucinated
    finding cannot reach the ledger.
    """
    findings: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(unit: str, verdict: str) -> None:
        if verdict in flagged and (unit, verdict) not in seen:
            seen.add((unit, verdict))
            findings.append({"artifact": unit, "verdict": verdict})

    lines = text.splitlines()
    in_flagged = False
    for index, line in enumerate(lines):
        # The FLAGGED list is read ONLY inside its own section. Applied to the
        # whole document the item pattern also matches the reviewer's prose --
        # a reviewer comparing guards writes exactly this shape inside
        # `## FINDINGS` -- and an append-only ledger cannot un-write a phantom
        # row: it escalates weekly forever.
        if line.startswith("###"):
            in_flagged = "FLAGGED" in line
        elif line.startswith("## "):
            in_flagged = False
        if in_flagged:
            item = re.fullmatch(r"-\s+`([^`]+)`.*?\*\*([A-Z_]+)\*\*\s*", line)
            if item:
                add(item.group(1), item.group(2))
        heading = re.fullmatch(r"##\s+(\S.*?)\s*", line)
        if heading:
            for following in lines[index + 1 : index + 4]:
                if not following.strip():
                    continue
                verdict_line = re.fullmatch(r"\*\*Verdict:\*\*\s*`([A-Z_]+)`\s*", following)
                if verdict_line:
                    add(heading.group(1), verdict_line.group(1))
                break
    return findings


class Source(NamedTuple):
    """One audit this ledger ingests.

    `name` is the ledger's Audit cell and is part of a row's identity, so
    two audits flagging the same file name are distinct rows and neither
    dedups the other away.

    NamedTuple, not a dataclass: this module is loaded by
    `spec_from_file_location` without a `sys.modules` entry (its own
    `_load_module`, and every test), and `@dataclass` needs that entry.
    """

    name: str
    default_report: Path
    module: str
    verdict_attr: str
    parser: Callable[[str, tuple[str, ...]], list[dict]]


SOURCES: tuple[Source, ...] = (
    # Claim audit stays first so a combined run appends its rows in the
    # same order as before the widening.
    Source(AUDIT_NAME, DEFAULT_REPORT, "experiment_claim_audit", "FLAGGED_VERDICTS", parse_report),
    Source(
        QA_AUDIT_NAME,
        DEFAULT_QA_REPORT,
        "qa_layer_authenticity_audit",
        "FLAGGED_VERDICTS",
        parse_qa_report,
    ),
)


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


def _normalize_artifact(name: str) -> str:
    """Compare artifacts by basename.

    A hand-entered row spells a guard `scripts/child_results_guard.py`; the
    QA-layer report writes the bare `child_results_guard.py`. Two spellings of
    one finding produced two rows on 2026-08-25, and an append-only ledger
    cannot un-write them -- both escalate weekly until a human dispositions
    each. Identity is the finding, not the spelling.
    """
    return name.strip().rsplit("/", 1)[-1]


def _identity(entry: dict) -> tuple[str, str, str]:
    return (entry["audit"], _normalize_artifact(entry["artifact"]), entry["verdict"])


def append_new_rows(
    ledger_path: Path,
    report_findings: list[dict],
    today: str,
    dry_run: bool = False,
    audit_name: str = AUDIT_NAME,
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
    fresh = [
        f
        for f in report_findings
        if (audit_name, _normalize_artifact(f["artifact"]), f["verdict"]) not in known
    ]
    if dry_run or not fresh:
        return len(fresh)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        if prefix:
            fh.write(prefix)
        for finding in fresh:
            fh.write(
                f"| {today} | {audit_name} | {finding['artifact']} | "
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
    qa_report_path: Path = DEFAULT_QA_REPORT,
    report_paths: dict[str, Path] | None = None,
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
    # Own state file, NOT the sentinel's: the state file doubles as this
    # tool's receipt (REQ-CONDUCTOR-RECEIPT-1), and a shared file would let
    # a fresh sentinel scan mask a ledger run that crashed before writing.
    state_path = state_path or REPO / "ops" / ".audit_findings_ledger_state.json"

    if report_paths is None:
        # Production shape: every source reads its own report.
        named = {AUDIT_NAME: report_path, QA_AUDIT_NAME: qa_report_path}
        selected = [(s, named.get(s.name, s.default_report)) for s in SOURCES]
    else:
        # An explicit map is the COMPLETE source list, so a source added
        # later cannot silently leak a real report into an isolated run.
        # Isolating one source and inheriting the rest is how a test starts
        # reading tracked state without anyone noticing.
        unknown = set(report_paths) - {s.name for s in SOURCES}
        if unknown:
            # A typo, or a stale literal after a rename, would otherwise
            # select ZERO sources, append nothing, and return success.
            raise KeyError(f"unknown audit source(s): {sorted(unknown)}")
        selected = [(s, report_paths[s.name]) for s in SOURCES if s.name in report_paths]
    appended = 0
    for source, path in selected:
        if not path.exists():
            # A missing report is a no-op, per source. The audit's own
            # receipt check owns "the audit did not run"; this tool owns
            # "its findings sat unanswered". An audit that did not run must
            # not make the ledger look clean, and an absent report appends
            # nothing rather than closing anything.
            continue
        findings = source.parser(path.read_text(encoding="utf-8"), source_flagged_verdicts(source))
        appended += append_new_rows(
            ledger_path, findings, today_str, dry_run=dry_run, audit_name=source.name
        )

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
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="claim-audit report")
    parser.add_argument("--qa-report", default=str(DEFAULT_QA_REPORT), help="QA-layer audit report")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        report_path=Path(args.report),
        qa_report_path=Path(args.qa_report),
        ledger_path=Path(args.ledger),
        dry_run=args.dry_run,
    )
    print(
        f"[audit-ledger] appended={summary['appended']} aging={summary['aging']} "
        f"malformed={summary['malformed']} escalated={summary['escalated']} "
        f"deduplicated={summary['deduplicated']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
