"""Tests for scripts/audit_findings_ledger.py — flagged audit verdicts
someone must answer.

REQ-OPS-AUDIT-LEDGER-1: flagged claim-audit verdicts enter an append-only
disposition ledger; OPEN rows age and escalate weekly through the run
sentinel's durable writer until a human writes a disposition.

All writes go under tmp_path; no test touches tracked state.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_findings_ledger", _REPO / "scripts" / "audit_findings_ledger.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load()

_REPORT = """# Experiment claim-refutation audit

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 1 |
| CLAIM_OVERSTATED | 1 |

## experiment_6478_identifiable_held_exact_energy_selection.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Reviewer prose that itself contains **BOLD** words and ## headings.

## experiment_9999_healthy.json

**CLAIM_SUPPORTED**

body text
"""

_TODAY = datetime(2026, 8, 22, tzinfo=UTC)


def _run(tmp_path, report_text=_REPORT, today=_TODAY, ledger_pre=None):
    report = tmp_path / "report.md"
    report.write_text(report_text)
    ledger = tmp_path / "ledger.md"
    if ledger_pre is not None:
        ledger.write_text(ledger_pre)
    summary = L.run(
        report_path=report,
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=today,
    )
    return summary, ledger


def test_ingest_appends_one_open_row(tmp_path):
    """SCENARIO-OPS-AUDIT-LEDGER-1-INGEST: one flagged verdict, one OPEN
    row; the supported verdict and the reviewer's inner headings/bold text
    do not produce rows."""
    summary, ledger = _run(tmp_path)
    assert summary["appended"] == 1
    entries, malformed = L.parse_ledger(ledger.read_text())
    assert malformed == []
    assert len(entries) == 1
    entry = entries[0]
    assert entry["artifact"].startswith("experiment_6478")
    assert entry["verdict"] == "CLAIM_OVERSTATED"
    assert entry["disposition"] == "OPEN"
    assert entry["first_seen"] == "2026-08-22"


def test_second_run_is_idempotent(tmp_path):
    """SCENARIO-OPS-AUDIT-LEDGER-1-IDEMPOTENT: nothing appended, nothing
    rewritten on a second pass over the same report."""
    _, ledger = _run(tmp_path)
    before = ledger.read_text()
    summary2 = L.run(
        report_path=tmp_path / "report.md",
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary2["appended"] == 0
    assert ledger.read_text() == before


def test_open_row_ages_and_escalates(tmp_path):
    """SCENARIO-OPS-AUDIT-LEDGER-1-AGING: an OPEN row first seen 8 days ago
    escalates, naming the finding and its age."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-14 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | OPEN | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 1
    log = (tmp_path / "log.md").read_text()
    assert "AUDIT_FINDING_UNTRIAGED" in log
    assert "OPEN 8 days" in log


def test_fresh_open_row_does_not_escalate(tmp_path):
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-20 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | OPEN | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 0


def test_human_close_silences_and_row_is_untouched(tmp_path):
    """SCENARIO-OPS-AUDIT-LEDGER-1-HUMAN-CLOSE: FIXED silences the aging
    escalation and the tool leaves the row bytes alone."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-01 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | FIXED | corrigendum landed |\n"
    )
    summary, ledger = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 0
    assert "corrigendum landed" in ledger.read_text()


def test_unrecognized_disposition_fails_closed(tmp_path):
    """A typo'd disposition must not silence the finding — only a
    recognized closed disposition does."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-01 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | FIXEDD | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 1
    # The log_step row format truncates the task cell at 50 chars, so the
    # code appears clipped — assert the clipped form the row really carries.
    assert "LEDGER_DISPOSITION_UNRECOGNI" in (tmp_path / "log.md").read_text()


def test_malformed_row_is_a_finding_not_a_skip(tmp_path):
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| not-a-date | experiment_claim_audit | x.json | CLAIM_OVERSTATED | OPEN | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["malformed"] == 1
    assert "LEDGER_ROW_MALFORMED" in (tmp_path / "log.md").read_text()


def test_missing_report_is_a_noop_ingest(tmp_path):
    summary = L.run(
        report_path=tmp_path / "absent.md",
        ledger_path=tmp_path / "ledger.md",
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 0
    assert not (tmp_path / "ledger.md").exists()


def test_weekly_rebucket_reescalates(tmp_path):
    """The same OPEN finding escalates again when it crosses into the next
    age-week bucket — silence never resumes until a disposition changes."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-01 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | OPEN | |\n"
    )
    report = tmp_path / "report.md"
    report.write_text("# empty report\n")
    ledger = tmp_path / "ledger.md"
    ledger.write_text(pre)
    common = dict(
        report_path=report,
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
    )
    week1 = L.run(today=datetime(2026, 8, 9, tzinfo=UTC), **common)
    week1_again = L.run(today=datetime(2026, 8, 10, tzinfo=UTC), **common)
    week2 = L.run(today=datetime(2026, 8, 16, tzinfo=UTC), **common)
    assert week1["escalated"] == 1
    assert week1_again["escalated"] == 0  # same bucket -> deduplicated
    assert week2["escalated"] == 1  # new bucket -> re-escalated
    assert (tmp_path / "log.md").read_text().count("AUDIT_FINDING_UNTRIAGED") == 2


def test_flagged_set_comes_from_the_audit_module():
    """One list, one home: the ledger's flagged set IS the claim audit's."""
    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "experiment_claim_audit", _REPO / "scripts" / "experiment_claim_audit.py"
    )
    audit = iu.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert L.flagged_verdicts() == tuple(audit.FLAGGED_VERDICTS)
