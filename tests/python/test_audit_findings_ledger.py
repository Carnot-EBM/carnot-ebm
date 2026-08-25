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
        # An explicit map names EVERY source this run may read, so adding a
        # source cannot leak a real tracked report into an isolated test.
        report_paths={L.AUDIT_NAME: report},
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
        report_paths={L.AUDIT_NAME: tmp_path / "report.md"},
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
    """Same-day rows stay silent; escalation begins at AGING_DAYS = 1 (the
    2026-08-23 amendment — a week of silence was structurally slow against
    a loop closing several milestones per day)."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-22 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | OPEN | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 0


def test_one_day_old_row_escalates(tmp_path):
    """The amendment's contract: an OPEN finding becomes visible on the
    NEXT day's closes, not a week later. The 2026-08-22 CLAIM_OVERSTATED
    findings sat OPEN until an operator prompt on 2026-08-23 — under this
    threshold they would have escalated automatically that morning."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-21 | experiment_claim_audit | experiment_6478_x.json "
        "| CLAIM_OVERSTATED | OPEN | |\n"
    )
    summary, _ = _run(tmp_path, report_text="# empty report\n", ledger_pre=pre)
    assert summary["escalated"] == 1
    log = (tmp_path / "log.md").read_text()
    assert "OPEN 1 days" in log


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
        report_paths={L.AUDIT_NAME: tmp_path / "absent.md"},
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
        report_paths={L.AUDIT_NAME: report},
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


# --- REQ-OPS-AUDIT-LEDGER-2: the ledger reads EVERY audit, not one -----------
#
# Incident, 2026-08-25. Milestone closes .572 and .573 produced 7
# SILENT_NON_FIRING verdicts between them and the ledger ingested ZERO,
# because DEFAULT_REPORT named the claim audit and nothing else. All 7 were
# hand-entered by the outer loop. The report is regenerated at every close,
# so an un-ingested finding is overwritten, not merely unread.

_QA_REPORT = """<!-- generated by scripts/qa_layer_authenticity_audit.py -->

# qa_layer_authenticity_audit_report — 2026-08-25

## Summary

| Verdict | Count |
|---|---|
| `CLEAN` | 1 |
| `SILENT_NON_FIRING` | 1 |

### MISSED INPUTS — a real input each guard does NOT catch
- `operator_curated_doc_guard.py` — docs/blog/caught-cheating.html, as a bare basename.

### FLAGGED — operator action recommended
- `operator_curated_doc_guard.py` — **SILENT_NON_FIRING**

---

## operator_curated_doc_guard.py

**Verdict:** `SILENT_NON_FIRING`

## VERDICT
SILENT_NON_FIRING

## CONCEPT
Prevent test-suite mutation of operator-curated documents.

## MISSED INPUT
`docs/blog/caught-cheating.html`, as the fd-relative bare basename.

## RECOMMENDATION
NEEDS_REDESIGN


## arc_count_integrity_lint.py

**Verdict:** `CLEAN`

## VERDICT
CLEAN
"""


def _run_qa(tmp_path, report_text=_QA_REPORT, today=_TODAY, ledger_pre=None):
    report = tmp_path / "qa_report.md"
    report.write_text(report_text)
    ledger = tmp_path / "ledger.md"
    if ledger_pre is not None:
        ledger.write_text(ledger_pre)
    summary = L.run(
        report_paths={L.QA_AUDIT_NAME: report},
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=today,
    )
    return summary, ledger


def test_qa_layer_silent_non_firing_produces_a_ledger_row(tmp_path):
    """SCENARIO-OPS-AUDIT-LEDGER-2-QA-INGEST — the incident input.

    A QA-layer report carrying a SILENT_NON_FIRING verdict must produce a
    ledger row. Named for `operator_curated_doc_guard.py`, one of the 7
    findings that reached no ledger on 2026-08-25.
    """
    summary, ledger = _run_qa(tmp_path)
    assert summary["appended"] == 1
    entries, malformed = L.parse_ledger(ledger.read_text())
    assert malformed == []
    assert len(entries) == 1
    assert entries[0]["audit"] == "qa_layer_authenticity_audit"
    assert entries[0]["artifact"] == "operator_curated_doc_guard.py"
    assert entries[0]["verdict"] == "SILENT_NON_FIRING"
    assert entries[0]["disposition"] == "OPEN"


def test_qa_clean_verdict_and_reviewer_prose_produce_no_row(tmp_path):
    """Only flagged verdicts enter. The reviewer's own `## VERDICT` /
    `## CONCEPT` headings are prose, not units, and CLEAN is not flagged."""
    _, ledger = _run_qa(tmp_path)
    text = ledger.read_text()
    assert "arc_count_integrity_lint.py" not in text
    assert "CONCEPT" not in text


def test_qa_ingest_is_idempotent(tmp_path):
    """A second close over the same report appends nothing and rewrites
    nothing — the report is regenerated every close, so re-reading it must
    not grow the ledger."""
    _, ledger = _run_qa(tmp_path)
    before = ledger.read_text()
    summary2, _ = _run_qa(tmp_path, ledger_pre=None)
    assert summary2["appended"] == 0
    assert ledger.read_text() == before


def test_two_audits_flagging_the_same_name_are_two_rows(tmp_path):
    """The audit name is part of a row's identity. Two audits flagging the
    same file must not collide, and neither may dedup the other away."""
    claim = tmp_path / "claim.md"
    claim.write_text("## shared_name.json\n\n**CLAIM_OVERSTATED**\n")
    qa = tmp_path / "qa.md"
    qa.write_text("## shared_name.json\n\n**Verdict:** `REAL_BUG`\n")
    ledger = tmp_path / "ledger.md"
    summary = L.run(
        report_paths={L.AUDIT_NAME: claim, L.QA_AUDIT_NAME: qa},
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 2
    entries, _ = L.parse_ledger(ledger.read_text())
    assert {e["audit"] for e in entries} == {L.AUDIT_NAME, L.QA_AUDIT_NAME}
    assert {e["artifact"] for e in entries} == {"shared_name.json"}


def test_missing_qa_report_is_a_noop_not_a_clean_bill(tmp_path):
    """An audit that did not run must not make the ledger look answered."""
    summary = L.run(
        report_paths={L.QA_AUDIT_NAME: tmp_path / "absent.md"},
        ledger_path=tmp_path / "ledger.md",
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 0
    assert not (tmp_path / "ledger.md").exists()


def test_every_source_imports_its_flagged_set_from_its_own_module():
    """One list, one home — for EVERY source, not just the first. A copy
    here is how a ledger silently stops ingesting a verdict the audit
    still emits."""
    for source in L.SOURCES:
        imported = L.source_flagged_verdicts(source)
        assert imported, f"{source.name} exposed no flagged verdicts"
        module = L._load_module(source.module, L.REPO / "scripts" / f"{source.module}.py")
        assert imported == tuple(sorted(getattr(module, source.verdict_attr)))


def test_qa_layer_audit_is_a_registered_source():
    """The incident in one assertion: the QA-layer audit must be a source.
    Its absence is what let 7 findings reach no ledger."""
    assert L.QA_AUDIT_NAME in {s.name for s in L.SOURCES}


def test_real_qa_report_parses_if_present():
    """Parse the REAL tracked report if it exists — the fixture above is a
    reduction, and a format drift in the writer must not pass unseen.
    Read-only; the report is evidence."""
    real = L.DEFAULT_QA_REPORT
    if not real.exists():
        assert L.QA_AUDIT_NAME in {s.name for s in L.SOURCES}
        return
    source = next(s for s in L.SOURCES if s.name == L.QA_AUDIT_NAME)
    findings = L.parse_qa_report(
        real.read_text(encoding="utf-8"), L.source_flagged_verdicts(source)
    )
    flagged_in_summary = real.read_text(encoding="utf-8").count("** — **SILENT_NON_FIRING**")
    assert len(findings) >= flagged_in_summary
    for finding in findings:
        assert finding["verdict"] in L.source_flagged_verdicts(source)


def test_two_spellings_of_one_finding_produce_one_row(tmp_path):
    """M9, measured live. A hand-entered row spells a guard `scripts/child_results_guard.py`;
    the QA report writes the bare basename. On 2026-08-25 the widening appended 4 rows that
    duplicated 4 hand-entered ones, and an append-only ledger cannot un-write them -- each
    duplicate escalates weekly until a human dispositions BOTH."""
    pre = (
        "| First seen | Audit | Artifact | Verdict | Disposition | Note |\n"
        "|---|---|---|---|---|---|\n"
        "| 2026-08-25 | qa_layer_authenticity_audit | scripts/child_results_guard.py "
        "| SILENT_NON_FIRING | OPEN | hand-entered by the outer loop |\n"
    )
    report = tmp_path / "qa.md"
    report.write_text(
        "### FLAGGED — operator action recommended\n"
        "- `child_results_guard.py` — **SILENT_NON_FIRING**\n"
    )
    ledger = tmp_path / "ledger.md"
    ledger.write_text(pre)
    summary = L.run(
        report_paths={L.QA_AUDIT_NAME: report},
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 0, "the bare basename is the same finding as the path form"
    assert "hand-entered by the outer loop" in ledger.read_text()


def test_reviewer_prose_shaped_like_a_flagged_item_produces_no_row(tmp_path):
    """M10. The item pattern applied to the whole document also matches the reviewer's own
    prose -- a reviewer comparing guards writes exactly this shape inside `## FINDINGS`. A
    phantom row is permanent in an append-only ledger and escalates weekly forever."""
    report = tmp_path / "qa.md"
    report.write_text(
        "## some_guard.py\n\n"
        "**Verdict:** `CLEAN`\n\n"
        "## FINDINGS\n"
        "1. Compare against its sibling:\n"
        "Its sibling was worse:\n"
        "- `arc_artifact_lint.py` — **REAL_BUG**\n"
    )
    summary, ledger = None, tmp_path / "ledger.md"
    summary = L.run(
        report_paths={L.QA_AUDIT_NAME: report},
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 0, "prose outside the FLAGGED section is not a finding"
    assert not ledger.exists()


def test_a_flagged_item_inside_the_flagged_section_still_ingests(tmp_path):
    """The scoping must not throw away the real list it was written to read."""
    report = tmp_path / "qa.md"
    report.write_text(
        "### FLAGGED — operator action recommended\n"
        "- `arc_artifact_lint.py` — **REAL_BUG**\n"
        "\n---\n"
    )
    summary, ledger = None, tmp_path / "ledger.md"
    summary = L.run(
        report_paths={L.QA_AUDIT_NAME: report},
        ledger_path=ledger,
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
        today=_TODAY,
    )
    assert summary["appended"] == 1
    entries, _ = L.parse_ledger(ledger.read_text())
    assert entries[0]["artifact"] == "arc_artifact_lint.py"


def test_an_unknown_source_key_raises_rather_than_selecting_nothing(tmp_path):
    """A typo, or a stale literal after a rename, would otherwise select ZERO sources,
    append nothing, and return success."""
    import pytest

    with pytest.raises(KeyError):
        L.run(
            report_paths={"qa_layer_audit": tmp_path / "x.md"},
            ledger_path=tmp_path / "ledger.md",
            conductor_log=tmp_path / "log.md",
            known_issues=tmp_path / "ki.md",
            state_path=tmp_path / "state.json",
            today=_TODAY,
        )
