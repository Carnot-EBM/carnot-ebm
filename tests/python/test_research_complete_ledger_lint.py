"""Tests for scripts/research_complete_ledger_lint.py.

REQ: REQ-CONDUCTOR-ARCHIVE-2 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-ARCHIVE-4,
SCENARIO-CONDUCTOR-ARCHIVE-5,
SCENARIO-CONDUCTOR-ARCHIVE-6,
SCENARIO-CONDUCTOR-ARCHIVE-7.

The lint is the regression lock for the truthful-archival fix (commit
87f209cdcf): the archiver no longer stamps "OK (conductor)" or appends
duplicate milestones, and this lint refuses a commit that would show
either behavior returning. Forward-only from cutoff 2026-08-22 so the
uncorrected historical corpus (57 phantom rows, 1,841 surplus duplicate
entries measured 2026-08-21) never blocks a commit.

Every ledger is built under tmp_path — no test reads or writes tracked
state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_complete_ledger_lint as lint  # noqa: E402

# Verbatim shape of a REAL phantom row from the historical corpus:
# milestone 2026.07.510, task exp5710 — result "OK (conductor)" with a
# deliverable that was never created (one of the 57).
REAL_PHANTOM_TASK = {
    "id": "exp5710-fr11-isolated-act-on-advice-canary",
    "title": "Gated on Exp5709 prospective promotion: isolated FR-11 act-on-advice canary",
    "deliverable": "results/experiment_5710_fr11_isolated_act_on_advice_canary.json",
    "result": "OK (conductor)",
}


def _entry(mid: str, completed: str, tasks: list[dict]) -> dict:
    return {"id": mid, "title": f"t-{mid}", "completed": completed, "tasks": tasks}


def _write(tmp_path: Path, milestones: list[dict]) -> Path:
    ledger = tmp_path / "research-complete.yaml"
    ledger.write_text(yaml.safe_dump({"milestones": milestones}, sort_keys=False))
    return ledger


def _ok_task(tmp_path: Path, name: str, create: bool) -> dict:
    deliverable = f"results/{name}.json"
    if create:
        p = tmp_path / deliverable
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return {"id": name, "title": name, "deliverable": deliverable, "result": "OK"}


def test_duplicate_post_cutoff_id_fails(tmp_path: Path) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-4: a post-cutoff entry re-using an id
    # that the file already holds is the retry-append regressing.
    ledger = _write(
        tmp_path,
        [
            _entry("2026.08.560", "2026-08-19", []),
            _entry("2026.08.560", "2026-08-25", []),
        ],
    )
    violations = lint.check_ledger(ledger)
    assert violations, "duplicate post-cutoff id must be a violation"
    assert any("2026.08.560" in v and "duplicate" in v.lower() for v in violations)
    assert lint.main([str(ledger)]) == 1


def test_ok_row_with_missing_deliverable_fails(tmp_path: Path) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-5 (missing half): OK must mean the
    # deliverable exists — that is the definition the archiver now derives.
    task = _ok_task(tmp_path, "experiment_9001_missing", create=False)
    ledger = _write(tmp_path, [_entry("2026.08.562", "2026-08-25", [task])])
    violations = lint.check_ledger(ledger)
    assert any("experiment_9001_missing" in v for v in violations)
    assert lint.main([str(ledger)]) == 1


def test_ok_row_with_present_deliverable_passes(tmp_path: Path) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-5 (present half).
    task = _ok_task(tmp_path, "experiment_9002_present", create=True)
    ledger = _write(tmp_path, [_entry("2026.08.562", "2026-08-25", [task])])
    assert lint.check_ledger(ledger) == []
    assert lint.main([str(ledger)]) == 0


def test_retired_literal_fails_even_with_deliverable(tmp_path: Path) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-6: "OK (conductor)" is the retired
    # hardcoded stamp. The derived vocabulary cannot produce it, so its
    # reappearance means the archiver regressed — deliverable or not.
    task = _ok_task(tmp_path, "experiment_9003_literal", create=True)
    task["result"] = "OK (conductor)"
    ledger = _write(tmp_path, [_entry("2026.08.562", "2026-08-25", [task])])
    violations = lint.check_ledger(ledger)
    assert any("OK (conductor)" in v for v in violations)
    assert lint.main([str(ledger)]) == 1


def test_historical_corpus_shape_passes_default_mode(tmp_path: Path) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-7: pre-cutoff duplicates and phantom
    # rows (verbatim real row from milestone .510) never block a commit.
    milestones = [
        _entry("2026.07.510", "2026-07-17", [dict(REAL_PHANTOM_TASK)]),
        _entry("2026.07.510", "2026-07-17", [dict(REAL_PHANTOM_TASK)]),
        _entry("2026.07.510", "2026-07-17", [dict(REAL_PHANTOM_TASK)]),
    ]
    ledger = _write(tmp_path, milestones)
    assert lint.check_ledger(ledger) == []
    assert lint.main([str(ledger)]) == 0


def test_report_historical_counts_pre_cutoff_violations(tmp_path: Path, capsys) -> None:
    # SCENARIO-CONDUCTOR-ARCHIVE-7 (evidence mode): the same corpus
    # shape is counted, not blocked, under --report-historical.
    milestones = [
        _entry("2026.07.510", "2026-07-17", [dict(REAL_PHANTOM_TASK)]),
        _entry("2026.07.510", "2026-07-17", [dict(REAL_PHANTOM_TASK)]),
    ]
    ledger = _write(tmp_path, milestones)
    assert lint.main([str(ledger), "--report-historical"]) == 0
    out = capsys.readouterr().out
    assert "duplicate" in out.lower()
    assert "OK (conductor)" in out
    # Two entries share one id (1 duplicate id) and both carry the
    # phantom row (2 phantom rows).
    assert "duplicate milestone ids: 1" in out
    assert "rows with the retired literal 'OK (conductor)': 2" in out


def test_ok_no_deliverable_is_honest_and_passes(tmp_path: Path) -> None:
    # REQ-CONDUCTOR-ARCHIVE-2 rule 3 carve-out: OK_NO_DELIVERABLE is the
    # archiver HONESTLY recording a missing deliverable. It must not be
    # punished, or the truthful vocabulary becomes worse than the lie.
    task = {
        "id": "exp-h",
        "title": "h",
        "deliverable": "results/never_created.json",
        "result": "OK_NO_DELIVERABLE",
    }
    ledger = _write(tmp_path, [_entry("2026.08.563", "2026-08-25", [task])])
    assert lint.check_ledger(ledger) == []


def test_missing_completed_date_is_checked_fail_closed(tmp_path: Path) -> None:
    # REQ-CONDUCTOR-ARCHIVE-2: the archiver always writes `completed`, so
    # an entry without it is an anomalous NEW entry — checked, not skipped.
    task = _ok_task(tmp_path, "experiment_9004_nodate", create=False)
    entry = {"id": "2026.08.564", "title": "t", "tasks": [task]}
    ledger = _write(tmp_path, [entry])
    assert lint.main([str(ledger)]) == 1


def test_unparseable_ledger_fails_closed(tmp_path: Path) -> None:
    # REQ-CONDUCTOR-ARCHIVE-2: a corrupted record fails the lint rather
    # than passing silently — the guard must never be trusted-and-silent.
    ledger = tmp_path / "research-complete.yaml"
    ledger.write_text("milestones: [unclosed")
    assert lint.main([str(ledger)]) == 1


def test_deliverable_free_ok_row_passes(tmp_path: Path) -> None:
    # A task that declares no deliverable can honestly archive OK
    # (derive_task_result returns OK for log-OK + empty deliverable).
    task = {"id": "exp-nod", "title": "nod", "deliverable": "", "result": "OK"}
    ledger = _write(tmp_path, [_entry("2026.08.565", "2026-08-25", [task])])
    assert lint.check_ledger(ledger) == []
