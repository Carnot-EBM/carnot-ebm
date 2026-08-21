"""Tests for truthful milestone archival in scripts/research_conductor.py.

REQ: REQ-CONDUCTOR-ARCHIVE-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-ARCHIVE-1,
SCENARIO-CONDUCTOR-ARCHIVE-2,
SCENARIO-CONDUCTOR-ARCHIVE-3.

Origin: the archiver stamped the literal "OK (conductor)" on every task,
including 57 whose deliverables were never created, and appended the same
milestone on every activation-refusal retry (684 copies of .510). These
tests pin the replacement: results derive from the conductor log plus
deliverable existence, and a milestone id archives at most once.

All file paths are redirected to tmp_path — no test writes tracked state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_conductor as rc  # noqa: E402

LOG_HEADER = (
    "# Research Conductor Log\n\n"
    "| Timestamp | Task | Status | Details |\n"
    "|-----------|------|--------|---------|\n"
)


def _row(title: str, status: str) -> str:
    return f"| 2026-08-21 12:00 UTC | {title[:50]} | {status} | details |\n"


class TestDeriveTaskResult:
    """derive_task_result: evidence in, result out. No literals."""

    def _status_map(self, *rows: tuple[str, str]) -> dict[str, list[str]]:
        text = LOG_HEADER + "".join(_row(t, s) for t, s in rows)
        return rc._statuses_since_last_activation(text)

    def test_ok_with_deliverable_is_ok(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "exp1.json").write_text("{}")
        task = {"title": "Exp 1: thing", "deliverable": "results/exp1.json"}
        smap = self._status_map(("Exp 1: thing", "OK"))
        assert rc.derive_task_result(task, smap, tmp_path) == "OK"

    def test_log_ok_without_deliverable_is_not_ok(self, tmp_path: Path) -> None:
        # SCENARIO-CONDUCTOR-ARCHIVE-1: the exact
        # phantom-OK shape — the log says OK, the file was never created.
        task = {"title": "Exp 2: phantom", "deliverable": "results/exp2.json"}
        smap = self._status_map(("Exp 2: phantom", "OK"))
        assert rc.derive_task_result(task, smap, tmp_path) == "OK_NO_DELIVERABLE"

    def test_three_fail_skip_is_skipped(self, tmp_path: Path) -> None:
        # SCENARIO-CONDUCTOR-ARCHIVE-2
        task = {"title": "Exp 3: skipped", "deliverable": "results/exp3.json"}
        smap = self._status_map(
            ("Exp 3: skipped", "FAIL"),
            ("Exp 3: skipped", "FAIL"),
            ("Exp 3: skipped", "SKIP"),
        )
        assert rc.derive_task_result(task, smap, tmp_path) == "SKIPPED (3-fail)"

    def test_gate_block_is_gate_blocked(self, tmp_path: Path) -> None:
        task = {"title": "Exp 4: gated", "deliverable": "results/exp4.json"}
        smap = self._status_map(
            ("Exp 4: gated", "GATE_BLOCK"),
            ("Exp 4: gated", "GATE_BLOCK"),
            ("Exp 4: gated", "GATE_BLOCK"),
        )
        assert rc.derive_task_result(task, smap, tmp_path) == "GATE_BLOCKED"

    def test_doomed_rerun_block_is_named(self, tmp_path: Path) -> None:
        task = {"title": "Exp 5: doomed", "deliverable": "results/exp5.json"}
        smap = self._status_map(("Exp 5: doomed", "DOOMED_RERUN_BLOCK"))
        assert rc.derive_task_result(task, smap, tmp_path) == "DOOMED_RERUN_BLOCKED"

    def test_flagged_log_row_is_flagged(self, tmp_path: Path) -> None:
        task = {"title": "Exp 6: fab", "deliverable": "results/exp6.json"}
        smap = self._status_map(("Exp 6: fab", "FLAGGED"))
        assert rc.derive_task_result(task, smap, tmp_path) == "FLAGGED"

    def test_flagged_artifact_stamp_overrides_ok(self, tmp_path: Path) -> None:
        # A quarantined artifact must not archive as OK even when the log
        # row predates the adversarial stamp.
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "exp7.json").write_text(json.dumps({"flagged_adversarial": True}))
        task = {"title": "Exp 7: stamped", "deliverable": "results/exp7.json"}
        smap = self._status_map(("Exp 7: stamped", "OK"))
        assert rc.derive_task_result(task, smap, tmp_path) == "FLAGGED"

    def test_no_evidence_is_not_run(self, tmp_path: Path) -> None:
        task = {"title": "Exp 8: ghost", "deliverable": "results/exp8.json"}
        assert rc.derive_task_result(task, {}, tmp_path) == "NOT_RUN"

    def test_deliverable_without_log_row_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "exp9.json").write_text("{}")
        task = {"title": "Exp 9: quiet", "deliverable": "results/exp9.json"}
        assert rc.derive_task_result(task, {}, tmp_path) == "OK_DELIVERABLE_ONLY"

    def test_prior_milestone_rows_do_not_count(self, tmp_path: Path) -> None:
        # Rows before the last activation line belong to a prior milestone
        # — same scoping pick_next_task uses.
        text = (
            LOG_HEADER
            + _row("Exp 10: old", "OK")
            + "| 2026-08-21 13:00 UTC | Milestone 2026.08.561 activated | OK | 5 tasks |\n"
        )
        smap = rc._statuses_since_last_activation(text)
        task = {"title": "Exp 10: old", "deliverable": "results/exp10.json"}
        assert rc.derive_task_result(task, smap, tmp_path) == "NOT_RUN"


class TestArchiveDedupAndDerivedResults:
    """_archive_current_milestone end-to-end against tmp_path files."""

    def _setup(self, tmp_path: Path, monkeypatch) -> None:
        roadmap = {
            "milestone": "2026.08.999",
            "milestone_title": "test milestone",
            "tasks": [
                {
                    "id": "exp9001-ok",
                    "title": "Exp 9001: succeeds",
                    "deliverable": "results/exp9001.json",
                    "prompt": "x",
                },
                {
                    "id": "exp9002-phantom",
                    "title": "Exp 9002: phantom",
                    "deliverable": "results/exp9002.json",
                    "prompt": "x",
                },
            ],
        }
        (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "exp9001.json").write_text("{}")
        log = (
            LOG_HEADER
            + "| t | Milestone 2026.08.999 activated | OK | 2 tasks |\n"
            + _row("Exp 9001: succeeds", "OK")
            + _row("Exp 9002: phantom", "OK")
        )
        (tmp_path / "conductor-log.md").write_text(log)
        monkeypatch.setattr(rc, "ROADMAP_FILE", tmp_path / "research-roadmap.yaml")
        monkeypatch.setattr(rc, "COMPLETE_FILE", tmp_path / "research-complete.yaml")
        monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "conductor-log.md")
        monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)

    def test_results_derive_from_evidence(self, tmp_path: Path, monkeypatch) -> None:
        self._setup(tmp_path, monkeypatch)
        assert rc._archive_current_milestone(push=False) is True
        data = yaml.safe_load((tmp_path / "research-complete.yaml").read_text())
        tasks = {t["id"]: t["result"] for t in data["milestones"][0]["tasks"]}
        assert tasks["exp9001-ok"] == "OK"
        # The log said OK but the deliverable was never created: never OK.
        assert tasks["exp9002-phantom"] == "OK_NO_DELIVERABLE"
        assert "OK (conductor)" not in (tmp_path / "research-complete.yaml").read_text()

    def test_duplicate_append_refused(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-ARCHIVE-3: the .510 re-archive
        # loop shape — a second call for the same milestone must not append.
        self._setup(tmp_path, monkeypatch)
        assert rc._archive_current_milestone(push=False) is True
        assert rc._archive_current_milestone(push=False) is True
        data = yaml.safe_load((tmp_path / "research-complete.yaml").read_text())
        ids = [m["id"] for m in data["milestones"]]
        assert ids == ["2026.08.999"]
