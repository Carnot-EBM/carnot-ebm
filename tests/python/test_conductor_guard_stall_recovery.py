"""Tests for activation-refusal guard-stall recovery in the conductor.

REQ: REQ-CONDUCTOR-STALL-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-STALL-1 (replan embeds the verbatim report),
SCENARIO-CONDUCTOR-STALL-2 (cap, then park), SCENARIO-CONDUCTOR-STALL-3
(parked iterations idle), SCENARIO-CONDUCTOR-STALL-4 (an operator edit
unparks with a fresh budget).

Origin: on an activation refusal the loop retried an identical activation
every two minutes, forever (~5,125 refusal lines; milestone .511 alone
stalled ~1,998 iterations). The recovery replans with the guard's own
violation report, capped, then parks. The guard itself is UNCHANGED — the
replanned roadmap goes back through the same lint.

All file paths are redirected to tmp_path — no test writes tracked state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import research_conductor as rc  # noqa: E402

VIOLATION_REPORT = (
    "exclusion-manifest lint HARD violations (milestone 2026.08.777):\n"
    "- SCOPE_MATCHED_PRIOR_FAILURE: task exp9101-doomed (Doomed rerun) — "
    "matches retired exp2091 with no prior_failures block"
)


def _wire(tmp_path: Path, monkeypatch) -> dict:
    """Redirect every file the recovery touches into tmp_path.

    Returns a recorder dict capturing planner invocations.
    """
    (tmp_path / "ops").mkdir()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text("milestone: '2026.08.777'\ntasks:\n  - id: exp9101-doomed\n")
    monkeypatch.setattr(rc, "NEXT_ROADMAP_FILE", roadmap)
    monkeypatch.setattr(rc, "REPLAN_STATE_FILE", tmp_path / "ops" / ".activation_replan_state.json")
    monkeypatch.setattr(rc, "ROADMAP_QUARANTINE_DIR", tmp_path / "ops" / "roadmap-quarantine")
    monkeypatch.setattr(rc, "KNOWN_ISSUES_FILE", tmp_path / "ops" / "known-issues.md")
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "ops" / "conductor-log.md")
    (tmp_path / "ops" / "known-issues.md").write_text("# Known issues\n")
    recorder: dict = {"calls": []}

    def fake_plan(push: bool = True, replan_context: str = "") -> bool:
        recorder["calls"].append(replan_context)
        # The real planner writes a fresh roadmap; mimic that.
        roadmap.write_text("milestone: '2026.08.777'\ntasks:\n  - id: exp9102-fixed\n")
        return True

    monkeypatch.setattr(rc, "_plan_next_milestone", fake_plan)
    return recorder


class TestBoundedReplan:
    def test_first_refusal_quarantines_and_replans_with_report(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # SCENARIO-CONDUCTOR-STALL-1
        recorder = _wire(tmp_path, monkeypatch)
        rc._handle_activation_refusal("2026.08.777", VIOLATION_REPORT, push=False)
        # The refused roadmap moved to quarantine (then the stub planner
        # rewrote research-roadmap-next.yaml).
        qdir = tmp_path / "ops" / "roadmap-quarantine"
        assert [p.name for p in qdir.iterdir()] == ["roadmap-2026.08.777-refusal1.yaml"]
        assert "exp9101-doomed" in (qdir / "roadmap-2026.08.777-refusal1.yaml").read_text()
        # The planner ran once with the VERBATIM violation report.
        assert recorder["calls"] == [VIOLATION_REPORT]
        state = json.loads((tmp_path / "ops" / ".activation_replan_state.json").read_text())
        assert state["replans"] == 1
        assert state["parked"] is False

    def test_cap_then_park(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-STALL-2: replans exhausted -> park, no planner.
        recorder = _wire(tmp_path, monkeypatch)
        rc._save_replan_state({"milestone": "2026.08.777", "replans": 2, "parked": False})
        rc._handle_activation_refusal("2026.08.777", VIOLATION_REPORT, push=False)
        assert recorder["calls"] == []  # no third replan
        state = json.loads((tmp_path / "ops" / ".activation_replan_state.json").read_text())
        assert state["parked"] is True
        # Durable OPERATOR-ATTENTION records: conductor log BLOCK line +
        # dated known-issues entry carrying the verbatim report.
        log_text = (tmp_path / "ops" / "conductor-log.md").read_text()
        assert "OPERATOR-ATTENTION" in log_text
        assert "BLOCK" in log_text
        ki = (tmp_path / "ops" / "known-issues.md").read_text()
        assert "activation PARKED" in ki
        assert VIOLATION_REPORT in ki
        # The refused roadmap stays in place for inspection.
        assert rc.NEXT_ROADMAP_FILE.exists()

    def test_refusal_for_new_milestone_resets_budget(self, tmp_path: Path, monkeypatch) -> None:
        recorder = _wire(tmp_path, monkeypatch)
        rc._save_replan_state({"milestone": "2026.08.700", "replans": 2, "parked": True})
        rc._handle_activation_refusal("2026.08.777", VIOLATION_REPORT, push=False)
        # Different milestone: stale state discarded, replan 1 of 2 runs.
        assert recorder["calls"] == [VIOLATION_REPORT]
        state = json.loads((tmp_path / "ops" / ".activation_replan_state.json").read_text())
        assert state == {"milestone": "2026.08.777", "replans": 1, "parked": False}


class TestParkedIdle:
    def test_parked_with_unchanged_roadmap_idles(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-STALL-3
        _wire(tmp_path, monkeypatch)
        text = rc.NEXT_ROADMAP_FILE.read_text()
        rc._save_replan_state(
            {
                "milestone": "2026.08.777",
                "replans": 2,
                "parked": True,
                "roadmap_sha256": rc._roadmap_content_hash(text),
            }
        )
        assert rc._activation_refusal_parked() is True
        # A second refusal while parked is a no-op: no planner, no state churn.
        recorder_calls_before = json.loads(
            (tmp_path / "ops" / ".activation_replan_state.json").read_text()
        )
        rc._handle_activation_refusal("2026.08.777", VIOLATION_REPORT, push=False)
        state_after = json.loads((tmp_path / "ops" / ".activation_replan_state.json").read_text())
        assert state_after == recorder_calls_before

    def test_operator_edit_unparks_with_fresh_budget(self, tmp_path: Path, monkeypatch) -> None:
        # SCENARIO-CONDUCTOR-STALL-4
        _wire(tmp_path, monkeypatch)
        rc._save_replan_state(
            {
                "milestone": "2026.08.777",
                "replans": 2,
                "parked": True,
                "roadmap_sha256": rc._roadmap_content_hash(rc.NEXT_ROADMAP_FILE.read_text()),
            }
        )
        # Operator hand-fixes the roadmap: content hash changes.
        rc.NEXT_ROADMAP_FILE.write_text(
            "milestone: '2026.08.777'\ntasks:\n  - id: exp9101-doomed\n"
            "    operator_override: 'hand-fixed 2026-08-21'\n"
        )
        assert rc._activation_refusal_parked() is False
        # Budget is fresh again: the state file was reset.
        assert json.loads((tmp_path / "ops" / ".activation_replan_state.json").read_text()) == {}
