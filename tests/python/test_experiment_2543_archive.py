"""Tests for Exp 2543 archive activation.

Spec: REQ-REPORT-2543, SCENARIO-REPORT-2543.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_2543_archive as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_fixture(
    root: Path,
    *,
    roadmap_milestone: str,
    complete_contains_244: bool,
    include_next: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {roadmap_milestone}\ntasks:\n- id: current-task\n",
        encoding="utf-8",
    )
    if include_next:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.05.245\ntasks:\n- id: exp2543-archive-and-activate\n",
            encoding="utf-8",
        )
    complete_text = "milestones:\n"
    if complete_contains_244:
        complete_text += (
            "- id: 2026.05.244\n"
            "  title: archived fixture\n"
            "  tasks:\n"
            "  - id: exp2542-retro-v244\n"
            "    result: OK (conductor)\n"
        )
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")


def test_scenario_report_2543_exits_cleanly_when_245_already_active(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2543: active .245 verifies the existing .244 archive only."""

    _write_fixture(tmp_path, roadmap_milestone="2026.05.245", complete_contains_244=True)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(100.0, 104.25))

    after_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    written = json.loads((tmp_path / "results/experiment_2543_archive.json").read_text())

    assert artifact == written
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_ready"] is True
    assert artifact["milestone_archived"] == "2026.05.244"
    assert artifact["preconditions_checked"]["roadmap_milestone"]["status"] == "already_activated"
    assert artifact["archive"]["appended_this_run"] is False
    assert artifact["activation"]["copied_this_run"] is False
    assert artifact["duration_s"] == 4.25
    assert "exp2531" in artifact["execution_gap_diagnosis"]["task_hypotheses"][1]
    assert after_complete == before_complete
    assert after_complete.count("id: 2026.05.244") == 1


def test_req_report_2543_archives_244_and_activates_245(tmp_path: Path) -> None:
    """REQ-REPORT-2543: active .244 appends archive state and swaps .245 roadmap."""

    _write_fixture(tmp_path, roadmap_milestone="2026.05.244", complete_contains_244=False)

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 2.0))

    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    roadmap_text = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    assert artifact["archive_ready"] is True
    assert (
        artifact["preconditions_checked"]["roadmap_milestone"]["status"] == "archived_and_activated"
    )
    assert artifact["archive"]["appended_this_run"] is True
    assert artifact["activation"]["copied_this_run"] is True
    assert "id: 2026.05.244" in complete_text
    assert "n_experiments_completed: 5" in complete_text
    assert "execution_gap: exp2530-exp2534 produced no artifacts" in complete_text
    assert roadmap_text.startswith("milestone: 2026.05.245")


def test_req_report_2543_unexpected_roadmap_is_reported_without_mutation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2543: unexpected active milestone records a terminal blocked verdict."""

    _write_fixture(
        tmp_path,
        roadmap_milestone="2026.05.246",
        complete_contains_244=False,
        include_next=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))

    assert artifact["preconditions_checked"]["roadmap_milestone"]["status"] == (
        "blocked_roadmap_unexpected_milestone"
    )
    assert "blocked_roadmap_unexpected_milestone" in artifact["honest_verdict"]
    assert artifact["archive_ready"] is False
    assert artifact["archive"]["appended_this_run"] is False
    assert artifact["activation"]["copied_this_run"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
