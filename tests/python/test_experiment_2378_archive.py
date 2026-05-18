"""Tests for Exp 2378 archive activation.

Spec: REQ-REPORT-2378, SCENARIO-REPORT-2378.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_2378_archive as exp


def _write_fixture(
    root: Path,
    *,
    roadmap_milestone: str,
    complete_contains_231: bool,
    include_next: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {roadmap_milestone}\ntasks:\n- id: current-task\n",
        encoding="utf-8",
    )
    if include_next:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.05.232\ntasks:\n- id: exp2378-archive-and-activate\n",
            encoding="utf-8",
        )
    complete_text = "milestones:\n"
    if complete_contains_231:
        complete_text += (
            "- id: 2026.05.231\n"
            "  title: archived fixture\n"
            "  tasks:\n"
            "  - id: exp2364-archive-and-activate\n"
            "    result: OK (conductor)\n"
        )
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")


def test_scenario_report_2378_archives_231_and_activates_232(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2378: active .231 appends archive state and swaps .232 roadmap."""

    _write_fixture(tmp_path, roadmap_milestone="2026.05.231", complete_contains_231=False)

    artifact = exp.run(root=tmp_path)

    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    roadmap_text = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    written = json.loads((tmp_path / "results/experiment_2378_archive.json").read_text())

    assert artifact == written
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_ready"] is True
    assert artifact["milestone_archived"] == "2026.05.231"
    assert artifact["archive"]["appended_this_run"] is True
    assert artifact["activation"]["copied_this_run"] is True
    assert "id: 2026.05.231" in complete_text
    assert "exp2365-fst-live-gen-v11" in complete_text
    assert "fst_live_validated=true" in complete_text
    assert roadmap_text.startswith("milestone: 2026.05.232")


def test_req_report_2378_exits_cleanly_when_232_already_active(tmp_path: Path) -> None:
    """REQ-REPORT-2378: already-active .232 does not duplicate the .231 archive."""

    _write_fixture(tmp_path, roadmap_milestone="2026.05.232", complete_contains_231=True)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path)

    after_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    assert artifact["preconditions"]["status"] == "already_activated"
    assert artifact["archive_ready"] is True
    assert artifact["archive"]["appended_this_run"] is False
    assert artifact["activation"]["copied_this_run"] is False
    assert after_complete == before_complete
    assert after_complete.count("id: 2026.05.231") == 1
    assert artifact["field_principles"]["honest_verdict"].endswith("Must start complete:.")


def test_req_report_2378_unexpected_roadmap_is_reported_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2378: unexpected active milestone records an honest terminal verdict."""

    _write_fixture(
        tmp_path,
        roadmap_milestone="2026.05.233",
        complete_contains_231=True,
        include_next=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path)

    assert artifact["preconditions"]["status"] == "blocked_roadmap_unexpected"
    assert "blocked_roadmap_unexpected" in artifact["honest_verdict"]
    assert artifact["archive_ready"] is True
    assert artifact["milestone_archived"] == "2026.05.231"
    assert artifact["archive"]["appended_this_run"] is False
    assert artifact["activation"]["copied_this_run"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
