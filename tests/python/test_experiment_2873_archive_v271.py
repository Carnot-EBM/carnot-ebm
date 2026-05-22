"""Tests for Exp 2873 milestone .271 archive and .272 activation.

Spec: REQ-REPORT-2873, SCENARIO-REPORT-2873.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_271_archive_272_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: .271 capstone synthesized; paper_ready=true; "
                    "clean_artifacts=8; blocked_artifacts=0; missing_artifacts=0; "
                    "adversarially_flagged_artifacts=3"
                ),
                "milestone": "2026.05.271",
                "paper_ready": True,
                "clean_artifacts": [
                    "exp2861",
                    "exp2863",
                    "exp2864",
                    "exp2865",
                    "exp2866",
                    "exp2867",
                    "exp2868",
                    "exp2869",
                ],
                "blocked_artifacts": [],
                "missing_artifacts": [],
                "adversarially_flagged_artifacts": ["exp2862", "exp2870", "exp2871"],
                "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
                "top_3_next_actions": [
                    "Re-run SOTA runtime with clean provenance.",
                    "Materialize clean MBPP, HumanEval, and TruthfulQA rows.",
                    "Resolve the KAN PWA/MILP tautology flag.",
                    "Ignored fourth action.",
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.272"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
    ),
    next_roadmap: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(active_roadmap, encoding="utf-8")
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# untouched by Exp 2873\n", encoding="utf-8")
    _write_capstone(root)


def test_scenario_report_2873_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2873: existing .271 archive confirms active .272 roadmap."""

    complete_text = """milestones:
- id: 2026.05.270
  completed: '2026-05-22'
  tasks: []
- id: 2026.05.271
  title: Runtime Repair + Manifest Reconciliation + Offline Self-Learning
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-22'
  tasks:
  - id: exp2872
    deliverable: results/experiment_2872_capstone_v271.json
    result: OK (conductor)
"""
    _write_common_files(tmp_path, complete_text=complete_text)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.75))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.05.271"
    assert artifact["activated_milestone"] == "2026.05.272"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2872_capstone_v271.json"
    assert artifact["paper_ready_from_capstone"] is True
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2861",
        "exp2863",
        "exp2864",
        "exp2865",
        "exp2866",
        "exp2867",
        "exp2868",
        "exp2869",
    ]
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["adversarially_flagged_artifacts_from_capstone"] == [
        "exp2862",
        "exp2870",
        "exp2871",
    ]
    assert artifact["headline_eligible_rows_from_capstone"] == ["FoVer", "HaluEval/FEVER"]
    assert len(artifact["top_3_next_actions"]) == 3
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == 1.75
    assert artifact["field_principles"]["duration_s"].endswith("no sleep padding.")
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2873_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2873: absent .271 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.270\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.272"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.271"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-22"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2873_blocks_on_wrong_roadmap_or_missing_capstone(tmp_path: Path) -> None:
    """REQ-REPORT-2873: missing source evidence produces a blocked terminal artifact."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.270\n  completed: '2026-05-22'\n",
        active_roadmap=(
            'milestone: "2026.05.271"\n'
            'milestone_doc: "openspec/change-proposals/stale-roadmap.md"\n'
        ),
    )
    (tmp_path / exp.CAPSTONE_SOURCE).unlink()

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert "research-complete.yaml does not archive 2026.05.271" in artifact["blocked_reasons"]
    assert "roadmap milestone is not 2026.05.272" in artifact["blocked_reasons"]
    assert (
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        in artifact["blocked_reasons"]
    )
    assert "capstone source missing or invalid" in artifact["blocked_reasons"]
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["adversarially_flagged_artifacts_from_capstone"] == []
    assert artifact["headline_eligible_rows_from_capstone"] == []
    assert artifact["top_3_next_actions"] == []


def test_req_report_2873_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2873: helper readers tolerate missing or malformed inputs."""

    assert exp._read_text(tmp_path / "missing.txt") == ""

    assert exp._read_json_mapping(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json", encoding="utf-8")
    assert exp._read_json_mapping(bad_json) == {}
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp._read_json_mapping(array_json) == {}

    assert exp._load_yaml_mapping(tmp_path / "missing.yaml") == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [unterminated\n", encoding="utf-8")
    assert exp._load_yaml_mapping(bad_yaml) == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp._load_yaml_mapping(list_yaml) == {}

    complete = (
        "milestones:\n"
        "- id: 2026.05.271\n"
        "  tasks: []\n"
        "- id: 2026.05.271\n"
        "  completed: '2026-05-22'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False


def test_req_report_2873_appends_to_empty_complete_file(tmp_path: Path) -> None:
    """REQ-REPORT-2873: empty archive files get a valid milestones root."""

    _write_common_files(
        tmp_path,
        complete_text="",
        next_roadmap=(
            'milestone: "2026.05.272"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.25))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))

    assert artifact["archive_appended_this_run"] is True
    assert complete["milestones"][0]["id"] == "2026.05.271"
