"""Tests for Exp 2885 milestone .272 archive and .273 activation.

Spec: REQ-REPORT-2885, SCENARIO-REPORT-2885.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_272_archive_273_activation as exp


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
                    "complete: .272 capstone synthesized; paper_ready=true; "
                    "clean_artifacts=7; flagged_artifacts=2; blocked_artifacts=1; "
                    "missing_artifacts=0; pilot_only_artifacts=1"
                ),
                "milestone": "2026.05.272",
                "paper_ready": True,
                "clean_artifacts": [
                    "exp2873",
                    "exp2874",
                    "exp2876",
                    "exp2877",
                    "exp2878",
                    "exp2880",
                    "exp2881",
                ],
                "flagged_artifacts": ["exp2875", "exp2882"],
                "blocked_artifacts": ["exp2883"],
                "missing_artifacts": [],
                "pilot_only_artifacts": ["exp2879"],
                "paper_v6_safe_claims": [
                    "FoVer and HaluEval/FEVER remain headline-eligible.",
                    "FR-11 RecMem trigger prototype is clean.",
                ],
                "paper_v6_forbidden_claims": [
                    "Do not cite MBPP or HumanEval as headline benchmark rows.",
                    "Do not claim THRML hardware acceleration.",
                ],
                "top_3_next_actions": [
                    "Re-run Exp 2875 with adversarial-clean duration.",
                    "Repair Exp 2882 with non-tautological metrics.",
                    "Promote MBPP/HumanEval only when clean evidence exists.",
                    "Ignored fourth action.",
                ],
                "field_principles": {"paper_ready": "Copied source principle."},
            }
        ),
        encoding="utf-8",
    )


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.273"\n'
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
    conductor.write_text("# untouched by Exp 2885\n", encoding="utf-8")
    _write_capstone(root)


def test_scenario_report_2885_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2885: existing .272 archive confirms active .273 roadmap."""

    complete_text = """milestones:
- id: 2026.05.271
  completed: '2026-05-22'
  tasks: []
- id: 2026.05.272
  title: Clean SOTA Corrigenda + RecMem Self-Learning + Evidence Expansion
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-22'
  tasks:
  - id: exp2884
    deliverable: results/experiment_2884_capstone_v272.json
    result: OK (conductor)
"""
    _write_common_files(tmp_path, complete_text=complete_text)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.75))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.05.272"
    assert artifact["activated_milestone"] == "2026.05.273"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2884_capstone_v272.json"
    assert artifact["paper_ready_from_capstone"] is True
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2873",
        "exp2874",
        "exp2876",
        "exp2877",
        "exp2878",
        "exp2880",
        "exp2881",
    ]
    assert artifact["flagged_artifacts_from_capstone"] == ["exp2875", "exp2882"]
    assert artifact["blocked_artifacts_from_capstone"] == ["exp2883"]
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["pilot_only_artifacts_from_capstone"] == ["exp2879"]
    assert artifact["paper_v6_safe_claims_from_capstone"] == [
        "FoVer and HaluEval/FEVER remain headline-eligible.",
        "FR-11 RecMem trigger prototype is clean.",
    ]
    assert artifact["paper_v6_forbidden_claims_from_capstone"] == [
        "Do not cite MBPP or HumanEval as headline benchmark rows.",
        "Do not claim THRML hardware acceleration.",
    ]
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


def test_req_report_2885_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2885: absent .272 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.271\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.273"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.272"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-22"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2885_blocks_on_wrong_roadmap_or_missing_capstone(tmp_path: Path) -> None:
    """REQ-REPORT-2885: missing source evidence produces a blocked terminal artifact."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.271\n  completed: '2026-05-22'\n",
        active_roadmap=(
            'milestone: "2026.05.272"\n'
            'milestone_doc: "openspec/change-proposals/stale-roadmap.md"\n'
        ),
    )
    (tmp_path / exp.CAPSTONE_SOURCE).unlink()

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert "research-complete.yaml does not archive 2026.05.272" in artifact["blocked_reasons"]
    assert "roadmap milestone is not 2026.05.273" in artifact["blocked_reasons"]
    assert (
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        in artifact["blocked_reasons"]
    )
    assert "capstone source missing or invalid" in artifact["blocked_reasons"]
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["flagged_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["pilot_only_artifacts_from_capstone"] == []
    assert artifact["paper_v6_safe_claims_from_capstone"] == []
    assert artifact["paper_v6_forbidden_claims_from_capstone"] == []
    assert artifact["top_3_next_actions"] == []


def test_req_report_2885_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2885: helper readers tolerate missing or malformed inputs."""

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
        "- id: 2026.05.272\n"
        "  tasks: []\n"
        "- id: 2026.05.272\n"
        "  completed: '2026-05-22'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False


def test_req_report_2885_appends_to_empty_complete_file(tmp_path: Path) -> None:
    """REQ-REPORT-2885: empty archive files get a valid milestones root."""

    _write_common_files(
        tmp_path,
        complete_text="",
        next_roadmap=(
            'milestone: "2026.05.273"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.25))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))

    assert artifact["archive_appended_this_run"] is True
    assert complete["milestones"][0]["id"] == "2026.05.272"
