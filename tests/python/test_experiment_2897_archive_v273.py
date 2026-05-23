"""Tests for Exp 2897 milestone .273 archive and .274 activation.

Spec: REQ-REPORT-2897, SCENARIO-REPORT-2897.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_273_archive_274_activation as exp


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
                    "complete: .273 capstone synthesized; paper_ready=true; "
                    "clean_artifacts=7; flagged_artifacts=2; blocked_artifacts=0; "
                    "missing_artifacts=0; pilot_only_artifacts=1"
                ),
                "milestone": "2026.05.273",
                "paper_ready": True,
                "clean_artifacts": [
                    "exp2886",
                    "exp2887",
                    "exp2890",
                    "exp2892",
                    "exp2893",
                    "exp2894",
                    "exp2895",
                ],
                "flagged_artifacts": ["exp2885", "exp2889"],
                "blocked_artifacts": [],
                "missing_artifacts": [],
                "pilot_only_artifacts": ["exp2891"],
            }
        ),
        encoding="utf-8",
    )


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.274"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
    ),
    next_roadmap: str | None = None,
    write_capstone: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(active_roadmap, encoding="utf-8")
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# untouched by Exp 2897\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root)


def test_scenario_report_2897_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2897: existing .273 archive confirms active .274 roadmap."""

    complete_text = """milestones:
- id: 2026.05.272
  completed: '2026-05-22'
  tasks: []
- id: 2026.05.273
  title: Clean Telemetry + Fast/Slow Memory + Constraint Benchmark Expansion
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-23'
  tasks:
  - id: exp2896
    deliverable: results/experiment_2896_capstone_v273.json
    result: OK (conductor)
"""
    _write_common_files(tmp_path, complete_text=complete_text)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.25))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.05.273"
    assert artifact["activated_milestone"] == "2026.05.274"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2896_capstone_v273.json"
    assert artifact["paper_ready_from_capstone"] is True
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2886",
        "exp2887",
        "exp2890",
        "exp2892",
        "exp2893",
        "exp2894",
        "exp2895",
    ]
    assert artifact["flagged_artifacts_from_capstone"] == ["exp2885", "exp2889"]
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["pilot_only_artifacts_from_capstone"] == ["exp2891"]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == 1.25
    assert artifact["field_principles"]["honest_verdict"].startswith("Self-declared")
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2897_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2897: absent .273 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.272\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.274"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.273"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-23"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2897_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2897: missing capstone exits with blocked_capstone_missing."""

    complete_text = "milestones:\n- id: 2026.05.272\n  tasks: []\n"
    _write_common_files(
        tmp_path,
        complete_text=complete_text,
        active_roadmap='milestone: "2026.05.274"\n',
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"] == "blocked_capstone_missing"
    assert artifact["archived_milestone"] == "2026.05.273"
    assert artifact["activated_milestone"] == "2026.05.274"
    assert artifact["capstone_source"] == "results/experiment_2896_capstone_v273.json"
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["flagged_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["pilot_only_artifacts_from_capstone"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.1
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2897_blocks_on_wrong_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-2897: wrong activation milestone is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=(
            "milestones:\n"
            "- id: 2026.05.273\n"
            "  completed: '2026-05-23'\n"
            "  tasks: []\n"
        ),
        active_roadmap='milestone: "2026.05.273"\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert "roadmap milestone is not 2026.05.274" in artifact["blocked_reasons"]
    assert artifact["archive"]["ready_after_run"] is True


def test_req_report_2897_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2897: helper readers tolerate missing or malformed inputs."""

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

    assert exp._as_str_list(["exp1", 2]) == ["exp1", "2"]
    assert exp._as_str_list("exp1") == []

    complete = (
        "milestones:\n"
        "- id: 2026.05.273\n"
        "  tasks: []\n"
        "- id: 2026.05.273\n"
        "  completed: '2026-05-23'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        roadmap={"milestone_matches": True},
        summary={
            "paper_ready_from_capstone": False,
            "clean_artifacts_from_capstone": [],
            "flagged_artifacts_from_capstone": [],
            "blocked_artifacts_from_capstone": [],
            "missing_artifacts_from_capstone": [],
            "pilot_only_artifacts_from_capstone": [],
        },
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.273"]
