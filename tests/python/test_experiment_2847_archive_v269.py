"""Tests for Exp 2847 milestone .269 archive and .270 activation.

Spec: REQ-REPORT-2847, SCENARIO-REPORT-2847.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_269_archive_270_activation as exp


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
                    "complete: .269 capstone synthesized; sota_runtime_ready=true; "
                    "paper_ready=false"
                ),
                "milestone": "2026.05.269",
                "sota_runtime_ready": True,
                "paper_ready": False,
                "blocked_artifacts": ["exp2838", "exp2839", "exp2840", "exp2844"],
                "missing_artifacts": ["exp2842", "exp2845"],
                "top_3_next_actions": [
                    "Clear adversarial verification flags on Exp 2836 and Exp 2837.",
                    "Materialize MBPP, HumanEval, and TruthfulQA local datasets.",
                    "Implement or select the live recurrence backend for Exp 2844.",
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
        'milestone: "2026.05.270"\n'
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
    conductor.write_text("# untouched by Exp 2847\n", encoding="utf-8")
    _write_capstone(root)


def test_scenario_report_2847_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2847: existing .269 archive confirms active .270 roadmap."""

    complete_text = """milestones:
- id: 2026.05.268
  completed: '2026-05-22'
  tasks: []
- id: 2026.05.269
  title: SOTA Runtime Gate + Multi-Corpus Evidence + LoopUS Self-Learning
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-22'
  tasks:
  - id: exp2846-capstone-v269
    deliverable: results/experiment_2846_capstone_v269.json
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
    assert artifact["archived_milestone"] == "2026.05.269"
    assert artifact["activated_milestone"] == "2026.05.270"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2846_capstone_v269.json"
    assert artifact["sota_runtime_ready_from_capstone"] is True
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["blocked_artifacts_from_capstone"] == [
        "exp2838",
        "exp2839",
        "exp2840",
        "exp2844",
    ]
    assert artifact["missing_artifacts_from_capstone"] == ["exp2842", "exp2845"]
    assert len(artifact["top_3_next_actions"]) == 3
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == 1.25
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2847_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2847: absent .269 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.268\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.270"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.269"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-22"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2847_blocks_on_wrong_roadmap_or_missing_capstone(tmp_path: Path) -> None:
    """REQ-REPORT-2847: missing source evidence produces a blocked terminal artifact."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.269\n  completed: '2026-05-22'\n",
        active_roadmap=(
            'milestone: "2026.05.269"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    (tmp_path / exp.CAPSTONE_SOURCE).unlink()

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert "roadmap milestone is not 2026.05.270" in artifact["blocked_reasons"]
    assert "capstone source missing or invalid" in artifact["blocked_reasons"]
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["top_3_next_actions"] == []


def test_req_report_2847_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2847: helper readers tolerate missing or malformed inputs."""

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
        "- id: 2026.05.269\n"
        "  tasks: []\n"
        "- id: 2026.05.269\n"
        "  completed: '2026-05-22'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False
