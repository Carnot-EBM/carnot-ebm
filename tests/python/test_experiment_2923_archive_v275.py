"""Tests for Exp 2923 milestone .275 archive and .276 activation.

Spec: REQ-REPORT-2923, SCENARIO-REPORT-2923.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_275_archive_276_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_capstone(root: Path, *, recommended_key: str = "top_3_next_actions") -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "honest_verdict": (
            "complete: .275 capstone synthesized; paper_ready=true; "
            "hardware_speedup_claim_eligible=true; clean_artifacts=6; "
            "flagged_artifacts=3; blocked_artifacts=1; missing_artifacts=1; "
            "diagnostic_only_artifacts=2"
        ),
        "milestone": "2026.05.275",
        "paper_ready": True,
        "hardware_speedup_claim_eligible": True,
        "clean_artifacts": [
            "exp2909",
            "exp2910",
            "exp2912",
            "exp2913",
            "exp2918",
            "exp2920",
        ],
        "flagged_artifacts": ["exp2911", "exp2919", "exp2921"],
        "blocked_artifacts": ["exp2914"],
        "missing_artifacts": ["exp2915"],
        "diagnostic_only_artifacts": ["exp2916", "exp2917"],
        recommended_key: [
            "Repair matrix v9 aggregation-only adversarial-verify metadata.",
            "Clear code hallucination taxonomy methodology flags.",
            "Rerun ConstraintBench mini with non-tautological metrics.",
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.276"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
    ),
    next_roadmap: str | None = None,
    write_capstone: bool = True,
    recommended_key: str = "top_3_next_actions",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(active_roadmap, encoding="utf-8")
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# untouched by Exp 2923\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root, recommended_key=recommended_key)


def test_scenario_report_2923_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2923: existing .275 archive confirms active .276 roadmap."""

    complete_text = """milestones:
- id: 2026.05.274
  completed: '2026-05-23'
  tasks: []
- id: 2026.05.275
  title: Hardware Baselines + Code Hallucination Repair + Verifier-Grounded Self-Learning
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-23'
  tasks:
  - id: exp2922
    deliverable: results/experiment_2922_capstone_v275.json
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
    assert artifact["honest_verdict"] == (
        "complete: archive_ready=true; archived_milestone=2026.05.275; "
        "activated_milestone=2026.05.276"
    )
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.275"
    assert artifact["activated_milestone"] == "2026.05.276"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2922_capstone_v275.json"
    assert artifact["paper_ready_from_capstone"] is True
    assert artifact["hardware_speedup_claim_eligible_from_capstone"] is True
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2909",
        "exp2910",
        "exp2912",
        "exp2913",
        "exp2918",
        "exp2920",
    ]
    assert artifact["flagged_artifacts_from_capstone"] == ["exp2911", "exp2919", "exp2921"]
    assert artifact["blocked_artifacts_from_capstone"] == ["exp2914"]
    assert artifact["missing_artifacts_from_capstone"] == ["exp2915"]
    assert artifact["diagnostic_artifacts_from_capstone"] == ["exp2916", "exp2917"]
    assert artifact["recommended_next_actions"] == [
        "Repair matrix v9 aggregation-only adversarial-verify metadata.",
        "Clear code hallucination taxonomy methodology flags.",
        "Rerun ConstraintBench mini with non-tautological metrics.",
    ]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == 1.25
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2923_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2923: absent .275 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.274\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.276"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
        recommended_key="recommended_next_actions",
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.275"]

    assert artifact["honest_verdict"] == (
        "complete: archive_ready=true; archived_milestone=2026.05.275; "
        "activated_milestone=2026.05.276"
    )
    assert artifact["archive_ready"] is True
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-23"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2923_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2923: missing capstone exits with blocked_capstone_missing."""

    complete_text = "milestones:\n- id: 2026.05.274\n  tasks: []\n"
    _write_common_files(
        tmp_path,
        complete_text=complete_text,
        active_roadmap='milestone: "2026.05.276"\n',
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"] == "blocked_capstone_missing"
    assert artifact["archive_ready"] is False
    assert artifact["archived_milestone"] == "2026.05.275"
    assert artifact["activated_milestone"] == "2026.05.276"
    assert artifact["capstone_source"] == "results/experiment_2922_capstone_v275.json"
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["hardware_speedup_claim_eligible_from_capstone"] is False
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["flagged_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["diagnostic_artifacts_from_capstone"] == []
    assert artifact["recommended_next_actions"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.1
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2923_blocks_on_wrong_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-2923: wrong activation milestone is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=(
            "milestones:\n"
            "- id: 2026.05.275\n"
            "  completed: '2026-05-23'\n"
            "  tasks: []\n"
        ),
        active_roadmap='milestone: "2026.05.275"\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.276" in artifact["blocked_reasons"]


def test_req_report_2923_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2923: helper readers tolerate missing or malformed inputs."""

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
        "- id: 2026.05.275\n"
        "  tasks: []\n"
        "- id: 2026.05.275\n"
        "  completed: '2026-05-23'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False
    assert exp._recommended_next_actions({"recommended_next_actions": ["a"]}) == ["a"]
    assert exp._recommended_next_actions({"top_3_next_actions": ["b"]}) == ["b"]
    assert exp._recommended_next_actions({"recommended_next_actions": "a"}) == []

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        roadmap={"milestone_matches": True},
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.275"]
