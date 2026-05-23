"""Tests for Exp 2937 milestone .276 archive and .277 activation.

Spec refs: REQ-REPORT-2937, SCENARIO-REPORT-2937.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_276_archive_277_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "experiment_2936_capstone_v276",
        "honest_verdict": (
            "complete: milestone=2026.05.276; paper_ready=true; "
            "hardware_speedup_claim_eligible=true; "
            "gate_mate_speedup_claim_eligible=false; "
            "evidence_boundary_repaired=true; "
            "sota_structured_generation_clean=false; "
            "fr11_self_learning_clean=true; clean=6; flagged=2; "
            "blocked=3; missing=1; projection_only=1"
        ),
        "milestone": "2026.05.276",
        "paper_ready": True,
        "hardware_speedup_claim_eligible": True,
        "gate_mate_speedup_claim_eligible": False,
        "evidence_boundary_repaired": True,
        "sota_structured_generation_clean": False,
        "fr11_self_learning_clean": True,
        "artifact_classification_counts": {
            "blocked": 3,
            "clean": 6,
            "diagnostic_only": 0,
            "flagged": 2,
            "missing": 1,
            "pilot_only": 0,
            "projection_only": 1,
        },
        "clean_artifacts": ["exp2923", "exp2924", "exp2925", "exp2926", "exp2933", "exp2935"],
        "flagged_artifacts": ["exp2932", "exp2934"],
        "blocked_artifacts": ["exp2927", "exp2929", "exp2931"],
        "missing_artifacts": ["exp2928"],
        "pilot_only_artifacts": [],
        "projection_only_artifacts": ["exp2930"],
        "diagnostic_only_artifacts": [],
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.277"\n'
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
    conductor.write_text("# untouched by Exp 2937\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root)


def test_req_report_2937_spec_is_declared() -> None:
    """REQ-REPORT-2937: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2937" in spec
    assert "SCENARIO-REPORT-2937" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_2937_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2937: existing .276 archive confirms active .277 roadmap."""

    complete_text = """milestones:
- id: 2026.05.275
  completed: '2026-05-23'
  tasks: []
- id: 2026.05.276
  title: Evidence Boundary Repair + GateMate Bring-Up + Solver-Grounded Self-Learning
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-23'
  tasks:
  - id: exp2936
    deliverable: results/experiment_2936_capstone_v276.json
    result: OK (conductor)
"""
    _write_common_files(tmp_path, complete_text=complete_text)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.5))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == (
        "complete: archive_ready=true; archived_milestone=2026.05.276; "
        "activated_milestone=2026.05.277"
    )
    assert artifact["field_principles"]["honest_verdict"] == (
        "Self-declared terminal state per Verdict Terminal-Prefix Discipline."
    )
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.276"
    assert artifact["activated_milestone"] == "2026.05.277"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2936_capstone_v276.json"
    assert artifact["capstone_honest_verdict"].startswith("complete: milestone=2026.05.276")
    assert artifact["paper_ready_from_capstone"] is True
    assert artifact["hardware_speedup_claim_eligible_from_capstone"] is True
    assert artifact["gate_mate_speedup_claim_eligible_from_capstone"] is False
    assert artifact["evidence_boundary_repaired_from_capstone"] is True
    assert artifact["sota_structured_generation_clean_from_capstone"] is False
    assert artifact["fr11_self_learning_clean_from_capstone"] is True
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2923",
        "exp2924",
        "exp2925",
        "exp2926",
        "exp2933",
        "exp2935",
    ]
    assert artifact["flagged_artifacts_from_capstone"] == ["exp2932", "exp2934"]
    assert artifact["blocked_artifacts_from_capstone"] == ["exp2927", "exp2929", "exp2931"]
    assert artifact["missing_artifacts_from_capstone"] == ["exp2928"]
    assert artifact["pilot_only_artifacts_from_capstone"] == []
    assert artifact["projection_only_artifacts_from_capstone"] == ["exp2930"]
    assert artifact["diagnostic_only_artifacts_from_capstone"] == []
    assert artifact["artifact_classification_counts_from_capstone"] == {
        "blocked": 3,
        "clean": 6,
        "diagnostic_only": 0,
        "flagged": 2,
        "missing": 1,
        "pilot_only": 0,
        "projection_only": 1,
    }
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == 1.5
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2937_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2937: absent .276 archive is appended without touching the roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.275\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.277"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.276"]

    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-23"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2937_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2937: missing capstone exits with blocked_capstone_missing."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.275\n  tasks: []\n",
        active_roadmap='milestone: "2026.05.277"\n',
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"] == "blocked_capstone_missing"
    assert artifact["archive_ready"] is False
    assert artifact["archived_milestone"] == "2026.05.276"
    assert artifact["activated_milestone"] == "2026.05.277"
    assert artifact["capstone_source"] == "results/experiment_2936_capstone_v276.json"
    assert artifact["capstone_honest_verdict"] == ""
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["flagged_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["pilot_only_artifacts_from_capstone"] == []
    assert artifact["artifact_classification_counts_from_capstone"] == {}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.1
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2937_blocks_on_wrong_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-2937: wrong activation milestone is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=("milestones:\n- id: 2026.05.276\n  completed: '2026-05-23'\n  tasks: []\n"),
        active_roadmap='milestone: "2026.05.276"\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.277" in artifact["blocked_reasons"]


def test_req_report_2937_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2937: helper readers tolerate missing or malformed inputs."""

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
    assert exp._as_int_mapping({"clean": 1, "flagged": True, "missing": "1"}) == {"clean": 1}
    assert exp._as_int_mapping(["not", "mapping"]) == {}

    complete = (
        "milestones:\n"
        "- id: 2026.05.276\n"
        "  tasks: []\n"
        "- id: 2026.05.276\n"
        "  completed: '2026-05-23'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        roadmap={"milestone_matches": True},
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.276"]
