"""Tests for Exp 3040 archive .284 and .285 handoff.

Spec refs: REQ-REPORT-3040, SCENARIO-REPORT-3040.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v284_activate_v285_3040 as mod


REQUIRED_FIELDS = {
    "archive_v284_activate_v285_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "carry_forward_blockers",
    "next_milestone",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_payload() -> dict[str, Any]:
    rows = [
        {"experiment_id": "exp3026", "status": "projection_only"},
        {"experiment_id": "exp3027", "status": "flagged"},
        {"experiment_id": "exp3028", "status": "flagged"},
        {"experiment_id": "exp3029", "status": "flagged", "repair_claim_status": "bounded"},
        {"experiment_id": "exp3030", "status": "clean"},
        {"experiment_id": "exp3031", "status": "flagged"},
        {"experiment_id": "exp3032", "status": "clean"},
        {
            "experiment_id": "exp3033",
            "status": "clean",
            "fr11_self_learning_promotable": True,
        },
        {
            "experiment_id": "exp3034",
            "status": "blocked",
            "gatemate_output_contract_ready": False,
        },
        {"experiment_id": "exp3035", "status": "gated_skipped"},
        {
            "experiment_id": "exp3036",
            "status": "gated_skipped",
            "host_visible_output_observed": False,
        },
        {"experiment_id": "exp3037", "status": "gated_skipped", "ssqa_gate_status": "gate_skipped"},
        {"experiment_id": "exp3038", "status": "clean"},
        {"experiment_id": "exp3039", "status": "missing"},
    ]
    return {
        "schema": "carnot.cross_corpus_matrix.v18_284_task_coverage.v1",
        "milestone": "2026.05.284",
        "matrix_v18_ready": True,
        "rows_total": 14,
        "clean": 4,
        "flagged": 4,
        "blocked": 1,
        "gated_skipped": 3,
        "projection_only": 1,
        "pilot_only": 0,
        "missing": 1,
        "retired": 0,
        "matrix_rows": rows,
        "honest_verdict": (
            "complete: matrix_v18_ready=true; rows_total=14; clean=4; flagged=4; "
            "blocked=1; gated_skipped=3; projection_only=1; pilot_only=0; missing=1; retired=0"
        ),
    }


def _capstone_payload(*, capstone_ready: bool = True, paper_ready: bool = False) -> dict[str, Any]:
    return {
        "schema": "carnot.milestone_capstone.v284_aggregation.v1",
        "milestone": "2026.05.284",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "repair_claim_status": "bounded",
        "fr11_self_learning_status": "controller_only_promotable",
        "gatemate_status": "blocked_pinout_missing_bounded",
        "ssqa_status": "gate_skipped_bounded_no_performance_claim",
        "matrix_v18_summary": {
            "matrix_v18_ready": True,
            "rows_total": 14,
            "clean": 4,
            "flagged": 4,
            "blocked": 1,
            "gated_skipped": 3,
            "missing": 1,
            "status_by_experiment": {"exp3039": "missing"},
        },
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; repair_claim_status=bounded; "
            "fr11_self_learning_status=controller_only_promotable; "
            "gatemate_status=blocked_pinout_missing_bounded"
        ),
    }


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.285",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3040-archive-v284-activate-v285\n"
        "    deliverable: results/experiment_3040_archive_v284_activate_v285.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "GateMate Output Unblock + Repair Flag Hygiene"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.285",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V18_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V284_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.285\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3039 and .285\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3039 and .285\n")


def test_req_report_3040_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3040: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3040" in spec
    assert "SCENARIO-REPORT-3040" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3040_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3040: staged .285 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.284")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["field"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v284_activate_v285_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["next_milestone"] == "2026.05.285"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v284_activate_v285_ready=true")

    assert artifact["status_summary_284"] == {
        "clean": 4,
        "flagged": 4,
        "bounded": {
            "repair_claim_status": "bounded",
            "fr11_self_learning_status": "controller_only_promotable",
            "gatemate_status": "blocked_pinout_missing_bounded",
        },
        "blocked": 1,
        "gated_skipped": 3,
        "missing": 1,
        "paper_ready": False,
    }
    assert blockers["repair_claim_status"]["value"] == "bounded"
    assert blockers["fr11_self_learning_status"]["value"] == "controller_only_promotable"
    assert blockers["gatemate_status"]["value"] == "blocked_pinout_missing_bounded"
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["milestone_doc_matches"] is True
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3040-archive-v284-activate-v285"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "source": "checked_in_artifacts",
    }


def test_req_report_3040_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3040: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v284_activate_v285_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.285"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3040_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3040: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v284_activate_v285_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3040_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3040: roadmap milestone/doc/tasks must match the .285 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.284")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v284_activate_v285_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.285" in artifact["blocked_reasons"]


def test_req_report_3040_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3040: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v284_activate_v285_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V284_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V284_REL_PATH
    )


def test_req_report_3040_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3040: helper edges preserve malformed and unusual states."""

    malformed_json = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed_yaml = tmp_path / "bad.yaml"
    list_yaml = tmp_path / "list.yaml"
    malformed_json.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1, 2, 3]\n", encoding="utf-8")
    malformed_yaml.write_text(": bad\n", encoding="utf-8")
    list_yaml.write_text("- x\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod.read_yaml_mapping(malformed_yaml) == {}
    assert mod.read_yaml_mapping(list_yaml) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert mod._int_or(True, 7) == 7
    assert mod._int_or("5", 7) == 5
    assert mod._int_or("x", 7) == 7
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={
            "source_present": False,
            "milestone_matches": False,
            "milestone_doc_matches": False,
            "non_empty_tasks": False,
        },
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.285",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
