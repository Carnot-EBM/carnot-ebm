"""Tests for Exp 3149 archive .292 and .293 handoff.

Spec refs: REQ-REPORT-3149, SCENARIO-REPORT-3149.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v292_activate_v293_3149 as mod


REQUIRED_FIELDS = {
    "archive_v292_activate_v293_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "blocker_delta_from_v25",
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
    return {
        "artifact": "experiment_3147_cross_corpus_matrix_v26",
        "matrix_v26_ready": True,
        "publication_blocker_count": 55,
        "blocker_delta_from_v25": 9,
        "honest_verdict": (
            "complete: matrix_v26_ready=true; rows_total=124; "
            "publication_blocker_count=55; blocker_delta_from_v25=9"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 55,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3148_capstone_v292",
        "milestone": "2026.05.292",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "blocker_delta_from_v25": 9,
        "next_top_gap": "false_accept_recovery_corrigendum_repair_gate",
        "false_accept_recovery_status": (
            "blocked_by_adversarial_corrigendum_false_accept_0.0_known_rows_blocked"
        ),
        "live_verifier_status": "flagged",
        "repair_gate_status": "blocked_repair_gate_state_blocked_other_blockers_6_disqualifiers_6",
        "repair_ladder_status": "gated_skipped_missing_artifact",
        "fr11_self_learning_status": (
            "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667"
        ),
        "ebt_arm_status": "projection_only_no_live_integration_blockers_6",
        "kan_status": "bounded_monitor_records_2_no_deployed_verifier",
        "sampler_hardware_status": (
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
        ),
        "source_artifacts": [
            {"path": mod.MATRIX_V26_REL_PATH.as_posix(), "role": "matrix_v26_authority"},
            {"path": "results/experiment_3139_live_sota_verifier_rerun_v7.json"},
        ],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; "
            "publication_blocker_count=55; blocker_delta_from_v25=9; "
            "next_top_gap=false_accept_recovery_corrigendum_repair_gate"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.293",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3149-archive-v292-activate-v293\n"
        "    deliverable: results/experiment_3149_archive_v292_activate_v293.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Verifier Evidence Corrigendum"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.293",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V26_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V292_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.293\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3148 and .293\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3148 and .293\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3148\n")


def test_req_report_3149_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3149: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3149" in spec
    assert "SCENARIO-REPORT-3149" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3149_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3149: staged .293 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.292")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v292_activate_v293_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 55
    assert artifact["blocker_delta_from_v25"] == 9
    assert artifact["next_milestone"] == "2026.05.293"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v292_activate_v293_ready=true")
    assert artifact["status_summary_292"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v26_ready": True,
        "publication_blocker_count": 55,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "blocker_delta_from_v25": 9,
        "blocker_delta_source": "capstone_blocker_delta_from_v25",
        "next_top_gap": "false_accept_recovery_corrigendum_repair_gate",
        "false_accept_recovery_status": (
            "blocked_by_adversarial_corrigendum_false_accept_0.0_known_rows_blocked"
        ),
        "live_verifier_status": "flagged",
        "repair_gate_status": "blocked_repair_gate_state_blocked_other_blockers_6_disqualifiers_6",
        "repair_ladder_status": "gated_skipped_missing_artifact",
        "fr11_self_learning_status": (
            "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667"
        ),
        "ebt_arm_status": "projection_only_no_live_integration_blockers_6",
        "kan_status": "bounded_monitor_records_2_no_deployed_verifier",
        "sampler_hardware_status": (
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
        ),
        "source_artifacts": _capstone_payload()["source_artifacts"],
    }
    assert set(blockers) == {
        "publication_blockers_55",
        "flagged_live_verifier_evidence",
        "repair_gate_blocked_other",
        "repair_ladder_missing_gated",
        "fr11_controller_only_memory",
        "ebt_arm_projection_only",
        "kan_bounded_only",
        "no_authenticated_hardware_speedup",
    }
    assert blockers["publication_blockers_55"]["value"] == 55
    assert blockers["flagged_live_verifier_evidence"]["value"] == "flagged"
    assert "blocked_other" in blockers["repair_gate_blocked_other"]["value"]
    assert blockers["repair_ladder_missing_gated"]["value"] == "gated_skipped_missing_artifact"
    assert blockers["fr11_controller_only_memory"]["value"] == (
        "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667"
    )
    assert blockers["ebt_arm_projection_only"]["value"] == "projection_only_no_live_integration_blockers_6"
    assert blockers["kan_bounded_only"]["value"] == "bounded_monitor_records_2_no_deployed_verifier"
    assert blockers["no_authenticated_hardware_speedup"]["value"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
    )
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3149-archive-v292-activate-v293"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["research_roadmap_next_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["ops_status_updated"] is False
    assert artifact["ops_changelog_updated"] is False
    assert artifact["traceability_updated"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "source": "checked_in_artifacts",
        "live_model_calls": 0,
        "hardware_commands_run": [],
    }


def test_req_report_3149_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3149: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v292_activate_v293_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.293"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3149_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3149: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v292_activate_v293_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3149_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3149: roadmap milestone/doc/tasks must match the .293 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.292")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v292_activate_v293_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.293" in artifact["blocked_reasons"]


def test_req_report_3149_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3149: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v292_activate_v293_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V292_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V292_REL_PATH
    )


def test_req_report_3149_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3149: helper edges preserve malformed and unusual states."""

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
    assert mod._int_field({"count": 7}, "count") == (7, "capstone_count")
    assert mod._int_field({"count": True}, "count") == (0, "missing")
    assert mod._int_field({"count": "5"}, "count") == (0, "missing")
    assert mod._text_field({"x": "value"}, "x") == "value"
    assert mod._text_field({"x": None}, "x") == ""
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._publication_blocker_count({}) == (0, "missing")
    assert mod._blocker_delta_from_v25({}) == (0, "missing")
    assert mod._next_milestone({"observed_milestone": "2026.05.294"}) == "2026.05.294"
    assert mod._next_milestone({}) == "2026.05.293"
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={"source_present": False, "milestone_matches": False},
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.293",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
