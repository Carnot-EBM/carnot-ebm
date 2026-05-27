"""Tests for Exp 3177 archive .294 and .295 handoff.

Spec refs: REQ-REPORT-3177, SCENARIO-REPORT-3177.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v294_activate_v295_3177 as mod


REQUIRED_FIELDS = {
    "archive_v294_activate_v295_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "blocker_delta_from_v27",
    "next_milestone",
    "carry_forward_blockers",
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


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    blocker_count: int = 73,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3176_capstone_v294",
        "milestone": "2026.05.294",
        "capstone_v294_ready": capstone_ready,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": blocker_count,
        "blocker_delta_from_v27": 8,
        "missing_artifact_count": 1,
        "verifier_status": "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only",
        "repair_gate_status": "blocked_flagged_verifier",
        "repair_ladder_status": (
            "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
        ),
        "fr11_self_learning_status": (
            "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
        ),
        "ebcn_kan_status": (
            "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
        ),
        "hardware_sampler_status": (
            "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made"
        ),
        "next_top_gap": "clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock",
        "honest_verdict": (
            "complete: capstone_v294_ready=true; capstone_ready=true; paper_ready=false; "
            "publication_blocker_count=73; blocker_delta_from_v27=8; missing_artifact_count=1"
        ),
    }


def _matrix_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3175_cross_corpus_matrix_v28",
        "matrix_v28_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 73,
        "blocker_delta_from_v27": 8,
        "missing_artifacts": [
            {
                "experiment_id": "exp3141",
                "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
                "reason": "carried_forward_unresolved_missing_artifact_from_v27",
            }
        ],
        "verifier_status": "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only",
        "repair_status": "blocked_flagged_verifier_repair_ladder_gated_skipped",
        "fr11_status": (
            "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
        ),
        "sidecar_status": (
            "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
        ),
        "hardware_status": "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made",
        "honest_verdict": "complete: matrix_v28_ready=true",
    }


def _replay_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3165_live_sota_authenticity_replay_v2",
        "preflight_passed": False,
        "live_call_count": 0,
        "blocked_reason": "CUDA/GPU substrate unavailable for mandated GGUF replay smoke",
        "honest_verdict": (
            "blocked_gpu_substrate: preflight_passed=false; live_call_count=0; "
            "detail=CUDA/GPU substrate unavailable for mandated GGUF replay smoke"
        ),
    }


def _verifier_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3167_clean_live_sota_verifier_rerun_v9",
        "clean_live_verifier_rerun_v9_ready": True,
        "gated_skip": True,
        "flagged_adversarial": True,
        "controlled_invariance_passed": False,
        "false_accept_gate_passed": False,
        "headline_claim_allowed": False,
        "live_call_count": 0,
        "honest_verdict": "complete: gated skip due to exp3165 preflight_passed=false",
    }


def _repair_gate_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3168_repair_gate_decision_v3",
        "repair_gate_decision_v3_ready": True,
        "repair_gate_state": "blocked_flagged_verifier",
        "repair_blockers": ["flagged_adversarial=true"],
        "honest_verdict": "blocked_flagged_verifier: flagged_adversarial=true",
    }


def _repair_ladder_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3169_repair_ladder_materializer_v4",
        "repair_ladder_materializer_v4_ready": True,
        "gated_skip": True,
        "gate_state": "blocked_flagged_verifier",
        "headline_repair_claim_allowed": False,
        "live_call_count": 0,
        "repair_attempt_count": 0,
        "honest_verdict": "blocked_repair_gate: repair gate blocked",
    }


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.295",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3177-archive-v294-activate-v295\n"
        "    deliverable: results/experiment_3177_archive_v294_activate_v295.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Receipt-Backed Live SOTA Clearance"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.295",
    capstone_ready: bool = True,
) -> None:
    _write_json(root, mod.CAPSTONE_V294_REL_PATH, _capstone_payload(capstone_ready=capstone_ready))
    _write_json(root, mod.MATRIX_V28_REL_PATH, _matrix_payload())
    _write_json(root, mod.GGUF_REPLAY_REL_PATH, _replay_payload())
    _write_json(root, mod.CLEAN_VERIFIER_REL_PATH, _verifier_payload())
    _write_json(root, mod.REPAIR_GATE_REL_PATH, _repair_gate_payload())
    _write_json(root, mod.REPAIR_LADDER_REL_PATH, _repair_ladder_payload())
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.295\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions .295 planning\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions .295 planning\n")


def test_req_report_3177_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3177: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3177" in spec
    assert "SCENARIO-REPORT-3177" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3177_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3177: staged .295 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.294")

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.5)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v294_activate_v295_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 73
    assert artifact["blocker_delta_from_v27"] == 8
    assert artifact["next_milestone"] == "2026.05.295"
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete: archive_v294_activate_v295_ready=true")
    assert artifact["status_summary_294"] == {
        "paper_ready": False,
        "capstone_v294_ready": True,
        "publication_blocker_count": 73,
        "blocker_delta_from_v27": 8,
        "missing_artifact_count": 1,
        "verifier_status": "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only",
        "repair_gate_status": "blocked_flagged_verifier",
        "repair_ladder_status": (
            "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
        ),
        "fr11_self_learning_status": (
            "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
        ),
        "ebcn_kan_status": (
            "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
        ),
        "hardware_boundary": "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made",
        "next_top_gap": "clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock",
        "gguf_replay_preflight_passed": False,
        "gguf_replay_live_call_count": 0,
        "gguf_replay_blocked_reason": (
            "CUDA/GPU substrate unavailable for mandated GGUF replay smoke"
        ),
        "clean_verifier_gated_skip": True,
        "clean_verifier_flagged_adversarial": True,
        "repair_ladder_live_repair_attempts": 0,
        "carried_forward_missing_artifact_count": 1,
    }
    assert set(blockers) == {
        "publication_blockers_73",
        "clean_live_sota_verifier_gated_skip",
        "gguf_replay_preflight_failed",
        "adversarial_verifier_state_flagged",
        "repair_gate_blocked",
        "no_live_repair_attempts",
        "projection_only_ebcn_kan_sidecar",
        "no_authenticated_hardware_speedup",
        "carried_forward_missing_artifact_1",
    }
    assert blockers["publication_blockers_73"]["value"] == 73
    assert blockers["clean_live_sota_verifier_gated_skip"]["value"] is True
    assert blockers["gguf_replay_preflight_failed"]["value"] is False
    assert blockers["adversarial_verifier_state_flagged"]["value"] is True
    assert blockers["repair_gate_blocked"]["value"] == "blocked_flagged_verifier"
    assert blockers["no_live_repair_attempts"]["value"] == 0
    assert "projection_only" in blockers["projection_only_ebcn_kan_sidecar"]["value"]
    assert "no_authenticated_speedup" in blockers["no_authenticated_hardware_speedup"]["value"]
    assert blockers["carried_forward_missing_artifact_1"]["value"] == 1
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3177-archive-v294-activate-v295"]
    assert artifact["activation_performed_by_this_task"] is False
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["research_roadmap_next_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["ops_status_updated"] is False
    assert artifact["ops_changelog_updated"] is False
    assert artifact["traceability_updated"] is False
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_dot294_capstone_matrix_and_roadmap_handoff_files",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "live_model_calls": 0,
        "hardware_commands_run": [],
    }


def test_req_report_3177_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3177: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v294_activate_v295_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.295"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3177_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3177: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v294_activate_v295_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_v294_ready=true" in artifact["blocked_reasons"]


def test_req_report_3177_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3177: roadmap milestone/doc/tasks must match the .295 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.294")

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)

    assert artifact["archive_v294_activate_v295_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.295" in artifact["blocked_reasons"]


def test_req_report_3177_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3177: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=4.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v294_activate_v295_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V294_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V294_REL_PATH
    )


def test_req_report_3177_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3177: malformed and unusual source states fail closed."""

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
    assert mod._bool_field({"x": True}, "x") is True
    assert mod._bool_field({"x": 1}, "x") is False
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._publication_blocker_count({}) == (0, "missing")
    assert mod._blocker_delta_from_v27({}) == (0, "missing")
    assert mod._missing_artifact_count({}, {}) == 0
    assert mod._missing_artifact_count({}, {"missing_artifacts": [1, 2]}) == 2
    assert mod._next_milestone({"observed_milestone": "2026.05.295"}) == "2026.05.295"
    assert mod._next_milestone({}) == "2026.05.295"
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={"source_present": False, "milestone_matches": False},
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.295",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
