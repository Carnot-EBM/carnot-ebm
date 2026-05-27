"""Tests for Exp 3163 archive .293 and .294 handoff.

Spec refs: REQ-REPORT-3163, SCENARIO-REPORT-3163.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v293_activate_v294_3163 as mod


REQUIRED_FIELDS = {
    "archive_v293_activate_v294_ready",
    "prior_capstone_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "blocker_delta_from_v26",
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


def _matrix_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3161_cross_corpus_matrix_v27",
        "matrix_v27_ready": True,
        "publication_blocker_count": 65,
        "blocker_delta_from_v26": 10,
        "false_accept_recovery_summary": {
            "preflight_status": "blocked",
            "preflight_passed": False,
            "clean_live_rerun_status": "gated_skipped",
            "live_verifier_evidence_trusted": False,
            "repair_gate_implication": "blocked_pending_clean_rerun",
        },
        "repair_summary": {
            "repair_gate_status": "blocked",
            "repair_gate_state": "blocked",
            "repair_ladder_status": "gated_skipped",
            "repair_attempt_count": 0,
            "gate_check_summary": "3 of 3 gate(s) failed",
        },
        "fr11_summary": {
            "ledger_consistency_rate": 0.857143,
            "ledger_consistent_count": 12,
            "ledger_status": "bounded",
            "promotion_recommendation": "block_fr11_promotion_until_ledger_consistency_reaches_1.0",
        },
        "architecture_boundary_summary": {
            "energy_sidecar_status": "projection_only",
            "live_integration_claim_allowed": False,
            "kan_status": "bounded",
            "kan_residual_blocker_count": 3,
            "hardware_status": "blocked",
            "authenticated_speedup_claim_allowed": False,
            "no_hardware_commands_run": True,
        },
        "missing_artifacts": [
            {
                "experiment_id": "exp3152",
                "path": mod.CLEAN_RERUN_REL_PATH.as_posix(),
                "reason": "missing_or_gated_dot293_artifact",
            },
            {
                "experiment_id": "exp3154",
                "path": mod.REPAIR_LADDER_REL_PATH.as_posix(),
                "reason": "missing_or_gated_dot293_artifact",
            },
        ],
        "honest_verdict": (
            "complete: matrix_v27_ready=true; publication_blocker_count=65; "
            "blocker_delta_from_v26=10"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 65,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3162_capstone_v293",
        "milestone": "2026.05.293",
        "capstone_ready": capstone_ready,
        "capstone_v293_ready": capstone_ready,
        "paper_ready": paper_ready,
        "blocker_delta_from_v26": 10,
        "next_top_gap": "clean_live_verifier_corrigendum_repair_gate",
        "verifier_evidence_status": (
            "corrigendum_preserved_exact_replay_but_did_not_unblock_repair_live_evidence_untrusted"
        ),
        "repair_gate_status": "blocked_pending_clean_rerun_gate_failed",
        "repair_ladder_status": "correctly_skipped_gate_blocked_no_live_repair_attempts",
        "fr11_self_learning_status": (
            "improved_to_0.857143_but_promotion_blocked_controller_memory_only_no_weight_update"
        ),
        "ebt_arm_status": "projection_only_scalar_auc_1.0_exact_rows_6_no_live_integration_blockers_6",
        "kan_status": "bounded_monitor_records_4_new_2_no_deployed_verifier_blockers_3",
        "sampler_hardware_status": (
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
        ),
        "missing_artifacts": _matrix_payload()["missing_artifacts"],
        "source_artifacts": [
            {"path": mod.MATRIX_V27_REL_PATH.as_posix(), "role": "matrix_v27_authority"},
            {"path": mod.PREFLIGHT_REL_PATH.as_posix(), "role": "blocked_preflight"},
        ],
        "honest_verdict": (
            "complete: capstone_v293_ready=true; capstone_ready=true; paper_ready=false; "
            "publication_blocker_count=65; blocker_delta_from_v26=10; "
            "next_top_gap=clean_live_verifier_corrigendum_repair_gate"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _preflight_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3151_live_inference_authenticity_preflight_v1",
        "live_inference_authenticity_preflight_ready": True,
        "preflight_passed": False,
        "blocked_reason": "duration_s=10.590346 is shorter than minimum plausible duration 60.0",
        "duration_s": 10.590346,
        "minimum_duration_requirement_s": 60.0,
        "live_call_count": 1,
        "honest_verdict": (
            "blocked_duration_too_short: preflight_passed=false; live_call_count=1"
        ),
    }


def _repair_gate_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3153_repair_gate_unlock_decision_v2",
        "status": "blocked_gate_check_failed",
        "gate_check_summary": "3 of 3 gate(s) failed",
        "honest_verdict": "blocked_gate_check_failed",
    }


def _fr11_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3156_fr11_ledger_consistency_closure_v1",
        "fr11_ledger_consistency_closure_v1_ready": True,
        "ledger_consistency_rate": 0.857143,
        "ledger_consistent_count": 12,
        "honest_verdict": "complete: ledger_consistency_rate=0.857143",
    }


def _ebcn_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3158_ebcn_energy_sidecar_calibration_v1",
        "ebcn_energy_sidecar_calibration_v1_ready": True,
        "scalar_energy_auc": 1.0,
        "exact_labeled_row_count": 6,
        "live_integration_claim_allowed": False,
        "honest_verdict": "complete: live_integration_claim_allowed=false",
    }


def _kan_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3159_kan_proof_carrying_monitor_expansion_v1",
        "kan_proof_carrying_monitor_expansion_v1_ready": True,
        "residual_blocker_count": 3,
        "deployed_verifier_claim_allowed": False,
        "honest_verdict": "complete: no deployed verifier claim",
    }


def _hardware_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3160_hardware_sampler_evidence_boundary_v7",
        "hardware_sampler_evidence_boundary_v7_ready": True,
        "authenticated_speedup_claim_allowed": False,
        "no_hardware_commands_run": True,
        "honest_verdict": "complete: no authenticated speedup claim",
    }


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.294",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3163-archive-v293-activate-v294\n"
        "    deliverable: results/experiment_3163_archive_v293_activate_v294.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Duration-Corrected Live Verifier Recovery"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.294",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V27_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V293_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    _write_json(root, mod.PREFLIGHT_REL_PATH, _preflight_payload())
    _write_json(root, mod.REPAIR_GATE_REL_PATH, _repair_gate_payload())
    _write_json(root, mod.FR11_LEDGER_REL_PATH, _fr11_payload())
    _write_json(root, mod.EBCN_REL_PATH, _ebcn_payload())
    _write_json(root, mod.KAN_REL_PATH, _kan_payload())
    _write_json(root, mod.HARDWARE_REL_PATH, _hardware_payload())
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.294\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3162 and .294\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3162 and .294\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3162\n")


def test_req_report_3163_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3163: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3163" in spec
    assert "SCENARIO-REPORT-3163" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3163_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3163: staged .294 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.293")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v293_activate_v294_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 65
    assert artifact["blocker_delta_from_v26"] == 10
    assert artifact["next_milestone"] == "2026.05.294"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v293_activate_v294_ready=true")
    assert artifact["status_summary_293"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v27_ready": True,
        "publication_blocker_count": 65,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "blocker_delta_from_v26": 10,
        "blocker_delta_source": "capstone_blocker_delta_from_v26",
        "next_top_gap": "clean_live_verifier_corrigendum_repair_gate",
        "verifier_evidence_status": (
            "corrigendum_preserved_exact_replay_but_did_not_unblock_repair_live_evidence_untrusted"
        ),
        "live_preflight_status": "blocked",
        "clean_verifier_rerun_status": "gated_skipped",
        "repair_gate_status": "blocked_pending_clean_rerun_gate_failed",
        "repair_ladder_status": "correctly_skipped_gate_blocked_no_live_repair_attempts",
        "fr11_self_learning_status": (
            "improved_to_0.857143_but_promotion_blocked_controller_memory_only_no_weight_update"
        ),
        "fr11_ledger_consistency": 0.857143,
        "ebcn_status": "projection_only_scalar_auc_1.0_exact_rows_6_no_live_integration_blockers_6",
        "kan_status": "bounded_monitor_records_4_new_2_no_deployed_verifier_blockers_3",
        "hardware_sampler_boundary": (
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
        ),
        "preflight_blocked_reason": (
            "duration_s=10.590346 is shorter than minimum plausible duration 60.0"
        ),
        "source_artifacts": _capstone_payload()["source_artifacts"],
    }
    assert set(blockers) == {
        "publication_blockers_65",
        "duration_authenticity_preflight_blocked",
        "clean_rerun_missing_gated",
        "thin_repair_gate_artifact",
        "repair_ladder_missing_gated",
        "fr11_ledger_consistency_0_857143",
        "bounded_ebcn_kan_diagnostics",
        "no_authenticated_hardware_speedup",
    }
    assert blockers["publication_blockers_65"]["value"] == 65
    assert blockers["duration_authenticity_preflight_blocked"]["value"] == "blocked"
    assert blockers["clean_rerun_missing_gated"]["value"] == "gated_skipped"
    assert blockers["thin_repair_gate_artifact"]["value"] == "blocked_gate_check_failed"
    assert blockers["repair_ladder_missing_gated"]["value"] == (
        "correctly_skipped_gate_blocked_no_live_repair_attempts"
    )
    assert blockers["fr11_ledger_consistency_0_857143"]["value"] == 0.857143
    assert blockers["bounded_ebcn_kan_diagnostics"]["value"] == {
        "ebcn_status": "projection_only_scalar_auc_1.0_exact_rows_6_no_live_integration_blockers_6",
        "kan_status": "bounded_monitor_records_4_new_2_no_deployed_verifier_blockers_3",
    }
    assert blockers["no_authenticated_hardware_speedup"]["value"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
    )
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3163-archive-v293-activate-v294"]
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


def test_req_report_3163_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3163: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v293_activate_v294_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.294"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3163_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3163: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v293_activate_v294_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3163_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3163: roadmap milestone/doc/tasks must match the .294 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.293")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v293_activate_v294_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.294" in artifact["blocked_reasons"]


def test_req_report_3163_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3163: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v293_activate_v294_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V293_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V293_REL_PATH
    )


def test_req_report_3163_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3163: helper edges preserve malformed and unusual states."""

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
    assert mod._float_field({"rate": 0.857143}, "rate") == (0.857143, "source_rate")
    assert mod._float_field({"rate": 1}, "rate") == (1.0, "source_rate")
    assert mod._float_field({"rate": True}, "rate") == (0.0, "missing")
    assert mod._text_field({"x": "value"}, "x") == "value"
    assert mod._text_field({"x": None}, "x") == ""
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._publication_blocker_count({}) == (0, "missing")
    assert mod._blocker_delta_from_v26({}) == (0, "missing")
    assert mod._next_milestone({"observed_milestone": "2026.05.295"}) == "2026.05.295"
    assert mod._next_milestone({}) == "2026.05.294"
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._preflight_status({}, {"preflight_passed": False}) == "blocked"
    assert mod._preflight_status({}, {"preflight_passed": True}) == "passed"
    assert mod._preflight_status({}, {}) == ""
    assert mod._ledger_consistency({}, {"ledger_consistency_rate": 0.857143}) == 0.857143
    assert (
        mod._repair_gate_artifact_status(
            {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"}
        )
        == "blocked_gate_check_failed"
    )
    assert mod._repair_gate_artifact_status({"status": "blocked"}) == "blocked"
    assert mod._repair_gate_artifact_status({}) == ""
    assert mod._blocked_reasons(
        capstone_present=False,
        prior_capstone_ready=False,
        roadmap_handoff={"source_present": False, "milestone_matches": False},
        vnext_doc_present=False,
    ) == [
        "prior capstone artifact missing or malformed",
        "roadmap handoff source is missing",
        "roadmap milestone is not 2026.05.294",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
