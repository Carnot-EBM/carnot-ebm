"""Tests for Exp 3109 archive .289 and .290 handoff.

Spec refs: REQ-REPORT-3109, SCENARIO-REPORT-3109.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v289_activate_v290_3109 as mod


REQUIRED_FIELDS = {
    "archive_v289_activate_v290_ready",
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
    return {
        "artifact": "experiment_3107_cross_corpus_matrix_v23",
        "milestone": "2026.05.289",
        "matrix_v23_ready": True,
        "publication_blocker_count": 36,
        "missing_artifacts": [
            {
                "path": "results/experiment_3102_gated_structured_repair_micro_panel_v3.json",
                "reason": "expected .289 structured repair micro-panel artifact is absent",
            }
        ],
        "headline_model_spec_gaps": [
            {
                "row_id": "dot289:exp3099_local_sota_confidence_abstention_panel",
                "source_artifact": (
                    "results/experiment_3099_local_sota_confidence_abstention_panel_v3.json"
                ),
                "missing_model_ids": [],
                "present_model_ids": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                ],
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ],
        "honest_verdict": (
            "complete: matrix_v23_ready=true; rows_total=89; "
            "publication_blocker_count=36; missing_artifacts=1"
        ),
    }


def _capstone_payload(
    *,
    capstone_ready: bool = True,
    paper_ready: bool = False,
    publication_blocker_count: int | None = 36,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact": "experiment_3108_capstone_v289",
        "milestone": "2026.05.289",
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "verifier_gain_status": "model_spec_gap_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "blocked_gated_missing_verifier_gated_repair_not_promoted",
        "fr11_self_learning_status": (
            "clean_controller_only_soundness_zero_completeness_promotion_blocked"
        ),
        "ebt_arm_status": "projection_only_sidecar_pipeline_no_model_integration",
        "sampler_hardware_status": "diagnostic_only_cpu_microbench_no_hardware_speedup",
        "gatemate_status": "blocked_no_rerun_operator_actions_required_no_speedup_claim",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
        "missing_capstone_input_artifacts": [
            {
                "path": "results/experiment_3102_gated_structured_repair_micro_panel_v3.json",
                "reason": "named by matrix v23 capstone_input_artifacts but not readable",
            }
        ],
        "headline_model_spec_gaps": _matrix_payload()["headline_model_spec_gaps"],
        "source_artifacts": [
            {"path": mod.MATRIX_V23_REL_PATH.as_posix(), "role": "matrix_v23"},
            {"path": "results/experiment_3100_z3_oracle_feedback_v2.json", "role": "z3"},
        ],
        "prd_gap_summary": {
            "verifier_repair": {
                "publication_blocker_count": 17,
                "publication_blocker_row_ids": ["vr-1", "vr-2"],
            },
            "fr11_self_learning": {
                "claim_boundary": "controller-only; completeness stress blocks promotion"
            },
            "ebt_arm_bridge": {"publication_blocker_count": 1},
            "sampler_hardware_adjacency": {"statuses_present": ["diagnostic_only"]},
            "gatemate_ssqa_evidence": {"publication_blocker_count": 11},
        },
        "matrix_v23_summary": {
            "matrix_v23_ready": True,
            "publication_blocker_count": 36,
            "rows_total": 89,
        },
        "next_milestone_recommendation": (
            "2026.05.290: clear verifier/repair first (17 blocker rows: model-spec "
            "gap, formal feedback, calibration, and missing repair micro-panel); keep "
            "FR-11 controller-only until completeness/retention promotion clears."
        ),
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; matrix_v23_ready=true; "
            "publication_blocker_count=36"
        ),
    }
    if publication_blocker_count is not None:
        payload["publication_blocker_count"] = publication_blocker_count
    return payload


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.290",
    milestone_doc: str = "openspec/change-proposals/research-roadmap-vNEXT.md",
    include_task: bool = True,
) -> str:
    tasks = (
        "tasks:\n"
        "  - id: exp3109-archive-v289-activate-v290\n"
        "    deliverable: results/experiment_3109_archive_v289_activate_v290.json\n"
        if include_task
        else "tasks: []\n"
    )
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Certified Verifier Recovery"\n'
        f'milestone_doc: "{milestone_doc}"\n'
        f"{tasks}"
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = True,
    active_milestone: str = "2026.05.290",
    capstone_ready: bool = True,
    paper_ready: bool = False,
) -> None:
    _write_json(root, mod.MATRIX_V23_REL_PATH, _matrix_payload())
    _write_json(
        root,
        mod.CAPSTONE_V289_REL_PATH,
        _capstone_payload(capstone_ready=capstone_ready, paper_ready=paper_ready),
    )
    _write_json(
        root,
        mod.FORMAL_FEEDBACK_REL_PATH,
        {
            "formal_feedback_v2_ready": False,
            "honest_verdict": "complete_blocked_headline: formal_feedback_v2_ready=false",
        },
    )
    _write_json(
        root,
        mod.VERIFIER_CALIBRATION_REL_PATH,
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: "
                "exp3100-z3-oracle-feedback-v2.formal_feedback_v2_ready "
                "(actual=False == expected=True)"
            ),
        },
    )
    _write_json(
        root,
        mod.FR11_STRESS_REL_PATH,
        {
            "promotion_decision": "blocked",
            "soundness_mistakes": 0,
            "completeness_mistakes": 12,
            "honest_verdict": "complete_fr11_stress_boundary_blocks_promotion",
        },
    )
    _write_json(
        root,
        mod.EBT_ARM_REL_PATH,
        {
            "sidecar_boundary_v2_ready": True,
            "honest_verdict": (
                "complete: sidecar_boundary_v2_ready=true; "
                "projection_only_no_live_model_integration_no_speedup"
            ),
        },
    )
    _write_json(
        root,
        mod.CLUT_SAMPLER_REL_PATH,
        {
            "hardware_claim_made": False,
            "hardware_commands_run": [],
            "honest_verdict": (
                "complete: CPU cLUT microbench ran; hardware_claim_made=false; "
                "hardware_commands_run=[]"
            ),
        },
    )
    _write_json(
        root,
        mod.GATEMATE_SSQA_REL_PATH,
        {
            "operator_evidence_ingestion_v3_ready": True,
            "gatemate_rerun_allowed": False,
            "ssqa_readback_allowed": False,
            "hardware_commands_run": [],
            "speedup_claim_made": False,
            "honest_verdict": (
                "complete: operator_evidence_ingestion_v3_ready=true; "
                "gatemate_rerun_allowed=false; ssqa_readback_allowed=false"
            ),
        },
    )
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.290\n")
    _write_text(root, mod.AGENTS_REL_PATH, "# repo instructions\n")
    _write_text(root, mod.CODEX_REL_PATH, "# codex instructions\n")
    _write_text(root, mod.CLAUDE_REL_PATH, "# claude instructions\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "status mentions exp3108 and .290\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "changelog mentions exp3108 and .290\n")
    _write_text(root, mod.TRACEABILITY_REL_PATH, "traceability mentions REQ-REPORT-3108\n")


def test_req_report_3109_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3109: OpenSpec declares the archive/handoff contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3109" in spec
    assert "SCENARIO-REPORT-3109" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3109_builds_ready_archive_from_staged_roadmap(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3109: staged .290 roadmap can drive a ready handoff."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.289")

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=6.25)
    blockers = {row["blocker_id"]: row for row in artifact["carry_forward_blockers"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["archive_v289_activate_v290_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["next_milestone"] == "2026.05.290"
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete: archive_v289_activate_v290_ready=true")
    assert artifact["status_summary_289"] == {
        "paper_ready": False,
        "capstone_ready": True,
        "matrix_v23_ready": True,
        "publication_blocker_count": 36,
        "publication_blocker_count_source": "capstone_publication_blocker_count",
        "verifier_gain_status": "model_spec_gap_or_gated_verifier_gain_recovery_incomplete",
        "repair_claim_status": "blocked_gated_missing_verifier_gated_repair_not_promoted",
        "fr11_self_learning_status": (
            "clean_controller_only_soundness_zero_completeness_promotion_blocked"
        ),
        "ebt_arm_status": "projection_only_sidecar_pipeline_no_model_integration",
        "sampler_hardware_status": "diagnostic_only_cpu_microbench_no_hardware_speedup",
        "gatemate_status": "blocked_no_rerun_operator_actions_required_no_speedup_claim",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
        "source_artifacts": _capstone_payload()["source_artifacts"],
        "missing_capstone_input_artifacts": [
            {
                "path": "results/experiment_3102_gated_structured_repair_micro_panel_v3.json",
                "reason": "named by matrix v23 capstone_input_artifacts but not readable",
            }
        ],
        "headline_model_spec_gaps": _matrix_payload()["headline_model_spec_gaps"],
    }
    assert set(blockers) == {
        "publication_blockers_36",
        "local_sota_metadata_cache_gap",
        "formal_feedback_v2_not_ready",
        "gated_verifier_calibration",
        "missing_repair_micro_panel",
        "fr11_completeness_retention_promotion_block",
        "ebt_arm_projection_only",
        "clut_cpu_diagnostic_only",
        "gatemate_ssqa_missing_operator_evidence",
    }
    assert blockers["publication_blockers_36"]["value"] == 36
    assert blockers["local_sota_metadata_cache_gap"]["value"] == [
        "dot289:exp3099_local_sota_confidence_abstention_panel"
    ]
    assert blockers["formal_feedback_v2_not_ready"]["value"] is False
    assert blockers["gated_verifier_calibration"]["value"]["honest_verdict"] == (
        "blocked_gate_check_failed"
    )
    assert blockers["missing_repair_micro_panel"]["value"] == [
        "results/experiment_3102_gated_structured_repair_micro_panel_v3.json"
    ]
    assert blockers["fr11_completeness_retention_promotion_block"]["value"] == {
        "fr11_self_learning_status": (
            "clean_controller_only_soundness_zero_completeness_promotion_blocked"
        ),
        "promotion_decision": "blocked",
        "soundness_mistakes": 0,
        "completeness_mistakes": 12,
    }
    assert blockers["ebt_arm_projection_only"]["value"] == (
        "projection_only_sidecar_pipeline_no_model_integration"
    )
    assert blockers["clut_cpu_diagnostic_only"]["value"] == {
        "sampler_hardware_status": "diagnostic_only_cpu_microbench_no_hardware_speedup",
        "hardware_claim_made": False,
        "hardware_commands_run": [],
    }
    assert blockers["gatemate_ssqa_missing_operator_evidence"]["value"] == {
        "gatemate_status": "blocked_no_rerun_operator_actions_required_no_speedup_claim",
        "ssqa_status": "gated_skipped_host_visible_readback_missing",
        "gatemate_rerun_allowed": False,
        "ssqa_readback_allowed": False,
    }
    assert all(row["matches_expected"] is True for row in blockers.values())
    assert artifact["roadmap_handoff"]["source_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is True
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_handoff"]["task_ids"] == ["exp3109-archive-v289-activate-v290"]
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
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "local_repo_only": True,
        "no_live_llm_inference": True,
        "source": "checked_in_artifacts",
    }


def test_req_report_3109_uses_active_roadmap_fallback_after_activation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3109: post-activation active roadmap is audited read-only."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["archive_v289_activate_v290_ready"] is True
    assert artifact["roadmap_handoff"]["source_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_handoff"]["requested_staged_roadmap_present"] is False
    assert artifact["roadmap_handoff"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_handoff"]["active_roadmap_milestone"] == "2026.05.290"
    assert mod.STAGED_ROADMAP_REL_PATH.as_posix() in artifact["missing_source_artifacts"]


def test_req_report_3109_blocks_when_capstone_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3109: prior capstone readiness is a hard precondition."""

    _write_common_sources(tmp_path, capstone_ready=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["archive_v289_activate_v290_ready"] is False
    assert artifact["prior_capstone_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_prior_capstone_not_ready:")
    assert "prior capstone is not capstone_ready=true" in artifact["blocked_reasons"]


def test_req_report_3109_blocks_when_roadmap_handoff_is_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-3109: roadmap milestone/doc/tasks must match the .290 handoff."""

    _write_common_sources(tmp_path, staged_roadmap=False, active_milestone="2026.05.289")

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert artifact["archive_v289_activate_v290_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_roadmap_handoff_not_ready:")
    assert "roadmap milestone is not 2026.05.290" in artifact["blocked_reasons"]


def test_req_report_3109_write_artifact_persists_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-3109: write_artifact emits the deliverable JSON."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v289_activate_v290_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)
    checksums = {row["path"]: row["sha256"] for row in saved["source_artifacts"]}
    assert checksums[mod.CAPSTONE_V289_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V289_REL_PATH
    )


def test_req_report_3109_helper_edges_remain_honest(tmp_path: Path) -> None:
    """REQ-REPORT-3109: helper edges preserve malformed and unusual states."""

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
    assert mod._first_int_from_text("publication_blocker_count=12") == 12
    assert mod._first_int_from_text("no count") is None
    assert mod._publication_blocker_count(
        {"publication_blocker_count": 9},
        {},
    ) == (9, "capstone_publication_blocker_count")
    assert mod._publication_blocker_count(
        {"publication_blockers": [1, 2]},
        {},
    ) == (2, "capstone_publication_blockers_length")
    assert mod._publication_blocker_count(
        {"matrix_v23_summary": {"publication_blocker_count": 8}},
        {},
    ) == (8, "capstone_matrix_v23_summary")
    assert mod._publication_blocker_count(
        {"honest_verdict": "complete: publication_blocker_count=4"},
        {},
    ) == (4, "capstone_honest_verdict")
    assert mod._publication_blocker_count(
        {},
        {"honest_verdict": "complete: publication_blocker_count=3"},
    ) == (3, "matrix_honest_verdict")
    assert mod._publication_blocker_count({}, {}) == (0, "missing")
    assert mod._status_value("repair_claim_status", {"repair_claim_status": "bounded"}, {}) == (
        "bounded"
    )
    assert mod._task_ids({"tasks": [{"id": "a"}, {"no": "id"}, "bad"]}) == ["a"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._source_artifact(tmp_path, "missing", Path("missing.txt"))["present"] is False
    assert mod._formal_feedback_ready({"formal_feedback_v2_ready": True}) is True
    assert (
        mod._formal_feedback_ready(
            {"honest_verdict": "complete_blocked_headline: formal_feedback_v2_ready=false"}
        )
        is False
    )
    assert (
        mod._formal_feedback_ready({"honest_verdict": "complete: formal_feedback_v2_ready=true"})
        is True
    )
    assert mod._formal_feedback_ready({}) is None
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
        "roadmap milestone is not 2026.05.290",
        "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md",
        "roadmap has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]
