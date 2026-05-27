"""Tests for Exp 3218 milestone .297 capstone.

Spec refs: REQ-REPORT-3218, SCENARIO-REPORT-3218.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v297_3218 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "matrix_artifact",
    "capstone_v297_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v30",
    "local_sota_receipt_status",
    "clean_verifier_status",
    "repair_gate_status",
    "repair_ladder_status",
    "context_fixture_status",
    "constraintbench_fixture_status",
    "fr11_self_learning_status",
    "hardware_sampler_status",
    "next_top_gap",
    "recommended_next_milestone_theme",
    "ops_status_updated",
    "ops_changelog_updated",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_v31(*, paper_ready: bool = False, blockers: int = 92) -> dict[str, Any]:
    return {
        "schema_version": "carnot.cross_corpus_matrix.v31_297_artifact_aggregation.v1",
        "experiment_id": "exp3217",
        "milestone": "2026.05.297",
        "cross_corpus_matrix_v31_ready": True,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v30": blockers - 85,
        "local_sota_receipt_status": (
            "blocked_selected_python_torch_cuda_cuda_receipt_ready_false_full_receipt_gated_skipped"
        ),
        "clean_verifier_status": (
            "gated_skipped_missing_clean_verifier_v12_after_full_receipt_gate"
        ),
        "repair_status": "repair_gate_blocked_v6_ladder_gated_skipped_v7",
        "repair_gate_status": "blocked",
        "repair_ladder_status": "gated_skipped",
        "context_fixture_status": "available_ready_for_clean_verifier_fixture_count_30",
        "constraintbench_fixture_status": (
            "available_exact_pilot_fixture_count_15_no_full_coverage_claimed"
        ),
        "fr11_self_learning_status": (
            "controller_trace_replay_promoted_nonforgetting_queue_audit_only_"
            "no_model_weight_update_claimed"
        ),
        "fr11_claim_boundaries": {
            "controller_memory_promotion_allowed": True,
            "queue_promotion_allowed": False,
            "model_weight_update_claimed": False,
        },
        "hardware_sampler_status": "no_authenticated_hardware_transcript_no_speedup_tsu_kona_claim",
        "hardware_claim_boundaries": {
            "authenticated_hardware_transcript_present": False,
            "speedup_claim_allowed": False,
            "tsu_or_kona_claim_allowed": False,
        },
        "required_evidence_blocked_or_missing": [
            "local_sota_receipt",
            "clean_verifier",
            "repair",
            "hardware_sampler",
        ],
        "next_top_gap": mod.NEXT_TOP_GAP,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "ops_docs_updated": False,
        "honest_verdict": (
            "complete: cross_corpus_matrix_v31_ready=true; "
            "paper_ready=false; publication_blocker_count=92"
        ),
    }


def _write_dot297_sources(root: Path) -> None:
    payloads: dict[Path, dict[str, Any]] = {
        mod.EXP3205_REL_PATH: {
            "schema_version": "carnot.archive_activation.v296_to_v297.v1",
            "experiment_id": "exp3205",
            "milestone": "2026.05.297",
            "activation_ready": True,
            "honest_verdict": "complete: archive_v296_activate_v297_ready=true",
        },
        mod.EXP3206_REL_PATH: {
            "schema_version": "carnot.cuda_env_forensics_ledger.v1",
            "experiment_id": "exp3206",
            "milestone": "2026.05.297",
            "cuda_env_diagnosed": True,
            "cuda_init_clean": False,
            "honest_verdict": "blocked_selected_python_torch_cuda",
        },
        mod.EXP3207_REL_PATH: {
            "schema_version": "carnot.llama_cpp_cuda_rebuild_clean_subprocess.v1",
            "experiment_id": "exp3207",
            "milestone": "2026.05.297",
            "cuda_receipt_ready": False,
            "honest_verdict": "blocked_selected_python_torch_cuda",
        },
        mod.EXP3208_REL_PATH: {
            "schema": "blocked_gate_check_v1",
            "experiment": 3208,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP3210_REL_PATH: {
            "schema_version": "carnot.context_cot_clbench_parametric_shortcut_fixtures.v1",
            "experiment_id": "exp3210",
            "milestone": "2026.05.297",
            "fixture_count": 30,
            "ready_for_clean_verifier": True,
            "honest_verdict": "complete: context fixtures ready",
        },
        mod.EXP3211_REL_PATH: {
            "schema_version": "carnot.constraintbench_feasibility_objective_pilot.v1",
            "experiment_id": "exp3211",
            "milestone": "2026.05.297",
            "fixture_count": 15,
            "ready_for_clean_verifier": True,
            "honest_verdict": "complete: exact pilot; no full ConstraintBench coverage claimed",
        },
        mod.EXP3213_REL_PATH: {
            "schema_version": "carnot.repair_gate_decision.v6",
            "experiment_id": "exp3213",
            "milestone": "2026.05.297",
            "repair_gate_state": "blocked",
            "repair_ladder_allowed": False,
            "honest_verdict": "complete: repair_gate_state=blocked",
        },
        mod.EXP3214_REL_PATH: {
            "schema": "blocked_gate_check_v1",
            "experiment": 3214,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP3215_REL_PATH: {
            "schema": "carnot.fr11.evidence_gated_trace_replay_controller.v2",
            "schema_version": "1.0",
            "experiment_id": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
            "milestone": "2026.05.297",
            "promotion_allowed": True,
            "negative_control_regression_count": 0,
            "model_weight_update_claimed": False,
            "honest_verdict": "complete: fr11 controller only",
        },
        mod.EXP3216_REL_PATH: {
            "schema": "carnot.fr11.grounded_continuation_nonforgetting_queue.v1",
            "schema_version": "1.0",
            "experiment_id": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
            "milestone": "2026.05.297",
            "nonforgetting_queue_defined": True,
            "controller_memory_promotion_allowed": False,
            "model_weight_update_claimed": False,
            "honest_verdict": "complete: fr11 queue audit only",
        },
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def _write_sources(root: Path, *, paper_ready: bool = False, blockers: int = 92) -> None:
    _write_json(
        root,
        mod.MATRIX_V31_REL_PATH,
        _matrix_v31(paper_ready=paper_ready, blockers=blockers),
    )
    _write_dot297_sources(root)


def test_req_report_3218_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3218: OpenSpec declares the capstone before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3218" in spec
    assert "SCENARIO-REPORT-3218" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3218_builds_capstone_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3218: .297 capstone preserves matrix v31 boundaries."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3218"
    assert artifact["milestone"] == "2026.05.297"
    assert artifact["matrix_artifact"] == mod.MATRIX_V31_REL_PATH.as_posix()
    assert artifact["capstone_v297_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 92
    assert artifact["blocker_delta_from_v30"] == 7
    assert artifact["local_sota_receipt_status"] == (
        "blocked_selected_python_torch_cuda_cuda_receipt_ready_false_full_receipt_gated_skipped"
    )
    assert artifact["clean_verifier_status"] == (
        "gated_skipped_missing_clean_verifier_v12_after_full_receipt_gate"
    )
    assert artifact["repair_gate_status"] == "blocked"
    assert artifact["repair_ladder_status"] == "gated_skipped"
    assert artifact["context_fixture_status"] == (
        "available_ready_for_clean_verifier_fixture_count_30"
    )
    assert artifact["constraintbench_fixture_status"] == (
        "available_exact_pilot_fixture_count_15_no_full_coverage_claimed"
    )
    assert artifact["fr11_self_learning_status"] == (
        "controller_trace_replay_promoted_nonforgetting_queue_audit_only_"
        "no_model_weight_update_claimed"
    )
    assert artifact["hardware_sampler_status"] == (
        "no_authenticated_hardware_transcript_no_speedup_tsu_kona_claim"
    )
    assert artifact["next_top_gap"] == mod.NEXT_TOP_GAP
    assert artifact["recommended_next_milestone_theme"] == (
        "cuda_environment_repair_and_clean_local_sota_receipt_recovery"
    )
    assert artifact["ops_status_updated"] is False
    assert artifact["ops_changelog_updated"] is False
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["claim_boundaries_preserved"] == {
        "paper_ready_claim_allowed": False,
        "repair_claim_allowed": False,
        "hardware_speedup_claim_allowed": False,
        "tsu_or_kona_claim_allowed": False,
        "model_weight_self_learning_claim_allowed": False,
    }
    assert artifact["phase_outcomes"]["cuda_receipt_recovery"]["verdict"] == "blocked"
    assert artifact["phase_outcomes"]["clean_verifier"]["verdict"] == "gated_skipped"
    assert artifact["phase_outcomes"]["context_fixtures"]["verdict"] == "available"
    assert artifact["phase_outcomes"]["constraintbench_fixtures"]["verdict"] == "available"
    assert artifact["phase_outcomes"]["structured_repair"]["repair_gate_status"] == "blocked"
    assert artifact["phase_outcomes"]["fr11_self_learning"]["model_weight_update_claimed"] is False
    assert artifact["phase_outcomes"]["hardware_boundary"]["speedup_claim_allowed"] is False
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert sources[mod.EXP3215_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3215_REL_PATH
    )


def test_req_report_3218_paper_ready_is_copied_from_matrix_only(tmp_path: Path) -> None:
    """REQ-REPORT-3218: `paper_ready` follows matrix v31 rather than sources."""

    _write_sources(tmp_path, paper_ready=True, blockers=0)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["capstone_v297_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v30"] == -85
    assert artifact["claim_boundaries_preserved"]["paper_ready_claim_allowed"] is True

    matrix = _matrix_v31(paper_ready=True, blockers=1)
    _write_json(tmp_path, mod.MATRIX_V31_REL_PATH, matrix)

    contradicted = mod.build_artifact(tmp_path)

    assert contradicted["capstone_v297_ready"] is False
    assert contradicted["paper_ready"] is True
    assert (
        "matrix_v31 paper_ready=true while publication blockers remain"
        in contradicted["invariant_violations"]
    )


def test_req_report_3218_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3218: writer and helper branches keep the capstone bounded."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v297_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    empty = mod.build_artifact(tmp_path / "empty", started_s=0.0, now_s=0.0)

    assert empty["capstone_v297_ready"] is False
    assert empty["paper_ready"] is False
    assert empty["publication_blocker_count"] == 0
    assert empty["blocker_delta_from_v30"] is None
    assert empty["local_sota_receipt_status"] == "missing_local_sota_receipt_status"
    assert empty["honest_verdict"].startswith("blocked:")

    assert mod._int_or_none(True) is None
    assert mod._int_or_none(3) == 3
    assert mod._int_or_none("3") is None
    assert mod._field_str({}, "x", "fallback") == "fallback"
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._source_experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._source_experiment_id({"experiment": 7}, "fallback") == "exp7"
    assert mod._source_experiment_id({}, "fallback") == "fallback"
    assert mod._phase_verdict("clean_live_verifier_ready") == "passed"
    assert mod._phase_verdict("available_ready_for_clean_verifier_fixture_count_30") == "available"
    assert mod._phase_verdict("diagnostic_only_hardware_boundary") == "diagnostic_only"
    assert mod._phase_verdict("blocked_cuda") == "blocked"
    assert mod._phase_verdict("gated_skipped_clean_verifier") == "gated_skipped"
    assert mod._phase_verdict("ambiguous") == "blocked"
    assert mod._invariant_violations(
        {
            "cross_corpus_matrix_v31_ready": True,
            "hardware_claim_boundaries": {
                "authenticated_hardware_transcript_present": False,
                "speedup_claim_allowed": True,
                "tsu_or_kona_claim_allowed": True,
            },
            "fr11_claim_boundaries": {"model_weight_update_claimed": True},
            "conductor_file_modified": True,
            "active_roadmap_modified": True,
        },
        0,
        False,
        {"model_weight_self_learning_claim_allowed": False},
    ) == [
        "hardware speedup claim lacks authenticated transcript in matrix_v31",
        "TSU/Kona claim lacks authenticated transcript in matrix_v31",
        "FR-11 model-weight self-learning claim is not proved by matrix_v31",
        "matrix_v31 reports conductor file modification",
        "matrix_v31 reports active roadmap modification",
    ]
    assert mod._recommended_next_milestone_theme("clean_live_verifier_v12_gate_clearance") == (
        "clean_live_verifier_v12_gate_clearance_after_receipt"
    )
    assert mod._recommended_next_milestone_theme("repair_gate_v6_unblock") == (
        "structured_repair_gate_unblock_and_ladder_execution"
    )
    assert mod._recommended_next_milestone_theme("fr11_controller_memory") == (
        "fr11_controller_memory_nonforgetting_promotion"
    )
    assert mod._recommended_next_milestone_theme("authenticated_hardware_transcript") == (
        "authenticated_hardware_transcript_or_explicit_no_speedup_boundary"
    )
    assert mod._recommended_next_milestone_theme("publication_blocker_retirement_review") == (
        "publication_blocker_retirement_review"
    )
