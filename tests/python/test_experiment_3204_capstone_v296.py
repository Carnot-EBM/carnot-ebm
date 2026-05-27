"""Tests for the Exp 3204 milestone .296 capstone.

Spec refs: REQ-REPORT-3204, SCENARIO-REPORT-3204.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v296_3204 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "matrix_artifact",
    "capstone_v296_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v29",
    "local_sota_receipt_status",
    "clean_verifier_status",
    "repair_gate_status",
    "repair_ladder_status",
    "fr11_self_learning_status",
    "hardware_sampler_status",
    "next_top_gap",
    "ops_docs_updated",
    "active_roadmap_modified",
    "conductor_file_modified",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_v30(*, paper_ready: bool = False, blockers: int = 85) -> dict[str, Any]:
    return {
        "schema_version": "carnot.cross_corpus_matrix.v30_296_artifact_aggregation.v1",
        "experiment_id": "exp3203",
        "matrix_version": "v30",
        "cross_corpus_matrix_v30_ready": True,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v29": blockers - 80,
        "source_artifacts_expected": [path.as_posix() for path in mod.CRITICAL_SOURCE_PATHS],
        "source_artifacts_loaded": [path.as_posix() for path in mod.CRITICAL_SOURCE_PATHS],
        "missing_artifact_count": 0,
        "status_counts": {
            "clean": 3,
            "blocked": 3,
            "gated_skipped": 2,
            "diagnostic_only": 5,
            "retired": 0,
            "missing": 0,
        },
        "local_sota_receipt_status": "blocked_cuda_unavailable_no_full_local_sota_receipt",
        "clean_verifier_status": (
            "gated_skipped_clean_verifier_v11_waiting_on_clean_rerun_allowed"
        ),
        "repair_status": "blocked_clean_verifier_gate_repair_ladder_gated_skipped",
        "fr11_self_learning_status": (
            "controller_memory_trace_policy_promoted_no_model_weight_update_sidecar_promotion_blocked"
        ),
        "hardware_sampler_status": (
            "diagnostic_only_sparse_potts_thrml_factor_boundary_no_authenticated_speedup"
        ),
        "next_top_gap": "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
        "required_evidence_blocked_or_missing": [
            "local_sota_receipt",
            "clean_verifier",
            "repair",
            "deployed_verifier_sidecar",
            "hardware_sampler",
        ],
        "paper_v6_narrowing": {
            "deployed_verifier_sidecar_claimed": False,
            "kv260_speedup_claimed": False,
            "model_weight_self_learning_claimed": False,
            "paper_ready_streak_claimed": False,
            "tsu_or_kona_execution_claimed": False,
        },
        "paper_v6_narrowing_preserved": True,
        "artifact_classifications": [
            {
                "row_id": "dot296:exp3198_repair_gate_v5",
                "source_artifact": mod.EXP3198_REL_PATH.as_posix(),
                "status": "blocked",
            },
            {
                "row_id": "dot296:exp3199_repair_ladder_v6",
                "source_artifact": mod.EXP3199_REL_PATH.as_posix(),
                "status": "gated_skipped",
            },
        ],
        "honest_verdict": "complete: cross_corpus_matrix_v30_ready=true",
    }


def _write_dot296_sources(root: Path) -> None:
    payloads: dict[Path, dict[str, Any]] = {
        mod.EXP3191_REL_PATH: {
            "schema_version": "carnot.archive_activation.v295_to_v296.v1",
            "experiment_id": "exp3191",
            "activation_ready": True,
            "honest_verdict": "complete: activation_ready=true",
        },
        mod.EXP3192_REL_PATH: {
            "schema_version": "carnot.receipt_adversarial_contract.v4",
            "experiment_id": "exp3192",
            "current_evidence_assessment": {
                "clean_rerun_allowed": False,
                "headline_claim_allowed": False,
                "proof_receipt_count": 2,
                "substrate_classification": "cpu_fallback_receipt_only",
            },
            "honest_verdict": "complete: receipt contract ready",
        },
        mod.EXP3193_REL_PATH: {
            "schema_version": "carnot.llama_cpp_cuda_offload_health_probe.v1",
            "experiment_id": "exp3193",
            "substrate_classification": "cuda_unavailable",
            "clean_rerun_allowed": False,
            "headline_claim_allowed": False,
            "receipt_count": 0,
            "blocker_reasons": ["selected Python torch.cuda.is_available() is false"],
            "honest_verdict": "blocked_cuda_unavailable: clean_rerun_allowed=false",
        },
        mod.EXP3194_REL_PATH: {
            "schema": "blocked_gate_check_v1",
            "experiment": 3194,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp3193.clean_rerun_allowed actual=False expected=True",
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP3195_REL_PATH: {
            "schema_version": "carnot.adaptive_verification_granularity_policy.v1",
            "experiment_id": "exp3195",
            "adaptive_verification_granularity_policy_v1_ready": True,
            "promotion_allowed": False,
            "estimated_verifier_call_delta": -71,
            "honest_verdict": "complete: adaptive policy ready",
        },
        mod.EXP3196_REL_PATH: {
            "schema_version": "carnot.gencp_domain_preview_repair_compiler.v1",
            "experiment_id": "exp3196",
            "preview_domain_count": 12,
            "repair_call_ready": False,
            "promotion_allowed": False,
            "source_errors": [],
            "honest_verdict": "complete: domain preview ready",
        },
        mod.EXP3197_REL_PATH: {
            "schema_version": "carnot.exverus_inductive_certificate_expansion.v1",
            "experiment_id": "exp3197",
            "invariant_record_count": 5,
            "repair_call_ready": False,
            "source_errors": [],
            "honest_verdict": "complete: certificate expansion ready",
        },
        mod.EXP3198_REL_PATH: {
            "schema_version": "carnot.repair_gate_decision.v5",
            "experiment_id": "exp3198",
            "repair_gate_state": "blocked_clean_verifier_gate_skipped",
            "downstream_gated_skip_expected": True,
            "blocker_reasons": [{"code": "exp3193_clean_rerun_not_allowed"}],
            "honest_verdict": "complete: repair gate blocked",
        },
        mod.EXP3199_REL_PATH: {
            "schema": "blocked_gate_check_v1",
            "experiment": 3199,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp3198.repair_gate_state actual=blocked expected=unblocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        mod.EXP3200_REL_PATH: {
            "schema_version": "1.0",
            "experiment_id": "experiment_3200_fr11_verify_trace_memory_controller_v1",
            "promotion_allowed": True,
            "model_weight_update_performed": False,
            "negative_control_regression_count": 0,
            "trace_count": 30,
            "honest_verdict": "complete: FR-11 trace-memory controller materialized",
        },
        mod.EXP3201_REL_PATH: {
            "schema_version": "1.0",
            "experiment_id": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
            "sidecar_promotion_allowed": False,
            "model_weight_update_performed": False,
            "heldout_regression_count": 0,
            "drift_regression_count": 0,
            "negative_control_regression_count": 0,
            "locality_violation_count": 0,
            "honest_verdict": "complete: sidecar audit finished",
        },
        mod.EXP3202_REL_PATH: {
            "schema_version": "carnot.sparse_potts_paoa_thrml_factor_boundary.v1",
            "experiment_id": "exp3202",
            "authenticated_hardware_transcript_present": False,
            "speedup_claim_allowed": False,
            "thrml_local_api_checked": True,
            "factor_record_count": 7,
            "hardware_claims_denied": [{"claim": "speedup", "denied": True}],
            "source_errors": [],
            "honest_verdict": "complete: hardware boundary materialized",
        },
    }
    for rel_path, payload in payloads.items():
        _write_json(root, rel_path, payload)


def _write_sources(root: Path, *, paper_ready: bool = False, blockers: int = 85) -> None:
    _write_json(
        root, mod.MATRIX_V30_REL_PATH, _matrix_v30(paper_ready=paper_ready, blockers=blockers)
    )
    _write_dot296_sources(root)


def test_req_report_3204_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3204: OpenSpec declares the capstone before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3204" in spec
    assert "SCENARIO-REPORT-3204" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3204_builds_capstone_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3204: .296 capstone preserves v30 claim boundaries."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.0)
    sources = {row["path"]: row for row in artifact["critical_source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3204"
    assert artifact["milestone"] == "2026.05.296"
    assert artifact["matrix_artifact"] == mod.MATRIX_V30_REL_PATH.as_posix()
    assert artifact["capstone_v296_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 85
    assert artifact["blocker_delta_from_v29"] == 5
    assert artifact["local_sota_receipt_status"] == (
        "blocked_cuda_unavailable_no_full_local_sota_receipt"
    )
    assert artifact["clean_verifier_status"] == (
        "gated_skipped_clean_verifier_v11_waiting_on_clean_rerun_allowed"
    )
    assert artifact["repair_gate_status"] == "blocked_clean_verifier_gate_skipped"
    assert artifact["repair_ladder_status"] == "gated_skipped_repair_gate_blocked"
    assert artifact["fr11_self_learning_status"] == (
        "controller_memory_trace_policy_promoted_no_model_weight_update_sidecar_promotion_blocked"
    )
    assert artifact["hardware_sampler_status"] == (
        "diagnostic_only_sparse_potts_thrml_factor_boundary_no_authenticated_speedup"
    )
    assert artifact["next_top_gap"] == (
        "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    )
    assert artifact["next_milestone_theme"] == (
        "cuda_offload_full_local_sota_receipt_and_clean_rerun_unblock"
    )
    assert artifact["ops_docs_updated"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_file_modified"] is False
    assert artifact["claim_boundaries_preserved"] == {
        "paper_ready_claim_allowed": False,
        "repair_claim_allowed": False,
        "hardware_speedup_claim_allowed": False,
        "tsu_or_kona_claim_allowed": False,
        "model_weight_self_learning_claim_allowed": False,
    }
    assert artifact["phase_outcomes"]["receipt_cuda"]["verdict"] == "blocked"
    assert artifact["phase_outcomes"]["clean_verifier"]["verdict"] == "gated_skipped"
    assert artifact["phase_outcomes"]["adaptive_repair_control"]["verdict"] == "blocked"
    assert artifact["phase_outcomes"]["fr11_self_learning"]["model_weight_update_claimed"] is False
    assert artifact["phase_outcomes"]["hardware_boundary"]["speedup_claim_allowed"] is False
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert sources[mod.EXP3193_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3193_REL_PATH
    )


def test_req_report_3204_paper_ready_is_copied_from_matrix_only(tmp_path: Path) -> None:
    """REQ-REPORT-3204: `paper_ready` follows matrix v30 rather than sources."""

    _write_sources(tmp_path, paper_ready=True, blockers=0)

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)

    assert artifact["capstone_v296_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v29"] == -80
    assert artifact["claim_boundaries_preserved"]["paper_ready_claim_allowed"] is True

    matrix = _matrix_v30(paper_ready=True, blockers=1)
    _write_json(tmp_path, mod.MATRIX_V30_REL_PATH, matrix)

    contradicted = mod.build_artifact(tmp_path)

    assert contradicted["capstone_v296_ready"] is False
    assert contradicted["paper_ready"] is True
    assert (
        "matrix_v30 paper_ready=true while publication blockers remain"
        in contradicted["invariant_violations"]
    )


def test_req_report_3204_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3204: missing or malformed evidence is visible and bounded."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v296_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["capstone_v296_ready"] is False
    assert empty["paper_ready"] is False
    assert empty["publication_blocker_count"] == 0
    assert empty["blocker_delta_from_v29"] is None
    assert empty["repair_gate_status"] == "missing_repair_gate_decision_v5"
    assert empty["repair_ladder_status"] == "missing_repair_ladder_v6"
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
    assert mod._repair_gate_status({}, []) == "missing_repair_gate_decision_v5"
    assert mod._repair_gate_status(
        {"repair_gate_state": "unblocked_for_bounded_repair_ladder"}, []
    ) == ("unblocked_for_bounded_repair_ladder")
    assert mod._repair_gate_status({}, [{"row_id": mod.REPAIR_GATE_ROW_ID, "status": "clean"}]) == (
        "clean_repair_gate_unblocked"
    )
    assert mod._repair_gate_status(
        {}, [{"row_id": mod.REPAIR_GATE_ROW_ID, "status": "blocked"}]
    ) == ("blocked_repair_gate_v5")
    assert mod._repair_ladder_status({}, []) == "missing_repair_ladder_v6"
    assert (
        mod._repair_ladder_status({"headline_claim_allowed": True}, []) == "repair_ladder_executed"
    )
    assert mod._repair_ladder_status({"status": "blocked"}, []) == "blocked_repair_ladder_v6"
    assert mod._repair_ladder_status(
        {}, [{"row_id": mod.REPAIR_LADDER_ROW_ID, "status": "clean"}]
    ) == ("repair_ladder_executed")
    assert mod._repair_ladder_status(
        {}, [{"row_id": mod.REPAIR_LADDER_ROW_ID, "status": "gated_skipped"}]
    ) == ("gated_skipped_repair_gate_blocked")
    assert mod._phase_verdict("clean_live_verifier_ready") == "passed"
    assert mod._phase_verdict("diagnostic_only_hardware_boundary") == "diagnostic_only"
    assert any(
        "source_artifacts_loaded omits critical" in violation
        for violation in mod._invariant_violations(
            {"cross_corpus_matrix_v30_ready": True, "source_artifacts_loaded": []},
            [],
            0,
            False,
        )
    )
    assert mod._next_milestone_theme("repair_gate_unblock_live_repair_attempts") == (
        "bounded_live_repair_ladder_execution_after_gate_unblock"
    )
    assert mod._next_milestone_theme("deployed_verifier_sidecar_promotion") == (
        "fr11_sidecar_promotion_without_model_weight_updates"
    )
    assert mod._next_milestone_theme(
        "authenticated_hardware_speedup_or_explicit_no_speedup_boundary"
    ) == ("authenticated_hardware_speedup_or_explicit_no_speedup_boundary")
    assert mod._next_milestone_theme("publication_blocker_retirement_review") == (
        "publication_blocker_retirement_review"
    )
