"""Tests for the Exp 3190 milestone .295 capstone.

Spec refs: REQ-REPORT-3190, SCENARIO-REPORT-3190.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v295_3190 as mod


REQUIRED_FIELDS = {
    "capstone_v295_ready",
    "matrix_authority",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v28",
    "missing_artifact_count",
    "verifier_status",
    "repair_gate_status",
    "repair_ladder_status",
    "fr11_self_learning_status",
    "sidecar_status",
    "hardware_sampler_status",
    "ops_docs_updated",
    "next_top_gap",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_v294() -> dict[str, Any]:
    return {
        "artifact": "experiment_3176_capstone_v294",
        "capstone_v294_ready": True,
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 73,
        "blocker_delta_from_v27": 8,
        "missing_artifact_count": 1,
        "honest_verdict": "complete: capstone_v294_ready=true",
    }


def _matrix_v29(*, clean: bool = False, paper_ready: bool | None = None) -> dict[str, Any]:
    blockers = 0 if clean else 80
    matrix_paper_ready = (blockers == 0) if paper_ready is None else paper_ready
    missing_artifacts = (
        []
        if clean
        else [
            {
                "experiment_id": "exp3141",
                "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
                "reason": "carried_forward_unresolved_missing_artifact_from_v28",
            }
        ]
    )
    return {
        "artifact": "experiment_3189_cross_corpus_matrix_v29",
        "cross_corpus_matrix_v29_ready": True,
        "prior_matrix_version": "v28",
        "prior_publication_blocker_count": 73,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v28": blockers - 73,
        "missing_artifacts": missing_artifacts,
        "missing_artifact_comparison": {
            "v28_missing_artifact_count": 1,
            "v29_missing_artifact_count": len(missing_artifacts),
            "missing_artifact_delta_from_v28": len(missing_artifacts) - 1,
        },
        "paper_ready": matrix_paper_ready,
        "paper_v6_narrowing_preserved": True,
        "paper_v6_narrowing": {
            "kv260_speedup_claimed": False,
            "tsu_or_kona_execution_claimed": False,
            "deployed_verifier_sidecar_claimed": False,
            "model_weight_self_learning_claimed": False,
            "paper_ready_streak_claimed": False,
        },
        "verifier_status": "clean_live_verifier_ready"
        if clean
        else "gated_skip_cpu_fallback_receipt_only_flagged_adversarial_controlled_invariance_passed_exact_authority_only",
        "repair_status": "clean_repair_ready"
        if clean
        else "blocked_receipt_precondition_repair_ladder_gated_skipped_certificate_expansion_flagged",
        "fr11_status": "controller_memory_promotion_allowed_cross_environment_replay_passed_no_model_weight_update",
        "sidecar_status": "clean_deployed_distributional_sidecar"
        if clean
        else "diagnostic_only_distributional_sidecar_no_deployed_verifier_claim",
        "hardware_status": "clean_authenticated_hardware_speedup"
        if clean
        else "diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution",
        "next_top_gap": "publication_scope_reconciliation"
        if clean
        else "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
        "status_counts": {
            "clean": 39 if not clean else 159,
            "flagged": 18 if not clean else 0,
            "blocked": 37 if not clean else 0,
            "gated_skipped": 14 if not clean else 0,
            "diagnostic_only": 9 if not clean else 0,
            "projection_only": 9 if not clean else 0,
            "missing": len(missing_artifacts),
            "retired": 31 if not clean else 0,
        },
        "inference_substrate": {
            "kind": "aggregation_from_checked_in_dot295_artifacts",
            "source": "matrix_v28_capstone_v294_archive_v295_and_dot295_artifacts",
            "executes_models": False,
            "executes_verifiers": False,
            "executes_repairs": False,
            "executes_solvers": False,
            "executes_hardware": False,
            "executes_conductor": False,
            "no_live_llm_inference": True,
            "local_repo_only": True,
        },
        "required_source_errors": [],
        "invariant_violations": [],
        "honest_verdict": "complete: cross_corpus_matrix_v29_ready=true",
    }


def _write_dot295_sources(root: Path, *, clean: bool = False) -> None:
    _write_json(
        root,
        mod.EXP3178_REL_PATH,
        {
            "receipt_backed_authenticity_contract_v3_ready": True,
            "flagged_adversarial": not clean,
            "honest_verdict": "complete: receipt contract ready",
        },
    )
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "substrate_classification": "gpu_receipt_clean"
            if clean
            else "cpu_fallback_receipt_only",
            "clean_rerun_allowed": clean,
            "headline_claim_allowed": clean,
            "live_call_count": 2,
            "flagged_adversarial": not clean,
            "honest_verdict": "complete: receipt smoke ready",
        },
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": True,
            "exact_row_count": 72,
            "receipt_backed_transcript_count": 2,
            "flagged_adversarial": not clean,
            "honest_verdict": "complete: invariance ready",
        },
    )
    _write_json(
        root,
        mod.EXP3181_REL_PATH,
        {
            "clean_live_sota_verifier_rerun_v10_ready": True,
            "gated_skip": not clean,
            "flagged_adversarial": not clean,
            "headline_claim_allowed": clean,
            "live_call_count": 2 if clean else 0,
            "gate_reasons": [] if clean else ["exp3179.clean_rerun_allowed=false"],
            "honest_verdict": "complete: clean rerun ready",
        },
    )
    _write_json(
        root,
        mod.EXP3182_REL_PATH,
        {
            "distributional_ebm_exact_row_sidecar_v1_ready": True,
            "deployed_verifier_claim_allowed": clean,
            "exact_labeled_row_count": 72,
            "known_false_accept_rows_scored": 2,
            "honest_verdict": "complete: sidecar ready",
        },
    )
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "repair_call_ready": clean,
            "flagged_adversarial": not clean,
            "counterexample_count": 4,
            "honest_verdict": "complete: certificate expansion ready",
        },
    )
    _write_json(
        root,
        mod.EXP3184_REL_PATH,
        {
            "repair_gate_decision_v4_ready": True,
            "repair_gate_state": "unblocked" if clean else "blocked_receipt_precondition",
            "blocker_reasons": [] if clean else ["exp3179.clean_rerun_allowed is not true"],
            "honest_verdict": "complete: repair gate ready",
        },
    )
    _write_json(
        root,
        mod.EXP3185_REL_PATH,
        {
            "multi_turn_repair_ladder_v5_ready": True,
            "gated_skip": not clean,
            "gate_state": "unblocked" if clean else "blocked_receipt_precondition",
            "headline_claim_allowed": clean,
            "repair_attempt_count": 3 if clean else 0,
            "repair_success_delta": 0.4 if clean else 0.0,
            "honest_verdict": "complete: repair ladder ready",
        },
    )
    _write_json(
        root,
        mod.EXP3186_REL_PATH,
        {
            "fr11_controller_memory_promotion_pack_v1_ready": True,
            "promotion_allowed": True,
            "no_model_weight_update_claimed": True,
            "honest_verdict": "complete: promotion pack ready",
        },
    )
    _write_json(
        root,
        mod.EXP3187_REL_PATH,
        {
            "fr11_cross_environment_drift_replay_v1_ready": True,
            "promotion_allowed": True,
            "rollback_triggered": False,
            "negative_control_regression_count": 0,
            "no_model_weight_update_claimed": True,
            "honest_verdict": "complete: drift replay ready",
        },
    )
    _write_json(
        root,
        mod.EXP3188_REL_PATH,
        {
            "thrml_factor_graph_api_boundary_v1_ready": True,
            "local_api_smoke_passed": True,
            "hardware_speedup_claim_allowed": clean,
            "kona_or_tsu_execution_claimed": False,
            "honest_verdict": "complete: THRML boundary ready",
        },
    )


def _write_sources(root: Path, *, clean: bool = False, paper_ready: bool | None = None) -> None:
    _write_json(root, mod.MATRIX_V29_REL_PATH, _matrix_v29(clean=clean, paper_ready=paper_ready))
    _write_json(root, mod.CAPSTONE_V294_REL_PATH, _capstone_v294())
    _write_dot295_sources(root, clean=clean)


def test_req_report_3190_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3190: OpenSpec declares the v295 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3190" in spec
    assert "SCENARIO-REPORT-3190" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3190_builds_blocked_paper_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3190: .295 closes honestly while paper remains blocked."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_v295_ready"] is True
    assert artifact["capstone_ready"] is True
    assert artifact["matrix_authority"] == mod.MATRIX_V29_REL_PATH.as_posix()
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 80
    assert artifact["blocker_delta_from_v28"] == 7
    assert artifact["missing_artifact_count"] == 1
    assert artifact["local_sota_receipt_status"] == (
        "cpu_fallback_receipt_only_non_headline_clean_rerun_blocked"
    )
    assert artifact["controlled_invariance_status"] == (
        "passed_controlled_invariance_exact_authority_receipts_flagged"
    )
    assert artifact["verifier_status"] == (
        "gated_skip_cpu_fallback_receipt_only_flagged_adversarial_controlled_invariance_passed_exact_authority_only"
    )
    assert artifact["repair_gate_status"] == "blocked_receipt_precondition"
    assert artifact["repair_ladder_status"] == (
        "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
    )
    assert artifact["fr11_self_learning_status"] == (
        "controller_memory_promotion_allowed_cross_environment_replay_passed_no_model_weight_update"
    )
    assert artifact["fr11_promotion_drift_replay_passed"] is True
    assert artifact["sidecar_status"] == (
        "diagnostic_only_distributional_sidecar_no_deployed_verifier_claim"
    )
    assert artifact["hardware_sampler_status"] == (
        "diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution"
    )
    assert artifact["thrml_boundary_status"] == (
        "local_api_smoke_only_no_speedup_no_tsu_kona_execution"
    )
    assert artifact["ops_docs_updated"] is False
    assert artifact["ops_reconciliation_decision"]["delegated_to_conductor"] is True
    assert artifact["next_top_gap"] == (
        "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    )
    assert artifact["paper_v6_narrowing_preserved"] is True
    assert artifact["phase_outcome_summary"] == {
        "receipt_backed_local_sota_path": "cpu_only_non_headline_evidence",
        "controlled_invariance_passed": True,
        "clean_verifier_unblocked": False,
        "repair_gate_unblocked": False,
        "repair_ladder_executed": False,
        "distributional_sidecar_deployed": False,
        "fr11_controller_memory_promoted_without_weight_update": True,
        "thrml_boundary_local_api_only": True,
    }
    assert artifact["inference_substrate"] == {
        "kind": "capstone_aggregation_from_checked_in_matrix_v29_and_dot295_artifacts",
        "source": "matrix_v29_capstone_v294_and_dot295_phase_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert sources[mod.MATRIX_V29_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V29_REL_PATH
    )


def test_req_report_3190_paper_ready_follows_matrix_and_all_gates(tmp_path: Path) -> None:
    """REQ-REPORT-3190: `paper_ready` is explicit and fail-closed."""

    _write_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["capstone_v295_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v28"] == -73
    assert artifact["missing_artifact_count"] == 0
    assert artifact["local_sota_receipt_status"] == "passed_receipt_clean_rerun_allowed"
    assert artifact["repair_gate_status"] == "clean_repair_gate_unblocked"
    assert artifact["repair_ladder_status"] == "clean_repair_ladder_materialized"
    assert artifact["thrml_boundary_status"] == "local_api_smoke_with_bounded_speedup_permission"
    assert artifact["next_top_gap"] == "publication_scope_reconciliation"

    _write_sources(tmp_path, clean=True, paper_ready=False)
    matrix_false = mod.build_artifact(tmp_path)

    assert matrix_false["capstone_v295_ready"] is True
    assert matrix_false["paper_ready"] is False
    assert matrix_false["next_top_gap"] == "publication_scope_reconciliation"

    _write_sources(tmp_path, clean=True, paper_ready=True)
    matrix = json.loads((tmp_path / mod.MATRIX_V29_REL_PATH).read_text(encoding="utf-8"))
    matrix["publication_blocker_count"] = 1
    _write_json(tmp_path, mod.MATRIX_V29_REL_PATH, matrix)

    contradicted = mod.build_artifact(tmp_path)

    assert contradicted["capstone_v295_ready"] is False
    assert contradicted["paper_ready"] is False
    assert (
        "matrix_v29 paper_ready cannot coexist with publication blockers"
        in contradicted["invariant_violations"]
    )


def test_req_report_3190_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3190: missing and malformed evidence blocks the capstone."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v295_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    empty = mod.build_artifact(tmp_path / "empty")
    assert empty["capstone_v295_ready"] is False
    assert empty["honest_verdict"].startswith("blocked:")
    assert empty["source_artifacts"][0]["present"] is False
    assert "matrix_v29 authority is missing or malformed" in empty["invariant_violations"]

    assert mod._local_sota_receipt_status({}) == "missing_local_sota_receipt"
    assert mod._local_sota_receipt_status({"local_sota_receipt_smoke_v3_ready": False}) == (
        "blocked_local_sota_receipt_not_ready"
    )
    assert mod._local_sota_receipt_status({"local_sota_receipt_smoke_v3_ready": True}) == (
        "blocked_local_sota_receipt_not_headline_eligible"
    )
    assert mod._controlled_invariance_status({}) == "missing_controlled_invariance"
    assert mod._controlled_invariance_status(
        {"controlled_invariance_executor_v2_ready": False}
    ) == ("blocked_controlled_invariance_not_ready")
    assert mod._controlled_invariance_status({"controlled_invariance_executor_v2_ready": True}) == (
        "blocked_controlled_invariance_not_passed"
    )
    assert mod._clean_verifier_status({}, "missing") == "missing_clean_verifier_rerun"
    assert mod._clean_verifier_status({"clean_live_sota_verifier_rerun_v10_ready": False}, "") == (
        "blocked_clean_verifier_rerun_not_ready"
    )
    assert mod._clean_verifier_status({"clean_live_sota_verifier_rerun_v10_ready": True}, "") == (
        "blocked_clean_verifier_not_headline_eligible"
    )
    assert mod._repair_gate_status({}) == "missing_repair_gate_decision"
    assert mod._repair_gate_status({"repair_gate_decision_v4_ready": False}) == (
        "blocked_repair_gate_decision_not_ready"
    )
    assert mod._repair_ladder_status({}) == "missing_repair_ladder"
    assert mod._repair_ladder_status({"multi_turn_repair_ladder_v5_ready": False}) == (
        "blocked_repair_ladder_not_ready"
    )
    assert mod._repair_ladder_status({"multi_turn_repair_ladder_v5_ready": True}) == (
        "blocked_repair_ladder_not_promotable"
    )
    assert mod._fr11_promotion_drift_replay_passed({}, {}) is False
    assert mod._thrml_boundary_status({}) == "missing_thrml_boundary"
    assert mod._thrml_boundary_status({"thrml_factor_graph_api_boundary_v1_ready": False}) == (
        "blocked_thrml_boundary_not_ready"
    )
    assert mod._thrml_boundary_status({"thrml_factor_graph_api_boundary_v1_ready": True}) == (
        "blocked_thrml_boundary_local_api_smoke_missing"
    )
    assert (
        mod._thrml_boundary_status(
            {
                "thrml_factor_graph_api_boundary_v1_ready": True,
                "local_api_smoke_passed": True,
                "hardware_speedup_claim_allowed": True,
                "kona_or_tsu_execution_claimed": True,
            }
        )
        == "blocked_thrml_boundary_overclaimed_execution"
    )
    assert mod._phase_receipt_outcome("passed_receipt_clean_rerun_allowed") == "passed"
    assert (
        mod._phase_receipt_outcome("cpu_fallback_receipt_only_non_headline_clean_rerun_blocked")
        == "cpu_only_non_headline_evidence"
    )
    assert mod._phase_receipt_outcome("blocked_local_sota_receipt_not_ready") == "blocked"
    assert (
        mod._next_top_gap("passed_receipt_clean_rerun_allowed", "blocked_receipt_precondition")
        == "repair_gate_unblock"
    )
    assert (
        mod._next_top_gap("passed_receipt_clean_rerun_allowed", "clean_repair_gate_unblocked")
        == "publication_scope_reconciliation"
    )
    assert mod._list(("not", "a", "list")) == []
