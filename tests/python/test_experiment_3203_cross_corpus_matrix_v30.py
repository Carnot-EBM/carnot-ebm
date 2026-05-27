"""Tests for Exp 3203 cross-corpus matrix v30.

Spec refs: REQ-REPORT-3203, SCENARIO-REPORT-3203.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v30_3203 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "matrix_version",
    "source_artifacts_expected",
    "source_artifacts_loaded",
    "missing_artifact_count",
    "status_counts",
    "publication_blocker_count",
    "blocker_delta_from_v29",
    "local_sota_receipt_status",
    "clean_verifier_status",
    "repair_status",
    "fr11_self_learning_status",
    "hardware_sampler_status",
    "paper_ready",
    "next_top_gap",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_v29() -> dict[str, Any]:
    return {
        "artifact": "experiment_3189_cross_corpus_matrix_v29",
        "cross_corpus_matrix_v29_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 80,
        "blocker_delta_from_v28": 7,
        "status_counts": {
            "blocked": 37,
            "clean": 39,
            "diagnostic_only": 9,
            "flagged": 18,
            "gated_skipped": 14,
            "missing": 2,
            "projection_only": 9,
            "retired": 31,
        },
        "next_top_gap": "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
        "paper_v6_narrowing_preserved": True,
        "honest_verdict": "complete: cross_corpus_matrix_v29_ready=true",
    }


def _write_matrix_v29(root: Path) -> None:
    _write_json(root, mod.MATRIX_V29_REL_PATH, _matrix_v29())


def _write_dot296_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3191_REL_PATH,
        {
            "schema_version": "carnot.archive_v295_activate_v296.v1",
            "experiment_id": "exp3191",
            "activation_ready": True,
            "prior_publication_blocker_count": 80,
            "prior_next_top_gap": "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock",
            "source_artifacts": [{"path": "results/experiment_3189_cross_corpus_matrix_v29.json"}],
            "source_checksums": {"results/experiment_3189_cross_corpus_matrix_v29.json": "abc"},
            "inference_substrate": {"kind": "aggregation_only"},
            "duration_s": 1.0,
            "honest_verdict": "complete: activation_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3192_REL_PATH,
        {
            "schema_version": "carnot.receipt_adversarial_contract.v4",
            "experiment_id": "exp3192",
            "receipt_adversarial_contract_v4_ready": True,
            "current_evidence_assessment": {
                "clean_rerun_allowed": False,
                "headline_claim_allowed": False,
                "proof_execution_sufficient": True,
                "proof_receipt_count": 2,
                "substrate_classification": "cpu_fallback_receipt_only",
            },
            "source_artifacts": [{"path": "results/experiment_3179_local_sota_receipt_smoke_v3.json"}],
            "source_checksums": {"results/experiment_3179_local_sota_receipt_smoke_v3.json": "def"},
            "inference_substrate": {"kind": "contract_artifact_only"},
            "duration_s": 0.1,
            "honest_verdict": (
                "complete: receipt_adversarial_contract_v4_ready=true; "
                "proof_execution_sufficient=true; clean_rerun_allowed=false"
            ),
        },
    )
    _write_json(
        root,
        mod.EXP3193_REL_PATH,
        {
            "schema_version": "carnot.llama_cpp_cuda_offload_health_probe.v1",
            "experiment_id": "exp3193",
            "substrate_classification": "cuda_unavailable",
            "clean_rerun_allowed": False,
            "headline_claim_allowed": False,
            "flagged_adversarial": True,
            "corrigendum_pending": False,
            "receipt_count": 0,
            "blocker_reasons": ["selected Python torch.cuda.is_available() is false"],
            "model_specs": [{"model_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
            "inference_substrate": {"kind": "cuda_offload_probe", "executes_models": False},
            "duration_s": 2.0,
            "honest_verdict": "blocked_cuda_unavailable: clean_rerun_allowed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3194_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3194,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp3193.clean_rerun_allowed actual=False expected=True",
            "gates_evaluated": [
                {
                    "upstream": "exp3193-llama-cpp-cuda-offload-health-probe-v1",
                    "artifact_field": "clean_rerun_allowed",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
            "duration_s": 0.0,
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3195_REL_PATH,
        {
            "schema_version": "carnot.adaptive_verification_granularity_policy.v1",
            "experiment_id": "exp3195",
            "adaptive_verification_granularity_policy_v1_ready": True,
            "promotion_allowed": False,
            "simulated_rows": 72,
            "estimated_verifier_call_delta": -71,
            "source_artifacts": [],
            "source_checksums": {},
            "inference_substrate": {"kind": "artifact_only_policy_simulation"},
            "duration_s": 0.2,
            "honest_verdict": "complete: adaptive_verification_granularity_policy_v1_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3196_REL_PATH,
        {
            "schema_version": "carnot.gencp_domain_preview_repair_compiler.v1",
            "experiment_id": "exp3196",
            "preview_domain_count": 12,
            "repair_call_ready": False,
            "promotion_allowed": False,
            "source_artifacts": [],
            "source_checksums": {},
            "source_errors": [],
            "inference_substrate": {"kind": "artifact_only_domain_preview_compiler"},
            "honest_verdict": "complete: gencp_domain_preview_repair_compiler_v1_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3197_REL_PATH,
        {
            "schema_version": "carnot.exverus_inductive_certificate_expansion.v1",
            "experiment_id": "exp3197",
            "invariant_record_count": 5,
            "repair_call_ready": False,
            "source_artifacts": [],
            "source_checksums": {},
            "source_errors": [],
            "inference_substrate": {"kind": "artifact_only_inductive_certificate_expansion"},
            "honest_verdict": "complete: exverus_inductive_certificate_expansion_v1_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3198_REL_PATH,
        {
            "schema_version": "carnot.repair_gate_decision.v5",
            "experiment_id": "exp3198",
            "repair_gate_state": "blocked_clean_verifier_gate_skipped",
            "clean_verifier_state": "blocked_gate_skipped_conductor_pre_gate",
            "receipt_gate_state": "blocked_cuda_unavailable",
            "downstream_gated_skip_expected": True,
            "blocker_reasons": [{"code": "exp3193_clean_rerun_not_allowed"}],
            "source_artifacts": [],
            "source_checksums": {},
            "inference_substrate": {"kind": "deterministic_repair_gate_decision_v5"},
            "honest_verdict": "complete: repair_gate_state=blocked_clean_verifier_gate_skipped",
        },
    )
    _write_json(
        root,
        mod.EXP3199_REL_PATH,
        {
            "schema": "blocked_gate_check_v1",
            "experiment": 3199,
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "exp3198.repair_gate_state actual=blocked expected=unblocked",
            "gates_evaluated": [
                {
                    "upstream": "exp3198-repair-gate-decision-v5",
                    "artifact_field": "repair_gate_state",
                    "expected": "unblocked_for_bounded_repair_ladder",
                    "actual": "blocked_clean_verifier_gate_skipped",
                    "passed": False,
                }
            ],
            "duration_s": 0.0,
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3200_REL_PATH,
        {
            "schema_version": "1.0",
            "experiment_id": "experiment_3200_fr11_verify_trace_memory_controller_v1",
            "continuous_self_learning_task": True,
            "promotion_allowed": True,
            "promotion_blockers": [],
            "model_weight_update_performed": False,
            "negative_control_regression_count": 0,
            "trace_count": 30,
            "inference_substrate": {
                "controller_memory_replay_only": True,
                "model_weight_mutation": False,
            },
            "duration_s": 0.3,
            "honest_verdict": "complete: fr11 verify trace-memory controller v1 materialized",
        },
    )
    _write_json(
        root,
        mod.EXP3201_REL_PATH,
        {
            "schema_version": "1.0",
            "experiment_id": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
            "sidecar_promotion_allowed": False,
            "model_weight_update_performed": False,
            "heldout_regression_count": 0,
            "drift_regression_count": 0,
            "negative_control_regression_count": 0,
            "locality_violation_count": 0,
            "inference_substrate": {"sidecar_audit_only": True, "sidecar_verifier_authority": False},
            "duration_s": 0.1,
            "honest_verdict": "complete: kan-cl nonforgetting sidecar audit finished",
        },
    )
    _write_json(
        root,
        mod.EXP3202_REL_PATH,
        {
            "schema_version": "carnot.sparse_potts_paoa_thrml_factor_boundary.v1",
            "experiment_id": "exp3202",
            "authenticated_hardware_transcript_present": False,
            "speedup_claim_allowed": False,
            "thrml_local_api_checked": True,
            "factor_record_count": 7,
            "hardware_claims_denied": [{"claim": "speedup", "denied": True}],
            "source_artifacts": [],
            "source_errors": [],
            "inference_substrate": {
                "kind": "local_factor_record_only_no_hardware_speedup",
                "executes_hardware": False,
            },
            "honest_verdict": "complete: sparse Potts/PAOA/THRML factor boundary materialized",
        },
    )


def test_req_report_3203_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3203: OpenSpec declares matrix v30 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3203" in spec
    assert "SCENARIO-REPORT-3203" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3203_builds_v30_from_dot296_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3203: .296 artifacts preserve claim boundaries."""

    _write_matrix_v29(tmp_path)
    _write_dot296_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.5)
    rows = {row["row_id"]: row for row in artifact["artifact_classifications"]}
    sources = {row["path"]: row for row in artifact["source_artifact_records"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["cross_corpus_matrix_v30_ready"] is True
    assert artifact["matrix_version"] == "v30"
    assert artifact["prior_matrix_version"] == "v29"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert len(artifact["source_artifacts_expected"]) == 13
    assert len(artifact["source_artifacts_loaded"]) == 13
    assert artifact["missing_artifact_count"] == 0
    assert artifact["status_counts"] == {
        "clean": 3,
        "blocked": 3,
        "gated_skipped": 2,
        "diagnostic_only": 5,
        "retired": 0,
        "missing": 0,
    }
    assert artifact["publication_blocker_count"] == 85
    assert artifact["blocker_delta_from_v29"] == 5
    assert artifact["paper_ready"] is False
    assert artifact["paper_v6_narrowing_preserved"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["authority:exp3189_matrix_v29"]["status"] == "clean"
    assert rows["dot296:exp3191_archive_activation"]["status"] == "clean"
    assert rows["dot296:exp3192_receipt_contract_v4"]["status"] == "blocked"
    assert rows["dot296:exp3193_cuda_offload_health_probe"]["status"] == "blocked"
    assert rows["dot296:exp3194_clean_verifier_rerun_v11"]["status"] == "gated_skipped"
    assert rows["dot296:exp3195_adaptive_granularity_policy"]["status"] == "diagnostic_only"
    assert rows["dot296:exp3196_domain_preview_compiler"]["status"] == "diagnostic_only"
    assert rows["dot296:exp3197_inductive_certificate_expansion"]["status"] == "diagnostic_only"
    assert rows["dot296:exp3198_repair_gate_v5"]["status"] == "blocked"
    assert rows["dot296:exp3199_repair_ladder_v6"]["status"] == "gated_skipped"
    assert rows["dot296:exp3200_fr11_trace_memory_controller"]["status"] == "clean"
    assert rows["dot296:exp3201_kan_cl_sidecar_audit"]["status"] == "diagnostic_only"
    assert rows["dot296:exp3202_sparse_potts_thrml_boundary"]["status"] == "diagnostic_only"

    assert rows["dot296:exp3194_clean_verifier_rerun_v11"]["contract_v4_adversarial"][
        "blocked_verdict"
    ] is True
    assert rows["dot296:exp3194_clean_verifier_rerun_v11"]["contract_v4_methodology"][
        "schema_version_present"
    ] is False
    assert rows["dot296:exp3193_cuda_offload_health_probe"]["contract_v4_adversarial"][
        "flagged_adversarial"
    ] is True

    assert artifact["local_sota_receipt_status"] == (
        "blocked_cuda_unavailable_no_full_local_sota_receipt"
    )
    assert artifact["clean_verifier_status"] == (
        "gated_skipped_clean_verifier_v11_waiting_on_clean_rerun_allowed"
    )
    assert artifact["repair_status"] == (
        "blocked_clean_verifier_gate_repair_ladder_gated_skipped"
    )
    assert artifact["fr11_self_learning_status"] == (
        "controller_memory_trace_policy_promoted_no_model_weight_update_sidecar_promotion_blocked"
    )
    assert artifact["hardware_sampler_status"] == (
        "diagnostic_only_sparse_potts_thrml_factor_boundary_no_authenticated_speedup"
    )
    assert artifact["next_top_gap"] == (
        "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    )
    assert artifact["required_evidence_blocked_or_missing"] == [
        "local_sota_receipt",
        "clean_verifier",
        "repair",
        "deployed_verifier_sidecar",
        "hardware_sampler",
    ]
    assert sources[mod.EXP3193_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3193_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_dot296_artifacts",
        "source": "matrix_v29_and_dot296_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3203_missing_dot296_artifacts_are_counted(tmp_path: Path) -> None:
    """REQ-REPORT-3203: missing expected `.296` artifacts are visible blockers."""

    _write_matrix_v29(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP3191_REL_PATH,
        {
            "schema_version": "carnot.archive_v295_activate_v296.v1",
            "experiment_id": "exp3191",
            "activation_ready": True,
            "source_artifacts": [],
            "source_checksums": {},
            "inference_substrate": {"kind": "aggregation_only"},
            "duration_s": 0.1,
            "honest_verdict": "complete: activation_ready=true",
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)
    missing_rows = [
        row for row in artifact["artifact_classifications"] if row["status"] == "missing"
    ]

    assert artifact["cross_corpus_matrix_v30_ready"] is True
    assert artifact["missing_artifact_count"] == 11
    assert len(missing_rows) == 11
    assert artifact["status_counts"] == {
        "clean": 2,
        "blocked": 0,
        "gated_skipped": 0,
        "diagnostic_only": 0,
        "retired": 0,
        "missing": 11,
    }
    assert artifact["publication_blocker_count"] == 91
    assert artifact["blocker_delta_from_v29"] == 11
    assert artifact["source_artifacts_loaded"] == [
        mod.MATRIX_V29_REL_PATH.as_posix(),
        mod.EXP3191_REL_PATH.as_posix(),
    ]

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["cross_corpus_matrix_v30_ready"] is False
    assert empty["blocker_delta_from_v29"] is None
    assert empty["honest_verdict"].startswith("blocked_matrix_v30_preconditions")


def test_req_report_3203_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3203: helper behavior is deterministic and fail-closed."""

    _write_matrix_v29(tmp_path)
    _write_dot296_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cross_corpus_matrix_v30_ready"] is True
    assert mod._status_counts([{"status": "bad"}])["missing"] == 1
    assert mod._normal_status("gated-skip") == "gated_skipped"
    assert mod._normal_status("unknown") == "missing"
    assert mod._experiment_id({"experiment_id": "exp1"}, "fallback") == "exp1"
    assert mod._experiment_id({"experiment": 7}, "fallback") == "exp7"
    assert mod._experiment_id({}, "fallback") == "fallback"
    assert mod._nested({}, "a.b") is None
    assert mod._nested({"a": {"b": False}}, "a.b") is False
    assert mod._bool_field({"a": {"b": True}}, "a.b") is True
    assert mod._text_list(["x", 2]) == ["x", "2"]
    assert mod._text_list("x") == ["x"]
    assert mod._source_payload(tmp_path / "missing", mod.SOURCE_SPECS[0])["status"] == "missing"
    assert mod._source_payload(tmp_path, mod.SOURCE_SPECS[0])["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V29_REL_PATH
    )


def test_req_report_3203_fail_closed_branch_matrix() -> None:
    """REQ-REPORT-3203: branch edges keep future matrix states fail-closed."""

    assert mod._classification_status("exp3189", {"cross_corpus_matrix_v29_ready": False}) == (
        "blocked",
        "matrix_v29 authority is not ready",
    )
    assert mod._classification_status("exp3191", {"activation_ready": False}) == (
        "blocked",
        "archive activation is not ready",
    )
    assert mod._classification_status(
        "exp3192",
        {
            "current_evidence_assessment": {
                "clean_rerun_allowed": True,
                "headline_claim_allowed": True,
                "substrate_classification": "full_local_sota_receipt",
            }
        },
    )[0] == "clean"
    assert mod._classification_status(
        "exp3193",
        {
            "clean_rerun_allowed": True,
            "headline_claim_allowed": True,
            "substrate_classification": "full_local_sota_receipt",
        },
    )[0] == "clean"
    assert mod._classification_status("exp3194", {"headline_claim_allowed": True})[0] == "clean"
    assert mod._classification_status("exp3194", {"headline_claim_allowed": False})[0] == "blocked"
    assert mod._classification_status(
        "exp3195", {"adaptive_verification_granularity_policy_v1_ready": False}
    )[0] == "blocked"
    assert mod._classification_status(
        "exp3195",
        {"adaptive_verification_granularity_policy_v1_ready": True, "promotion_allowed": True},
    )[0] == "clean"
    assert mod._classification_status("exp3196", {"source_errors": ["bad"]})[0] == "blocked"
    assert mod._classification_status("exp3196", {"repair_call_ready": True})[0] == "clean"
    assert mod._classification_status("exp3197", {"source_errors": ["bad"]})[0] == "blocked"
    assert mod._classification_status("exp3197", {"repair_call_ready": True})[0] == "clean"
    assert mod._classification_status(
        "exp3198", {"repair_gate_state": "unblocked_for_bounded_repair_ladder"}
    )[0] == "clean"
    assert mod._classification_status(
        "exp3200", {"promotion_allowed": False, "model_weight_update_performed": False}
    )[0] == "blocked"
    assert mod._classification_status("exp3201", {"sidecar_promotion_allowed": True})[0] == "clean"
    assert mod._classification_status(
        "exp3201", {"model_weight_update_performed": True}
    )[0] == "blocked"
    assert mod._classification_status("exp3202", {"source_errors": ["bad"]})[0] == "blocked"
    assert mod._classification_status(
        "exp3202",
        {"authenticated_hardware_transcript_present": True, "speedup_claim_allowed": True},
    )[0] == "clean"
    assert mod._classification_status("exp3202", {"speedup_claim_allowed": True})[0] == "blocked"
    assert mod._classification_status("exp9999", {"artifact": "unknown"})[0] == "blocked"

    assert (
        mod._local_sota_receipt_status(
            {
                "exp3192": {},
                "exp3193": {
                    "clean_rerun_allowed": True,
                    "headline_claim_allowed": True,
                    "substrate_classification": "full_local_sota_receipt",
                },
            }
        )
        == "passed_full_local_sota_receipt_clean_rerun_allowed"
    )
    assert (
        mod._local_sota_receipt_status(
            {
                "exp3192": {
                    "current_evidence_assessment": {
                        "substrate_classification": "cpu_fallback_receipt_only"
                    }
                },
                "exp3193": {},
            }
        )
        == "blocked_cpu_fallback_receipt_only_non_headline"
    )
    assert (
        mod._local_sota_receipt_status({"exp3192": {}, "exp3193": {}})
        == "missing_local_sota_receipt_evidence"
    )
    assert (
        mod._local_sota_receipt_status({"exp3192": {"current_evidence_assessment": {}}, "exp3193": {}})
        == "blocked_no_full_local_sota_receipt"
    )

    assert (
        mod._clean_verifier_status(
            [{"row_id": "dot296:exp3194_clean_verifier_rerun_v11", "status": "clean"}]
        )
        == "clean_live_verifier_ready"
    )
    assert (
        mod._clean_verifier_status(
            [{"row_id": "dot296:exp3194_clean_verifier_rerun_v11", "status": "blocked"}]
        )
        == "blocked_clean_verifier_v11"
    )
    assert (
        mod._repair_status(
            {"exp3198": {}},
            [
                {"row_id": "dot296:exp3198_repair_gate_v5", "status": "clean"},
                {"row_id": "dot296:exp3199_repair_ladder_v6", "status": "clean"},
            ],
        )
        == "repair_ready"
    )
    assert (
        mod._repair_status(
            {"exp3198": {"downstream_gated_skip_expected": True}},
            [
                {"row_id": "dot296:exp3198_repair_gate_v5", "status": "blocked"},
                {"row_id": "dot296:exp3199_repair_ladder_v6", "status": "blocked"},
            ],
        )
        == "blocked_repair_gate_downstream_gated_skip_expected"
    )
    assert (
        mod._repair_status(
            {"exp3198": {}},
            [
                {"row_id": "dot296:exp3198_repair_gate_v5", "status": "blocked"},
                {"row_id": "dot296:exp3199_repair_ladder_v6", "status": "blocked"},
            ],
        )
        == "blocked_repair"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3200": {}, "exp3201": {}},
            [
                {"row_id": "dot296:exp3200_fr11_trace_memory_controller", "status": "clean"},
                {"row_id": "dot296:exp3201_kan_cl_sidecar_audit", "status": "clean"},
            ],
        )
        == "controller_memory_and_sidecar_promoted_no_model_weight_update"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3200": {"model_weight_update_performed": True}, "exp3201": {}},
            [{"row_id": "dot296:exp3200_fr11_trace_memory_controller", "status": "blocked"}],
        )
        == "blocked_fr11_model_weight_update_claimed"
    )
    assert (
        mod._fr11_self_learning_status(
            {"exp3200": {"model_weight_update_performed": False}, "exp3201": {}},
            [{"row_id": "dot296:exp3200_fr11_trace_memory_controller", "status": "blocked"}],
        )
        == "blocked_fr11_trace_memory_controller"
    )
    assert (
        mod._hardware_sampler_status(
            {"exp3202": {}},
            [{"row_id": "dot296:exp3202_sparse_potts_thrml_boundary", "status": "clean"}],
        )
        == "authenticated_hardware_speedup_claim_allowed"
    )
    assert (
        mod._hardware_sampler_status(
            {"exp3202": {"speedup_claim_allowed": True}},
            [{"row_id": "dot296:exp3202_sparse_potts_thrml_boundary", "status": "blocked"}],
        )
        == "blocked_unsupported_hardware_speedup_claim"
    )
    assert (
        mod._hardware_sampler_status(
            {"exp3202": {"speedup_claim_allowed": False}},
            [{"row_id": "dot296:exp3202_sparse_potts_thrml_boundary", "status": "blocked"}],
        )
        == "blocked_hardware_sampler_boundary"
    )

    assert (
        mod._next_top_gap(
            "passed_full_local_sota_receipt_clean_rerun_allowed",
            "blocked_clean_verifier_v11",
            "blocked_repair",
            "blocked_fr11",
            "blocked_hardware",
        )
        == "clean_live_verifier_adversarial_flag_clearance"
    )
    assert (
        mod._next_top_gap(
            "passed_full_local_sota_receipt_clean_rerun_allowed",
            "clean_live_verifier_ready",
            "blocked_repair",
            "blocked_fr11",
            "blocked_hardware",
        )
        == "repair_gate_unblock_live_repair_attempts"
    )
    assert (
        mod._next_top_gap(
            "passed_full_local_sota_receipt_clean_rerun_allowed",
            "clean_live_verifier_ready",
            "repair_ready",
            "blocked_fr11",
            "blocked_hardware",
        )
        == "deployed_verifier_sidecar_promotion"
    )
    assert (
        mod._next_top_gap(
            "passed_full_local_sota_receipt_clean_rerun_allowed",
            "clean_live_verifier_ready",
            "repair_ready",
            "controller_memory_and_sidecar_promoted_no_model_weight_update",
            "blocked_hardware",
        )
        == "authenticated_hardware_speedup_or_explicit_no_speedup_boundary"
    )
    assert (
        mod._next_top_gap(
            "passed_full_local_sota_receipt_clean_rerun_allowed",
            "clean_live_verifier_ready",
            "repair_ready",
            "controller_memory_and_sidecar_promoted_no_model_weight_update",
            "authenticated_hardware_speedup_claim_allowed",
        )
        == "publication_blocker_retirement_review"
    )

    assert mod._prior_publication_blocker_count({"publication_blocker_count": True}) is None
    assert any(
        "status_counts keys" in violation
        for violation in mod._invariant_violations({}, [], {}, 0, None, [])
    )
    assert any(
        "status_counts do not sum" in violation
        for violation in mod._invariant_violations(
        {"cross_corpus_matrix_v29_ready": True}, [], {status: 1 for status in mod.STATUSES}, 0, None, []
    )
    )
    assert any(
        "publication_blocker_count does not reconcile" in violation
        for violation in mod._invariant_violations(
        {"cross_corpus_matrix_v29_ready": True},
        [],
        {status: 0 for status in mod.STATUSES},
        5,
        1,
        [],
    )
    )
    assert mod._row_status([], "missing") == "missing"
    assert mod._is_gate_skip({"gated_skip": True}) is True
    assert mod._is_gate_skip({"schema": "blocked_gate_check_v1"}) is True
    assert mod._is_gate_skip({}) is False
    assert mod._corrigendum_pending(["pending"]) is True
    assert mod._int_value(True) == 0
    assert mod._int_value("5") == 0
