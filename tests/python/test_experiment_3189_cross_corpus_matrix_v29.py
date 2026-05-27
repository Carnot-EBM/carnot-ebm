"""Tests for Exp 3189 cross-corpus matrix v29.

Spec refs: REQ-REPORT-3189, SCENARIO-REPORT-3189.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v29_3189 as mod


REQUIRED_FIELDS = {
    "cross_corpus_matrix_v29_ready",
    "prior_matrix_version",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v28",
    "clean_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skip_rows",
    "diagnostic_only_rows",
    "missing_artifacts",
    "verifier_status",
    "repair_status",
    "fr11_status",
    "hardware_status",
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


def _row(row_id: str, status: str, *, claim_scope: str = "headline") -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id}.json",
        "source_field": "status",
        "evidence_class": "v28_carry",
        "blocker_class": mod.blocker_class(status, claim_scope),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
        "row_origin": "matrix_v28_test",
    }


def _matrix_v28(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("carry_clean", "clean"),
        _row("carry_flagged", "flagged"),
        _row("carry_blocked", "blocked"),
        _row("carry_gated", "gated_skipped"),
        _row("carry_diagnostic", "diagnostic_only"),
        _row("carry_projection", "projection_only", claim_scope="architecture_sidecar"),
        _row("carry_missing", "missing"),
        _row("carry_retired", "retired"),
    ]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": row["status"],
            "blocker_class": row["blocker_class"],
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in rows
        if row["status"] in mod.PUBLICATION_BLOCKING_STATUSES
    ]
    return {
        "artifact": "experiment_3175_cross_corpus_matrix_v28",
        "matrix_v28_ready": ready,
        "rows_total": len(rows),
        "rows": rows,
        "status_counts": {
            status: sum(row["status"] == status for row in rows) for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "missing_artifacts": [
            {
                "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
                "experiment_id": "exp3141",
                "reason": "carried_forward_unresolved_missing_artifact_from_v27",
            }
        ],
        "verifier_status": "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only",
        "repair_status": "blocked_flagged_verifier_repair_ladder_gated_skipped_certificate_pilot_flagged",
        "fr11_status": "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update",
        "sidecar_status": "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier",
        "hardware_status": "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made",
        "paper_ready": False,
        "honest_verdict": "complete: matrix_v28_ready=true",
    }


def _write_required_sources(root: Path, *, matrix_ready: bool = True) -> None:
    _write_json(root, mod.MATRIX_V28_REL_PATH, _matrix_v28(ready=matrix_ready))
    _write_json(
        root,
        mod.CAPSTONE_V294_REL_PATH,
        {
            "artifact": "experiment_3176_capstone_v294",
            "capstone_v294_ready": True,
            "paper_ready": False,
            "publication_blocker_count": 73,
            "next_top_gap": "clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock",
            "honest_verdict": "complete: capstone_v294_ready=true",
        },
    )
    _write_json(
        root,
        mod.ARCHIVE_V295_REL_PATH,
        {
            "artifact": "experiment_3177_archive_v294_activate_v295",
            "archive_v294_activate_v295_ready": True,
            "prior_publication_blocker_count": 73,
            "honest_verdict": "complete: archive_v294_activate_v295_ready=true",
        },
    )


def _write_dot295_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3178_REL_PATH,
        {
            "receipt_backed_authenticity_contract_v3_ready": True,
            "flagged_adversarial": True,
            "contract_blockers": [],
            "honest_verdict": "complete: receipt contract",
        },
    )
    _write_json(
        root,
        mod.EXP3179_REL_PATH,
        {
            "local_sota_receipt_smoke_v3_ready": True,
            "preflight_passed": True,
            "clean_rerun_allowed": False,
            "flagged_adversarial": True,
            "headline_claim_allowed": False,
            "live_call_count": 2,
            "substrate_classification": "cpu_fallback_receipt_only",
            "throughput_plausibility_passed": True,
            "proof_receipts": [{"transcript_hash": "a"}, {"transcript_hash": "b"}],
            "honest_verdict": "complete: receipt smoke",
        },
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "controlled_invariance_passed": True,
            "flagged_adversarial": True,
            "known_false_accept_regression_count": 2,
            "semantic_false_accept_count": 0,
            "shortcut_failure_count": 0,
            "exact_row_count": 72,
            "source_errors": [],
            "honest_verdict": "complete: controlled invariance",
        },
    )
    _write_json(
        root,
        mod.EXP3181_REL_PATH,
        {
            "clean_live_sota_verifier_rerun_v10_ready": True,
            "gated_skip": True,
            "gate_reasons": [
                "exp3179.clean_rerun_allowed=false",
                "exp3179.substrate_classification=cpu_fallback_receipt_only",
            ],
            "controlled_invariance_passed": True,
            "flagged_adversarial": True,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 0.0,
            "headline_claim_allowed": False,
            "live_call_count": 0,
            "known_false_accept_regression_count": 2,
            "honest_verdict": "blocked_receipt_precondition: gated skip",
        },
    )
    _write_json(
        root,
        mod.EXP3182_REL_PATH,
        {
            "distributional_ebm_exact_row_sidecar_v1_ready": True,
            "deployed_verifier_claim_allowed": False,
            "known_false_accept_rows_scored": 2,
            "exact_labeled_row_count": 72,
            "false_accept_separation_auc": 1.0,
            "source_errors": [],
            "honest_verdict": "complete: diagnostic sidecar",
        },
    )
    _write_json(
        root,
        mod.EXP3183_REL_PATH,
        {
            "counterexample_certificate_expansion_v3_ready": True,
            "flagged_adversarial": True,
            "repair_call_ready": False,
            "counterexample_count": 4,
            "exact_row_count": 72,
            "known_false_accept_rows_covered": 2,
            "blocker_reasons": ["flagged_adversarial_evidence_present"],
            "source_errors": [],
            "honest_verdict": "complete: certificate expansion",
        },
    )
    _write_json(
        root,
        mod.EXP3184_REL_PATH,
        {
            "repair_gate_decision_v4_ready": True,
            "repair_gate_state": "blocked_receipt_precondition",
            "blocker_reasons": ["exp3179.clean_rerun_allowed is not true"],
            "missing_artifacts": [],
            "allowed_repair_attempt_budget": {"enabled": False, "max_total_repair_attempts": 0},
            "honest_verdict": "blocked_receipt_precondition: repair gate",
        },
    )
    _write_json(
        root,
        mod.EXP3185_REL_PATH,
        {
            "multi_turn_repair_ladder_v5_ready": True,
            "gated_skip": True,
            "gate_state": "blocked_receipt_precondition",
            "headline_claim_allowed": False,
            "flagged_adversarial": True,
            "repair_attempt_count": 0,
            "repair_success_delta": 0.0,
            "remaining_blockers": ["repair gate not unblocked"],
            "honest_verdict": "blocked_repair_gate_precondition: repair ladder",
        },
    )
    _write_json(
        root,
        mod.EXP3186_REL_PATH,
        {
            "fr11_controller_memory_promotion_pack_v1_ready": True,
            "continuous_self_learning_task": True,
            "learning_tier": "Tier 2: Constraint Memory",
            "no_model_weight_update_claimed": True,
            "promotion_allowed": True,
            "promotion_manifest": {"promotion_decision": "promote_controller_memory_only"},
            "honest_verdict": "complete: FR-11 promotion",
        },
    )
    _write_json(
        root,
        mod.EXP3187_REL_PATH,
        {
            "fr11_cross_environment_drift_replay_v1_ready": True,
            "continuous_self_learning_task": True,
            "replay_mode_only": True,
            "no_model_weight_update_claimed": True,
            "promotion_allowed": True,
            "cross_environment_row_count": 9,
            "heldout_row_count": 12,
            "negative_control_regression_count": 0,
            "rollback_triggered": False,
            "honest_verdict": "complete: drift replay",
        },
    )
    _write_json(
        root,
        mod.EXP3188_REL_PATH,
        {
            "thrml_factor_graph_api_boundary_v1_ready": True,
            "thrml_import_available": True,
            "thrml_version": "0.1.3",
            "local_api_smoke_passed": True,
            "hardware_speedup_claim_allowed": False,
            "kona_or_tsu_execution_claimed": False,
            "selected_exact_rows": [{"row_id": "r1"}, {"row_id": "r2"}],
            "api_gap_records": [{"gap_id": "adapter_needed"}],
            "source_errors": [],
            "honest_verdict": "complete: THRML boundary",
        },
    )


def test_req_report_3189_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3189: OpenSpec declares the v29 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3189" in spec
    assert "SCENARIO-REPORT-3189" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3189_builds_v29_from_dot295_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3189: .295 rows preserve blocked/gated/flagged accounting."""

    _write_required_sources(tmp_path)
    _write_dot295_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=4.0)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["cross_corpus_matrix_v29_ready"] is True
    assert artifact["prior_matrix_version"] == "v28"
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert artifact["rows_total"] == len(artifact["rows"]) == 19
    assert artifact["publication_blocker_count"] == 12
    assert artifact["blocker_delta_from_v28"] == 7
    assert artifact["status_counts"] == {
        "clean": 3,
        "blocked": 2,
        "gated_skipped": 3,
        "flagged": 5,
        "diagnostic_only": 3,
        "projection_only": 1,
        "missing": 1,
        "retired": 1,
    }
    assert artifact["clean_rows"] == 3
    assert artifact["flagged_rows"] == 5
    assert artifact["blocked_rows"] == 2
    assert artifact["gated_skip_rows"] == 3
    assert artifact["diagnostic_only_rows"] == 3
    assert artifact["missing_artifacts"] == [
        {
            "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
            "experiment_id": "exp3141",
            "reason": "carried_forward_unresolved_missing_artifact_from_v28",
        }
    ]
    assert artifact["missing_artifact_comparison"] == {
        "v28_missing_artifact_count": 1,
        "v29_missing_artifact_count": 1,
        "missing_artifact_delta_from_v28": 0,
        "new_missing_dot295_artifacts": [],
    }
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["dot295:exp3178_receipt_contract"]["status"] == "flagged"
    assert rows["dot295:exp3179_sota_receipt_smoke"]["status"] == "flagged"
    assert rows["dot295:exp3180_controlled_invariance"]["status"] == "flagged"
    assert rows["dot295:exp3181_clean_live_rerun_v10"]["status"] == "gated_skipped"
    assert rows["dot295:exp3182_distributional_sidecar"]["status"] == "diagnostic_only"
    assert rows["dot295:exp3183_certificate_expansion"]["status"] == "flagged"
    assert rows["dot295:exp3184_repair_gate_v4"]["status"] == "blocked"
    assert rows["dot295:exp3185_repair_ladder_v5"]["status"] == "gated_skipped"
    assert rows["dot295:exp3186_fr11_promotion_pack"]["status"] == "clean"
    assert rows["dot295:exp3187_fr11_drift_replay"]["status"] == "clean"
    assert rows["dot295:exp3188_thrml_boundary"]["status"] == "diagnostic_only"

    assert artifact["verifier_status"] == (
        "gated_skip_cpu_fallback_receipt_only_flagged_adversarial_controlled_invariance_passed_exact_authority_only"
    )
    assert artifact["repair_status"] == (
        "blocked_receipt_precondition_repair_ladder_gated_skipped_certificate_expansion_flagged"
    )
    assert artifact["fr11_status"] == (
        "controller_memory_promotion_allowed_cross_environment_replay_passed_no_model_weight_update"
    )
    assert artifact["hardware_status"] == (
        "diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution"
    )
    assert artifact["next_top_gap"] == (
        "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    )
    assert artifact["paper_v6_narrowing_preserved"] is True
    assert artifact["paper_v6_narrowing"] == {
        "kv260_speedup_claimed": False,
        "tsu_or_kona_execution_claimed": False,
        "deployed_verifier_sidecar_claimed": False,
        "model_weight_self_learning_claimed": False,
        "paper_ready_streak_claimed": False,
    }
    assert artifact["paper_readiness_implications"]["blocked_headline_claims"] == [
        "live_verifier_headline",
        "repair_headline",
        "deployed_verifier_sidecar",
        "hardware_speedup",
    ]
    assert sources[mod.EXP3181_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3181_REL_PATH
    )
    assert artifact["inference_substrate"] == {
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
    }


def test_req_report_3189_missing_dot295_artifacts_are_visible(tmp_path: Path) -> None:
    """REQ-REPORT-3189: missing `.295` deliverables are counted separately."""

    _write_required_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.0)
    dot295_rows = [row for row in artifact["rows"] if row["row_id"].startswith("dot295:")]

    assert artifact["cross_corpus_matrix_v29_ready"] is True
    assert len(dot295_rows) == 11
    assert {row["status"] for row in dot295_rows} == {"missing"}
    assert artifact["publication_blocker_count"] == 16
    assert artifact["blocker_delta_from_v28"] == 11
    assert len(artifact["missing_artifacts"]) == 12
    assert artifact["missing_artifact_comparison"]["new_missing_dot295_artifacts"] == [
        spec.path.as_posix() for spec in mod.DOT295_SOURCE_SPECS
    ]

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["cross_corpus_matrix_v29_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v29_preconditions")
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V28_REL_PATH.as_posix(),
        mod.CAPSTONE_V294_REL_PATH.as_posix(),
        mod.ARCHIVE_V295_REL_PATH.as_posix(),
    ]


def test_req_report_3189_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3189: helper behavior is deterministic and fail-closed."""

    _write_required_sources(tmp_path)
    _write_dot295_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cross_corpus_matrix_v29_ready"] is True
    assert mod._ready_status({}, "ready") == "missing"
    assert mod._ready_status({"ready": False}, "ready") == "blocked"
    assert mod._ready_status({"ready": True}, "ready") == "clean"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean")]})[0]["row_id"] == "carry"

    assert mod._receipt_contract_row({})["status"] == "missing"
    assert (
        mod._receipt_contract_row({"receipt_backed_authenticity_contract_v3_ready": False})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._receipt_contract_row({"receipt_backed_authenticity_contract_v3_ready": True})["status"]
        == "clean"
    )
    assert mod._sota_receipt_smoke_row({})["status"] == "missing"
    assert (
        mod._sota_receipt_smoke_row({"local_sota_receipt_smoke_v3_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._sota_receipt_smoke_row(
            {"local_sota_receipt_smoke_v3_ready": True, "preflight_passed": False}
        )["status"]
        == "blocked"
    )
    assert (
        mod._sota_receipt_smoke_row(
            {
                "local_sota_receipt_smoke_v3_ready": True,
                "preflight_passed": True,
                "flagged_adversarial": True,
            }
        )["status"]
        == "flagged"
    )
    assert (
        mod._sota_receipt_smoke_row(
            {
                "local_sota_receipt_smoke_v3_ready": True,
                "preflight_passed": True,
                "clean_rerun_allowed": False,
                "flagged_adversarial": False,
            }
        )["status"]
        == "blocked"
    )
    assert (
        mod._sota_receipt_smoke_row(
            {
                "local_sota_receipt_smoke_v3_ready": True,
                "preflight_passed": True,
                "clean_rerun_allowed": True,
                "substrate_classification": "full_local_sota_receipt",
            }
        )["status"]
        == "clean"
    )

    assert mod._controlled_invariance_row({})["status"] == "missing"
    assert (
        mod._controlled_invariance_row({"controlled_invariance_executor_v2_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._controlled_invariance_row(
            {
                "controlled_invariance_executor_v2_ready": True,
                "controlled_invariance_passed": True,
                "flagged_adversarial": True,
            }
        )["status"]
        == "flagged"
    )
    assert (
        mod._controlled_invariance_row(
            {"controlled_invariance_executor_v2_ready": True, "controlled_invariance_passed": True}
        )["status"]
        == "diagnostic_only"
    )
    assert (
        mod._controlled_invariance_row({"controlled_invariance_executor_v2_ready": True})["status"]
        == "blocked"
    )

    assert mod._clean_rerun_v10_row({})["status"] == "missing"
    assert (
        mod._clean_rerun_v10_row({"clean_live_sota_verifier_rerun_v10_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._clean_rerun_v10_row(
            {"clean_live_sota_verifier_rerun_v10_ready": True, "gated_skip": True}
        )["status"]
        == "gated_skipped"
    )
    assert (
        mod._clean_rerun_v10_row(
            {
                "clean_live_sota_verifier_rerun_v10_ready": True,
                "flagged_adversarial": True,
            }
        )["status"]
        == "flagged"
    )
    assert (
        mod._clean_rerun_v10_row(
            {
                "clean_live_sota_verifier_rerun_v10_ready": True,
                "controlled_invariance_passed": True,
                "headline_claim_allowed": True,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._clean_rerun_v10_row({"clean_live_sota_verifier_rerun_v10_ready": True})["status"]
        == "blocked"
    )

    assert mod._distributional_sidecar_row({})["status"] == "missing"
    assert (
        mod._distributional_sidecar_row({"distributional_ebm_exact_row_sidecar_v1_ready": False})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._distributional_sidecar_row(
            {"distributional_ebm_exact_row_sidecar_v1_ready": True, "source_errors": ["bad"]}
        )["status"]
        == "blocked"
    )
    assert (
        mod._distributional_sidecar_row(
            {
                "distributional_ebm_exact_row_sidecar_v1_ready": True,
                "deployed_verifier_claim_allowed": True,
            }
        )["status"]
        == "clean"
    )

    assert mod._certificate_expansion_row({})["status"] == "missing"
    assert (
        mod._certificate_expansion_row({"counterexample_certificate_expansion_v3_ready": False})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._certificate_expansion_row(
            {
                "counterexample_certificate_expansion_v3_ready": True,
                "repair_call_ready": True,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._certificate_expansion_row(
            {
                "counterexample_certificate_expansion_v3_ready": True,
                "flagged_adversarial": True,
            }
        )["status"]
        == "flagged"
    )
    assert (
        mod._certificate_expansion_row({"counterexample_certificate_expansion_v3_ready": True})[
            "status"
        ]
        == "blocked"
    )

    assert mod._repair_gate_v4_row({})["status"] == "missing"
    assert mod._repair_gate_v4_row({"repair_gate_decision_v4_ready": False})["status"] == "blocked"
    assert (
        mod._repair_gate_v4_row(
            {"repair_gate_decision_v4_ready": True, "repair_gate_state": "unblocked"}
        )["status"]
        == "clean"
    )
    assert mod._repair_ladder_v5_row({})["status"] == "missing"
    assert (
        mod._repair_ladder_v5_row({"multi_turn_repair_ladder_v5_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._repair_ladder_v5_row({"multi_turn_repair_ladder_v5_ready": True, "gated_skip": True})[
            "status"
        ]
        == "gated_skipped"
    )
    assert (
        mod._repair_ladder_v5_row(
            {"multi_turn_repair_ladder_v5_ready": True, "flagged_adversarial": True}
        )["status"]
        == "flagged"
    )
    assert (
        mod._repair_ladder_v5_row(
            {"multi_turn_repair_ladder_v5_ready": True, "headline_claim_allowed": True}
        )["status"]
        == "clean"
    )

    assert mod._fr11_promotion_pack_row({})["status"] == "missing"
    assert (
        mod._fr11_promotion_pack_row({"fr11_controller_memory_promotion_pack_v1_ready": False})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._fr11_promotion_pack_row(
            {
                "fr11_controller_memory_promotion_pack_v1_ready": True,
                "promotion_allowed": True,
                "no_model_weight_update_claimed": False,
            }
        )["status"]
        == "blocked"
    )
    assert mod._fr11_drift_replay_row({})["status"] == "missing"
    assert (
        mod._fr11_drift_replay_row({"fr11_cross_environment_drift_replay_v1_ready": False})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._fr11_drift_replay_row(
            {
                "fr11_cross_environment_drift_replay_v1_ready": True,
                "promotion_allowed": True,
                "no_model_weight_update_claimed": True,
                "negative_control_regression_count": 1,
            }
        )["status"]
        == "blocked"
    )
    assert mod._thrml_boundary_row({})["status"] == "missing"
    assert (
        mod._thrml_boundary_row({"thrml_factor_graph_api_boundary_v1_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._thrml_boundary_row(
            {
                "thrml_factor_graph_api_boundary_v1_ready": True,
                "hardware_speedup_claim_allowed": True,
            }
        )["status"]
        == "blocked"
    )
    assert (
        mod._thrml_boundary_row(
            {
                "thrml_factor_graph_api_boundary_v1_ready": True,
                "local_api_smoke_passed": True,
            }
        )["status"]
        == "diagnostic_only"
    )

    assert (
        mod._verifier_status(
            {
                "exp3179": {"clean_rerun_allowed": True},
                "exp3180": {},
                "exp3181": {},
            },
            [{"row_id": "dot295:exp3181_clean_live_rerun_v10", "status": "clean"}],
        )
        == "clean_live_sota_verifier_ready"
    )
    assert (
        mod._repair_status(
            {"exp3183": {}, "exp3184": {}, "exp3185": {}},
            [
                {"row_id": "dot295:exp3184_repair_gate_v4", "status": "clean"},
                {"row_id": "dot295:exp3185_repair_ladder_v5", "status": "clean"},
            ],
        )
        == "repair_ready"
    )
    assert (
        mod._fr11_status(
            {"exp3186": {}, "exp3187": {}},
            [{"row_id": "dot295:exp3186_fr11_promotion_pack", "status": "clean"}],
        )
        == "controller_memory_promotion_pack_ready_pending_drift_replay"
    )
    assert (
        mod._hardware_status(
            {"exp3188": {"hardware_speedup_claim_allowed": True}},
            [{"row_id": "dot295:exp3188_thrml_boundary", "status": "blocked"}],
        )
        == "blocked_unsupported_thrml_or_hardware_claim"
    )

    violations = mod._invariant_violations(
        {"matrix_v28_ready": False},
        {"capstone_v294_ready": False},
        {"archive_v294_activate_v295_ready": False},
        [_row("flagged", "flagged")],
        {"clean": 0},
        [],
        [],
        True,
    )
    assert violations == [
        "matrix v28 authority is not ready",
        "capstone v294 authority is not ready",
        "archive v295 handoff is not ready",
        "status_counts keys do not match required v29 statuses",
        "status_counts do not sum to rows_total",
    ]
    full_counts = {status: 0 for status in mod.STATUSES}
    full_counts["flagged"] = 1
    blocker_violation = mod._invariant_violations(
        {"matrix_v28_ready": True},
        {"capstone_v294_ready": True},
        {"archive_v294_activate_v295_ready": True},
        [_row("flagged", "flagged")],
        full_counts,
        [],
        [],
        False,
    )
    assert blocker_violation == [
        "publication_blocker_count does not match row statuses",
        "paper-v6 narrowing was not preserved",
    ]
