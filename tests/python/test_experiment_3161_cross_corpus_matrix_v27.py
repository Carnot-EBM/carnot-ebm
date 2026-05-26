"""Tests for Exp 3161 cross-corpus matrix v27.

Spec refs: REQ-REPORT-3161, SCENARIO-REPORT-3161.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v27_3161 as mod


REQUIRED_FIELDS = {
    "matrix_v27_ready",
    "rows_total",
    "status_counts",
    "publication_blocker_count",
    "blocker_delta_from_v26",
    "inherited_adversarial_flag_count",
    "missing_artifacts",
    "false_accept_recovery_summary",
    "repair_summary",
    "fr11_summary",
    "architecture_boundary_summary",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(row_id: str, status: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id}.json",
        "source_field": "status",
        "evidence_class": "v26_carry",
        "blocker_class": mod.blocker_class(status),
        "claim_scope": "carry",
        "summary": {"source_status": status},
        "row_origin": "matrix_v26_test",
    }


def _matrix_v26(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("carry_clean", "clean"),
        _row("carry_flagged", "flagged"),
        _row("carry_bounded", "bounded"),
        _row("carry_retired", "retired"),
        _row("carry_diagnostic", "diagnostic_only"),
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
        "artifact": "experiment_3147_cross_corpus_matrix_v26",
        "matrix_v26_ready": ready,
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
                "reason": "missing_or_malformed_dot292_artifact",
            }
        ],
        "false_accept_recovery_summary": {
            "recovery_claim_status": "blocked_by_adversarial_corrigendum",
            "source_false_accept_rate": 0.5,
            "rerun_false_accept_rate": 0.0,
            "known_false_accept_rows_blocked": True,
        },
        "repair_gate_summary": {"repair_gate_state": "blocked_other"},
        "fr11_summary": {
            "model_weight_learning_allowed": False,
            "no_weight_update_claim": True,
        },
        "architecture_boundary_summary": {
            "speedup_claim_allowed": False,
            "live_integration": False,
            "deployed_kan_verifier_claim": False,
        },
        "honest_verdict": "complete: matrix_v26_ready=true",
    }


def _capstone_v292(*, ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3148_capstone_v292",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 2,
        "blocker_delta_from_v25": 9,
        "next_top_gap": "false_accept_recovery_corrigendum_repair_gate",
        "live_verifier_status": "flagged",
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_required_baseline(
    root: Path, *, matrix_ready: bool = True, capstone_ready: bool = True
) -> None:
    _write_json(root, mod.MATRIX_V26_REL_PATH, _matrix_v26(ready=matrix_ready))
    _write_json(root, mod.CAPSTONE_V292_REL_PATH, _capstone_v292(ready=capstone_ready))


def _write_dot293_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3149_REL_PATH,
        {
            "archive_v292_activate_v293_ready": True,
            "prior_publication_blocker_count": 2,
            "carry_forward_blockers": [{"blocker_id": "publication_blockers_2"}],
            "honest_verdict": "complete: handoff ready",
        },
    )
    _write_json(
        root,
        mod.EXP3150_REL_PATH,
        {
            "adversarial_corrigendum_v1_ready": True,
            "flagged_artifact_count": 5,
            "adversarial_flag_counts": {"aggregate_inherited_flag": 3},
            "known_false_accept_recovery_preserved": True,
            "live_verifier_evidence_trusted": False,
            "repair_gate_implication": "blocked_pending_clean_rerun",
            "blocked_downstream_fields": ["exp3139.false_accept_rate"],
            "safe_downstream_fields": ["exp3137.replay_false_accept_rate"],
            "methodology_requirements_for_rerun": ["record transcript hashes"],
            "honest_verdict": "complete: corrigendum ready",
        },
    )
    _write_json(
        root,
        mod.EXP3151_REL_PATH,
        {
            "live_inference_authenticity_preflight_ready": True,
            "preflight_passed": False,
            "headline_claim_allowed": False,
            "live_call_count": 1,
            "minimum_duration_requirement_s": 60.0,
            "duration_s": 10.5,
            "blocked_reason": "duration_s=10.5 is shorter than minimum plausible duration 60.0",
            "selected_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "locally_usable_model_ids": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "honest_verdict": "blocked_duration_too_short: preflight_passed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3153_REL_PATH,
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "3 of 3 gate(s) failed",
            "gates_evaluated": [
                {"upstream": "exp3152-clean-live-sota-verifier-rerun-v8", "passed": False}
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3155_ALIAS_REL_PATH,
        {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {"upstream": "exp3154-multi-turn-repair-ladder-v3", "passed": False}
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3156_REL_PATH,
        {
            "fr11_ledger_consistency_closure_v1_ready": True,
            "continuous_self_learning_targeted": True,
            "replay_panel_count": 14,
            "ledger_consistency_rate": 0.857143,
            "ledger_consistent_count": 12,
            "soundness_errors": 0,
            "completeness_errors": 0,
            "methodology_complete": True,
            "no_weight_update_claim": True,
            "promotion_recommendation": "block_fr11_promotion_until_ledger_consistency_reaches_1.0",
            "residual_mismatch_rows": [{"row_id": "resyn-3084-arith-003"}],
            "honest_verdict": "complete: fr11 closure ready",
        },
    )
    _write_json(
        root,
        mod.EXP3157_REL_PATH,
        {
            "fr11_attractor_residual_memory_audit_v1_ready": True,
            "continuous_self_learning_targeted": True,
            "replay_panel_count": 14,
            "ledger_consistency_rate": 0.857143,
            "no_weight_update_claim": True,
            "promotion_recommendation": "block_fr11_promotion_until_ledger_consistency_reaches_1.0",
            "redundant_check_suppression_rate": 0.428571,
            "unsafe_skip_count": 0,
            "risky_family_escalation_rate": 1.0,
            "honest_verdict": "complete: residual memory diagnostic ready",
        },
    )
    _write_json(
        root,
        mod.EXP3158_REL_PATH,
        {
            "ebcn_energy_sidecar_calibration_v1_ready": True,
            "exact_labeled_row_count": 6,
            "known_false_accept_rows_scored": 2,
            "scalar_energy_auc": 1.0,
            "violation_localization_coverage": 1.0,
            "live_integration_claim_allowed": False,
            "residual_blockers": ["no live verifier integration implemented"],
            "honest_verdict": "complete: energy calibration ready",
        },
    )
    _write_json(
        root,
        mod.EXP3159_REL_PATH,
        {
            "kan_proof_carrying_monitor_expansion_v1_ready": True,
            "monitor_record_count": 4,
            "new_monitor_record_count": 2,
            "exact_row_coverage_count": 4,
            "deployed_verifier_claim_allowed": False,
            "claim_boundary": {
                "does_not_prove": ["deployed verifier improvement"],
                "proves": "four replayable exact-row KAN records",
            },
            "implementation_blockers": [],
            "residual_blockers": ["No deployed accept/reject gate consumes these proof records."],
            "honest_verdict": "complete_kan_records_added_no_deployed_verifier",
        },
    )
    _write_json(
        root,
        mod.EXP3160_REL_PATH,
        {
            "hardware_sampler_evidence_boundary_v7_ready": True,
            "authenticated_speedup_claim_allowed": False,
            "no_hardware_commands_run": True,
            "cuda_status": "runtime_ready_no_sampler_speedup_claim_flagged_adversarial",
            "kv260_status": "authenticated_historical_board_evidence_scoped_no_fresh_speedup_claim",
            "gatemate_status": "blocked_operator_evidence_incomplete_no_speedup_claim",
            "polarfire_status": "authenticated_historical_dispatch_evidence_no_speedup_claim",
            "extropic_thrml_status": "architecture_reference_only_no_local_tsu_or_xtr_execution",
            "kona_status": "architecture_reference_only_no_local_kona_or_aleph_execution",
            "missing_operator_evidence": [{"missing_item": "authenticated_speedup_claim"}],
            "missing_required_source_artifacts": [],
            "hardware_commands_run": [],
            "honest_verdict": "complete: hardware boundary ready",
        },
    )


def test_req_report_3161_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3161: OpenSpec declares the v27 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3161" in spec
    assert "SCENARIO-REPORT-3161" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3161_builds_v27_from_dot293_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3161: .293 blockers and skips stay visible."""

    _write_required_baseline(tmp_path)
    _write_dot293_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.5)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v27_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 17
    assert artifact["publication_blocker_count"] == 12
    assert artifact["blocker_delta_from_v26"] == 10
    assert artifact["inherited_adversarial_flag_count"] == 3
    assert artifact["status_counts"] == {
        "clean": 2,
        "blocked": 4,
        "flagged": 2,
        "gated_skipped": 2,
        "diagnostic_only": 2,
        "projection_only": 1,
        "bounded": 3,
        "missing": 0,
        "retired": 1,
    }
    assert artifact["missing_artifacts"] == [
        {
            "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
            "experiment_id": "exp3141",
            "reason": "carried_forward_missing_artifact_from_v26",
        },
        {
            "path": mod.EXP3152_REL_PATH.as_posix(),
            "experiment_id": "exp3152",
            "reason": "missing_or_gated_dot293_artifact",
        },
        {
            "path": mod.EXP3154_REL_PATH.as_posix(),
            "experiment_id": "exp3154",
            "reason": "missing_or_gated_dot293_artifact",
        },
        {
            "path": mod.EXP3155_REL_PATH.as_posix(),
            "experiment_id": "exp3155",
            "reason": "missing_expected_dot293_deliverable_alias_loaded",
            "loaded_alias_path": mod.EXP3155_ALIAS_REL_PATH.as_posix(),
        },
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["carry_clean"]["status"] == "clean"
    assert rows["carry_flagged"]["status"] == "flagged"
    assert rows["dot293:exp3149_archive_handoff"]["status"] == "clean"
    assert rows["dot293:exp3150_adversarial_corrigendum"]["status"] == "flagged"
    assert rows["dot293:exp3151_live_preflight"]["status"] == "blocked"
    assert rows["dot293:exp3152_clean_live_rerun"]["status"] == "gated_skipped"
    assert rows["dot293:exp3153_repair_gate_v2"]["status"] == "blocked"
    assert rows["dot293:exp3154_repair_ladder_v3"]["status"] == "gated_skipped"
    assert rows["dot293:exp3155_tracefix_repair"]["status"] == "blocked"
    assert rows["dot293:exp3156_fr11_ledger_closure"]["status"] == "bounded"
    assert rows["dot293:exp3157_fr11_residual_memory"]["status"] == "diagnostic_only"
    assert rows["dot293:exp3158_ebcn_energy_sidecar"]["status"] == "projection_only"
    assert rows["dot293:exp3159_kan_monitor_expansion"]["status"] == "bounded"
    assert rows["dot293:exp3160_hardware_boundary"]["status"] == "blocked"

    recovery = artifact["false_accept_recovery_summary"]
    assert recovery["known_false_accept_recovery_preserved"] is True
    assert recovery["live_verifier_evidence_trusted"] is False
    assert recovery["preflight_passed"] is False
    assert recovery["clean_live_rerun_status"] == "gated_skipped"
    assert recovery["recovery_claim_status"] == (
        "exact_replay_preserved_but_live_verifier_untrusted"
    )

    repair = artifact["repair_summary"]
    assert repair["repair_gate_status"] == "blocked"
    assert repair["repair_ladder_status"] == "gated_skipped"
    assert repair["tracefix_status"] == "blocked"
    assert repair["repair_claim_allowed"] is False
    assert repair["live_repair_executed"] is False

    fr11 = artifact["fr11_summary"]
    assert fr11["ledger_status"] == "bounded"
    assert fr11["residual_memory_status"] == "diagnostic_only"
    assert fr11["ledger_consistency_rate"] == pytest.approx(0.857143)
    assert fr11["model_weight_learning_allowed"] is False
    assert fr11["unsafe_skip_count"] == 0

    architecture = artifact["architecture_boundary_summary"]
    assert architecture["energy_sidecar_status"] == "projection_only"
    assert architecture["kan_status"] == "bounded"
    assert architecture["hardware_status"] == "blocked"
    assert architecture["live_integration_claim_allowed"] is False
    assert architecture["deployed_verifier_claim_allowed"] is False
    assert architecture["authenticated_speedup_claim_allowed"] is False
    assert architecture["hardware_commands_run"] == []

    assert artifact["paper_readiness_implications"] == {
        "paper_ready": False,
        "publication_blocker_count": 12,
        "blocked_headline_claims": [
            "live_verifier_headline",
            "repair_headline",
            "fr11_model_weight_learning",
            "energy_sidecar_live_integration",
            "kan_deployed_verifier",
            "hardware_speedup",
        ],
    }
    assert sources[mod.EXP3150_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3150_REL_PATH
    )
    assert sources[mod.EXP3155_REL_PATH.as_posix()]["loaded_path"] == (
        mod.EXP3155_ALIAS_REL_PATH.as_posix()
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_dot293_artifacts",
        "source": "matrix_v26_capstone_v292_and_dot293_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3161_missing_optional_artifacts_are_visible(tmp_path: Path) -> None:
    """REQ-REPORT-3161: absent `.293` evidence is visible but not smoothed."""

    _write_required_baseline(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    dot293_rows = [row for row in artifact["rows"] if row["row_id"].startswith("dot293:")]

    assert artifact["matrix_v27_ready"] is True
    assert len(dot293_rows) == 12
    assert {row["status"] for row in dot293_rows} == {"missing"}
    assert artifact["publication_blocker_count"] == 14
    assert artifact["blocker_delta_from_v26"] == 12
    assert len(artifact["missing_artifacts"]) == 13
    assert all(
        row["reason"]
        in {
            "carried_forward_missing_artifact_from_v26",
            "missing_or_gated_dot293_artifact",
        }
        for row in artifact["missing_artifacts"]
    )

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["matrix_v27_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v27_preconditions")
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V26_REL_PATH.as_posix(),
        mod.CAPSTONE_V292_REL_PATH.as_posix(),
    ]


def test_req_report_3161_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3161: helper behavior is deterministic and fail-closed."""

    _write_required_baseline(tmp_path)
    _write_dot293_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v27_ready"] is True
    assert mod.normal_status("model_spec_gap") == "blocked"
    assert mod.normal_status("unknown") == "missing"
    assert mod._ready_status({}, "ready") == "missing"
    assert mod._ready_status({"ready": False}, "ready") == "blocked"
    assert mod._ready_status({"ready": True}, "ready") == "clean"
    alias_spec = mod.SourceSpec(
        "exp-test",
        Path("primary.json"),
        "alias_break_coverage",
        aliases=(Path("alias.json"),),
    )
    _write_json(tmp_path, Path("primary.json"), {"ready": True})
    _write_json(tmp_path, Path("alias.json"), {"ready": True})
    assert mod._source_payload(tmp_path, alias_spec)["loaded_path"] == "primary.json"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean")]})[0]["row_id"] == "carry"
    assert mod._corrigendum_row({})["status"] == "missing"
    assert mod._corrigendum_row({"adversarial_corrigendum_v1_ready": False})["status"] == "blocked"
    assert mod._corrigendum_row({"adversarial_corrigendum_v1_ready": True})["status"] == "clean"
    assert (
        mod._preflight_row({"live_inference_authenticity_preflight_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._preflight_row(
            {"live_inference_authenticity_preflight_ready": True, "preflight_passed": True}
        )["status"]
        == "diagnostic_only"
    )
    assert mod._clean_rerun_row({}, {"preflight_passed": True}, {})["status"] == "missing"
    assert mod._clean_rerun_row({}, {"preflight_passed": False}, {})["status"] == "gated_skipped"
    assert (
        mod._clean_rerun_row(
            {"clean_live_verifier_rerun_v8_ready": True, "flagged_adversarial": True},
            {"preflight_passed": True},
            {},
        )["status"]
        == "flagged"
    )
    assert (
        mod._clean_rerun_row(
            {
                "clean_live_verifier_rerun_v8_ready": True,
                "flagged_adversarial": False,
                "false_accept_gate_passed": True,
                "headline_claim_allowed": True,
            },
            {"preflight_passed": True},
            {},
        )["status"]
        == "clean"
    )
    assert (
        mod._clean_rerun_row(
            {"clean_live_verifier_rerun_v8_ready": False, "status": "blocked"},
            {"preflight_passed": True},
            {},
        )["status"]
        == "blocked"
    )
    assert (
        mod._clean_rerun_row(
            {
                "clean_live_verifier_rerun_v8_ready": True,
                "flagged_adversarial": False,
                "false_accept_gate_passed": False,
            },
            {"preflight_passed": True},
            {},
        )["status"]
        == "bounded"
    )
    assert (
        mod._repair_gate_v2_row(
            {"repair_gate_decision_v2_ready": True, "repair_gate_state": "unblocked"}
        )["status"]
        == "clean"
    )
    assert mod._repair_ladder_row({}, {"status": "blocked"})["status"] == "gated_skipped"
    assert mod._repair_ladder_row({}, {"repair_gate_state": "unblocked"})["status"] == "missing"
    assert (
        mod._repair_ladder_row({"multi_turn_repair_ladder_v3_ready": True}, {})["status"]
        == "bounded"
    )
    assert (
        mod._repair_ladder_row(
            {"multi_turn_repair_ladder_v3_ready": False, "status": "blocked"}, {}
        )["status"]
        == "blocked"
    )
    assert (
        mod._tracefix_row({"tracefix_counterexample_repair_pilot_v1_ready": True})["status"]
        == "clean"
    )
    assert (
        mod._fr11_ledger_row({"fr11_ledger_consistency_closure_v1_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._fr11_ledger_row(
            {
                "fr11_ledger_consistency_closure_v1_ready": True,
                "ledger_consistency_rate": 1.0,
                "soundness_errors": 0,
                "completeness_errors": 0,
                "no_weight_update_claim": False,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._fr11_residual_row({"fr11_attractor_residual_memory_audit_v1_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._fr11_residual_row(
            {"fr11_attractor_residual_memory_audit_v1_ready": True, "unsafe_skip_count": 1}
        )["status"]
        == "blocked"
    )
    assert (
        mod._fr11_residual_row(
            {"fr11_attractor_residual_memory_audit_v1_ready": True, "unsafe_skip_count": 0}
        )["status"]
        == "diagnostic_only"
    )
    assert (
        mod._energy_row({"ebcn_energy_sidecar_calibration_v1_ready": False})["status"] == "blocked"
    )
    assert (
        mod._energy_row(
            {
                "ebcn_energy_sidecar_calibration_v1_ready": True,
                "live_integration_claim_allowed": True,
                "residual_blockers": [],
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._kan_row({"kan_proof_carrying_monitor_expansion_v1_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._kan_row(
            {
                "kan_proof_carrying_monitor_expansion_v1_ready": True,
                "deployed_verifier_claim_allowed": True,
                "residual_blockers": [],
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._hardware_row({"hardware_sampler_evidence_boundary_v7_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_sampler_evidence_boundary_v7_ready": True,
                "authenticated_speedup_claim_allowed": True,
                "missing_operator_evidence": [],
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_sampler_evidence_boundary_v7_ready": True,
                "authenticated_speedup_claim_allowed": False,
                "missing_operator_evidence": [],
                "gatemate_status": "architecture_bounded",
            }
        )["status"]
        == "bounded"
    )
    clean_recovery = mod._false_accept_recovery_summary(
        {
            "exp3150": {
                "live_verifier_evidence_trusted": True,
                "known_false_accept_recovery_preserved": True,
            },
            "exp3151": {"preflight_passed": True},
        },
        [
            {
                "row_id": "dot293:exp3152_clean_live_rerun",
                "status": "clean",
            }
        ],
        {"false_accept_recovery_summary": {}},
    )
    assert clean_recovery["recovery_claim_status"] == "clean_live_verifier_recovery_ready"

    violations = mod._invariant_violations(
        {"matrix_v26_ready": False},
        {"capstone_ready": False},
        [_row("flagged", "flagged")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v26 authority is not ready",
        "capstone v292 authority is not ready",
        "status_counts keys do not match required v27 statuses",
        "status_counts do not sum to rows_total",
    ]
    full_counts = {status: 0 for status in mod.STATUSES}
    full_counts["flagged"] = 1
    blocker_violation = mod._invariant_violations(
        {"matrix_v26_ready": True},
        {"capstone_ready": True},
        [_row("flagged", "flagged")],
        full_counts,
        [],
        [],
    )
    assert blocker_violation == ["publication_blocker_count does not match row statuses"]
