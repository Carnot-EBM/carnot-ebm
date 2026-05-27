"""Tests for Exp 3175 cross-corpus matrix v28.

Spec refs: REQ-REPORT-3175, SCENARIO-REPORT-3175.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v28_3175 as mod


REQUIRED_FIELDS = {
    "matrix_v28_ready",
    "rows_total",
    "publication_blocker_count",
    "blocker_delta_from_v27",
    "clean_rows",
    "flagged_rows",
    "blocked_rows",
    "gated_skip_rows",
    "diagnostic_only_rows",
    "missing_artifacts",
    "inherited_adversarial_flag_count",
    "verifier_status",
    "repair_status",
    "fr11_status",
    "hardware_status",
    "paper_ready",
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


def _row(row_id: str, status: str, *, claim_scope: str = "headline") -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id}.json",
        "source_field": "status",
        "evidence_class": "v27_carry",
        "blocker_class": mod.blocker_class(status, claim_scope),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
        "row_origin": "matrix_v27_test",
    }


def _matrix_v27(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("carry_clean", "clean"),
        _row("carry_flagged", "flagged"),
        _row("carry_bounded_repair", "bounded", claim_scope="repair_headline_boundary"),
        _row("carry_bounded_sidecar", "bounded", claim_scope="architecture_kan_boundary"),
        _row("carry_projection", "projection_only", claim_scope="architecture_energy_boundary"),
        _row("carry_diagnostic", "diagnostic_only"),
        _row("carry_missing", "missing"),
        _row("carry_retired", "retired"),
    ]
    blockers = [
        {
            "row_id": row["row_id"],
            "status": mod.normal_status(row["status"], row["claim_scope"]),
            "blocker_class": mod.blocker_class(row["status"], row["claim_scope"]),
            "source_artifact": row["source_artifact"],
            "source_field": row["source_field"],
            "claim_scope": row["claim_scope"],
        }
        for row in rows
        if mod.normal_status(row["status"], row["claim_scope"])
        in mod.PUBLICATION_BLOCKING_STATUSES
    ]
    return {
        "artifact": "experiment_3161_cross_corpus_matrix_v27",
        "matrix_v27_ready": ready,
        "rows_total": len(rows),
        "rows": rows,
        "status_counts": {
            status: sum(
                mod.normal_status(row["status"], row["claim_scope"]) == status for row in rows
            )
            for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "missing_artifacts": [
            {
                "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
                "experiment_id": "exp3141",
                "reason": "carried_forward_missing_artifact_from_v26",
            },
            {
                "path": "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
                "experiment_id": "exp3152",
                "reason": "missing_or_gated_dot293_artifact",
            },
            {
                "path": "results/experiment_3154_multi_turn_repair_ladder_v3.json",
                "experiment_id": "exp3154",
                "reason": "missing_or_gated_dot293_artifact",
            },
            {
                "path": "results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json",
                "experiment_id": "exp3155",
                "reason": "missing_expected_dot293_deliverable_alias_loaded",
            },
        ],
        "inherited_adversarial_flag_count": 3,
        "honest_verdict": "complete: matrix_v27_ready=true",
    }


def _capstone_v293(*, ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3162_capstone_v293",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 65,
        "blocker_delta_from_v26": 10,
        "verifier_evidence_status": (
            "corrigendum_preserved_exact_replay_but_did_not_unblock_repair"
        ),
        "repair_gate_status": "blocked_pending_clean_rerun_gate_failed",
        "repair_ladder_status": "correctly_skipped_gate_blocked_no_live_repair_attempts",
        "fr11_self_learning_status": (
            "improved_to_0.857143_but_promotion_blocked_controller_memory_only"
        ),
        "sampler_hardware_status": (
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
        ),
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_required_baseline(
    root: Path, *, matrix_ready: bool = True, capstone_ready: bool = True
) -> None:
    _write_json(root, mod.MATRIX_V27_REL_PATH, _matrix_v27(ready=matrix_ready))
    _write_json(root, mod.CAPSTONE_V293_REL_PATH, _capstone_v293(ready=capstone_ready))


def _write_dot294_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3163_REL_PATH,
        {
            "archive_v293_activate_v294_ready": True,
            "prior_publication_blocker_count": 65,
            "honest_verdict": "complete: handoff ready",
        },
    )
    _write_json(
        root,
        mod.EXP3164_REL_PATH,
        {
            "duration_corrected_authenticity_contract_v2_ready": True,
            "flagged_adversarial": True,
            "old_fixed_duration_rule_retired_as_hard_gate": True,
            "observed_source_assessment": {"passed": True},
            "honest_verdict": "complete: v2 duration contract ready",
        },
    )
    _write_json(
        root,
        mod.EXP3165_REL_PATH,
        {
            "live_sota_authenticity_replay_v2_ready": True,
            "preflight_passed": False,
            "headline_claim_allowed": False,
            "live_call_count": 0,
            "blocked_reason": "CUDA/GPU substrate unavailable",
            "honest_verdict": "blocked_gpu_substrate: preflight_passed=false",
        },
    )
    _write_json(
        root,
        mod.EXP3166_REL_PATH,
        {
            "verifier_invariance_token_suspicion_audit_ready": True,
            "diagnostics_allowed_to_gate_repair": [{"name": "known_false_accept_rows_blocked"}],
            "blocked_checks": [{"name": "future_first_token_logprob_telemetry"}],
            "trusted_exact_rows": [{"row_id": "exact-1"}],
            "source_errors": [],
            "honest_verdict": "complete: exact authority only",
        },
    )
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "clean_live_verifier_rerun_v9_ready": True,
            "gated_skip": True,
            "gated_skip_reason": "exp3165 preflight_passed=false",
            "flagged_adversarial": True,
            "controlled_invariance_passed": False,
            "false_accept_gate_passed": False,
            "false_accept_rate": 0.0,
            "headline_claim_allowed": False,
            "live_call_count": 0,
            "exact_ground_truth_count": 72,
            "honest_verdict": "complete: gated skip",
        },
    )
    _write_json(
        root,
        mod.EXP3168_REL_PATH,
        {
            "repair_gate_decision_v3_ready": True,
            "repair_gate_state": "blocked_flagged_verifier",
            "gated_skip": True,
            "repair_blockers": ["flagged_adversarial=true"],
            "selected_repair_rows": [],
            "honest_verdict": "blocked_flagged_verifier: gated skip",
        },
    )
    _write_json(
        root,
        mod.EXP3169_REL_PATH,
        {
            "repair_ladder_materializer_v4_ready": True,
            "gated_skip": True,
            "gated_skip_reason": "repair gate blocked",
            "gate_state": "blocked_flagged_verifier",
            "headline_repair_claim_allowed": False,
            "live_call_count": 0,
            "repair_attempt_count": 0,
            "honest_verdict": "blocked_repair_gate: repair gate blocked",
        },
    )
    _write_json(
        root,
        mod.EXP3170_REL_PATH,
        {
            "counterexample_certificate_repair_pilot_v2_ready": True,
            "flagged_adversarial": True,
            "counterexample_count": 4,
            "exact_accept_count": 0,
            "exact_row_count": 5,
            "prior_repair_candidates_scored": 0,
            "repair_call_required_for_next_step": True,
            "honest_verdict": "complete: counterexample certificates ready",
        },
    )
    _write_json(
        root,
        mod.EXP3171_REL_PATH,
        {
            "fr11_ledger_counterexample_isolation_ready": True,
            "continuous_self_learning_task": True,
            "prior_ledger_consistency_rate": 0.857143,
            "ledger_consistent_count": 12,
            "ledger_inconsistent_count": 2,
            "replay_panel_count": 14,
            "no_model_weight_update_claimed": True,
            "promotion_allowed": False,
            "isolated_counterexample_families": [{"fixture_family": "arith"}],
            "honest_verdict": "complete: isolation ready",
        },
    )
    _write_json(
        root,
        mod.EXP3172_REL_PATH,
        {
            "fr11_nonforgetting_self_learning_pilot_v2_ready": True,
            "continuous_self_learning_task": True,
            "before_ledger_consistency_rate": 0.857143,
            "after_ledger_consistency_rate": 1.0,
            "heldout_consistency_rate": 1.0,
            "nonforgetting_passed": True,
            "controller_memory_update_applied": True,
            "model_weight_update_claimed": False,
            "promotion_allowed": True,
            "promotion_recommendation": "promote_controller_memory_update_only",
            "honest_verdict": "complete: nonforgetting passed",
        },
    )
    _write_json(
        root,
        mod.EXP3173_REL_PATH,
        {
            "ebcn_kan_bounded_diagnostic_expansion_v2_ready": True,
            "live_integration_claim_allowed": False,
            "deployed_verifier_claim_allowed": False,
            "exact_labeled_row_count": 72,
            "known_false_accept_rows_scored": 2,
            "kan_monitor_record_count": 4,
            "promotion_blockers": ["no live integration"],
            "honest_verdict": "complete: bounded diagnostics ready",
        },
    )
    _write_json(
        root,
        mod.EXP3174_REL_PATH,
        {
            "hardware_tooling_boundary_v8_ready": True,
            "authenticated_speedup_claim_allowed": False,
            "speedup_claim_made": False,
            "hardware_commands_run": [],
            "cuda_status": "runtime_ready_no_sampler_speedup_claim_flagged_adversarial",
            "kv260_status": "authenticated_historical_board_evidence_scoped",
            "gatemate_status": "blocked_operator_evidence_incomplete_no_speedup_claim",
            "polarfire_status": "authenticated_historical_dispatch_evidence_no_speedup_claim",
            "extropic_thrml_status": "architecture_reference_only",
            "kona_status": "architecture_reference_only",
            "missing_required_source_artifacts": [],
            "honest_verdict": "complete: hardware boundary ready",
        },
    )


def test_req_report_3175_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3175: OpenSpec declares the v28 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3175" in spec
    assert "SCENARIO-REPORT-3175" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3175_builds_v28_from_dot294_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3175: .294 blockers and gated skips become rows."""

    _write_required_baseline(tmp_path)
    _write_dot294_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.5)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v28_ready"] is True
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["rows_total"] == len(artifact["rows"]) == 20
    assert artifact["publication_blocker_count"] == 13
    assert artifact["blocker_delta_from_v27"] == 8
    assert artifact["clean_rows"] == 3
    assert artifact["flagged_rows"] == 3
    assert artifact["blocked_rows"] == 4
    assert artifact["gated_skip_rows"] == 2
    assert artifact["diagnostic_only_rows"] == 3
    assert artifact["status_counts"] == {
        "clean": 3,
        "blocked": 4,
        "gated_skipped": 2,
        "flagged": 3,
        "diagnostic_only": 3,
        "projection_only": 3,
        "missing": 1,
        "retired": 1,
    }
    assert artifact["missing_artifacts"] == [
        {
            "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
            "experiment_id": "exp3141",
            "reason": "carried_forward_unresolved_missing_artifact_from_v27",
        }
    ]
    assert artifact["missing_artifact_comparison"] == {
        "v27_missing_artifact_count": 4,
        "v28_missing_artifact_count": 1,
        "missing_artifact_delta_from_v27": -3,
        "materialized_v27_missing_artifacts": [
            "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
            "results/experiment_3154_multi_turn_repair_ladder_v3.json",
            "results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json",
        ],
    }
    assert artifact["inherited_adversarial_flag_count"] == 3
    assert artifact["paper_ready"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["carry_bounded_repair"]["status"] == "blocked"
    assert rows["carry_bounded_sidecar"]["status"] == "projection_only"
    assert rows["dot294:exp3163_archive_handoff"]["status"] == "clean"
    assert rows["dot294:exp3164_duration_contract"]["status"] == "flagged"
    assert rows["dot294:exp3165_live_sota_replay"]["status"] == "blocked"
    assert rows["dot294:exp3166_invariance_audit"]["status"] == "diagnostic_only"
    assert rows["dot294:exp3167_clean_live_rerun"]["status"] == "gated_skipped"
    assert rows["dot294:exp3168_repair_gate_v3"]["status"] == "blocked"
    assert rows["dot294:exp3169_repair_ladder_v4"]["status"] == "gated_skipped"
    assert rows["dot294:exp3170_certificate_repair"]["status"] == "flagged"
    assert rows["dot294:exp3171_fr11_counterexample_isolation"]["status"] == "diagnostic_only"
    assert rows["dot294:exp3172_fr11_nonforgetting"]["status"] == "clean"
    assert rows["dot294:exp3173_ebcn_kan_diagnostics"]["status"] == "projection_only"
    assert rows["dot294:exp3174_hardware_tooling"]["status"] == "blocked"

    assert artifact["verifier_status"] == (
        "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only"
    )
    assert artifact["repair_status"] == (
        "blocked_flagged_verifier_repair_ladder_gated_skipped_certificate_pilot_flagged"
    )
    assert artifact["fr11_status"] == (
        "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
    )
    assert artifact["sidecar_status"] == (
        "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
    )
    assert artifact["hardware_status"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made"
    )
    assert artifact["paper_readiness_implications"]["blocked_headline_claims"] == [
        "live_verifier_headline",
        "repair_headline",
        "energy_sidecar_live_integration",
        "kan_deployed_verifier",
        "hardware_speedup",
    ]
    assert sources[mod.EXP3167_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3167_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_dot294_artifacts",
        "source": "matrix_v27_capstone_v293_and_dot294_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3175_optional_dot294_absences_are_visible(tmp_path: Path) -> None:
    """REQ-REPORT-3175: absent `.294` evidence becomes explicit matrix rows."""

    _write_required_baseline(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    dot294_rows = [row for row in artifact["rows"] if row["row_id"].startswith("dot294:")]

    assert artifact["matrix_v28_ready"] is True
    assert len(dot294_rows) == 12
    assert {row["status"] for row in dot294_rows} == {"missing"}
    assert artifact["publication_blocker_count"] == 17
    assert artifact["blocker_delta_from_v27"] == 12
    assert len(artifact["missing_artifacts"]) == 16
    assert artifact["missing_artifact_comparison"]["missing_artifact_delta_from_v27"] == 12
    assert all(
        row["reason"]
        in {
            "carried_forward_unresolved_missing_artifact_from_v27",
            "missing_expected_dot294_artifact",
        }
        for row in artifact["missing_artifacts"]
    )

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["matrix_v28_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v28_preconditions")
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V27_REL_PATH.as_posix(),
        mod.CAPSTONE_V293_REL_PATH.as_posix(),
    ]


def test_req_report_3175_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3175: helper behavior is deterministic and fail-closed."""

    _write_required_baseline(tmp_path)
    _write_dot294_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v28_ready"] is True
    assert mod.normal_status("model_spec_gap") == "blocked"
    assert mod.normal_status("bounded", "future_adapter_context") == "projection_only"
    assert mod.normal_status("bounded", "repair_headline_boundary") == "blocked"
    assert mod.normal_status("unknown") == "missing"
    assert mod._ready_status({}, "ready") == "missing"
    assert mod._ready_status({"ready": False}, "ready") == "blocked"
    assert mod._ready_status({"ready": True}, "ready") == "clean"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean")]})[0]["row_id"] == "carry"
    assert mod._archive_row({})["status"] == "missing"
    assert mod._archive_row({"archive_v293_activate_v294_ready": False})["status"] == "blocked"
    assert mod._contract_row({})["status"] == "missing"
    assert mod._contract_row({"duration_corrected_authenticity_contract_v2_ready": False})[
        "status"
    ] == "blocked"
    assert mod._contract_row({"duration_corrected_authenticity_contract_v2_ready": True})[
        "status"
    ] == "clean"
    assert mod._live_sota_row({})["status"] == "missing"
    assert mod._live_sota_row({"live_sota_authenticity_replay_v2_ready": False})[
        "status"
    ] == "blocked"
    assert mod._live_sota_row(
        {
            "live_sota_authenticity_replay_v2_ready": True,
            "preflight_passed": True,
            "headline_claim_allowed": True,
        }
    )["status"] == "clean"
    assert mod._invariance_audit_row({})["status"] == "missing"
    assert mod._invariance_audit_row(
        {"verifier_invariance_token_suspicion_audit_ready": True, "source_errors": ["bad"]}
    )["status"] == "blocked"
    assert mod._clean_rerun_row({})["status"] == "missing"
    assert mod._clean_rerun_row({"clean_live_verifier_rerun_v9_ready": False})[
        "status"
    ] == "blocked"
    assert mod._clean_rerun_row(
        {"clean_live_verifier_rerun_v9_ready": True, "flagged_adversarial": True}
    )["status"] == "flagged"
    assert mod._clean_rerun_row(
        {
            "clean_live_verifier_rerun_v9_ready": True,
            "controlled_invariance_passed": True,
            "false_accept_gate_passed": True,
            "headline_claim_allowed": True,
        }
    )["status"] == "clean"
    assert mod._clean_rerun_row({"clean_live_verifier_rerun_v9_ready": True})[
        "status"
    ] == "blocked"
    assert mod._repair_gate_row({})["status"] == "missing"
    assert mod._repair_gate_row({"repair_gate_decision_v3_ready": False})[
        "status"
    ] == "blocked"
    assert mod._repair_gate_row(
        {"repair_gate_decision_v3_ready": True, "repair_gate_state": "unblocked"}
    )["status"] == "clean"
    assert mod._repair_ladder_row({})["status"] == "missing"
    assert mod._repair_ladder_row({"repair_ladder_materializer_v4_ready": False})[
        "status"
    ] == "blocked"
    assert mod._repair_ladder_row(
        {"repair_ladder_materializer_v4_ready": True, "headline_repair_claim_allowed": True}
    )["status"] == "clean"
    assert mod._repair_ladder_row({"repair_ladder_materializer_v4_ready": True})[
        "status"
    ] == "blocked"
    assert mod._certificate_repair_row({})["status"] == "missing"
    assert mod._certificate_repair_row(
        {"counterexample_certificate_repair_pilot_v2_ready": False}
    )["status"] == "blocked"
    assert mod._certificate_repair_row(
        {"counterexample_certificate_repair_pilot_v2_ready": True}
    )["status"] == "diagnostic_only"
    assert mod._certificate_repair_row(
        {
            "counterexample_certificate_repair_pilot_v2_ready": True,
            "repair_call_required_for_next_step": True,
        }
    )["status"] == "blocked"
    assert mod._fr11_isolation_row({})["status"] == "missing"
    assert mod._fr11_isolation_row({"fr11_ledger_counterexample_isolation_ready": False})[
        "status"
    ] == "blocked"
    assert mod._fr11_nonforgetting_row({})["status"] == "missing"
    assert mod._fr11_nonforgetting_row(
        {"fr11_nonforgetting_self_learning_pilot_v2_ready": False}
    )["status"] == "blocked"
    assert mod._fr11_nonforgetting_row(
        {
            "fr11_nonforgetting_self_learning_pilot_v2_ready": True,
            "nonforgetting_passed": False,
        }
    )["status"] == "blocked"
    assert mod._sidecar_row({})["status"] == "missing"
    assert mod._sidecar_row({"ebcn_kan_bounded_diagnostic_expansion_v2_ready": False})[
        "status"
    ] == "blocked"
    assert mod._sidecar_row(
        {
            "ebcn_kan_bounded_diagnostic_expansion_v2_ready": True,
            "live_integration_claim_allowed": True,
            "deployed_verifier_claim_allowed": True,
        }
    )["status"] == "clean"
    assert mod._hardware_row({})["status"] == "missing"
    assert mod._hardware_row({"hardware_tooling_boundary_v8_ready": False})[
        "status"
    ] == "blocked"
    assert mod._hardware_row(
        {
            "hardware_tooling_boundary_v8_ready": True,
            "authenticated_speedup_claim_allowed": True,
            "speedup_claim_made": True,
        }
    )["status"] == "clean"
    assert (
        mod._verifier_status(
            {"exp3167": {"flagged_adversarial": True}, "exp3165": {}},
            [{"row_id": "dot294:exp3167_clean_live_rerun", "status": "flagged"}],
        )
        == "flagged_adversarial_clean_rerun_not_headline_safe"
    )
    assert (
        mod._verifier_status(
            {"exp3167": {}, "exp3165": {}},
            [{"row_id": "dot294:exp3167_clean_live_rerun", "status": "clean"}],
        )
        == "clean_live_verifier_ready"
    )
    assert (
        mod._verifier_status(
            {"exp3167": {}, "exp3165": {"preflight_passed": False}},
            [{"row_id": "dot294:exp3167_clean_live_rerun", "status": "blocked"}],
        )
        == "blocked_live_sota_replay_preflight_failed"
    )
    assert (
        mod._repair_status(
            {"exp3168": {}, "exp3169": {"gated_skip": True}},
            [
                {"row_id": "dot294:exp3168_repair_gate_v3", "status": "blocked"},
                {"row_id": "dot294:exp3169_repair_ladder_v4", "status": "gated_skipped"},
                {"row_id": "dot294:exp3170_certificate_repair", "status": "diagnostic_only"},
            ],
        )
        == "blocked_repair_gate_ladder_gated_skipped"
    )
    assert (
        mod._repair_status(
            {"exp3168": {}, "exp3169": {}},
            [
                {"row_id": "dot294:exp3168_repair_gate_v3", "status": "clean"},
                {"row_id": "dot294:exp3169_repair_ladder_v4", "status": "clean"},
            ],
        )
        == "repair_ready"
    )
    assert (
        mod._fr11_status(
            {"exp3172": {}, "exp3171": {"fr11_ledger_counterexample_isolation_ready": True}},
            [{"row_id": "dot294:exp3172_fr11_nonforgetting", "status": "blocked"}],
        )
        == "blocked_fr11_promotion_counterexamples_isolated"
    )
    assert (
        mod._sidecar_status(
            {"exp3173": {}},
            [{"row_id": "dot294:exp3173_ebcn_kan_diagnostics", "status": "clean"}],
        )
        == "clean_ebcn_kan_live_integration_and_deployed_verifier_allowed"
    )
    assert (
        mod._hardware_status(
            {"exp3174": {}},
            [{"row_id": "dot294:exp3174_hardware_tooling", "status": "clean"}],
        )
        == "clean_authenticated_speedup_claim_present"
    )
    assert (
        mod._hardware_status(
            {
                "exp3174": {
                    "hardware_tooling_boundary_v8_ready": True,
                    "speedup_claim_made": True,
                    "hardware_commands_run": ["flash"],
                }
            },
            [{"row_id": "dot294:exp3174_hardware_tooling", "status": "blocked"}],
        )
        == "blocked_no_authenticated_speedup_hardware_commands_present_speedup_claim_made"
    )

    violations = mod._invariant_violations(
        {"matrix_v27_ready": False},
        {"capstone_ready": False},
        [_row("flagged", "flagged")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v27 authority is not ready",
        "capstone v293 authority is not ready",
        "status_counts keys do not match required v28 statuses",
        "status_counts do not sum to rows_total",
    ]
    full_counts = {status: 0 for status in mod.STATUSES}
    full_counts["flagged"] = 1
    blocker_violation = mod._invariant_violations(
        {"matrix_v27_ready": True},
        {"capstone_ready": True},
        [_row("flagged", "flagged")],
        full_counts,
        [],
        [],
    )
    assert blocker_violation == ["publication_blocker_count does not match row statuses"]
