"""Tests for Exp 3147 cross-corpus matrix v26.

Spec refs: REQ-REPORT-3147, SCENARIO-REPORT-3147.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v26_3147 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "matrix_v26_ready",
    "rows_total",
    "prior_publication_blocker_count",
    "publication_blocker_count",
    "blocker_delta_from_v25",
    "missing_artifacts",
    "status_counts",
    "headline_claim_allowance_summary",
    "false_accept_recovery_summary",
    "repair_gate_summary",
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


def _row(row_id: str, status: str, claim_scope: str = "carry") -> dict[str, Any]:
    return {
        "row_id": row_id,
        "status": status,
        "source_artifact": f"results/{row_id.replace(':', '_')}.json",
        "source_field": "status",
        "evidence_class": "v25_carry",
        "blocker_class": mod.blocker_class(status),
        "claim_scope": claim_scope,
        "summary": {"source_status": status},
        "row_origin": "matrix_v25_test",
    }


def _matrix_v25(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("carry:clean", "clean"),
        _row("carry:bounded", "bounded", "baseline_bounded"),
        _row("carry:gated", "gated_skipped", "baseline_gate"),
        _row("carry:diagnostic", "diagnostic_only", "baseline_diagnostic"),
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
        "artifact": "experiment_3133_cross_corpus_matrix_v25",
        "matrix_v25_ready": ready,
        "rows_total": len(rows),
        "rows": rows,
        "status_counts": {
            status: sum(row["status"] == status for row in rows) for status in mod.STATUSES
        },
        "publication_blocker_count": len(blockers),
        "publication_blockers": blockers,
        "headline_claim_allowance_summary": {
            "comparative_sota_pair_allowed": False,
            "sota_cache_headline_allowed": True,
            "present_model_ids": [GEMMA26],
            "missing_model_ids": [QWEN, GEMMA31],
            "selected_headline_model_ids": [GEMMA26],
        },
        "honest_verdict": "complete: matrix_v25_ready=true",
    }


def _capstone_v291(*, ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3134_capstone_v291",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 46,
        "honest_verdict": "complete: capstone_ready=true; paper_ready=false",
    }


def _write_required_baseline(
    root: Path, *, matrix_ready: bool = True, capstone_ready: bool = True
) -> None:
    _write_json(root, mod.MATRIX_V25_REL_PATH, _matrix_v25(ready=matrix_ready))
    _write_json(root, mod.CAPSTONE_V291_REL_PATH, _capstone_v291(ready=capstone_ready))


def _write_dot292_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3135_REL_PATH,
        {
            "archive_v291_activate_v292_ready": True,
            "prior_publication_blocker_count": 46,
            "honest_verdict": "complete: archive_v291_activate_v292_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "false_accept_autopsy_v1_ready": True,
            "source_false_accept_rate": 0.5,
            "recomputed_false_accept_rate": 0.5,
            "source_false_accept_count": 2,
            "source_live_row_count": 6,
            "false_accept_row_ids": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "false_accept_mechanism_counts": {
                "SAT/validity-token confusion": 1,
                "contradiction miss": 1,
            },
            "regression_row_set": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            "honest_verdict": "complete: autopsy ready but flagged",
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "acceptance_contract_v1_ready": True,
            "known_false_accept_rows_blocked": True,
            "replay_false_accept_rate": 0.0,
            "replay_false_reject_rate": 0.0,
            "replay_abstention_rate": 0.025641,
            "replay_counts": {"total_rows": 78},
            "repair_gate_prerequisites": {"repair_gate_opens_only_after_live_rerun_replay": True},
            "honest_verdict": "complete: acceptance contract ready",
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "canonical_grounding_pilot_v1_ready": True,
            "false_accept_rows_blocked": 2,
            "regression_rows_evaluated": 2,
            "residual_false_accept_rows": [],
            "canonicalizer_implemented": True,
            "premise_grounding_block_count": 2,
            "canonicalization_block_count": 2,
            "ledger_replay_block_count": 2,
            "honest_verdict": "complete: canonical grounding ready",
        },
    )
    _write_json(
        root,
        mod.EXP3139_REL_PATH,
        {
            "live_verifier_rerun_v7_ready": True,
            "false_accept_gate_passed": True,
            "headline_claim_allowed": True,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 0.5,
            "verifier_gain_delta": 0.5,
            "live_call_count": 6,
            "repair_gate_candidate_state": "candidate_ready",
            "source_false_accept_rate": 0.5,
            "regression_rows_included": True,
            "selected_model_ids": [GEMMA26],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "honest_verdict": "complete: live rerun metrics ready but flagged",
        },
    )
    _write_json(
        root,
        mod.EXP3140_REL_PATH,
        {
            "repair_gate_decision_v1_ready": True,
            "repair_gate_state": "blocked_other",
            "false_accept_gate_passed": True,
            "false_accept_rate": 0.0,
            "known_false_accepts_blocked": True,
            "regression_rows_included": True,
            "exact_authority_ready": True,
            "live_model_ready": True,
            "monitor_ledger_ready": True,
            "repair_rows_available": True,
            "repair_blockers": ["exp3139 flagged_adversarial=true"],
            "headline_disqualifiers": ["exp3139 flagged_adversarial=true"],
            "selected_repair_rows": [],
            "honest_verdict": "blocked_other: exp3139 flagged_adversarial=true",
        },
    )
    _write_json(
        root,
        mod.EXP3142_REL_PATH,
        {
            "fr11_vera_evoenv_v2_ready": True,
            "continuous_self_learning_targeted": True,
            "live_model_variant_generation": False,
            "live_call_count": 0,
            "admitted_environment_count": 3,
            "hardened_variant_count": 3,
            "equivalent_variant_count": 3,
            "ledger_consistency_rate": 0.833333,
            "no_weight_update_claim": True,
            "promotion_recommendation": (
                "promote_controller_environment_memory_only_block_model_weight_learning"
            ),
            "soundness_errors": 0,
            "completeness_errors": 0,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}],
            "honest_verdict": "complete: solver-only FR-11",
        },
    )
    _write_json(
        root,
        mod.EXP3143_REL_PATH,
        {
            "fr11_experience_verifier_memory_v1_ready": True,
            "continuous_self_learning_targeted": True,
            "replay_row_count": 12,
            "ledger_consistency_rate": 0.666667,
            "no_weight_update_claim": True,
            "promotion_recommendation": "promote_controller_routing_memory_only",
            "suppressed_check_count": 7,
            "escalated_check_count": 5,
            "estimated_check_savings_rate": 0.166667,
            "residual_false_accept_risk": 0.0,
            "residual_false_reject_risk": 0.166667,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            "honest_verdict": "complete: controller routing memory only",
        },
    )
    _write_json(
        root,
        mod.EXP3144_REL_PATH,
        {
            "ebt_arm_false_accept_calibration_v3_ready": True,
            "live_integration": False,
            "live_call_count": 6,
            "live_row_count": 6,
            "false_accept_rows_evaluated": 2,
            "false_accept_row_ids": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
            "integration_blockers": ["no generation-path sidecar hook exercised under tests"],
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}],
            "honest_verdict": "complete: calibration boundary only",
        },
    )
    _write_json(
        root,
        mod.EXP3145_REL_PATH,
        {
            "kan_proof_carrying_monitor_v2_ready": True,
            "kan_code_present": True,
            "deployed_verifier_claim": False,
            "attached_monitor_record_count": 2,
            "milp_property_check_count": 1,
            "implementation_blockers": [],
            "claim_boundary": {
                "proves": "two replayable KAN PWA/MILP proof records",
                "does_not_prove": ["deployed verifier improvement", "hardware execution"],
            },
            "false_accept_relevance": {"would_prevent_live_false_accept": False},
            "honest_verdict": "complete_kan_records_attached_no_deployed_claim",
        },
    )
    _write_json(
        root,
        mod.EXP3146_REL_PATH,
        {
            "hardware_sampler_evidence_boundary_v6_ready": True,
            "gatemate_evidence_complete": False,
            "ssqa_readback_ready": False,
            "speedup_claim_allowed": False,
            "kona_claim_allowed": False,
            "thrml_tsu_claim_allowed": False,
            "hardware_commands_run": [],
            "missing_operator_evidence": [{"missing_item": "ssqa:host_visible_smoke_evidence"}],
            "sampler_boundary_decisions": {
                "gatemate": "blocked",
                "ssqa": "blocked",
                "kona": "out-of-scope",
            },
            "honest_verdict": "complete: hardware boundary ready; evidence blocked",
        },
    )


def test_req_report_3147_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3147: OpenSpec declares the v26 matrix contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3147" in spec
    assert "SCENARIO-REPORT-3147" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3147_builds_v26_from_dot292_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3147: .292 rows stay flagged, bounded, blocked, or gated."""

    _write_required_baseline(tmp_path)
    _write_dot292_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=14.25)
    rows = {row["row_id"]: row for row in artifact["rows"]}
    blockers = {row["row_id"] for row in artifact["publication_blockers"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["matrix_v26_ready"] is True
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["rows_total"] == len(artifact["rows"]) == 15
    assert artifact["prior_publication_blocker_count"] == 2
    assert artifact["publication_blocker_count"] == 11
    assert artifact["blocker_delta_from_v25"] == 9
    assert artifact["status_counts"] == {
        "clean": 3,
        "flagged": 4,
        "bounded": 2,
        "blocked": 2,
        "gated_skipped": 2,
        "missing": 0,
        "retired": 0,
        "projection_only": 1,
        "diagnostic_only": 1,
        "model_spec_gap": 0,
    }
    assert artifact["missing_artifacts"] == [
        {
            "path": mod.EXP3141_REL_PATH.as_posix(),
            "experiment_id": "exp3141",
            "reason": "missing_or_malformed_dot292_artifact",
        }
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    assert rows["carry:bounded"]["status"] == "bounded"
    assert rows["dot292:exp3136_false_accept_autopsy"]["status"] == "flagged"
    assert rows["dot292:exp3137_accept_abstain_contract"]["status"] == "clean"
    assert rows["dot292:exp3138_canonical_grounding"]["status"] == "clean"
    assert rows["dot292:exp3139_live_verifier_rerun"]["status"] == "flagged"
    assert rows["dot292:exp3140_repair_gate"]["status"] == "blocked"
    assert rows["dot292:exp3141_repair_ladder"]["status"] == "gated_skipped"
    assert rows["dot292:exp3142_fr11_vera_evoenv"]["status"] == "flagged"
    assert rows["dot292:exp3143_fr11_experience_memory"]["status"] == "flagged"
    assert rows["dot292:exp3144_ebt_arm_calibration"]["status"] == "projection_only"
    assert rows["dot292:exp3145_kan_monitor_boundary"]["status"] == "bounded"
    assert rows["dot292:exp3146_hardware_boundary"]["status"] == "blocked"
    assert "carry:diagnostic" not in blockers

    allowance = artifact["headline_claim_allowance_summary"]
    assert allowance["comparative_sota_pair_allowed"] is False
    assert allowance["live_verifier_headline_allowed"] is False
    assert allowance["repair_headline_claim_allowed"] is False
    assert allowance["exact_safe_contract_claim_allowed"] is True
    assert allowance["canonical_grounding_claim_allowed"] is True
    assert allowance["missing_model_ids"] == [QWEN, GEMMA31]
    assert "live_verifier_lift_adversarial_flag" in allowance["blocked_headline_claims"]

    recovery = artifact["false_accept_recovery_summary"]
    assert recovery["autopsy_status"] == "flagged"
    assert recovery["accept_abstain_contract_status"] == "clean"
    assert recovery["canonical_grounding_status"] == "clean"
    assert recovery["live_verifier_status"] == "flagged"
    assert recovery["source_false_accept_rate"] == pytest.approx(0.5)
    assert recovery["rerun_false_accept_rate"] == pytest.approx(0.0)
    assert recovery["known_false_accept_rows_blocked"] is True
    assert recovery["recovery_claim_status"] == "blocked_by_adversarial_corrigendum"

    repair = artifact["repair_gate_summary"]
    assert repair["repair_gate_status"] == "blocked"
    assert repair["repair_ladder_status"] == "gated_skipped"
    assert repair["repair_gate_state"] == "blocked_other"
    assert repair["selected_repair_rows"] == []
    assert repair["repair_blocker_count"] == 1

    fr11 = artifact["fr11_summary"]
    assert fr11["vera_evoenv_status"] == "flagged"
    assert fr11["experience_memory_status"] == "flagged"
    assert fr11["no_weight_update_claim"] is True
    assert fr11["model_weight_learning_allowed"] is False
    assert fr11["vera_ledger_consistency_rate"] == pytest.approx(0.833333)
    assert fr11["experience_ledger_consistency_rate"] == pytest.approx(0.666667)

    architecture = artifact["architecture_boundary_summary"]
    assert architecture["ebt_arm_status"] == "projection_only"
    assert architecture["kan_monitor_status"] == "bounded"
    assert architecture["hardware_boundary_status"] == "blocked"
    assert architecture["live_integration"] is False
    assert architecture["deployed_kan_verifier_claim"] is False
    assert architecture["speedup_claim_allowed"] is False
    assert architecture["hardware_commands_run"] == []

    assert artifact["diagnostic_only_rows"] == ["carry:diagnostic"]
    assert {row["row_id"] for row in artifact["gated_skips"]} == {
        "carry:gated",
        "dot292:exp3141_repair_ladder",
    }
    assert set(artifact["architecture_boundary_rows"]) == {
        "dot292:exp3144_ebt_arm_calibration",
        "dot292:exp3145_kan_monitor_boundary",
        "dot292:exp3146_hardware_boundary",
    }
    assert sources[mod.EXP3139_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3139_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_dot292_artifacts",
        "source": "matrix_v25_capstone_v291_and_dot292_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def test_req_report_3147_missing_optional_artifacts_are_rows_not_successes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3147: absent `.292` evidence stays visible without blocking the matrix."""

    _write_required_baseline(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    dot292_rows = [row for row in artifact["rows"] if row["row_id"].startswith("dot292:")]

    assert artifact["matrix_v26_ready"] is True
    assert len(dot292_rows) == 11
    assert {row["status"] for row in dot292_rows} == {"missing"}
    assert artifact["publication_blocker_count"] == 13
    assert artifact["blocker_delta_from_v25"] == 11
    assert len(artifact["missing_artifacts"]) == 12
    assert all(
        row["reason"] == "missing_or_malformed_dot292_artifact"
        for row in artifact["missing_artifacts"]
    )

    empty = mod.build_artifact(tmp_path / "empty")

    assert empty["matrix_v26_ready"] is False
    assert empty["honest_verdict"].startswith("blocked_matrix_v26_preconditions")
    assert [row["path"] for row in empty["required_source_errors"]] == [
        mod.MATRIX_V25_REL_PATH.as_posix(),
        mod.CAPSTONE_V291_REL_PATH.as_posix(),
    ]


def test_req_report_3147_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3147: helper behavior is deterministic and fail-closed."""

    _write_required_baseline(tmp_path)
    _write_dot292_sources(tmp_path)
    malformed = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    malformed.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v26_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._has_flags({"flagged_adversarial": True}) is True
    assert mod._has_flags({"corrigendum_pending": [1]}) is True
    assert mod._has_flags({}) is False
    assert mod._ready_flagged_status({}, "ready") == "missing"
    assert mod._ready_flagged_status({"ready": False}, "ready") == "blocked"
    assert mod._ready_flagged_status({"ready": True, "flagged_adversarial": True}, "ready") == (
        "flagged"
    )
    assert mod._ready_flagged_status({"ready": True}, "ready") == "clean"
    assert mod._carry_forward_rows({"rows": [1, _row("carry", "clean")]})[0]["row_id"] == "carry"
    assert mod._contract_row({})["status"] == "missing"
    assert (
        mod._contract_row({"acceptance_contract_v1_ready": False})["status"]
        == "blocked"
    )
    assert mod._contract_row({"acceptance_contract_v1_ready": True})["status"] == "blocked"
    assert mod._canonical_grounding_row({"canonical_grounding_pilot_v1_ready": False})[
        "status"
    ] == "blocked"
    assert (
        mod._canonical_grounding_row(
            {"canonical_grounding_pilot_v1_ready": True, "residual_false_accept_rows": ["x"]}
        )["status"]
        == "blocked"
    )
    assert mod._canonical_grounding_row({"canonical_grounding_pilot_v1_ready": True})[
        "status"
    ] == "bounded"
    assert mod._live_verifier_row({"live_verifier_rerun_v7_ready": False})["status"] == "blocked"
    assert (
        mod._live_verifier_row(
            {
                "live_verifier_rerun_v7_ready": True,
                "headline_claim_allowed": True,
                "false_accept_gate_passed": True,
                "false_accept_rate": 0.0,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._live_verifier_row(
            {
                "live_verifier_rerun_v7_ready": True,
                "headline_claim_allowed": True,
                "false_accept_gate_passed": True,
                "false_accept_rate": 0.1,
            }
        )["status"]
        == "blocked"
    )
    assert (
        mod._live_verifier_row(
            {
                "live_verifier_rerun_v7_ready": True,
                "headline_claim_allowed": False,
                "false_accept_gate_passed": True,
                "false_accept_rate": 0.0,
            }
        )["status"]
        == "bounded"
    )
    assert mod._repair_gate_row({"repair_gate_decision_v1_ready": False})["status"] == "blocked"
    assert (
        mod._repair_gate_row(
            {"repair_gate_decision_v1_ready": True, "repair_gate_state": "unblocked"}
        )["status"]
        == "clean"
    )
    assert (
        mod._repair_gate_row(
            {"repair_gate_decision_v1_ready": True, "repair_gate_state": "review"}
        )["status"]
        == "bounded"
    )
    assert mod._repair_ladder_row({}, {"repair_gate_state": "unblocked"})["status"] == "missing"
    assert (
        mod._repair_ladder_row({"multi_turn_repair_ladder_v2_ready": True}, {})["status"]
        == "clean"
    )
    assert (
        mod._repair_ladder_row({"multi_turn_repair_ladder_v2_ready": False, "status": "blocked"}, {})[
            "status"
        ]
        == "blocked"
    )
    assert (
        mod._repair_ladder_row({"multi_turn_repair_ladder_v2_ready": False}, {})["status"]
        == "blocked"
    )
    assert mod._fr11_vera_row({"fr11_vera_evoenv_v2_ready": False})["status"] == "blocked"
    assert (
        mod._fr11_vera_row(
            {
                "fr11_vera_evoenv_v2_ready": True,
                "live_model_variant_generation": False,
                "no_weight_update_claim": True,
                "ledger_consistency_rate": 0.5,
            }
        )["status"]
        == "bounded"
    )
    assert (
        mod._fr11_vera_row(
            {
                "fr11_vera_evoenv_v2_ready": True,
                "live_model_variant_generation": True,
                "no_weight_update_claim": False,
                "ledger_consistency_rate": 1.0,
            }
        )["status"]
        == "clean"
    )
    assert mod._fr11_experience_row({"fr11_experience_verifier_memory_v1_ready": False})[
        "status"
    ] == "blocked"
    assert (
        mod._fr11_experience_row(
            {
                "fr11_experience_verifier_memory_v1_ready": True,
                "no_weight_update_claim": True,
                "ledger_consistency_rate": 0.5,
            }
        )["status"]
        == "bounded"
    )
    assert (
        mod._fr11_experience_row(
            {
                "fr11_experience_verifier_memory_v1_ready": True,
                "no_weight_update_claim": False,
                "ledger_consistency_rate": 1.0,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._ebt_arm_row({"ebt_arm_false_accept_calibration_v3_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._ebt_arm_row(
            {
                "ebt_arm_false_accept_calibration_v3_ready": True,
                "live_integration": True,
                "flagged_adversarial": True,
            }
        )["status"]
        == "flagged"
    )
    assert (
        mod._ebt_arm_row(
            {"ebt_arm_false_accept_calibration_v3_ready": True, "live_integration": True}
        )["status"]
        == "clean"
    )
    assert (
        mod._kan_monitor_row({"kan_proof_carrying_monitor_v2_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._kan_monitor_row(
            {"kan_proof_carrying_monitor_v2_ready": True, "implementation_blockers": ["x"]}
        )["status"]
        == "blocked"
    )
    assert (
        mod._kan_monitor_row(
            {"kan_proof_carrying_monitor_v2_ready": True, "deployed_verifier_claim": True}
        )["status"]
        == "clean"
    )
    assert (
        mod._hardware_row({"hardware_sampler_evidence_boundary_v6_ready": False})["status"]
        == "blocked"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_sampler_evidence_boundary_v6_ready": True,
                "gatemate_evidence_complete": True,
                "ssqa_readback_ready": True,
                "speedup_claim_allowed": True,
            }
        )["status"]
        == "clean"
    )
    assert (
        mod._hardware_row(
            {
                "hardware_sampler_evidence_boundary_v6_ready": True,
                "gatemate_evidence_complete": True,
                "ssqa_readback_ready": True,
                "speedup_claim_allowed": False,
            }
        )["status"]
        == "bounded"
    )

    exact_safe_payloads = {
        "exp3136": {"false_accept_autopsy_v1_ready": True, "source_false_accept_rate": 0.5},
        "exp3137": {
            "acceptance_contract_v1_ready": True,
            "known_false_accept_rows_blocked": True,
            "replay_false_accept_rate": 0.0,
            "replay_false_reject_rate": 0.0,
        },
        "exp3138": {
            "canonical_grounding_pilot_v1_ready": True,
            "canonicalizer_implemented": True,
            "false_accept_rows_blocked": 1,
            "regression_rows_evaluated": 1,
            "residual_false_accept_rows": [],
        },
        "exp3139": {
            "live_verifier_rerun_v7_ready": True,
            "headline_claim_allowed": True,
            "false_accept_gate_passed": True,
            "false_accept_rate": 0.0,
        },
    }
    exact_safe_rows = [
        mod._false_accept_autopsy_row(exact_safe_payloads["exp3136"]),
        mod._contract_row(exact_safe_payloads["exp3137"]),
        mod._canonical_grounding_row(exact_safe_payloads["exp3138"]),
        mod._live_verifier_row(exact_safe_payloads["exp3139"]),
    ]
    assert (
        mod._false_accept_recovery_summary(exact_safe_payloads, exact_safe_rows)[
            "recovery_claim_status"
        ]
        == "exact_safe_recovery_ready"
    )

    violations = mod._invariant_violations(
        {"matrix_v25_ready": False},
        {"capstone_ready": False},
        [_row("flagged", "flagged")],
        {"clean": 0},
        [],
        [],
    )
    assert violations == [
        "matrix v25 authority is not ready",
        "capstone v291 authority is not ready",
        "status_counts keys do not match required v26 statuses",
        "status_counts do not sum to rows_total",
    ]
    full_counts = {status: 0 for status in mod.STATUSES}
    full_counts["flagged"] = 1
    blocker_violation = mod._invariant_violations(
        {"matrix_v25_ready": True},
        {"capstone_ready": True},
        [_row("flagged", "flagged")],
        full_counts,
        [],
        [],
    )
    assert blocker_violation == ["publication_blocker_count does not match row statuses"]
