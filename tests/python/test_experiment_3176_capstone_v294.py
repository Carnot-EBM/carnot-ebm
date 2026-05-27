"""Tests for the Exp 3176 milestone .294 capstone.

Spec refs: REQ-REPORT-3176, SCENARIO-REPORT-3176.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v294_3176 as mod


REQUIRED_FIELDS = {
    "capstone_v294_ready",
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v27",
    "missing_artifact_count",
    "verifier_status",
    "repair_gate_status",
    "repair_ladder_status",
    "fr11_self_learning_status",
    "ebcn_kan_status",
    "hardware_sampler_status",
    "next_top_gap",
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


def _matrix_v27() -> dict[str, Any]:
    return {
        "artifact": "experiment_3161_cross_corpus_matrix_v27",
        "matrix_v27_ready": True,
        "rows_total": 136,
        "publication_blocker_count": 65,
        "blocker_delta_from_v26": 10,
        "missing_artifacts": [
            {"path": f"results/missing_{idx}.json", "experiment_id": f"exp{idx}"}
            for idx in range(4)
        ],
        "honest_verdict": "complete: matrix_v27_ready=true",
    }


def _capstone_v293() -> dict[str, Any]:
    return {
        "artifact": "experiment_3162_capstone_v293",
        "capstone_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 65,
        "next_top_gap": "clean_live_verifier_corrigendum_repair_gate",
        "honest_verdict": "complete: capstone_ready=true",
    }


def _matrix_v28(*, clean: bool = False) -> dict[str, Any]:
    blockers = 0 if clean else 73
    delta = blockers - 65
    return {
        "artifact": "experiment_3175_cross_corpus_matrix_v28",
        "matrix_v28_ready": True,
        "rows_total": 148 if not clean else 3,
        "publication_blocker_count": blockers,
        "prior_publication_blocker_count": 65,
        "blocker_delta_from_v27": delta,
        "missing_artifacts": []
        if clean
        else [
            {
                "path": "results/experiment_3141_multi_turn_repair_ladder_v2.json",
                "experiment_id": "exp3141",
                "reason": "carried_forward_unresolved_missing_artifact_from_v27",
            }
        ],
        "missing_artifact_comparison": {
            "v27_missing_artifact_count": 4,
            "v28_missing_artifact_count": 0 if clean else 1,
            "missing_artifact_delta_from_v27": -4 if clean else -3,
            "materialized_v27_missing_artifacts": []
            if clean
            else [
                "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
                "results/experiment_3154_multi_turn_repair_ladder_v3.json",
                "results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json",
            ],
        },
        "verifier_status": "clean_live_verifier_ready"
        if clean
        else "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only",
        "repair_status": "repair_ready"
        if clean
        else "blocked_flagged_verifier_repair_ladder_gated_skipped_certificate_pilot_flagged",
        "fr11_status": "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update",
        "sidecar_status": "clean_ebcn_kan_live_integration_and_deployed_verifier_allowed"
        if clean
        else "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier",
        "hardware_status": "clean_authenticated_speedup_claim_present"
        if clean
        else "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made",
        "paper_ready": clean,
        "paper_readiness_implications": {
            "paper_ready": clean,
            "publication_blocker_count": blockers,
            "blocked_headline_claims": []
            if clean
            else [
                "live_verifier_headline",
                "repair_headline",
                "energy_sidecar_live_integration",
                "kan_deployed_verifier",
                "hardware_speedup",
            ],
        },
        "inference_substrate": {
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
        },
        "required_source_errors": [],
        "invariant_violations": [],
        "honest_verdict": "complete: matrix_v28_ready=true",
    }


def _write_phase_sources(root: Path, *, clean: bool = False) -> None:
    _write_json(
        root,
        mod.EXP3164_REL_PATH,
        {
            "duration_corrected_authenticity_contract_v2_ready": True,
            "old_fixed_duration_rule_retired_as_hard_gate": True,
            "observed_source_assessment": {"passed": True, "violations": []},
            "honest_verdict": "complete: duration contract ready",
        },
    )
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "clean_live_verifier_rerun_v9_ready": True,
            "gated_skip": not clean,
            "flagged_adversarial": not clean,
            "controlled_invariance_passed": clean,
            "false_accept_gate_passed": clean,
            "headline_claim_allowed": clean,
            "live_call_count": 2 if clean else 0,
            "honest_verdict": "complete: verifier source ready",
        },
    )
    _write_json(
        root,
        mod.EXP3168_REL_PATH,
        {
            "repair_gate_decision_v3_ready": True,
            "repair_gate_state": "unblocked" if clean else "blocked_flagged_verifier",
            "repair_blockers": [] if clean else ["flagged_adversarial=true"],
            "honest_verdict": "complete: repair gate source ready",
        },
    )
    _write_json(
        root,
        mod.EXP3169_REL_PATH,
        {
            "repair_ladder_materializer_v4_ready": True,
            "gated_skip": not clean,
            "gate_state": "unblocked" if clean else "blocked_flagged_verifier",
            "headline_repair_claim_allowed": clean,
            "live_call_count": 2 if clean else 0,
            "repair_attempt_count": 2 if clean else 0,
            "honest_verdict": "complete: repair ladder source ready",
        },
    )
    _write_json(
        root,
        mod.EXP3172_REL_PATH,
        {
            "fr11_nonforgetting_self_learning_pilot_v2_ready": True,
            "after_ledger_consistency_rate": 1.0,
            "heldout_consistency_rate": 1.0,
            "nonforgetting_passed": True,
            "controller_memory_update_applied": True,
            "model_weight_update_claimed": False,
            "promotion_allowed": True,
            "promotion_recommendation": "promote_controller_memory_update_only",
            "honest_verdict": "complete: fr11 source ready",
        },
    )
    _write_json(
        root,
        mod.EXP3173_REL_PATH,
        {
            "ebcn_kan_bounded_diagnostic_expansion_v2_ready": True,
            "live_integration_claim_allowed": clean,
            "deployed_verifier_claim_allowed": clean,
            "exact_labeled_row_count": 72,
            "known_false_accept_rows_scored": 2,
            "kan_monitor_record_count": 4,
            "honest_verdict": "complete: sidecar source ready",
        },
    )
    _write_json(
        root,
        mod.EXP3174_REL_PATH,
        {
            "hardware_tooling_boundary_v8_ready": True,
            "authenticated_speedup_claim_allowed": clean,
            "speedup_claim_made": clean,
            "hardware_commands_run": ["authenticated-smoke"] if clean else [],
            "honest_verdict": "complete: hardware source ready",
        },
    )


def _write_sources(root: Path, *, clean: bool = False) -> None:
    _write_json(root, mod.MATRIX_V27_REL_PATH, _matrix_v27())
    _write_json(root, mod.CAPSTONE_V293_REL_PATH, _capstone_v293())
    _write_json(root, mod.MATRIX_V28_REL_PATH, _matrix_v28(clean=clean))
    _write_phase_sources(root, clean=clean)


def test_req_report_3176_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3176: OpenSpec declares the v294 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3176" in spec
    assert "SCENARIO-REPORT-3176" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3176_builds_blocked_paper_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3176: .294 closes honestly while paper remains blocked."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_v294_ready"] is True
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 73
    assert artifact["blocker_delta_from_v27"] == 8
    assert artifact["missing_artifact_count"] == 1
    assert artifact["duration_corrected_authenticity_status"] == (
        "passed_duration_corrected_contract_old_fixed_floor_retired"
    )
    assert artifact["clean_live_verifier_evidence_exists"] is False
    assert artifact["verifier_status"] == (
        "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only"
    )
    assert artifact["repair_gate_status"] == "blocked_flagged_verifier"
    assert artifact["repair_ladder_status"] == (
        "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
    )
    assert artifact["fr11_self_learning_status"] == (
        "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
    )
    assert artifact["fr11_promotion_grade_consistency"] is True
    assert artifact["fr11_model_weight_update_claimed"] is False
    assert artifact["ebcn_kan_status"] == (
        "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
    )
    assert artifact["hardware_sampler_status"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made"
    )
    assert artifact["hardware_claims_blocked"] is True
    assert artifact["next_top_gap"] == (
        "clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock"
    )
    assert artifact["matrix_comparison"] == {
        "v27_publication_blocker_count": 65,
        "v28_publication_blocker_count": 73,
        "capstone_v293_publication_blocker_count": 65,
        "blocker_delta_from_v27": 8,
        "v27_missing_artifact_count": 4,
        "v28_missing_artifact_count": 1,
        "missing_artifact_delta_from_v27": -3,
    }
    assert artifact["inference_substrate"] == {
        "kind": "capstone_aggregation_from_checked_in_matrix_v28_and_phase_artifacts",
        "source": "matrix_v28_matrix_v27_capstone_v293_and_dot294_phase_artifacts",
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
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert sources[mod.MATRIX_V28_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V28_REL_PATH
    )


def test_req_report_3176_paper_ready_requires_every_headline_gate(tmp_path: Path) -> None:
    """REQ-REPORT-3176: `paper_ready` is explicit and fail-closed."""

    _write_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["capstone_v294_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v27"] == -65
    assert artifact["missing_artifact_count"] == 0
    assert artifact["clean_live_verifier_evidence_exists"] is True
    assert artifact["repair_gate_status"] == "clean_repair_gate_unblocked"
    assert artifact["repair_ladder_status"] == "clean_repair_ladder_materialized"
    assert artifact["ebcn_kan_status"] == (
        "clean_ebcn_kan_live_integration_and_deployed_verifier_allowed"
    )
    assert artifact["hardware_sampler_status"] == "clean_authenticated_speedup_claim_present"
    assert artifact["next_top_gap"] == "publication_scope_reconciliation"

    matrix = json.loads((tmp_path / mod.MATRIX_V28_REL_PATH).read_text(encoding="utf-8"))
    matrix["paper_ready"] = False
    _write_json(tmp_path, mod.MATRIX_V28_REL_PATH, matrix)

    contradicted = mod.build_artifact(tmp_path)

    assert contradicted["paper_ready"] is False
    assert (
        "matrix_v28 paper_ready disagrees with derived capstone paper_ready"
        in contradicted["invariant_violations"]
    )


def test_req_report_3176_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3176: missing and malformed evidence blocks the capstone."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v294_ready"] is True
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None

    empty = mod.build_artifact(tmp_path / "empty")
    assert empty["capstone_v294_ready"] is False
    assert empty["honest_verdict"].startswith("blocked:")
    assert empty["source_artifacts"][0]["present"] is False
    assert "matrix_v28 authority is missing or malformed" in empty["invariant_violations"]

    assert mod._duration_corrected_authenticity_status({}) == "missing_duration_contract"
    assert (
        mod._duration_corrected_authenticity_status(
            {"duration_corrected_authenticity_contract_v2_ready": False}
        )
        == "blocked_duration_contract_not_ready"
    )
    assert (
        mod._duration_corrected_authenticity_status(
            {
                "duration_corrected_authenticity_contract_v2_ready": True,
                "old_fixed_duration_rule_retired_as_hard_gate": True,
                "observed_source_assessment": {"passed": False},
            }
        )
        == "blocked_duration_contract_measurement_violations"
    )
    assert mod._repair_gate_status({}) == "missing_repair_gate_decision"
    assert mod._repair_gate_status({"repair_gate_decision_v3_ready": False}) == (
        "blocked_repair_gate_decision_not_ready"
    )
    assert mod._repair_ladder_status({}) == "missing_repair_ladder_materializer"
    assert mod._repair_ladder_status({"repair_ladder_materializer_v4_ready": False}) == (
        "blocked_repair_ladder_materializer_not_ready"
    )
    assert mod._repair_ladder_status({"repair_ladder_materializer_v4_ready": True}) == (
        "blocked_repair_ladder_not_promotable"
    )
    assert mod._fr11_promotion_grade_consistency({}) is False
    assert mod._sidecar_headline_clean({}) is False
    assert mod._hardware_headline_clean({}) is False
    assert (
        mod._next_top_gap(
            "passed_duration_corrected_contract_old_fixed_floor_retired",
            True,
            "clean_repair_gate_unblocked",
            "clean_repair_ladder_materialized",
            False,
            True,
            0,
        )
        == "ebcn_kan_live_integration_boundary"
    )
    assert (
        mod._next_top_gap(
            "passed_duration_corrected_contract_old_fixed_floor_retired",
            True,
            "clean_repair_gate_unblocked",
            "clean_repair_ladder_materialized",
            True,
            False,
            0,
        )
        == "hardware_sampler_authenticated_speedup_evidence"
    )
    assert (
        mod._next_top_gap(
            "passed_duration_corrected_contract_old_fixed_floor_retired",
            True,
            "clean_repair_gate_unblocked",
            "clean_repair_ladder_materialized",
            True,
            True,
            1,
        )
        == "publication_blocker_retirement"
    )
    assert mod._float("not-a-number") is None
