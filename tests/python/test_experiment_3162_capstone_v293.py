"""Tests for the Exp 3162 milestone .293 capstone.

Spec refs: REQ-REPORT-3162, SCENARIO-REPORT-3162.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v293_3162 as mod


REQUIRED_FIELDS = {
    "capstone_v293_ready",
    "capstone_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v26",
    "next_top_gap",
    "verifier_evidence_status",
    "repair_gate_status",
    "repair_ladder_status",
    "fr11_self_learning_status",
    "ebt_arm_status",
    "kan_status",
    "sampler_hardware_status",
    "what_293_proved",
    "next_milestone_recommendations",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_blockers(count: int) -> list[dict[str, Any]]:
    scopes = [
        "live_verifier_source_trust",
        "live_verifier_preflight",
        "repair_gate_decision",
        "repair_execution",
        "fr11_self_learning_promotion_gate",
        "architecture_hardware_sampler_boundary",
    ]
    return [
        {
            "row_id": f"blocker:{idx}",
            "status": "blocked",
            "blocker_class": "publication_blocker_blocked",
            "source_artifact": "results/source.json",
            "source_field": "status",
            "claim_scope": scopes[idx % len(scopes)],
        }
        for idx in range(count)
    ]


def _source_artifacts() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "exp3148",
            "path": mod.CAPSTONE_V292_REL_PATH.as_posix(),
            "loaded_path": mod.CAPSTONE_V292_REL_PATH.as_posix(),
            "present": True,
            "primary_present": True,
            "alias_loaded": False,
            "role": "capstone_v292_authority",
            "ready_field": "capstone_ready",
            "source_type": "json",
        },
        {
            "experiment_id": "exp3150",
            "path": "results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json",
            "loaded_path": "results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json",
            "present": True,
            "primary_present": True,
            "alias_loaded": False,
            "role": "adversarial_corrigendum",
            "ready_field": "adversarial_corrigendum_v1_ready",
            "source_type": "json",
        },
        {
            "experiment_id": "exp3152",
            "path": "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
            "loaded_path": "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
            "present": False,
            "primary_present": False,
            "alias_loaded": False,
            "role": "clean_live_verifier_rerun",
            "ready_field": "clean_live_verifier_rerun_v8_ready",
            "source_type": "json",
        },
        {
            "experiment_id": "exp3155",
            "path": "results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json",
            "loaded_path": "results/experiment_3155_tracefix_counterexample_repair_pilot.json",
            "present": True,
            "primary_present": False,
            "alias_loaded": True,
            "role": "tracefix_counterexample_repair",
            "ready_field": "tracefix_counterexample_repair_pilot_v1_ready",
            "source_type": "json",
        },
    ]


def _capstone_v292(*, ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3148_capstone_v292",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 55,
        "blocker_delta_from_v25": 9,
        "next_top_gap": "false_accept_recovery_corrigendum_repair_gate",
        "fr11_self_learning_status": (
            "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667"
        ),
        "honest_verdict": "complete: capstone_ready=true",
    }


def _matrix_v27(*, ready: bool = True, blockers: int = 65) -> dict[str, Any]:
    clean = blockers == 0
    blocked_claims = []
    if not clean:
        blocked_claims = [
            "live_verifier_headline",
            "repair_headline",
            "fr11_model_weight_learning",
            "energy_sidecar_live_integration",
            "kan_deployed_verifier",
            "hardware_speedup",
        ]
    return {
        "artifact": "experiment_3161_cross_corpus_matrix_v27",
        "matrix_v27_ready": ready,
        "rows_total": 1 if clean else 136,
        "prior_publication_blocker_count": 55,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v26": blockers - 55,
        "status_counts": {
            "blocked": 0 if clean else 14,
            "bounded": 0 if clean else 22,
            "clean": 1 if clean else 35,
            "diagnostic_only": 0 if clean else 5,
            "flagged": 0 if clean else 12,
            "gated_skipped": 0 if clean else 10,
            "missing": 0 if clean else 2,
            "projection_only": 0 if clean else 5,
            "retired": 0 if clean else 31,
        },
        "publication_blockers": _publication_blockers(blockers),
        "missing_artifacts": []
        if clean
        else [
            {
                "experiment_id": "exp3152",
                "path": "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json",
                "reason": "missing_or_gated_dot293_artifact",
            },
            {
                "experiment_id": "exp3154",
                "path": "results/experiment_3154_multi_turn_repair_ladder_v3.json",
                "reason": "missing_or_gated_dot293_artifact",
            },
        ],
        "false_accept_recovery_summary": {
            "known_false_accept_recovery_preserved": True,
            "known_false_accept_rows_blocked": True,
            "live_verifier_evidence_trusted": clean,
            "methodology_requirement_count": 0 if clean else 7,
            "preflight_passed": clean,
            "preflight_status": "clean" if clean else "blocked",
            "clean_live_rerun_status": "clean" if clean else "gated_skipped",
            "recovery_claim_status": "clean_live_verifier_evidence_trusted"
            if clean
            else "exact_replay_preserved_but_live_verifier_untrusted",
            "repair_gate_implication": "unblocked" if clean else "blocked_pending_clean_rerun",
            "source_false_accept_rate": 0.5,
            "v26_rerun_false_accept_rate": 0.0,
            "blocked_downstream_field_count": 0 if clean else 13,
            "safe_downstream_field_count": 8,
        },
        "repair_summary": {
            "blocked_at_layer": "" if clean else "conductor_pre_gate",
            "gate_check_summary": "" if clean else "3 of 3 gate(s) failed",
            "gates_evaluated_count": 3,
            "live_repair_executed": clean,
            "repair_attempt_count": 3 if clean else 0,
            "repair_claim_allowed": clean,
            "repair_gate_state": "unblocked" if clean else "blocked",
            "repair_gate_status": "clean" if clean else "blocked",
            "repair_ladder_status": "clean" if clean else "gated_skipped",
            "selected_repair_rows": ["repair-row"] if clean else [],
            "tracefix_counterexample_count": 2 if clean else 0,
            "tracefix_status": "clean" if clean else "blocked",
        },
        "fr11_summary": {
            "continuous_self_learning_targeted": True,
            "ledger_consistency_rate": 1.0 if clean else 0.857143,
            "ledger_consistent_count": 14 if clean else 12,
            "ledger_status": "clean" if clean else "bounded",
            "model_weight_learning_allowed": clean,
            "no_weight_update_claim": not clean,
            "promotion_recommendation": "allow_fr11_promotion"
            if clean
            else "block_fr11_promotion_until_ledger_consistency_reaches_1.0",
            "replay_panel_count": 14,
            "residual_memory_status": "clean" if clean else "diagnostic_only",
            "residual_mismatch_count": 0 if clean else 2,
            "soundness_errors": 0,
            "completeness_errors": 0,
            "unsafe_skip_count": 0,
        },
        "architecture_boundary_summary": {
            "authenticated_speedup_claim_allowed": clean,
            "deployed_verifier_claim_allowed": clean,
            "energy_residual_blocker_count": 0 if clean else 6,
            "energy_sidecar_status": "clean" if clean else "projection_only",
            "exact_labeled_row_count": 6,
            "exact_row_coverage_count": 4,
            "hardware_commands_run": ["authenticated-smoke"] if clean else [],
            "hardware_status": "clean" if clean else "blocked",
            "kan_residual_blocker_count": 0 if clean else 3,
            "kan_status": "clean" if clean else "bounded",
            "known_false_accept_rows_scored": 2,
            "live_integration_claim_allowed": clean,
            "missing_operator_evidence_count": 0 if clean else 8,
            "monitor_record_count": 4,
            "new_monitor_record_count": 2,
            "no_hardware_commands_run": not clean,
            "scalar_energy_auc": 1.0,
            "violation_localization_coverage": 1.0,
        },
        "paper_readiness_implications": {
            "blocked_headline_claims": blocked_claims,
            "paper_ready": clean,
            "publication_blocker_count": blockers,
        },
        "source_artifacts": _source_artifacts(),
        "inference_substrate": {
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
        },
        "invariant_violations": [],
        "required_source_errors": [],
        "honest_verdict": "complete: matrix_v27_ready=true",
    }


def _write_matrix_capstone_and_sources(root: Path, *, blockers: int = 65) -> None:
    matrix = _matrix_v27(blockers=blockers)
    _write_json(root, mod.MATRIX_V27_REL_PATH, matrix)
    _write_json(root, mod.CAPSTONE_V292_REL_PATH, _capstone_v292())
    for source in matrix["source_artifacts"]:
        if not source["present"]:
            continue
        loaded = Path(source.get("loaded_path") or source["path"])
        _write_json(
            root, loaded, {"artifact": source["experiment_id"], source["ready_field"]: True}
        )


def test_req_report_3162_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3162: OpenSpec declares the .293 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3162" in spec
    assert "SCENARIO-REPORT-3162" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3162_builds_capstone_from_matrix_v27(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3162: .293 closes without unblocking repair."""

    _write_matrix_capstone_and_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["capstone_v293_ready"] is True
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 65
    assert artifact["blocker_delta_from_v26"] == 10
    assert artifact["next_top_gap"] == "clean_live_verifier_corrigendum_repair_gate"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert artifact["verifier_evidence_status"] == (
        "corrigendum_preserved_exact_replay_but_did_not_unblock_repair_live_evidence_untrusted"
    )
    assert artifact["repair_gate_status"] == "blocked_pending_clean_rerun_gate_failed"
    assert artifact["repair_ladder_status"] == (
        "correctly_skipped_gate_blocked_no_live_repair_attempts"
    )
    assert artifact["repair_was_executed"] is False
    assert artifact["repair_skip_was_correct"] is True
    assert artifact["fr11_self_learning_status"] == (
        "improved_to_0.857143_but_promotion_blocked_controller_memory_only_no_weight_update"
    )
    assert artifact["fr11_improved_from_v292"] is True
    assert artifact["fr11_promotion_allowed"] is False
    assert artifact["ebt_arm_status"] == (
        "projection_only_scalar_auc_1.0_exact_rows_6_no_live_integration_blockers_6"
    )
    assert artifact["kan_status"] == (
        "bounded_monitor_records_4_new_2_no_deployed_verifier_blockers_3"
    )
    assert artifact["sampler_hardware_status"] == (
        "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8"
    )
    assert artifact["hardware_energy_kan_headline_safe"] is False

    assert "corrigendum preserved exact known-false-accept replay" in artifact["what_293_proved"][0]
    assert artifact["next_milestone_recommendations"][0].startswith(
        "Run a clean live verifier rerun"
    )
    assert sources[mod.MATRIX_V27_REL_PATH.as_posix()]["role"] == "matrix_v27_authority"
    assert sources[mod.CAPSTONE_V292_REL_PATH.as_posix()]["role"] == ("capstone_v292_authority")
    assert sources[mod.MATRIX_V27_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.MATRIX_V27_REL_PATH
    )
    assert artifact["inference_substrate"] == {
        "kind": "aggregation_from_checked_in_matrix_v27_capstone_v292_and_dot293_artifacts",
        "source": mod.MATRIX_V27_REL_PATH.as_posix(),
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "live_model_calls_run_by_capstone": 0,
        "hardware_commands_run_by_capstone": [],
    }


def test_req_report_3162_clean_matrix_becomes_paper_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3162: paper readiness requires all five closeout gates."""

    _write_matrix_capstone_and_sources(tmp_path, blockers=0)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["capstone_v293_ready"] is True
    assert artifact["capstone_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["publication_blocker_count"] == 0
    assert artifact["blocker_delta_from_v26"] == -55
    assert artifact["next_top_gap"] == "publication_scope_reconciliation"
    assert artifact["verifier_evidence_status"] == "clean_live_verifier_evidence_trusted"
    assert artifact["repair_gate_status"] == "clean_repair_gate_unblocked"
    assert artifact["repair_ladder_status"] == "clean_repair_ladder_executed"
    assert artifact["fr11_self_learning_status"] == "clean_fr11_promotion_allowed"
    assert artifact["ebt_arm_status"] == "clean_energy_sidecar_live_integration"
    assert artifact["kan_status"] == "clean_deployed_kan_verifier_claim"
    assert artifact["sampler_hardware_status"] == "clean_authenticated_sampler_hardware_speedup"
    assert artifact["hardware_energy_kan_headline_safe"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3162_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3162: malformed inputs fail closed and helpers are deterministic."""

    _write_matrix_capstone_and_sources(tmp_path)
    bad = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    bad.write_text("{bad-json}\n", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))
    missing = mod.build_artifact(tmp_path / "empty")
    not_ready_root = tmp_path / "not_ready"
    _write_json(not_ready_root, mod.MATRIX_V27_REL_PATH, _matrix_v27(ready=False))
    _write_json(not_ready_root, mod.CAPSTONE_V292_REL_PATH, _capstone_v292())
    not_ready = mod.build_artifact(not_ready_root)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v293_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._duration(5.0, 4.0) == 0.0
    assert (
        mod._next_top_gap(0, True, True, True, False) == "architecture_boundary_headline_evidence"
    )
    assert missing["capstone_v293_ready"] is False
    assert missing["honest_verdict"].startswith("blocked:")
    assert "matrix_v27 authority is missing or malformed" in missing["invariant_violations"]
    assert not_ready["capstone_v293_ready"] is False
    assert "matrix_v27_ready is not true" in not_ready["invariant_violations"]
