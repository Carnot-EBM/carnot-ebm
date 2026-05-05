"""Tests for the Exp 1350 milestone .104 retrospective.

Spec: REQ-REPORT-029, SCENARIO-REPORT-029.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_104 import (
    CRITERION_NAMES,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[int, dict[str, object]]:
    return {
        1337: {
            "status": "complete",
            "honest_verdict": "environment_ready_stale_103_artifacts_classified",
            "environment_ready": True,
            "disk_quota_ok": True,
            "focused_pretest_status": "passed",
            "repeated_pretest_signature": {
                "focused_pretest_signature_active": False,
                "pretest_signature_occurrences": 21,
            },
            "stale_skeleton_count": 1,
        },
        1338: {
            "status": "complete",
            "honest_verdict": "exp1325_stale_environment_failure_gates_closed_recovery_ready",
            "exp1325_terminal_classification": "stale_skeleton_environment_failure",
            "minimum_parseable_attempts_to_recover": 6,
            "certificate_recovery_ready": True,
            "rerun_is_materially_different": True,
            "stale_artifacts_not_modified": True,
            "downstream_tasks_to_keep_closed": ["exp1342", "exp1343", "exp1345", "exp1346"],
        },
        1339: {
            "status": "complete",
            "honest_verdict": "dryrun_ready_pure_python_tagdispatch_xgrammar_absent",
            "dynamic_grammar_ready": True,
            "state_transition_error_count": 0,
            "unknown_state_supported": True,
            "certificate_states_supported": ["REPAIR_HINT", "SAT", "UNKNOWN", "UNSAT"],
        },
        1341: {
            "status": "complete",
            "honest_verdict": "local_certificate_slice_diagnostic_exp1340_missing_no_universal_detector_claim",
            "source_cases_available": 134,
            "repair_policy_by_failure_type": {
                "parser_schema_mismatch": {"policy": "request_fresh_verifier"},
                "semantic_invalidity": {"policy": "promote"},
                "unknown_state_mishandling": {"policy": "quarantine"},
            },
            "universal_detector_claim_allowed": False,
        },
        1343: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp1342-chopchop-nsvif-semantic-validator-gated.validator_execution_pass_rate "
                "(upstream artifact not found)"
            ),
        },
        1344: {
            "status": "complete",
            "honest_verdict": "failure_type_memory_policy_dvi_ready_replay_non_headline",
            "nonforgetting_certificate_rate": 1.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": -0.846154,
            "dvi_ready": True,
            "headline_result_allowed": False,
            "failure_type_policy": {
                "semantic_invalidity": {"certificate_tail_update_allowed": True},
                "unknown_state_mishandling": {"certificate_tail_update_allowed": False},
            },
        },
        1347: {
            "status": "complete",
            "honest_verdict": "thrml_unavailable_mapping_notes_only_no_hardware_claim",
            "thrml_import_available": False,
            "hardware_claim_allowed": False,
            "tsu_mapping_notes": ["THRML package unavailable in the local environment"],
        },
        1348: {
            "status": "complete",
            "honest_verdict": "cpu_only_update_dynamics_dual_bram_packet_ready_hardware_not_run",
            "hardware_claim_allowed": False,
            "kv260_claim_allowed": False,
            "sync_async_regime": {"sync_supported": True, "async_supported": True},
            "bram_layout": {"dual_bram_packet": True},
        },
        1349: {
            "status": "complete",
            "honest_verdict": (
                "external_parity_gap_audit_complete_local_evidence_only_no_kona_or_external_dependency_claim"
            ),
            "external_dependency_claim_allowed": False,
            "parity_gaps": ["native continuous latent EBT evidence", "external Kona parity execution"],
            "phase3_obligations": ["keep local evidence separate from external positioning"],
            "publication_claim_changes_needed": ["keep publication hold active"],
        },
    }


def test_scenario_report_029_counts_milestone_104_source_criteria() -> None:
    """SCENARIO-REPORT-029: Exp1350 reports .104 9/12 from source fields."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids={1340, 1342, 1345, 1346},
        roadmap_next_present=False,
        active_roadmap_present=True,
    )

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "environment_gate_ready": "MET",
        "exp1325_stale_gate_state_closed": "MET",
        "dynamic_grammar_ready_or_terminal_blocker": "MET",
        "triggered_certificate_branch_recovered_or_retired": "MISSING",
        "halluguard_failure_split_no_universal_detector": "MET",
        "semantic_validator_executed_unknown_preserved": "MISSING",
        "margin_aware_scheduler_false_acceptance_risk_reported": "GATED",
        "continuous_self_learning_accounted": "MET",
        "dvi_grpo_gate_discipline_preserved": "MET",
        "hardware_portability_evidence_without_unverified_claims": "MET",
        "external_ebt_kona_parity_mapped": "MET",
        "retro_104_complete": "MET",
    }
    assert artifact["criteria_met"] == 9
    assert artifact["criteria_total"] == 12
    assert artifact["status"] == "complete"
    assert artifact["experiment_statuses"]["exp1340"]["status"] == "missing"
    assert artifact["experiment_statuses"]["exp1343"]["status"] == "blocked"
    assert artifact["publication_hold_state"]["hold_active"] is True
    assert artifact["publication_hold_state"]["hold_lift_evidence"] == "absent"
    assert artifact["certificate_branch_verdict"]["headline_ready"] is False
    assert artifact["self_learning_verdict"]["headline_ready"] is False
    assert artifact["hardware_verdict"]["hardware_execution_claim_allowed"] is False
    assert artifact["prior_failure_hygiene_notes"]["stale_skeleton_closed_cleanly"] is True
    assert artifact["prior_failure_hygiene_notes"]["focused_pretest_closed_cleanly"] is True
    assert len(artifact["carry_forward_tasks"]) >= 5
    assert artifact["honest_verdict"] == "milestone_104_9_of_12_criteria_met_carryforward_required"
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)


def test_req_report_029_writes_bootstrap_and_final_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-029: Exp1350 writes in-progress first and then final JSON."""

    out_path = tmp_path / "results" / "experiment_1350_milestone_104_retro_carryforward.json"
    bootstrap = write_in_progress_artifact(out_path)

    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    results_dir = tmp_path / "results"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    _write_json(tmp_path / "research-roadmap.yaml", {"milestone": "2026.04.104"})
    _write_json(
        tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md",
        {"title": "Research Roadmap vNEXT: Milestone 2026.04.104"},
    )

    artifact = run(root=tmp_path, out_path=out_path)

    assert artifact["status"] == "complete"
    assert artifact["criteria_met"] == 9
    assert json.loads(out_path.read_text(encoding="utf-8"))["honest_verdict"] == (
        "milestone_104_9_of_12_criteria_met_carryforward_required"
    )
