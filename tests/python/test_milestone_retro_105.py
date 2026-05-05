"""Tests for the Exp 1363 milestone .105 retrospective.

Spec: REQ-REPORT-031, SCENARIO-REPORT-031.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_105 import (
    CRITERION_NAMES,
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
        1351: {
            "status": "complete",
            "terminal_certificate_required": True,
            "honest_verdict": "handoff_state_missing_exp1340_terminal_certificate_semantic_scheduler_dvi_grpo_closed",
        },
        1352: {
            "status": "complete",
            "sota_run_allowed": True,
            "max_token_budget_sufficient": True,
            "dynamic_dispatch_preserved": True,
            "honest_verdict": "preflight_allows_exp1353_pure_python_fallback_xgrammar_absent",
        },
        1353: {
            "status": "complete",
            "certificate_case_count": 4,
            "certificate_parse_rate": 0.0,
            "certificate_truthfulness_rate": 0.0,
            "trigger_token_hit_rate": 0.0,
            "unknown_preservation_rate": 0.0,
            "terminal_blocker": None,
            "headline_result_allowed": True,
            "honest_verdict": "sota_triggered_certificate_v7_measured",
        },
        1354: {
            "status": "complete",
            "certificate_cases_used": 4,
            "symbolization_pass_rate": 0.0,
            "countermodel_pass_rate": 0.0,
            "validity_pass_rate": 0.0,
            "dominant_skill_gap": "symbolization",
            "skill_split_claim_allowed": True,
            "honest_verdict": "logic_skill_split_supported_symbolization_dominates_exp1353",
        },
        1355: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp1353-triggered-certificate-v7-truncproof-sota.certificate_parse_rate "
                "(actual=0.0 >= expected=0.75)"
            ),
        },
        1357: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp1356-verge-mcs-repair-localization.repair_hint_precision "
                "(upstream artifact not found for task id 'exp1356-verge-mcs-repair-localization')"
            ),
        },
        1358: {
            "status": "complete",
            "replay_cases_used": 282,
            "fresh_verified_sample_count": 0,
            "self_learning_delta_overall": 1.596429,
            "nonforgetting_certificate_rate": 1.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": -0.846154,
            "dvi_ready": True,
            "headline_result_allowed": False,
            "update_is_replay_only": True,
            "honest_verdict": "verifier_selected_memory_replay_only_dvi_ready_non_headline",
        },
        1360: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: "
                "exp1359-dvi-certificate-tail-v4-gated.lossless_acceptance_claim_allowed "
                "(upstream artifact not found for task id 'exp1359-dvi-certificate-tail-v4-gated')"
            ),
        },
        1361: {
            "status": "complete",
            "certificate_states_mapped": ["SAT", "UNSAT", "UNKNOWN", "REPAIR"],
            "binary_spin_count": 4,
            "pdit_variable_count": 1,
            "state_expansion_ratio": 4.0,
            "energy_equivalence_error": 0.0,
            "hardware_claim_allowed": False,
            "kv260_claim_allowed": False,
            "honest_verdict": "cpu_only_pdit_certificate_state_mapping_ready_hardware_not_run",
        },
        1362: {
            "status": "complete",
            "publication_hold_state": "active",
            "external_dependency_claim_allowed": False,
            "honest_verdict": "publication_hold_active_local_evidence_does_not_support_ebt_arm_kona_or_hardware_claims",
        },
    }


def test_scenario_report_031_counts_only_evidence_and_gate_discipline() -> None:
    """SCENARIO-REPORT-031: .105 closeout does not overclaim blocked branches."""

    artifact = build_artifact(
        _scenario_sources(),
        {1356, 1359},
        roadmap_next_present=False,
        active_roadmap_present=True,
    )

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_total"] == 12
    assert artifact["criteria_met"] == 9
    assert artifact["criteria_results"]["exp1353_terminal_certificate_evidence"] == "MET"
    assert artifact["criteria_results"]["exp1355_unknown_preserving_semantic_validation"] == "GATED"
    assert artifact["criteria_results"]["exp1356_mcs_repair_hints_or_terminal_blocker"] == "MISSING"
    assert artifact["criteria_results"]["exp1357_false_acceptance_risk_before_savings"] == "GATED"
    assert artifact["criteria_results"]["exp1359_1360_structured_gate_discipline"] == "MET"

    assert artifact["certificate_branch_verdict"]["terminal_sota_evidence"] is True
    assert artifact["certificate_branch_verdict"]["branch_success"] is False
    assert artifact["certificate_branch_verdict"]["dominant_blocker"] == "missing_structural_tag"
    assert artifact["semantic_repair_verdict"]["semantic_repair_evidence_produced"] is False
    assert artifact["self_learning_verdict"]["mandatory_self_learning_satisfied"] is True
    assert artifact["self_learning_verdict"]["headline_evidence_produced"] is False
    assert artifact["hardware_verdict"]["hardware_execution_claim_allowed"] is False
    assert artifact["publication_hold_state"] == "active"
    assert artifact["experiment_statuses"]["exp1356"]["status"] == "missing"
    assert artifact["experiment_statuses"]["exp1359"]["status"] == "missing"
    assert artifact["roadmap_inputs"]["missing_requested_inputs"] == ["research-roadmap-next.yaml"]
    assert artifact["honest_verdict"].startswith("milestone_105_9_of_12_criteria_met")


def test_req_report_031_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-REPORT-031: a run leaves an auditable bootstrap artifact first."""

    out_path = tmp_path / "results" / "experiment_1363_milestone_105_retro_carryforward.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260505"
    assert written["criteria_total"] == 12
    assert set(artifact) >= {
        "status",
        "criteria_total",
        "criteria_met",
        "experiment_statuses",
        "certificate_branch_verdict",
        "semantic_repair_verdict",
        "self_learning_verdict",
        "hardware_verdict",
        "publication_hold_state",
        "carry_forward_tasks",
        "prior_failure_hygiene_notes",
        "honest_verdict",
    }


def test_req_report_031_run_loads_sources_and_marks_missing_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-031: run loads source JSON and writes final schema."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1363_milestone_105_retro_carryforward.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    _write_json(tmp_path / "research-roadmap.yaml", {"milestone": "2026.04.105"})
    _write_json(tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md", {})

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1363_milestone_105_retro_carryforward"
    assert written["schema"] == "milestone_retro_105_carryforward_v1"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 9
    assert any(
        item["experiment_id"] == "exp1356" and not item["exists"]
        for item in written["source_artifacts_checked"]
    )
    assert any(
        item["experiment_id"] == "exp1359" and not item["exists"]
        for item in written["source_artifacts_checked"]
    )
    assert written["prior_failure_hygiene_notes"]["missing_artifacts"] == ["exp1356", "exp1359"]
    assert len(written["carry_forward_tasks"]) >= 6
