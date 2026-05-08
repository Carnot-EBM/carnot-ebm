"""Tests for the Exp 1559 milestone .119 retrospective.

Spec: REQ-REPORT-061, SCENARIO-REPORT-061.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_119 as retro119
from carnot.reporting.milestone_retro_119 import (
    HONESTLY_TERMINAL,
    MET,
    NOT_MET,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _blocker_reason,
    _load_sources,
    _protected_files_clean,
    _read_json,
    _read_text,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1547": {
            "status": "complete",
            "activation_manifest_complete": True,
            "predecessor_criteria_met": 13,
            "predecessor_criteria_total": 14,
            "thrml_independent_rng_required": True,
            "honest_verdict": "complete: milestone_119_activation_complete",
        },
        "exp1548": {
            "status": "complete",
            "independent_rng_audit_ready": False,
            "rng_path_independent": True,
            "byte_identical_pairs": [],
            "bounded_kl_passed": False,
            "max_kl_divergence": 0.169802350136,
            "nonzero_stochastic_delta_observed": True,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": "complete: independent_rng_thrml_carnot_parity_not_ready",
        },
        "exp1549": {
            "status": "complete",
            "satquest_oracle_repair_ready": True,
            "satquest_zero_false_accepts": True,
            "solver_oracle_false_accepts_after": 0,
            "assignment_witnesses_checked": 10,
            "unsat_certificates_checked": 11,
            "perturbation_checks_passed": True,
            "honest_verdict": "complete: satquest_oracle_repair_zero_false_accepts",
        },
        "exp1550": {
            "status": "complete",
            "satquest_sota_reeval_ready": True,
            "repaired_gate": {"ready": True, "satquest_zero_false_accepts": True},
            "live_sota_model_inference_used": True,
            "model_availability_blockers": [],
            "solver_oracle_false_accepts": 0,
            "false_accept_rate": 0.0,
            "model_specs": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "models_attempted": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            "honest_verdict": "complete: satquest_sota_reeval_zero_false_accepts",
        },
        "exp1551": {
            "status": "complete",
            "unified_contract_gate_ready": True,
            "automata_masks_used": True,
            "semantic_repair_layer_used": True,
            "sat_oracle_used": True,
            "runtime_contracts_used": True,
            "deterministic_validators_final_authority": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "soft_signal_override_count": 0,
            "honest_verdict": "complete: automata_sat_unified_contract_gate_ready",
        },
        "exp1552": {
            "status": "complete",
            "residual_drift_repair_ready": True,
            "localized_repairs_attempted": 64,
            "repair_attempts": 64,
            "repaired_drift_cases": 64,
            "drift_reduction_delta": 1.0,
            "contradiction_cases_untouched": 2,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "replay_pass_rate": 1.0,
            "honest_verdict": "complete: residual_drift_repair_policy_ready",
        },
        "exp1553": {
            "status": "complete",
            "claim_isolation_router_scale_ready": True,
            "cases_total": 75,
            "claims_extracted": 75,
            "routed_cases": 23,
            "budget_delta": -52,
            "budget_reduced": True,
            "false_accept_rate": 0.0,
            "missed_failure_count": 0,
            "unified_contract_gate_ready": True,
            "honest_verdict": "complete: claim-isolation router scale ready",
        },
        "exp1554": {
            "status": "complete",
            "product_line_scale_v4_ready": True,
            "branch_retired": False,
            "cases_total": 120,
            "parse_rate": 1.0,
            "feasibility_rate": 1.0,
            "oracle_agreement_rate": 1.0,
            "objective_gap_mean": 0.0,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: product-line staged scale v4 ready",
        },
        "exp1555": {
            "status": "complete",
            "fr11_positive_utility_gate_ready": True,
            "continuous_self_learning_task": "fr11_positive_utility_or_retire_v14",
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "utility_delta": 1.0,
            "positive_utility_achieved": True,
            "positive_utility_claim_retired": False,
            "external_feedback_used": True,
            "honest_verdict": "complete: fr11 positive utility gate passed",
        },
        "exp1556": {
            "status": "complete",
            "arm_ebm_logprob_telemetry_ready": True,
            "logprob_available": True,
            "topk_available": True,
            "deterministic_validators_final_authority": True,
            "diagnostic_cases": 4,
            "routing_auc": 1.0,
            "telemetry_blockers": ["token_logprobs_missing"],
            "honest_verdict": "complete: ARM/EBT logprob telemetry ready diagnostic-only",
        },
        "exp1557": {
            "status": "complete",
            "verification_compute_router_ready": True,
            "satquest_sota_reeval_ready": True,
            "unified_contract_gate_ready": True,
            "verification_cost_baseline": 399,
            "verification_cost_router": 358,
            "verification_cost_delta": -41,
            "weak_verifiers_used": ["energy_diagnostic"],
            "deterministic_validators_used": ["sat_solver", "unified_contract_gate"],
            "soft_signals_used_for_routing_only": ["mean_logprob"],
            "false_accept_rate": 0.0,
            "missed_failure_count": 0,
            "honest_verdict": "complete: weaver_verification_compute_router_ready",
        },
        "exp1558": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "2 of 3 gate(s) failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gates_evaluated": [
                {
                    "artifact_field": "independent_rng_audit_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                },
                {
                    "artifact_field": "rng_path_independent",
                    "expected": True,
                    "actual": True,
                    "passed": True,
                },
                {
                    "artifact_field": "bounded_kl_passed",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                },
            ],
        },
    }


def test_req_report_061_scores_119_and_records_120_carry_forward_gates() -> None:
    """REQ-REPORT-061: .119 criteria use source fields and expose gates."""

    artifact = build_artifact(
        sources=_terminal_sources(),
        missing_source_ids=[],
        roadmap_doc_text="| THRML RNG | independent_rng_audit_ready=true |",
        research_roadmap_yaml_text="milestone: 2026.04.119\n",
        research_roadmap_next_text="milestone: 2026.04.120\n",
        research_complete_text="- id: 2026.04.118\n",
        ops_status_text="THRML KL=0.17; FR-11 positive utility passed",
        ops_changelog_text="Exp 1558 gated by Exp 1548",
        ops_known_issues_text="THRML vendoring; soft-signal authority boundaries",
        conductor_log_text="| exp1558 | GATE_BLOCK | 2 of 3 gate(s) failed |",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.119"
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_total"] == 13
    assert artifact["criteria_results"]["thrml_rng"]["status"] == NOT_MET
    assert artifact["criteria_results"]["hardware_readiness"]["status"] == HONESTLY_TERMINAL
    assert artifact["thrml_independent_rng_gate"]["rng_path_independent"] is True
    assert artifact["thrml_independent_rng_gate"]["independent_rng_audit_ready"] is False
    assert artifact["thrml_independent_rng_gate"]["carry_forward_to_120"] == (
        "vendor_thrml_or_repair_sampler_mismatch_before_any_parity_headline"
    )
    assert artifact["satquest_oracle_repair_gate"]["satquest_zero_false_accepts"] is True
    assert artifact["satquest_sota_gate"]["solver_oracle_false_accepts"] == 0
    assert artifact["unified_contract_gate"]["acceptance_authority"] == (
        "deterministic_validators_only"
    )
    assert artifact["fr11_positive_utility_or_retire_gate"]["positive_utility_achieved"] is True
    assert artifact["arm_ebt_telemetry_gate"]["acceptance_authority"] == (
        "deterministic_validators_only"
    )
    assert artifact["verification_compute_router_gate"]["soft_signals_authority"] == (
        "routing_only"
    )
    assert artifact["extropic_readiness_gate"]["status"] == "blocked_by_thrml_rng_gate"
    assert artifact["ops_reconciliation_needed"]["needed"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    completed_ids = {task["experiment_id"] for task in artifact["completed_tasks"]}
    terminal_ids = {task["experiment_id"] for task in artifact["honestly_terminal_tasks"]}
    failed_ids = {task["experiment_id"] for task in artifact["failed_or_blocked_tasks"]}
    assert "exp1549" in completed_ids
    assert "exp1558" in terminal_ids
    assert "exp1548" in failed_ids


def test_scenario_report_061_successful_thrml_branch_and_missing_source() -> None:
    """SCENARIO-REPORT-061: missing and repaired branches stay explicit."""

    sources = _terminal_sources()
    sources["exp1548"].update(
        {
            "independent_rng_audit_ready": True,
            "bounded_kl_passed": True,
            "max_kl_divergence": 0.02,
            "honest_verdict": "complete: independent_rng_audit_ready",
        }
    )
    sources["exp1558"].update(
        {
            "status": "complete",
            "thrml_post_rng_scale_decision_ready": True,
            "extropic_packet_updated": True,
            "no_hardware_execution_claim": True,
            "honest_verdict": "complete: extropic packet updated after rng gate",
        }
    )

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1556"],
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_roadmap_next_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        conductor_log_text="",
        protected_files_unchanged=False,
    )

    assert artifact["criteria_results"]["thrml_rng"]["status"] == MET
    assert artifact["criteria_results"]["arm_ebt_telemetry"]["status"] == NOT_MET
    assert artifact["criteria_results"]["retrospective"]["status"] == NOT_MET
    assert artifact["extropic_readiness_gate"]["status"] == "readiness_packet_updated"
    assert artifact["active_roadmap_modified"] is True
    assert artifact["conductor_modified"] is True


def test_req_report_061_missing_sources_and_run_write_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-061: run writes in-progress and terminal JSON artifacts."""

    out_path = tmp_path / "results" / "experiment_1559_milestone_119_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        if exp_id != "exp1558":
            _write_json(tmp_path / "results" / filename, _terminal_sources()[exp_id])

    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "Success Criteria",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.119\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.04.120\n",
        encoding="utf-8",
    )
    (tmp_path / "research-complete.yaml").write_text("- id: 2026.04.118\n", encoding="utf-8")
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text("| exp1557 | OK |\n", encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text("status", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("changelog", encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text("known", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_results"]["hardware_readiness"]["status"] == NOT_MET
    assert any(task["experiment_id"] == "exp1558" for task in written["failed_or_blocked_tasks"])
    assert written["source_inputs_read"]["ops/known-issues.md"]["exists"] is True
    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _blocker_reason({"status": "blocked", "honest_verdict": "blocked reason"}) == (
        "blocked reason"
    )
    assert _blocker_reason({"blockers": ["a", "b"]}) == "a, b"
    assert retro119._arm_telemetry_honest_blocker(
        {"status": "blocked", "deterministic_validators_final_authority": True}
    )
    loaded, missing = _load_sources(tmp_path / "results")
    assert "exp1547" in loaded
    assert missing == ["exp1558"]


def test_req_report_061_protected_file_helper_is_defensive(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-061: git guard reports dirty but treats git errors as unknown-clean."""

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(retro119.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(retro119.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(retro119.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
