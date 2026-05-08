"""Tests for the Exp 1546 milestone .118 retrospective.

Spec: REQ-REPORT-059, SCENARIO-REPORT-059.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_118 as retro118
from carnot.reporting.milestone_retro_118 import (
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
        "exp1533": {
            "status": "complete",
            "activation_manifest_complete": True,
            "predecessor_criteria_met": 14,
            "predecessor_criteria_total": 14,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: activation ready",
        },
        "exp1534": {
            "status": "complete",
            "orphan_test_guard_ready": True,
            "import_targets_checked": 2464,
            "orphan_imports_detected": 0,
            "active_roadmap_modified": False,
            "conductor_modified": False,
            "honest_verdict": "complete: orphan test guard ready",
        },
        "exp1535": {
            "status": "complete",
            "contract_decoder_adapter_ready": True,
            "model_specs": [
                {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            "baseline_parse_rate": 0.0,
            "automata_parse_rate": 1.0,
            "baseline_contract_accept_rate": 0.0,
            "automata_contract_accept_rate": 1.0,
            "latency_delta_seconds": -0.2,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "abs_dfa_masks_used": True,
            "honest_verdict": "complete: decoder ready",
        },
        "exp1536": {
            "status": "complete",
            "satquest_benchmark_ready": True,
            "solver_oracle_used": "exact_exhaustive_fallback",
            "solver_oracle_false_accepts": 3,
            "false_accept_rate": 0.166667,
            "cnf_instances": 6,
            "formats_tested": ["machine", "narrative", "symbolic"],
            "blockers": ["solver_oracle_false_accepts"],
            "honest_verdict": "complete: satquest benchmark ready with false accepts",
        },
        "exp1537": {
            "status": "complete",
            "beaver_bound_ready": True,
            "bounded_prefixes": 78,
            "bound_violations": [],
            "deterministic_validator_final_authority": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "blockers": [
                "token_logprobs_unavailable_structural_simulation_used",
                "topk_unavailable_structural_simulation_used",
            ],
            "honest_verdict": "complete: beaver prefix-risk audit ready",
        },
        "exp1538": {
            "status": "complete",
            "residual_drift_ledger_ready": True,
            "multi_turn_cases": 134,
            "contradiction_cases": 2,
            "satisfiable_drift_cases": 64,
            "repaired_drift_cases": 2,
            "other_blocker_cases": 68,
            "drift_rate": 0.477612,
            "solver_oracle_used": True,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: residual drift ledger ready",
        },
        "exp1539": {
            "status": "complete",
            "fr11_external_feedback_ready": True,
            "continuous_self_learning_task": True,
            "no_model_weight_mutation": True,
            "soundness_mistakes": 0,
            "utility_delta": 0.0,
            "positive_utility_promotion_ready": False,
            "externally_verified_updates": ["daily_eval:verified"],
            "promoted_updates": ["daily_eval:verified"],
            "honest_verdict": "complete: fr11 ready; positive utility not demonstrated",
        },
        "exp1540": {
            "status": "complete",
            "product_line_scale_ready": True,
            "branch_retired": False,
            "cases_total": 40,
            "automata_constraints_used": True,
            "syntax_stage_pass_rate": 1.0,
            "feasibility_stage_pass_rate": 1.0,
            "oracle_agreement_rate": 1.0,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "deterministic_validator_final_authority": True,
            "honest_verdict": "complete: product line scaled",
        },
        "exp1541": {
            "status": "complete",
            "uncertainty_router_ready": True,
            "cases_loaded": 18,
            "claims_extracted": 18,
            "routed_cases": 7,
            "budget_delta": -11,
            "budget_improvement_claimed": True,
            "verifier_calls_claim_isolated": 7,
            "verifier_calls_full_context": 18,
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "honest_verdict": "complete: router ready",
        },
        "exp1542": {
            "status": "complete",
            "arm_ebm_diagnostic_ready": True,
            "deterministic_validators_final_authority": True,
            "no_model_weight_mutation": True,
            "diagnostic_cases": 24,
            "routing_auc": 1.0,
            "energy_label_correlation": 0.683698,
            "soft_value_label_correlation": None,
            "logprob_available": False,
            "honest_verdict": "complete: arm ebm diagnostic ready",
        },
        "exp1543": {
            "status": "complete",
            "thrml_parity_n256_schedule_ready": True,
            "parity_passed": True,
            "n_spins": 256,
            "schedules_tested": 3,
            "kl_divergence": 0.002662339801,
            "max_energy_delta": 0.011440429688,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": "complete_thrml_n256",
        },
        "exp1544": {
            "status": "complete",
            "diverse_topology_parity_n64_ready": True,
            "parity_passed": True,
            "n_spins": 64,
            "topologies_tested": ["complete", "sparse_random", "lattice", "scale_free"],
            "topologies_passed": ["complete", "sparse_random", "lattice", "scale_free"],
            "kl_divergence": 0.000728807813,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "honest_verdict": "complete_thrml_n64_diverse",
        },
        "exp1545": {
            "status": "complete",
            "extropic_z1_readiness_packet_ready": True,
            "no_hardware_execution_claim": True,
            "benchmark_cases_included": 7,
            "access_blockers": ["no_authenticated_extropic_z1_or_xtr0_device_access"],
            "required_device_evidence_fields": ["authenticated_access_proof"],
            "protected_files_unchanged": {
                "research-roadmap.yaml": True,
                "scripts/research_conductor.py": True,
            },
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: extropic packet ready no hardware claim",
        },
    }


def test_req_report_059_scores_118_and_records_carry_forward_gates() -> None:
    """REQ-REPORT-059: .118 criteria use source fields and preserve limits."""

    artifact = build_artifact(
        sources=_terminal_sources(),
        missing_source_ids=[],
        roadmap_doc_text="Target threshold: at least 12 of 14 tasks complete",
        research_roadmap_yaml_text="milestone: 2026.04.118\n",
        research_roadmap_next_text="milestone: 2026.04.119\n",
        research_complete_text="- id: 2026.04.117\n",
        ops_status_text="FR-11 utility_delta=0.0; SATQuest false accepts remain",
        ops_changelog_text="Extropic readiness only; no hardware execution",
        ops_known_issues_text="SATQuest false accepts; no authenticated Extropic access",
        conductor_log_text="| exp1533 | OK |\n| exp1545 | OK |",
        protected_files_unchanged=True,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.118"
    assert artifact["criteria_met"] == 13
    assert artifact["criteria_total"] == 14
    assert artifact["criteria_results"]["satquest_benchmark"]["status"] == NOT_MET
    assert artifact["criteria_results"]["continuous_self_learning"]["status"] == MET
    assert artifact["satquest_verifier_gate"]["zero_solver_oracle_false_accepts"] is False
    assert artifact["fr11_positive_utility_gate"]["positive_utility_achieved"] is False
    assert artifact["fr11_positive_utility_gate"]["status"] == "safety_only"
    assert artifact["automata_contract_gate"]["automata_constraints_improved_contract_generation"]
    assert artifact["residual_drift_gate"]["satisfiable_drift_cases"] == 64
    assert artifact["product_line_carry_forward_gate"]["decision"] == "continue"
    assert artifact["claim_isolation_router_gate"]["budget_reduced"] is True
    assert artifact["arm_ebm_diagnostic_boundary"]["acceptance_authority"] == (
        "deterministic_validators_only"
    )
    assert artifact["thrml_next_scaling_gate"]["can_scale_further_in_software"] is True
    assert artifact["extropic_access_readiness_gate"]["hardware_execution_claimed"] is False
    assert artifact["ops_reconciliation_needed"]["needed"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    completed_ids = {task["experiment_id"] for task in artifact["completed_tasks"]}
    terminal_ids = {task["experiment_id"] for task in artifact["honestly_terminal_tasks"]}
    failed_ids = {task["experiment_id"] for task in artifact["failed_or_blocked_tasks"]}
    assert "exp1535" in completed_ids
    assert "exp1536" not in completed_ids
    assert {"exp1536", "exp1539", "exp1545"} <= terminal_ids
    assert failed_ids == {"exp1536"}


def test_scenario_report_059_honestly_terminal_thrml_and_positive_fr11_branch() -> None:
    """SCENARIO-REPORT-059: blockers and positive utility stay explicit."""

    sources = _terminal_sources()
    sources["exp1539"].update(
        {
            "utility_delta": 0.25,
            "positive_utility_promotion_ready": True,
            "honest_verdict": "complete: fr11 positive utility demonstrated",
        }
    )
    sources["exp1543"].update(
        {
            "status": "blocked",
            "thrml_parity_n256_schedule_ready": False,
            "parity_passed": False,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "blockers": ["thrml_runtime_unavailable"],
        }
    )

    artifact = build_artifact(
        sources=sources,
        missing_source_ids=[],
        roadmap_doc_text="",
        research_roadmap_yaml_text="",
        research_roadmap_next_text="",
        research_complete_text="",
        ops_status_text="",
        ops_changelog_text="",
        ops_known_issues_text="",
        conductor_log_text="",
        protected_files_unchanged=True,
    )

    assert artifact["criteria_results"]["thrml_n256"]["status"] == HONESTLY_TERMINAL
    assert artifact["fr11_positive_utility_gate"]["status"] == "positive_utility_ready"
    assert artifact["fr11_positive_utility_gate"]["positive_utility_achieved"] is True
    assert artifact["criteria_met"] == 13
    assert any(task["experiment_id"] == "exp1543" for task in artifact["honestly_terminal_tasks"])


def test_req_report_059_missing_sources_and_run_write_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-059: missing artifacts fail explicitly and run writes JSON."""

    out_path = tmp_path / "results" / "experiment_1546_milestone_118_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        if exp_id != "exp1545":
            _write_json(tmp_path / "results" / filename, _terminal_sources()[exp_id])

    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "Target threshold: at least 12 of 14 tasks complete",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.118\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.04.119\n",
        encoding="utf-8",
    )
    (tmp_path / "research-complete.yaml").write_text("- id: 2026.04.117\n", encoding="utf-8")
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text("| exp1544 | OK |\n", encoding="utf-8")
    (tmp_path / "ops" / "status.md").write_text("status", encoding="utf-8")
    (tmp_path / "ops" / "changelog.md").write_text("changelog", encoding="utf-8")
    (tmp_path / "ops" / "known-issues.md").write_text("known", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, protected_files_unchanged=True)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_results"]["hardware_readiness"]["status"] == NOT_MET
    assert any(task["experiment_id"] == "exp1545" for task in written["failed_or_blocked_tasks"])
    assert written["source_inputs_read"]["ops/known-issues.md"]["exists"] is True
    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _blocker_reason({"status": "blocked", "honest_verdict": "blocked reason"}) == (
        "blocked reason"
    )
    loaded, missing = _load_sources(tmp_path / "results")
    assert "exp1533" in loaded
    assert missing == ["exp1545"]


def test_req_report_059_protected_file_helper_is_defensive(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-059: git guard reports dirty but treats git errors as unknown-clean."""

    class CleanResult:
        returncode = 0

    class DirtyResult:
        returncode = 1

    monkeypatch.setattr(retro118.subprocess, "run", lambda *args, **kwargs: CleanResult())
    assert _protected_files_clean(tmp_path) is True
    monkeypatch.setattr(retro118.subprocess, "run", lambda *args, **kwargs: DirtyResult())
    assert _protected_files_clean(tmp_path) is False

    def raise_os_error(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(retro118.subprocess, "run", raise_os_error)
    assert _protected_files_clean(tmp_path) is True
