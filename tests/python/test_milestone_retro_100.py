"""Tests for the Exp 1295 milestone .100 retrospective.

Spec: REQ-REPORT-025, SCENARIO-REPORT-025.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_100 import (
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
        1282: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "prior_failures field is missing or incomplete",
        },
        1283: {
            "status": "complete",
            "honest_verdict": "selected_llama_cpp_gbnf",
            "grammar_backend_available": True,
            "grammar_backend_selected": "llama_cpp_gbnf",
            "cdot_expressiveness_note": "route targets require post-decode checks",
            "static_trie_note": "bounded enum fields are trie-friendly",
            "bounded_vocab_constraint_count": 9,
            "automata_fallback_viable": True,
            "dfa_checkable_fields": ["final_answer"],
            "structure_snowballing_risk": "medium",
        },
        1286: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "upstream artifact not found for task id",
            "gates_evaluated": [
                {
                    "upstream": "exp1285-triggered-certificate-extraction-v2",
                    "artifact_field": "certificate_parse_rate",
                    "expected": 0.8,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
        1288: {
            "status": "complete",
            "honest_verdict": "online_verifier_feedback_neutral_non_headline",
            "headline_result_allowed": False,
            "dvi_acceptance_delta": 0.357143,
            "online_acceptance_delta": 0.357143,
            "claim_level_memory_entries": 7,
            "clause_prediction_records": [{"constraint_pattern": "arithmetic:addition"}],
            "memory_update_written": True,
            "self_learning_delta_overall": 0.0,
        },
        1291: {
            "status": "complete",
            "honest_verdict": "hardnetpp_nonlinear_repair_viable",
            "hardnetpp_delta_over_snarenet": 1.2207222442957435,
            "nonlinear_repair_viable": True,
            "construct_refine_iterations": 5.611111111111111,
            "copy_as_decode_verified_span_reuse": 1.0,
        },
        1292: {
            "status": "complete",
            "honest_verdict": "feasibility_channel_predictive_marginal",
            "feasibility_channel_auc": 0.6604651162790698,
            "feasibility_channel_predictive": True,
            "repair_help_prediction_accuracy": 0.44871794871794873,
            "false_continue_rate": 0.7714285714285715,
            "false_stop_rate": 0.0,
        },
        1293: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "prior_failures field is missing or incomplete",
        },
        1294: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "prior_failures field is missing or incomplete",
        },
    }


def test_scenario_report_025_counts_milestone_100_source_criteria() -> None:
    """SCENARIO-REPORT-025: Exp1295 reports .100 5/14 from source fields."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "sota_gguf_cache_provenance_preflight": "BLOCKED",
        "certificate_grammar_backend_bakeoff": "MET",
        "ars_uqlm_answer_stability_sota_audit": "GATED",
        "triggered_certificate_extraction_v2": "GATED",
        "grad_beaver_nsvif_semantic_routing": "GATED",
        "token_guard_cactus_constrained_acceptance_v2": "GATED",
        "interwhen_dvi_verifier_feedback_replay": "MET",
        "leanabell_grpo_v9_sota_headline_gated": "GATED",
        "skill_graph_promotion_demotion": "MISSING",
        "hardnetpp_nonlinear_repair_benchmark": "MET",
        "dsp_feasibility_channel_diagnostic": "MET",
        "ebt_arm_ebm_cot_energy_bridge_audit": "BLOCKED",
        "arxiv_v10_submission_receipt_or_blocker": "BLOCKED",
        "retro_100_complete": "MET",
    }
    assert artifact["criteria_met"] == 5
    assert artifact["criteria_total"] == 14
    assert artifact["retro_complete"] is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"] == "milestone_100_5_of_14_criteria_met"
    assert artifact["self_learning_result"]["dvi_acceptance_delta"] == 0.357143
    assert artifact["sota_model_usage_summary"]["headline_result_allowed"] is False
    assert artifact["continuous_repair_summary"]["hardnetpp_nonlinear_repair_viable"] is True
    assert artifact["publication_state"]["status"] == "blocked"
    assert any("experiment_1290" in item["path"] for item in artifact["stale_artifacts"])
    assert len(artifact["top_successes"]) == 4
    assert len(artifact["top_gaps"]) == 5
    assert len(artifact["key_carry_forwards"]) == 5


def test_req_report_025_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-REPORT-025: a run can leave an auditable in-progress artifact."""

    out_path = tmp_path / "results" / "experiment_1295_milestone_retro_100.json"

    artifact = write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260504"
    assert written["criteria_total"] == 14
    assert written["retro_complete"] is False


def test_req_report_025_run_loads_sources_and_writes_schema(tmp_path: Path) -> None:
    """REQ-REPORT-025: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1295_milestone_retro_100.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1295_milestone_retro_100"
    assert written["schema"] == "milestone_retro_v5"
    assert written["milestone"] == "2026.04.100"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 5
    assert written["criteria_total"] == 14


def test_req_report_025_dependency_gates_are_mechanical() -> None:
    """REQ-REPORT-025: unmet gates, passed gates, and missing artifacts differ."""

    sources = _scenario_sources()
    artifact = build_artifact(sources)
    assert artifact["criteria_results"]["ars_uqlm_answer_stability_sota_audit"] == "GATED"
    assert artifact["criteria_results"]["skill_graph_promotion_demotion"] == "MISSING"

    sources[1282] = {"status": "complete", "cached_sota_ready": True}
    sources[1284] = {"status": "complete", "answer_stability_score": 0.7}
    sources[1285] = {
        "status": "complete",
        "certificate_parse_rate": 0.79,
        "headline_result_allowed": True,
    }
    gated_artifact = build_artifact(sources)
    assert gated_artifact["criteria_results"]["token_guard_cactus_constrained_acceptance_v2"] == "GATED"

    sources[1285]["certificate_parse_rate"] = 0.8
    ungated_artifact = build_artifact(sources)
    assert ungated_artifact["criteria_results"]["token_guard_cactus_constrained_acceptance_v2"] == "MISSING"

    sources[1288]["memory_update_written"] = False
    memory_gated_artifact = build_artifact(sources)
    assert memory_gated_artifact["criteria_results"]["skill_graph_promotion_demotion"] == "GATED"


def test_req_report_025_complete_and_stale_branches_are_mechanical() -> None:
    """REQ-REPORT-025: complete, terminal partial, and stale artifacts are distinct."""

    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    sources = _scenario_sources()
    sources[1282] = {"status": "complete", "cached_sota_ready": True, "headline_result_possible": True}
    sources[1284] = {
        "status": "complete",
        "answer_stability_score": 0.7,
        "models_used": [{"hf_id": model_id, "used_for_generation": True}],
    }
    sources[1285] = {"status": "complete", "certificate_parse_rate": 0.8, "headline_result_allowed": True}
    sources[1286] = {"status": "complete", "semantic_routing_coverage": 0.75, "routed_claim_count": 12}
    sources[1287] = {
        "status": "complete",
        "cactus_acceptance_rate": 0.4,
        "risk_bound_proxy": 0.1,
        "token_guard_risk_score": 0.2,
        "low_risk_acceptance_rate": 0.3,
        "speedbench_eval_mode": "replay",
    }
    sources[1289] = {"status": "complete", "headline_result_allowed": True, "grpo_v9_delta": 0.2}
    sources[1290] = {"status": "complete", "skill_replay_delta": 0.1}
    sources[1293] = {"status": "complete", "energy_bridge_written": True}
    sources[1294] = {"status": "complete", "external_blocker": "operator_submission_required"}

    complete_artifact = build_artifact(sources)

    assert complete_artifact["criteria_results"]["ars_uqlm_answer_stability_sota_audit"] == "MET"
    assert complete_artifact["criteria_results"]["triggered_certificate_extraction_v2"] == "MET"
    assert complete_artifact["criteria_results"]["grad_beaver_nsvif_semantic_routing"] == "MET"
    assert complete_artifact["criteria_results"]["token_guard_cactus_constrained_acceptance_v2"] == "MET"
    assert complete_artifact["criteria_results"]["leanabell_grpo_v9_sota_headline_gated"] == "MET"
    assert complete_artifact["criteria_results"]["skill_graph_promotion_demotion"] == "MET"
    assert complete_artifact["criteria_results"]["ebt_arm_ebm_cot_energy_bridge_audit"] == "MET"
    assert complete_artifact["criteria_results"]["arxiv_v10_submission_receipt_or_blocker"] == "MET"
    assert complete_artifact["sota_model_usage_summary"]["headline_model_ids_used"] == [model_id]

    sources[1282] = {"status": "in_progress", "cached_sota_ready": True}
    sources[1285] = {"status": "complete", "certificate_parse_rate": 0.8, "headline_result_allowed": False}
    stale_artifact = build_artifact(sources)

    assert stale_artifact["criteria_results"]["sota_gguf_cache_provenance_preflight"] == "NOT_MET"
    assert stale_artifact["criteria_results"]["triggered_certificate_extraction_v2"] == "NOT_MET"
