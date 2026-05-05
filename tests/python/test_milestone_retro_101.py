"""Tests for the Exp 1308 milestone .101 retrospective.

Spec: REQ-REPORT-027, SCENARIO-REPORT-027.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_101 import (
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
        1296: {
            "status": "complete",
            "honest_verdict": "activation_audit_passed",
            "prior_failures_coverage_ok": True,
            "roadmap_gate_audit_passed": True,
            "n_prior_failures_missing": 0,
            "activation_blockers": [],
            "exp1283_grammar_backend_available": True,
            "exp1288_memory_update_written": True,
        },
        1297: {
            "status": "complete",
            "honest_verdict": "sota_gguf_cache_not_ready",
            "cached_sota_ready": False,
            "provenance_ok": True,
            "missing_models": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "cached_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "headline_result_possible": False,
        },
        1298: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "exp1297-sota-gguf-cache-provenance-preflight-v2.cached_sota_ready "
                "(actual=False == expected=True)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp1297-sota-gguf-cache-provenance-preflight-v2",
                    "artifact_field": "cached_sota_ready",
                    "actual": False,
                    "expected": True,
                    "passed": False,
                }
            ],
        },
        1300: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "upstream artifact not found for task id "
                "'exp1299-triggered-certificate-extraction-v3'"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp1299-triggered-certificate-extraction-v3",
                    "artifact_field": "certificate_parse_rate",
                    "actual": None,
                    "expected": 0.8,
                    "passed": False,
                }
            ],
        },
        1302: {
            "status": "complete",
            "honest_verdict": "skill_graph_candidates_written_sandboxed",
            "memory_update_written": True,
            "skill_graph_candidate_count": 7,
            "promoted_memory_count": 5,
            "demoted_memory_count": 1,
            "expired_memory_count": 1,
            "replay_evidence_count": 140,
        },
        1303: {
            "status": "complete",
            "honest_verdict": "online_memory_policy_improved_non_headline",
            "bandit_regret": 13.3,
            "accepted_violation_delta": -0.807143,
            "self_learning_delta_overall": 1.596429,
            "memory_demotion_count": 18,
            "headline_result_allowed": False,
        },
        1305: {
            "status": "complete",
            "honest_verdict": (
                "complete: conservative replay policy is useful as an operator gate, "
                "but DSP feasibility is still marginal and this is not a learned general stop rule"
            ),
            "feasibility_stop_policy_written": True,
            "stop_policy_precision": 1.0,
            "feasibility_channel_auc": 0.6604651162790698,
            "hardnetpp_delta_over_snarenet": 1.2207222442957435,
        },
        1306: {
            "status": "complete",
            "honest_verdict": "energy_bridge_completed_local_alignment_only_strategic_context_not_implemented",
            "energy_bridge_completed": True,
            "ebt_citation_count_checked": {"citation_count": 232},
            "arm_ebm_alignment_note": "local alignment only",
            "ebm_cot_sequence_energy_note": "sequence-energy context",
            "extropic_kona_status_checked": "future sampler context",
            "hardware_sampler_context_recorded": "p-bit and Kona context",
        },
        1307: {
            "status": "complete",
            "honest_verdict": "operator_hold_active_no_local_arxiv_receipt",
            "arxiv_receipt_present": False,
            "credentialed_submission_attempted": False,
            "operator_hold_active": True,
            "publication_state": "operator_hold",
            "blocker": "operator_publication_hold_active_no_local_receipt",
        },
    }


def test_scenario_report_027_counts_milestone_101_source_criteria() -> None:
    """SCENARIO-REPORT-027: Exp1308 reports .101 8/13 from source fields."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "prior_failures_activation_audit": "MET",
        "sota_gguf_cache_provenance_preflight_v2": "MET",
        "sota_answer_stability_falcon_audit": "GATED",
        "triggered_certificate_extraction_v3": "GATED",
        "semantic_routing_v2": "GATED",
        "safe_prefix_cactus_acceptance_v3": "GATED",
        "skill_graph_promotion_demotion_v2": "MET",
        "querybandits_ngc_online_memory_policy": "MET",
        "grpo_vprm_v10_sota_gated": "GATED",
        "hardnetpp_dsp_feasibility_stop_policy": "MET",
        "ebt_arm_ebm_cot_energy_bridge_audit_v2": "MET",
        "arxiv_v10_hold_receipt_v2": "MET",
        "retro_101_complete": "MET",
    }
    assert artifact["criteria_met"] == 8
    assert artifact["criteria_total"] == 13
    assert artifact["status"] == "complete"
    assert artifact["retro_complete"] is True
    assert artifact["docs_reconciled"] is False
    assert artifact["honest_verdict"] == "milestone_101_8_of_13_criteria_met"
    assert artifact["activation_failures"] == []
    assert {item["experiment_id"] for item in artifact["gated_or_skipped_tasks"]} == {
        1298,
        1299,
        1300,
        1301,
        1304,
    }
    assert artifact["scientific_negative_results"] == [
        {
            "experiment_id": 1305,
            "criterion": "hardnetpp_dsp_feasibility_stop_policy",
            "status": "MET",
            "honest_verdict": _scenario_sources()[1305]["honest_verdict"],
            "interpretation": "terminal repair-policy artifact; DSP remains marginal and the rule is not a learned general stop policy",
        }
    ]


def test_req_report_027_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-REPORT-027: a run can leave an auditable in-progress artifact."""

    out_path = tmp_path / "results" / "experiment_1308_milestone_retro_101.json"

    artifact = write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260505"
    assert written["criteria_total"] == 13
    assert written["retro_complete"] is False


def test_req_report_027_run_loads_sources_and_writes_schema(tmp_path: Path) -> None:
    """REQ-REPORT-027: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1308_milestone_retro_101.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1308_milestone_retro_101"
    assert written["schema"] == "milestone_retro_v6"
    assert written["milestone"] == "2026.04.101"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 8
    assert written["criteria_total"] == 13


def test_req_report_027_carry_forward_prior_failures_are_exact() -> None:
    """REQ-REPORT-027: carry-forwards include planner-ready prior_failures."""

    artifact = build_artifact(_scenario_sources())

    carry_forwards = {item["task_id"]: item for item in artifact["carry_forward_tasks"]}
    assert carry_forwards["sota_gguf_cache_readiness"]["prior_failures"] == [
        {
            "experiment_id": "exp1297-sota-gguf-cache-provenance-preflight-v2",
            "verdict": "sota_gguf_cache_not_ready",
            "addressed_by": (
                "Provision or deliberately replace the missing "
                "unsloth/gemma-4-26B-A4B-it-GGUF cache entry before rerunning SOTA certificate work."
            ),
            "retire_if_same_verdict": False,
        }
    ]
    assert carry_forwards["triggered_certificate_path"]["prior_failures"] == [
        {
            "experiment_id": "exp1298-sota-answer-stability-falcon-audit",
            "verdict": "blocked_gate_check_failed",
            "addressed_by": "Rerun only after exp1297.cached_sota_ready == true and record answer_stability_score.",
            "retire_if_same_verdict": False,
        },
        {
            "experiment_id": "exp1299-triggered-certificate-extraction-v3",
            "verdict": "missing_gated_by_sota_cache_and_answer_stability",
            "addressed_by": (
                "Produce certificate_parse_rate, headline_result_allowed, grammar cost, "
                "truthfulness, and FALCON repair metrics after SOTA readiness gates open."
            ),
            "retire_if_same_verdict": False,
        },
    ]
    assert carry_forwards["publication_hold"]["prior_failures"] == [
        {
            "experiment_id": "exp1307-arxiv-v10-hold-receipt-v2",
            "verdict": "operator_hold_active_no_local_arxiv_receipt",
            "addressed_by": (
                "Keep publication tasks terminal by recording the operator hold or a local receipt; "
                "do not attempt credentialed submission without operator approval."
            ),
            "retire_if_same_verdict": False,
        }
    ]


def test_req_report_027_failed_and_blocked_branches_are_mechanical() -> None:
    """REQ-REPORT-027: failed science, blocked activation, and gated work differ."""

    sources = _scenario_sources()
    sources[1296] = {
        "status": "complete",
        "honest_verdict": "activation_audit_failed",
        "prior_failures_coverage_ok": False,
        "roadmap_gate_audit_passed": False,
        "n_prior_failures_missing": 2,
        "activation_blockers": ["exp9999-missing-priors"],
    }
    sources[1298] = {
        "status": "complete",
        "honest_verdict": "sota_answer_stability_failed",
        "answer_stability_score": 0.5,
    }
    sources[1302] = {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "operator",
        "gate_check_summary": "manual operator blocker",
    }

    artifact = build_artifact(sources)

    assert artifact["criteria_results"]["prior_failures_activation_audit"] == "FAILED"
    assert artifact["criteria_results"]["sota_answer_stability_falcon_audit"] == "FAILED"
    assert artifact["criteria_results"]["skill_graph_promotion_demotion_v2"] == "BLOCKED"
    assert artifact["activation_failures"] == [
        {
            "experiment_id": 1296,
            "criterion": "prior_failures_activation_audit",
            "honest_verdict": "activation_audit_failed",
            "details": ["exp9999-missing-priors"],
        }
    ]


def test_req_report_027_open_gates_cover_terminal_metric_branches() -> None:
    """REQ-REPORT-027: opened gates let downstream terminal metrics count."""

    sources = _scenario_sources()
    sources[1297]["cached_sota_ready"] = True
    sources[1297]["missing_models"] = []
    sources[1298] = {
        "status": "complete",
        "honest_verdict": "answer_stability_measured",
        "answer_stability_score": 0.7,
    }
    sources[1299] = {
        "status": "complete",
        "honest_verdict": "certificate_metrics_measured",
        "certificate_parse_rate": 0.8,
        "headline_result_allowed": True,
    }
    sources[1300] = {
        "status": "complete",
        "honest_verdict": "semantic_routing_measured",
        "semantic_routing_coverage": 0.6,
    }
    sources[1301] = {
        "status": "complete",
        "honest_verdict": "cactus_acceptance_measured",
        "cactus_acceptance_rate": 0.35,
    }
    sources[1304] = {
        "status": "complete",
        "honest_verdict": "grpo_vprm_delta_measured",
        "grpo_vprm_delta": 0.01,
    }

    artifact = build_artifact(sources)

    assert artifact["criteria_results"]["triggered_certificate_extraction_v3"] == "MET"
    assert artifact["criteria_results"]["semantic_routing_v2"] == "MET"
    assert artifact["criteria_results"]["safe_prefix_cactus_acceptance_v3"] == "MET"
    assert artifact["criteria_results"]["grpo_vprm_v10_sota_gated"] == "MET"

    del sources[1302]
    missing_artifact = build_artifact(sources)
    assert missing_artifact["criteria_results"]["skill_graph_promotion_demotion_v2"] == "MISSING"

    del sources[1299]
    missing_certificate_artifact = build_artifact(sources)
    assert (
        missing_certificate_artifact["criteria_results"]["triggered_certificate_extraction_v3"]
        == "MISSING"
    )

    sources[1299] = {
        "status": "complete",
        "honest_verdict": "certificate_metrics_measured",
        "certificate_parse_rate": 0.8,
        "headline_result_allowed": True,
    }
    del sources[1301]
    missing_cactus_artifact = build_artifact(sources)
    assert (
        missing_cactus_artifact["criteria_results"]["safe_prefix_cactus_acceptance_v3"] == "MISSING"
    )

    del sources[1304]
    missing_grpo_artifact = build_artifact(sources)
    assert missing_grpo_artifact["criteria_results"]["grpo_vprm_v10_sota_gated"] == "MISSING"
