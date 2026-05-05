"""Tests for the Exp 1322 milestone .102 retrospective.

Spec: REQ-REPORT-028, SCENARIO-REPORT-028.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_102 import (
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
        1309: {
            "status": "complete",
            "honest_verdict": (
                "resolver_repaired_two_cached_headline_models_ready_full_suite_not_green_unrelated_failures"
            ),
            "sota_pair_ready": True,
            "cached_pair_specs_count": 2,
            "cached_mandated_models": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "missing_optional_models": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "headline_result_possible": True,
            "focused_tests_passed": True,
        },
        1310: {
            "status": "complete",
            "honest_verdict": "sota_pair_llamacpp_smoke_loaded",
            "models_loaded": 2,
            "llama_cpp_import_ok": True,
            "tokens_per_second": 20.2226,
            "gpu_memory_gb": {"0": 0.2949, "1": 0.2949},
            "model_specs_count": 2,
            "models_used": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "headline_result_possible": True,
        },
        1311: {
            "status": "complete",
            "honest_verdict": "sota_constraint_satquest_stability_audit_complete",
            "answer_stability_score": 0.9,
            "cross_model_disagreement_rate": 0.8,
            "constraintbench_items": 5,
            "satquest_items": 5,
            "pysat_verified_rate": 0.525,
            "feasibility_rate": 0.5,
            "unknown_or_abstain_rate": 0.55,
            "headline_result_allowed": True,
            "models_used": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
        },
        1312: {
            "status": "complete",
            "honest_verdict": "triggered_certificate_dccd_gbnf_comparison_complete",
            "certificate_parse_rate": 0.71223,
            "certificate_truthfulness_rate": 0.69697,
            "dccd_delta_over_grammar_only": 0.2,
            "grammar_projection_tax_proxy": {"rows_measured": 40},
            "repair_success_rate": 1.0,
            "models_used": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "headline_result_allowed": True,
        },
        1313: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp1312-triggered-certificate-extraction-dccd-gbnf.certificate_parse_rate "
                "(actual=0.71223 >= expected=0.75)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp1312-triggered-certificate-extraction-dccd-gbnf",
                    "artifact_field": "certificate_parse_rate",
                    "op": ">=",
                    "expected": 0.75,
                    "actual": 0.71223,
                    "passed": False,
                }
            ],
        },
        1315: {
            "status": "complete",
            "honest_verdict": "cerce_nonforgetting_preserved_improved_non_headline",
            "nonforgetting_certificate_rate": 1.0,
            "memory_regression_count": 0,
            "self_learning_delta_overall": 1.596429,
            "lagrangian_violation_penalty": 0.0,
            "accepted_violation_delta": -0.846154,
            "promoted_memory_count": 5,
            "demoted_memory_count": 2,
            "headline_result_allowed": False,
        },
        1316: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 2 gate(s) failed; first failure: "
                "exp1312-triggered-certificate-extraction-dccd-gbnf.certificate_parse_rate "
                "(actual=0.71223 >= expected=0.75)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp1312-triggered-certificate-extraction-dccd-gbnf",
                    "artifact_field": "certificate_parse_rate",
                    "op": ">=",
                    "expected": 0.75,
                    "actual": 0.71223,
                    "passed": False,
                },
                {
                    "upstream": "exp1315-continuous-self-learning-cerce-nonforgetting-audit",
                    "artifact_field": "nonforgetting_certificate_rate",
                    "op": ">=",
                    "expected": 0.9,
                    "actual": 1.0,
                    "passed": True,
                },
            ],
        },
        1317: {
            "status": "complete",
            "honest_verdict": "grpo_vprm_v11_positive_headline_gate",
            "grpo_vprm_delta": 0.45,
            "verifier_feedback_token_mask_delta": 0.25,
            "nonforgetting_preserved": True,
            "self_verification_gain": 0.45,
            "headline_result_allowed": True,
        },
        1318: {
            "status": "complete",
            "honest_verdict": (
                "complete: learned policy matched the conservative replay policy on a "
                "deterministic held-out seed split, but this is still replay-distribution "
                "generalization and not a broad general stop rule"
            ),
            "learned_stop_policy_written": True,
            "generalization_split": {"held_out_count": 36, "train_count": 120},
            "stop_policy_precision": 1.0,
            "stop_policy_recall": 1.0,
            "hardnetpp_delta_over_replay_policy": 0.0,
            "dsp_feasibility_auc": 0.640625,
        },
        1319: {
            "status": "complete",
            "honest_verdict": "hardware_portability_audit_only_no_fpga_npu_or_analog_execution",
            "rm_per_inference": 24,
            "bop_per_inference": 192,
            "nabs_per_inference": 75,
            "lookup_table_bytes": 6144,
            "analog_kan_candidate": True,
            "npu_or_fpga_best_target": "FPGA",
            "hardware_claim_allowed": False,
            "hardware_execution": {
                "analog_execution": False,
                "cpu_artifact_generation_only": True,
                "fpga_execution": False,
                "npu_execution": False,
            },
        },
        1320: {
            "status": "complete",
            "honest_verdict": "cpu_portability_packet_ready_hardware_not_run",
            "dual_bram_mapping_ready": True,
            "reuse_factor_sweep": [{"reuse_factor": 4}],
            "dac_bits_sweep": [{"dac_bits": 6}],
            "kl_to_cpu_gibbs": 0.000412165565,
            "vivado_required_for_next_step": True,
            "hardware_claim_allowed": False,
        },
        1321: {
            "status": "complete",
            "honest_verdict": "operator_hold_active_related_work_delta_written_no_submission",
            "publication_state": "operator_hold",
            "operator_hold_active": True,
            "credentialed_submission_attempted": False,
            "related_work_delta_written": True,
            "new_references_count": 19,
        },
    }


def test_scenario_report_028_counts_milestone_102_source_criteria() -> None:
    """SCENARIO-REPORT-028: Exp1322 reports .102 11/14 from source fields."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "sota_gguf_pair_resolver_repair": "MET",
        "sota_gguf_llamacpp_smoke_load": "MET",
        "sota_constraintbench_satquest_answer_stability": "MET",
        "triggered_certificate_extraction_dccd_gbnf": "MET",
        "constrainprompt_nsvif_semantic_validator_mus_repair": "GATED",
        "beaver_lite_cactus_safe_prefix_acceptance": "GATED",
        "continuous_self_learning_cerce_nonforgetting_audit": "MET",
        "dvi_certificate_tail_online_update": "GATED",
        "grpo_vprm_v11_headline_gate": "MET",
        "hardnetpp_dsp_learned_stop_policy": "MET",
        "kan_hardware_complexity_audit": "MET",
        "pbit_sampler_portability_packet": "MET",
        "publication_hold_related_work_delta_v11": "MET",
        "retro_102_complete": "MET",
    }
    assert artifact["criteria_met"] == 11
    assert artifact["criteria_total"] == 14
    assert artifact["status"] == "complete"
    assert artifact["retro_complete"] is True
    assert artifact["sota_runtime_recovered"] is True
    assert artifact["certificate_path_headline_ready"] is False
    assert artifact["continuous_self_learning_advanced"] is True
    assert artifact["repair_generalization_advanced"] is True
    assert artifact["hardware_claims_honest"] is True
    assert artifact["publication_state"] == "operator_hold"
    assert artifact["honest_verdict"] == "milestone_102_11_of_14_criteria_met"

    carry_forwards = {item["task_id"]: item for item in artifact["carry_forward_tasks"]}
    assert {
        "certificate_path_headline_readiness",
        "dvi_certificate_tail_update",
        "repair_generalization_breadth",
        "publication_operator_hold",
    } <= set(carry_forwards)
    assert carry_forwards["certificate_path_headline_readiness"]["prior_failures"][0] == {
        "experiment_id": "exp1312-triggered-certificate-extraction-dccd-gbnf",
        "verdict": "triggered_certificate_dccd_gbnf_comparison_complete",
        "addressed_by": (
            "Raise certificate_parse_rate from 0.71223 to at least 0.75, then rerun "
            "semantic validators and safe-prefix acceptance from the parsed certificate corpus."
        ),
        "retire_if_same_verdict": False,
    }


def test_req_report_028_in_progress_artifact_is_durable(tmp_path: Path) -> None:
    """REQ-REPORT-028: a run can leave an auditable in-progress artifact."""

    out_path = tmp_path / "results" / "experiment_1322_milestone_retro_102.json"

    artifact = write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["run_date"] == "20260505"
    assert written["criteria_total"] == 14
    assert written["retro_complete"] is False


def test_req_report_028_run_loads_sources_and_writes_schema(tmp_path: Path) -> None:
    """REQ-REPORT-028: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1322_milestone_retro_102.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1322_milestone_retro_102"
    assert written["schema"] == "milestone_retro_v7"
    assert written["milestone"] == "2026.04.102"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 11
    assert written["criteria_total"] == 14
    assert any(item["experiment_id"] == 1314 and not item["loaded"] for item in written["source_artifacts_checked"])


def test_req_report_028_open_gates_allow_terminal_downstream_metrics() -> None:
    """REQ-REPORT-028: opened gates let downstream terminal metrics count."""

    sources = _scenario_sources()
    sources[1312]["certificate_parse_rate"] = 0.8
    sources[1313] = {
        "status": "complete",
        "honest_verdict": "semantic_validators_measured",
        "compiled_validator_count": 20,
        "validator_execution_pass_rate": 0.6,
        "semantic_violation_reduction": 0.2,
        "mus_repair_hint_count": 5,
        "residual_drift_cases": [],
        "unknown_or_abstain_rate": 0.1,
    }
    sources[1314] = {
        "status": "complete",
        "honest_verdict": "safe_prefix_acceptance_measured",
        "low_risk_acceptance_rate": 0.3,
        "false_acceptance_rate": 0.0,
        "safe_prefix_repair_delta": 0.1,
        "full_verifier_call_reduction": 0.2,
        "risk_bound_proxy": 0.05,
        "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
        "headline_result_allowed": True,
    }
    sources[1316] = {
        "status": "complete",
        "honest_verdict": "dvi_certificate_tail_updated",
        "drafter_acceptance_delta": 0.2,
        "accepted_violation_delta": -0.1,
        "online_update_count": 4,
        "nonforgetting_preserved": True,
        "lossless_acceptance_claim_allowed": True,
        "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
        "headline_result_allowed": True,
    }

    artifact = build_artifact(sources)

    assert artifact["criteria_results"]["constrainprompt_nsvif_semantic_validator_mus_repair"] == "MET"
    assert artifact["criteria_results"]["beaver_lite_cactus_safe_prefix_acceptance"] == "MET"
    assert artifact["criteria_results"]["dvi_certificate_tail_online_update"] == "MET"
    assert artifact["criteria_met"] == 14
    assert artifact["certificate_path_headline_ready"] is True
    assert artifact["honest_verdict"] == "milestone_102_14_of_14_criteria_met"


def test_req_report_028_failed_blocked_and_missing_branches_are_mechanical() -> None:
    """REQ-REPORT-028: failed metrics, operator blockers, and missing open gates differ."""

    sources = _scenario_sources()
    sources[1309]["sota_pair_ready"] = False
    sources[1310] = {
        "status": "blocked",
        "honest_verdict": "operator_blocked_runtime",
        "blocked_at_layer": "operator",
    }
    sources[1312]["certificate_parse_rate"] = 0.8
    sources[1313] = {
        "status": "blocked",
        "honest_verdict": "operator_blocked_validator",
        "blocked_at_layer": "operator",
    }
    sources[1312]["headline_result_allowed"] = False
    del sources[1317]
    del sources[1321]

    artifact = build_artifact(sources)

    assert artifact["criteria_results"]["sota_gguf_pair_resolver_repair"] == "FAILED"
    assert artifact["criteria_results"]["sota_gguf_llamacpp_smoke_load"] == "BLOCKED"
    assert artifact["criteria_results"]["constrainprompt_nsvif_semantic_validator_mus_repair"] == "BLOCKED"
    assert artifact["criteria_results"]["beaver_lite_cactus_safe_prefix_acceptance"] == "GATED"
    assert artifact["criteria_results"]["grpo_vprm_v11_headline_gate"] == "GATED"
    assert artifact["criteria_results"]["publication_hold_related_work_delta_v11"] == "MISSING"
    assert artifact["sota_runtime_recovered"] is False
    assert artifact["publication_state"] == "missing"

    missing_open_gate_sources = _scenario_sources()
    missing_open_gate_sources[1312]["certificate_parse_rate"] = 0.8
    missing_open_gate_sources[1313] = {
        "status": "complete",
        "honest_verdict": "semantic_validators_measured",
        "compiled_validator_count": 20,
        "validator_execution_pass_rate": 0.6,
        "semantic_violation_reduction": 0.2,
        "mus_repair_hint_count": 5,
        "residual_drift_cases": [],
        "unknown_or_abstain_rate": 0.1,
    }
    missing_open_gate_artifact = build_artifact(missing_open_gate_sources)
    assert (
        missing_open_gate_artifact["criteria_results"]["beaver_lite_cactus_safe_prefix_acceptance"]
        == "MISSING"
    )
