"""Tests for Exp 3315 verifier-guided backtracking repair policy v1.

Spec refs: REQ-VERIFY-3315, SCENARIO-VERIFY-3315.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import vgb_backtracking_repair_policy_v1 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
DENSE_ID = "unsloth/gemma-4-31B-it-GGUF"


REQUIRED_FIELDS = {
    "vgb_repair_policy_ready",
    "backtracking_policy",
    "proposal_budget",
    "exact_acceptance_rules",
    "verifier_confidence_thresholds",
    "no_new_model_execution",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(30):
        clean_reject = index in {7, 11, 23}
        rows.append(
            {
                "case_id": f"case-{index:02d}",
                "case_hash": f"{index + 1:064x}",
                "exact_check_passed": True,
                "exact_checker_type": "exact_integer_string",
                "calibrated_clean_verifier_decision": "reject" if clean_reject else "accept",
                "uncertainty_score": 0.7 if clean_reject else 0.1,
                "abstention": {
                    "policy_blocked": clean_reject,
                    "reason_codes": ["exact_clean_disagreement"] if clean_reject else [],
                },
                "false_accept": False,
                "model_identity": {"model_id": MODEL_ID, "model_family": "gemma"},
            }
        )
    return rows


def _stage_sources(root: Path, *, clean: bool = False) -> None:
    _write_json(
        root,
        mod.EXP3313_REL_PATH,
        {
            "artifact": "experiment_3313_repair_substrate_root_cause_autopsy_v1",
            "experiment_id": "exp3313",
            "repair_substrate_autopsy_ready": True,
            "rerun_contract": {
                "exp3315": {
                    "deliverable": mod.OUTPUT_REL_PATH.as_posix(),
                    "acceptance_requirements": [
                        "vgb_repair_policy_ready=true",
                        "exact_verifiers_not_llm_judges_are_final",
                    ],
                },
                "exp3316": {
                    "deliverable": "results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json",
                    "acceptance_requirements": [
                        "exact_acceptance_authority_no_llm_judge",
                        "headline_claim_allowed_true_only_if_all_gates_pass_else_honestly_blocked",
                    ],
                },
            },
            "no_new_model_execution": True,
            "honest_verdict": "complete: fixture",
        },
    )
    _write_json(
        root,
        mod.EXP3314_REL_PATH,
        {
            "artifact": "experiment_3314_distributional_ebm_repair_uncertainty_audit_v1",
            "experiment_id": "exp3314",
            "distributional_repair_audit_ready": True,
            "repair_case_count": 30,
            "repair_row_scores": _rows(),
            "uncertainty_abstention_policy": {
                "policy_name": "exp3316_headline_promotion_abstention_policy_v1",
                "uncertainty_score_block_threshold": 0.6,
                "provenance_risk_score_block_threshold": 0.5,
                "model_identity_coverage_risk_block_threshold": 0.5,
                "high_uncertainty_case_count": 0 if clean else 3,
                "high_uncertainty_case_ids": [] if clean else ["case-07", "case-11", "case-23"],
                "row_abstention_count": 0 if clean else 30,
                "provenance_risk_blocks_headline": not clean,
                "model_identity_risk_blocks_headline": not clean,
                "headline_promotion_blocked": not clean,
                "exact_acceptance_remains_final_authority": True,
            },
            "distributional_energy_schema": {
                "row_components": {
                    "deterministic_constraint_penalty": {"authority": "final_exact_checker"},
                    "abstention": {"authority": "advisory_policy_only"},
                }
            },
            "model_identity_confound_check": {
                "confound_detected": not clean,
                "model_identity_coverage_risk": 0.0 if clean else 0.666667,
                "used_model_ids": [MODEL_ID, QWEN_ID, DENSE_ID] if clean else [MODEL_ID],
                "missing_mandated_model_ids": [] if clean else [QWEN_ID, DENSE_ID],
            },
            "provenance_risk_features": {
                "provenance_risk_score": 0.0 if clean else 1.0,
                "critical_adversarial_flag_count": 0 if clean else 1,
                "critical_adversarial_flags": []
                if clean
                else [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}],
                "runtime_contract_ready": True,
            },
            "exact_acceptance_authority_preserved": True,
            "no_new_model_execution": True,
            "honest_verdict": "complete: fixture",
        },
    )


def test_req_verify_3315_spec_anchor_declares_vgb_policy() -> None:
    """REQ-VERIFY-3315: OpenSpec declares the policy before implementation."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3315" in spec
    assert "SCENARIO-VERIFY-3315" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3313_REL_PATH.as_posix() in spec
    assert mod.EXP3314_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3315_writes_policy_with_abstention_handoff(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3315: policy is ready but blocks dirty Exp3316 promotion."""

    _stage_sources(tmp_path)

    output = mod.write_artifact(tmp_path, output_path=Path("results/out.json"), started_s=10.0, now_s=12.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["vgb_repair_policy_ready"] is True
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_repair_generation"] is True
    assert artifact["no_new_verifier_run"] is True
    assert artifact["inference_substrate"] == "deterministic_policy_artifact_no_model_calls"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    budget = artifact["proposal_budget"]
    assert budget["max_attempts_per_case"] == 4
    assert budget["max_backtracks_per_case"] == 3
    assert budget["max_total_attempts"] == 120
    assert "stop_on_exact_acceptance" in budget["stop_conditions"]
    assert "stop_on_advisory_gate_closed_before_generation" in budget["stop_conditions"]

    thresholds = artifact["verifier_confidence_thresholds"]
    assert thresholds["process_accept_confidence_min"] == pytest.approx(0.8)
    assert thresholds["row_uncertainty_abstain_threshold"] == pytest.approx(0.6)
    assert thresholds["provenance_risk_abstain_threshold"] == pytest.approx(0.5)
    assert thresholds["model_identity_coverage_risk_abstain_threshold"] == pytest.approx(0.5)

    exact = artifact["exact_acceptance_rules"]
    assert exact["final_acceptance_authority"] == "exact_verifier_only"
    assert exact["llm_judge_final_acceptance_allowed"] is False
    assert "exact_check_passed=true" in exact["required_acceptance_conditions"]
    assert "exact_checker_type_present" in exact["required_acceptance_conditions"]

    policy = artifact["backtracking_policy"]
    assert policy["actions"]["accepted"]["requires_exact_acceptance"] is True
    assert "clean_process_verifier_rejects_exact_success" in policy["actions"]["backtracked"]["trigger_conditions"]
    assert "source_provenance_or_uncertainty_policy_blocks" in policy["actions"]["abstained"]["trigger_conditions"]
    assert "candidate_attempt_logging" in policy
    for field in mod.REQUIRED_ATTEMPT_LOG_FIELDS:
        assert field in policy["candidate_attempt_logging"]["required_fields"]

    handoff = artifact["exp3316_handoff"]
    assert handoff["headline_promotion_blocked_until_policy_clears"] is True
    assert "vgb_repair_policy_ready" in handoff["required_artifact_fields"]
    assert "candidate_attempts" in handoff["required_artifact_fields"]
    mod.validate_artifact(artifact)


def test_req_verify_3315_routes_candidate_attempts_deterministically() -> None:
    """REQ-VERIFY-3315: exact outcomes decide accept/reject; process signals route backtracking."""

    thresholds = mod.default_verifier_confidence_thresholds({})

    accepted = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "false_accept": False,
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.93,
            "uncertainty_score": 0.1,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    exact_failed = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": False,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.99,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    clean_reject = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "reject",
            "process_verifier_confidence": 0.91,
            "uncertainty_score": 0.7,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    clean_abstain = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "abstain",
            "process_verifier_confidence": 0.5,
            "uncertainty_score": 0.2,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    unknown_decision = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "maybe",
            "process_verifier_confidence": 0.91,
            "uncertainty_score": 0.2,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    exhausted = mod.route_candidate_attempt(
        {
            "attempt_index": 4,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "reject",
            "process_verifier_confidence": 0.91,
            "uncertainty_score": 0.7,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )
    risk_block = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.95,
            "uncertainty_score": 0.1,
            "advisory_policy_blocked": True,
        },
        thresholds=thresholds,
        max_attempts_per_case=4,
    )

    assert accepted["policy_action"] == "accepted"
    assert accepted["exact_acceptance_authority"] is True
    assert exact_failed["policy_action"] == "rejected"
    assert exact_failed["backtrack_next"] is True
    assert "deterministic_exact_check_failed" in exact_failed["reason_codes"]
    assert clean_reject["policy_action"] == "backtracked"
    assert clean_reject["backtrack_next"] is True
    assert "clean_process_verifier_rejects_exact_success" in clean_reject["reason_codes"]
    assert clean_abstain["policy_action"] == "backtracked"
    assert "clean_process_verifier_abstains_on_exact_success" in clean_abstain["reason_codes"]
    assert unknown_decision["policy_action"] == "backtracked"
    assert "unknown_clean_process_verifier_decision" in unknown_decision["reason_codes"]
    assert exhausted["policy_action"] == "abstained"
    assert exhausted["backtrack_next"] is False
    assert "proposal_budget_exhausted" in exhausted["reason_codes"]
    assert risk_block["policy_action"] == "abstained"
    assert "advisory_risk_gate_closed" in risk_block["reason_codes"]
    assert mod.inferred_process_confidence("ABSTAIN") == 0.5
    assert mod.inferred_process_confidence("unknown") == 0.0


def test_req_verify_3315_clean_sources_allow_policy_without_handoff_block(tmp_path: Path) -> None:
    """REQ-VERIFY-3315: clean upstream policy leaves Exp3316 unblocked by Exp3315."""

    _stage_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)

    assert artifact["vgb_repair_policy_ready"] is True
    assert artifact["exp3316_handoff"]["headline_promotion_blocked_until_policy_clears"] is False
    assert artifact["source_policy_summary"]["headline_promotion_blocked_by_exp3314"] is False
    assert artifact["proposal_budget"]["max_total_attempts"] == 120
    assert mod.duration(4.0, 3.0) == 0.0
    assert mod.rate(2, 4) == 0.5
    assert mod.rate(1, 0) == 0.0


def test_req_verify_3315_fail_closed_for_missing_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-3315: missing evidence cannot produce a ready policy."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["vgb_repair_policy_ready"] is False
    assert artifact["no_new_model_execution"] is True
    assert artifact["source_policy_summary"]["exp3314_ready"] is False
    assert artifact["exp3316_handoff"]["headline_promotion_blocked_until_policy_clears"] is True
    assert artifact["proposal_budget"]["max_total_attempts"] == 0
    assert artifact["sample_candidate_routing"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    assert mod.file_status(tmp_path / "missing.json")["readable"] is False
    assert mod.sha256_file(tmp_path / "missing.bin") is None
    assert mod.mapping_list("not rows") == []
    assert mod.string_list([1, "x", None]) == ["1", "x", "None"]
    assert mod.numeric(True) == 0.0
    assert mod.count_value(True) == 0
    assert mod.count_value("7") == 7
    assert mod.count_value("bad") == 0

    missing_exact_type = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "",
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.99,
        },
        thresholds=mod.default_verifier_confidence_thresholds({}),
        max_attempts_per_case=4,
    )
    false_accept = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "false_accept": True,
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.99,
        },
        thresholds=mod.default_verifier_confidence_thresholds({}),
        max_attempts_per_case=4,
    )
    low_confidence = mod.route_candidate_attempt(
        {
            "attempt_index": 1,
            "exact_check_passed": True,
            "exact_checker_type": "exact_integer_string",
            "calibrated_clean_verifier_decision": "accept",
            "process_verifier_confidence": 0.2,
        },
        thresholds=mod.default_verifier_confidence_thresholds({}),
        max_attempts_per_case=4,
    )
    assert missing_exact_type["policy_action"] == "abstained"
    assert "missing_exact_checker_type" in missing_exact_type["reason_codes"]
    assert false_accept["policy_action"] == "rejected"
    assert "false_accept_recorded" in false_accept["reason_codes"]
    assert low_confidence["policy_action"] == "backtracked"
    assert "process_verifier_confidence_below_accept_threshold" in low_confidence["reason_codes"]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})

    invalid = json.loads(json.dumps(artifact))
    invalid["vgb_repair_policy_ready"] = "yes"
    with pytest.raises(ValueError, match="vgb_repair_policy_ready must be a bool"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["backtracking_policy"] = {}
    with pytest.raises(ValueError, match="backtracking_policy must be non-empty"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["proposal_budget"] = {}
    with pytest.raises(ValueError, match="proposal_budget must be non-empty"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["exact_acceptance_rules"]["final_acceptance_authority"] = "llm_judge"
    with pytest.raises(ValueError, match="exact verifier must be final authority"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["exact_acceptance_rules"]["llm_judge_final_acceptance_allowed"] = True
    with pytest.raises(ValueError, match="LLM judges cannot be final acceptance authority"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["verifier_confidence_thresholds"] = {}
    with pytest.raises(ValueError, match="verifier_confidence_thresholds must be non-empty"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["no_new_model_execution"] = False
    with pytest.raises(ValueError, match="no_new_model_execution must be true"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["backtracking_policy"]["actions"].pop("accepted")
    with pytest.raises(ValueError, match="backtracking_policy must define all four actions"):
        mod.validate_artifact(invalid)

    invalid = json.loads(json.dumps(artifact))
    invalid["honest_verdict"] = "blocked"
    with pytest.raises(ValueError, match="honest_verdict must start"):
        mod.validate_artifact(invalid)
