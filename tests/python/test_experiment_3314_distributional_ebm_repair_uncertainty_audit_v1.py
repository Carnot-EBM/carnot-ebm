"""Tests for Exp 3314 distributional EBM repair uncertainty audit v1.

Spec refs: REQ-VERIFY-3314, SCENARIO-VERIFY-3314.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import distributional_ebm_repair_uncertainty_audit_v1 as mod


MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
DENSE_ID = "unsloth/gemma-4-31B-it-GGUF"


REQUIRED_FIELDS = {
    "distributional_repair_audit_ready",
    "uncertainty_abstention_policy",
    "distributional_energy_schema",
    "model_identity_confound_check",
    "provenance_risk_features",
    "repair_case_count",
    "no_new_model_execution",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_rows(*, clean: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reject_ids = set() if clean else {7, 11, 23}
    for index in range(30):
        rejected_exact_success = index in reject_ids
        family = "arithmetic_exact_rows" if index < 12 else "code_output_checks" if index < 18 else "symbolic_aliases"
        rows.append(
            {
                "case_id": f"case-{index:02d}",
                "case_hash": f"{index + 1:064x}",
                "family": family,
                "exact_check_passed": True,
                "exact_checker_type": "exact_integer_string",
                "verified_success": not rejected_exact_success,
                "false_accept": False,
                "abstained": False,
                "calibrated_clean_verifier_decision": "reject" if rejected_exact_success else "accept",
                "calibrated_clean_verifier_output": "REJECT" if rejected_exact_success else "ACCEPT",
                "failure_class": "clean_verifier_rejected_exact_success"
                if rejected_exact_success
                else "unknown_verifier_decision",
                "candidate_answer": str(index + 3),
                "expected_answer": str(index + 3),
                "failing_candidate": str(index),
                "localized_repair_feedback": f"Replace {index} with {index + 3}.",
                "model_id": MODEL_ID,
                "model_path": "/cache/gemma-4-26b.gguf",
                "token_counts": {
                    "prompt_tokens": 230 + index,
                    "completion_tokens": 2 + (index % 3),
                    "total_tokens": 232 + index,
                },
            }
        )
    return rows


def _panel_payload(*, clean: bool = False, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    candidate_rows = rows if rows is not None else _candidate_rows(clean=clean)
    used_models = [
        {
            "model_id": MODEL_ID,
            "hf_id": MODEL_ID,
            "name": "Gemma4-26B-A4B-it",
            "role": "moe",
            "model_path": "/cache/gemma-4-26b.gguf",
            "size_bytes": 16_947_539_744,
            "quantization": "Q4_K_M",
            "legacy_small_model": False,
        }
    ]
    missing = [] if clean else [
        {"model_id": QWEN_ID, "hf_id": QWEN_ID, "role": "moe", "reason": "not_cached"},
        {"model_id": DENSE_ID, "hf_id": DENSE_ID, "role": "dense", "reason": "not_cached"},
    ]
    if clean:
        used_models.extend(
            [
                {"model_id": QWEN_ID, "hf_id": QWEN_ID, "role": "moe", "model_path": "/cache/qwen.gguf"},
                {"model_id": DENSE_ID, "hf_id": DENSE_ID, "role": "dense", "model_path": "/cache/gemma31.gguf"},
            ]
        )
    return {
        "artifact": "experiment_3302_headline_sota_repair_panel_v11",
        "experiment_id": "exp3302",
        "headline_repair_panel_ready": True,
        "repair_panel_ran": True,
        "headline_claim_allowed": clean,
        "panel_case_count": len(candidate_rows),
        "verified_success_count": sum(row["verified_success"] is True for row in candidate_rows),
        "false_accept_count": 0,
        "abstention_count": 0,
        "repair_success_rate": 1.0 if clean else 0.9,
        "repair_success_ci95": [0.85, 1.0] if clean else [0.74, 0.97],
        "false_accept_rate_ci95": [0.0, 0.11],
        "candidate_results": candidate_rows,
        "manifest_case_hashes": [str(row["case_hash"]) for row in candidate_rows],
        "manifest_case_hashes_match": True,
        "provenance_clean": clean,
        "flagged_adversarial": not clean,
        "corrigendum_pending": [] if clean else [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=15.4 is below the live GGUF floor.",
            }
        ],
        "models_used": used_models,
        "missing_model_specs": missing,
        "model_specs": {
            "mandated_model_ids": [QWEN_ID, DENSE_ID, MODEL_ID],
            "mandated_models": {
                MODEL_ID: {"model_id": MODEL_ID, "role": "moe", "cached": True},
                QWEN_ID: {"model_id": QWEN_ID, "role": "moe", "cached": clean},
                DENSE_ID: {"model_id": DENSE_ID, "role": "dense", "cached": clean},
            },
        },
        "duration_s": 90.0 if clean else 15.496424,
        "inference_substrate": "live_local_sota_gguf_repair_plus_calibrated_clean_verifier",
        "random_seed": 3302,
        "reproducibility_checksum": "b" * 64,
        "honest_verdict": "complete: fixture",
    }


def _audit_payload(*, clean: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3303_repair_headline_evidence_audit_v1",
        "experiment_id": "exp3303",
        "repair_headline_evidence_audit_ready": True,
        "headline_claim_allowed_after_audit": clean,
        "source_provenance_clean": clean,
        "substrate_consistency_passed": clean,
        "no_new_model_execution": True,
        "adversarial_verify_flags": [] if clean else [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration below floor",
            }
        ],
        "model_invocation_summary": {
            "actual_model_declarations_present": True,
            "legacy_small_model_used": False,
            "used_model_count": 3 if clean else 1,
            "used_model_ids": [MODEL_ID, QWEN_ID, DENSE_ID] if clean else [MODEL_ID],
            "missing_model_ids": [] if clean else [QWEN_ID, DENSE_ID],
            "mandated_model_ids": [QWEN_ID, MODEL_ID, DENSE_ID],
        },
        "exact_check_provenance": {
            "all_claimed_successes_exact_checked": True,
            "candidate_result_count": 30,
            "llm_judge_dependency_count": 0,
        },
        "duration_s": 0.01,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "random_seed": 3303,
        "reproducibility_checksum": "c" * 64,
        "honest_verdict": "complete: fixture",
    }


def _autopsy_payload(*, clean: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3313_repair_substrate_root_cause_autopsy_v1",
        "experiment_id": "exp3313",
        "repair_substrate_autopsy_ready": True,
        "runtime_contract_reference": {
            "experiment_id": "exp3309",
            "runtime_contract_ready": True,
            "minimum_live_duration_s": 60.0,
            "repair_substrate_rules_present": True,
        },
        "candidate_outcome_summary": {
            "candidate_result_count": 30,
            "exact_check_passed_count": 30,
            "verified_success_count": 30 if clean else 27,
            "false_accept_count": 0,
            "clean_verifier_rejected_exact_success_count": 0 if clean else 3,
            "clean_verifier_rejected_exact_success_case_ids": [] if clean else ["case-07", "case-11", "case-23"],
        },
        "source_provenance_failure_modes": [] if clean else [
            {"id": "source_panel_provenance_dirty", "classification": "evidence_hygiene_failure"},
            {"id": "incomplete_mandated_model_coverage", "classification": "model_identity_hygiene_failure"},
        ],
        "substrate_consistency_failure_modes": [] if clean else [
            {"id": "source_duration_below_live_floor", "is_blocker": True, "minimum_live_duration_s": 60.0},
            {"id": "critical_adversarial_flag_present", "is_blocker": True},
        ],
        "rerun_contract": {
            "exp3314": {"acceptance_requirements": ["distributional_repair_audit_ready=true"]},
            "exp3316": {"acceptance_requirements": ["headline_claim_allowed_true_only_if_all_gates_pass_else_honestly_blocked"]},
        },
        "no_new_model_execution": True,
        "duration_s": 0.01,
        "random_seed": 3313,
        "reproducibility_checksum": "d" * 64,
        "honest_verdict": "complete: fixture",
    }


def _stage_sources(root: Path, *, clean: bool = False, rows: list[dict[str, Any]] | None = None) -> None:
    _write_json(root, mod.EXP3302_REL_PATH, _panel_payload(clean=clean, rows=rows))
    _write_json(root, mod.EXP3303_REL_PATH, _audit_payload(clean=clean))
    _write_json(root, mod.EXP3313_REL_PATH, _autopsy_payload(clean=clean))


def test_req_verify_3314_spec_anchor_declares_distributional_audit() -> None:
    """REQ-VERIFY-3314: OpenSpec declares the audit before implementation."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3314" in spec
    assert "SCENARIO-VERIFY-3314" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.EXP3302_REL_PATH.as_posix() in spec
    assert mod.EXP3303_REL_PATH.as_posix() in spec
    assert mod.EXP3313_REL_PATH.as_posix() in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3314_writes_uncertainty_sidecar(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3314: dirty provenance and row disagreement block promotion."""

    _stage_sources(tmp_path)

    output = mod.write_artifact(tmp_path, output_path=Path("results/out.json"), started_s=10.0, now_s=12.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["distributional_repair_audit_ready"] is True
    assert artifact["repair_case_count"] == 30
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_repair_generation"] is True
    assert artifact["no_new_verifier_run"] is True
    assert artifact["exact_acceptance_authority_preserved"] is True
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")

    schema = artifact["distributional_energy_schema"]["row_components"]
    assert set(schema) == {
        "deterministic_constraint_penalty",
        "learned_proxy_quality",
        "provenance_risk",
        "model_identity",
        "uncertainty",
        "abstention",
    }
    assert schema["deterministic_constraint_penalty"]["authority"] == "final_exact_checker"
    assert schema["abstention"]["authority"] == "advisory_policy_only"

    provenance = artifact["provenance_risk_features"]
    assert provenance["source_provenance_clean"] is False
    assert provenance["critical_adversarial_flag_count"] == 1
    assert provenance["runtime_contract_ready"] is True
    assert provenance["source_duration_below_live_floor"] is True
    assert provenance["provenance_risk_score"] >= artifact["uncertainty_abstention_policy"][
        "provenance_risk_score_block_threshold"
    ]

    confounds = artifact["model_identity_confound_check"]
    assert confounds["confound_detected"] is True
    assert confounds["used_model_ids"] == [MODEL_ID]
    assert confounds["missing_mandated_model_ids"] == [QWEN_ID, DENSE_ID]
    assert confounds["used_model_families"] == ["gemma"]
    assert confounds["missing_model_families"] == ["gemma", "qwen"]
    assert confounds["model_identity_coverage_risk"] >= artifact["uncertainty_abstention_policy"][
        "model_identity_coverage_risk_block_threshold"
    ]

    policy = artifact["uncertainty_abstention_policy"]
    assert policy["headline_promotion_blocked"] is True
    assert policy["provenance_risk_blocks_headline"] is True
    assert policy["model_identity_risk_blocks_headline"] is True
    assert policy["high_uncertainty_case_count"] == 3
    assert policy["row_abstention_count"] == 30
    assert policy["exact_acceptance_remains_final_authority"] is True
    assert "uncertainty_abstention_policy" in policy["exp3316_required_fields"]

    scores = artifact["repair_row_scores"]
    assert len(scores) == 30
    accept_row = scores[0]
    reject_row = next(row for row in scores if row["case_id"] == "case-07")
    assert accept_row["deterministic_constraint_penalty"] == 0.0
    assert accept_row["uncertainty_score"] < policy["uncertainty_score_block_threshold"]
    assert accept_row["abstention"]["policy_blocked"] is True
    assert accept_row["abstention"]["reason_codes"] == ["source_provenance_risk", "model_identity_coverage_risk"]
    assert reject_row["deterministic_constraint_penalty"] == 0.0
    assert reject_row["exact_acceptance_authority"] is True
    assert reject_row["uncertainty"]["exact_clean_disagreement"] is True
    assert reject_row["uncertainty_score"] >= policy["uncertainty_score_block_threshold"]
    assert "exact_clean_disagreement" in reject_row["abstention"]["reason_codes"]
    assert "source_provenance_risk" in reject_row["abstention"]["reason_codes"]
    mod.validate_artifact(artifact)


def test_req_verify_3314_clean_sources_do_not_block_policy(tmp_path: Path) -> None:
    """REQ-VERIFY-3314: advisory abstention clears when uncertainty and provenance are clean."""

    _stage_sources(tmp_path, clean=True)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.2)

    assert artifact["distributional_repair_audit_ready"] is True
    assert artifact["provenance_risk_features"]["provenance_risk_score"] == 0.0
    assert artifact["model_identity_confound_check"]["confound_detected"] is False
    assert artifact["model_identity_confound_check"]["model_identity_coverage_risk"] == 0.0
    assert artifact["uncertainty_abstention_policy"]["headline_promotion_blocked"] is False
    assert artifact["uncertainty_abstention_policy"]["high_uncertainty_case_count"] == 0
    assert artifact["uncertainty_abstention_policy"]["row_abstention_count"] == 0
    assert all(row["abstention"]["policy_blocked"] is False for row in artifact["repair_row_scores"])
    assert mod.score_clean_verifier_quality("ACCEPT") == 1.0
    assert mod.score_clean_verifier_quality("ABSTAIN") == 0.5
    assert mod.score_clean_verifier_quality("REJECT") == 0.25
    assert mod.score_clean_verifier_quality("anything") == 0.0
    assert mod.rate(3, 6) == 0.5
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(5.0, 4.0) == 0.0


def test_req_verify_3314_fail_closed_for_missing_or_malformed_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-3314: incomplete evidence cannot masquerade as audit-ready."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["distributional_repair_audit_ready"] is False
    assert artifact["repair_case_count"] == 0
    assert artifact["no_new_model_execution"] is True
    assert artifact["uncertainty_abstention_policy"]["headline_promotion_blocked"] is True
    assert artifact["provenance_risk_features"]["source_artifacts_readable"] is False
    assert artifact["model_identity_confound_check"]["confound_detected"] is True
    assert artifact["repair_row_scores"] == []
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}
    assert mod.file_status(tmp_path / "missing.json")["present"] is False
    assert mod.file_status(tmp_path)["readable"] is False
    assert mod.mapping_list("not rows") == []
    assert mod.string_list([1, "x", None]) == ["1", "x", "None"]
    assert mod.model_ids_from_rows([{"model_id": "m1"}, {"hf_id": "m2"}, {"model_id": "m1"}]) == [
        "m1",
        "m2",
    ]
    assert mod.numeric(True) == 0.0
    assert mod.count_value(True) == 0
    assert mod.count_value(4.0) == 4
    assert mod.count_value("5") == 5
    assert mod.count_value("bad") == 0
    assert mod.model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert mod.model_family("custom/Llama-70B") == "llama"
    assert mod.model_family("unknown/model") == "unknown"
    assert mod.sha256_file(tmp_path / "missing.bin") is None

    clean_provenance = {"provenance_risk_score": 0.0}
    clean_model = {"model_identity_coverage_risk": 0.0}
    exact_fail = mod.repair_row_score(
        {
            "case_id": "exact-fail",
            "exact_check_passed": False,
            "calibrated_clean_verifier_decision": "accept",
            "candidate_answer": "x",
            "token_counts": {"total_tokens": 1},
        },
        clean_provenance,
        clean_model,
    )
    unknown_decision = mod.repair_row_score(
        {
            "case_id": "unknown",
            "exact_check_passed": True,
            "calibrated_clean_verifier_decision": "maybe",
            "candidate_answer": "x",
            "token_counts": {"total_tokens": 1},
        },
        clean_provenance,
        clean_model,
    )
    missing_answer = mod.repair_row_score(
        {
            "case_id": "missing-answer",
            "exact_check_passed": True,
            "calibrated_clean_verifier_decision": "accept",
            "candidate_answer": "",
            "token_counts": {"total_tokens": 1},
        },
        clean_provenance,
        clean_model,
    )
    assert exact_fail["abstention"]["reason_codes"] == ["deterministic_constraint_failure"]
    assert unknown_decision["abstention"]["reason_codes"] == ["unknown_clean_verifier_decision"]
    assert missing_answer["abstention"]["reason_codes"] == ["row_uncertainty"]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="distributional_repair_audit_ready"):
        mod.validate_artifact(artifact | {"distributional_repair_audit_ready": "true"})
    with pytest.raises(ValueError, match="uncertainty_abstention_policy"):
        mod.validate_artifact(artifact | {"uncertainty_abstention_policy": {}})
    with pytest.raises(ValueError, match="distributional_energy_schema"):
        mod.validate_artifact(artifact | {"distributional_energy_schema": {}})
    with pytest.raises(ValueError, match="model_identity_confound_check"):
        mod.validate_artifact(artifact | {"model_identity_confound_check": {}})
    with pytest.raises(ValueError, match="provenance_risk_features"):
        mod.validate_artifact(artifact | {"provenance_risk_features": {}})
    with pytest.raises(ValueError, match="repair_case_count"):
        mod.validate_artifact(artifact | {"repair_case_count": "0"})
    with pytest.raises(ValueError, match="no_new_model_execution"):
        mod.validate_artifact(artifact | {"no_new_model_execution": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
