"""Tests for Exp 3143 FR-11 experience-driven verifier memory.

Spec refs: REQ-LEARN-3143, SCENARIO-LEARN-3143,
SCENARIO-LEARN-3143-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_experience_driven_verifier_memory_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verifier_row(
    row_id: str,
    family: str,
    *,
    expected_action: str,
    live_decision: str,
    exact_label: str,
    answer_format: str = "validity_token",
    difficulty: list[str] | None = None,
    mechanism: str = "no_failure",
    ledger_consistent: bool = True,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "fixture_family": family,
        "difficulty_buckets": difficulty or ["easy"],
        "answer_extraction_format": answer_format,
        "failure_mechanism_from_exp3124": mechanism,
        "expected_action": expected_action,
        "live_decision": live_decision,
        "exact_label": exact_label,
        "monitor_events": [
            {
                "event_type": "constraint_ledger",
                "payload": {"ledger_action": expected_action},
            },
            {
                "event_type": "candidate_final_answer",
                "payload": {
                    "final_answer_consistent_with_ledger": ledger_consistent,
                    "final_answer_consistent_with_exact": ledger_consistent,
                },
            },
        ],
    }


def _exp3136_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
        "false_accept_autopsy_v1_ready": True,
        "false_accept_mechanism_counts": {"contradiction miss": 1},
        "false_accept_row_ids": ["unsafe-reject"],
        "source_false_accept_count": 1,
        "source_live_row_count": 4,
        "verifier_rows": [
            _verifier_row(
                "safe-accept",
                "safe_math",
                expected_action="accept",
                live_decision="accept",
                exact_label="VALID",
                difficulty=["easy", "satisfiable_drift"],
            ),
            _verifier_row(
                "safe-reject",
                "repair_json",
                expected_action="reject",
                live_decision="reject",
                exact_label="REPAIRABLE",
                answer_format="repairability_token",
                difficulty=["hard"],
            ),
            _verifier_row(
                "unsafe-reject",
                "unsafe_math",
                expected_action="reject",
                live_decision="accept",
                exact_label="INVALID",
                difficulty=["easy", "contradiction"],
                mechanism="contradiction",
                ledger_consistent=False,
            ),
            _verifier_row(
                "unsafe-valid",
                "unsafe_math",
                expected_action="accept",
                live_decision="accept",
                exact_label="VALID",
                difficulty=["easy"],
            ),
        ],
    }


def _variant_record(
    variant_id: str,
    family_id: str,
    *,
    kind: str = "equivalent",
    exact_replay_passed: bool = True,
    soundness_errors: int = 0,
    completeness_errors: int = 0,
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "source_environment_id": f"{family_id}-source",
        "variant_kind": kind,
        "environment": {"family_id": family_id},
        "exact_replay_passed": exact_replay_passed,
        "no_answer_leakage_passed": True,
        "solve_verify_asymmetry_passed": True,
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
    }


def _exp3142_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3142_fr11_vera_evoenv_hardening_v2",
        "fr11_vera_evoenv_v2_ready": True,
        "variant_records": [
            _variant_record("safe-variant-e", "modular_balance_vera_equivalent"),
            _variant_record(
                "safe-variant-h",
                "modular_balance_vera_hardened",
                kind="hardened",
            ),
        ],
        "ledger_consistency_rate": 0.8,
        "ledger_replay_summary": {
            "prior_ledger_consistency_rate": 0.5,
            "ledger_consistency_rate": 0.8,
        },
        "no_weight_update_claim": True,
    }


def _exp3129_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3129_fr11_constraint_memory_retention_drift_audit_v1",
        "fr11_constraint_memory_audit_v1_ready": True,
        "ledger_consistency_rate": 0.5,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    (root / "research-program.md").write_text("continuous self-learning\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3143\nSCENARIO-LEARN-3143\nSCENARIO-LEARN-3143-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3136_REL_PATH, _exp3136_payload())
    _write_json(root, mod.EXP3142_REL_PATH, _exp3142_payload())
    _write_json(root, mod.EXP3129_REL_PATH, _exp3129_payload())


def test_req_learn_3143_spec_anchor_exists() -> None:
    """REQ-LEARN-3143: OpenSpec declares the experience-memory artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3143" in spec
    assert "SCENARIO-LEARN-3143" in spec
    assert "SCENARIO-LEARN-3143-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_experience_verifier_memory_v1_ready" in spec
    assert "memory_key_schema" in spec
    assert "residual_false_accept_risk" in spec
    assert "residual_false_reject_risk" in spec


def test_req_learn_3143_memory_keys_and_routing_policy() -> None:
    """REQ-LEARN-3143-1/2/3/4/5: exact keys suppress; risky families escalate."""

    rows = mod.load_replay_rows(_exp3136_payload(), _exp3142_payload())
    routed = mod.simulate_routing_policy(rows, ledger_consistency_rate=0.5)
    decisions = {row["row_id"]: row["routing_decision"] for row in routed["routing_rows"]}
    key = mod.memory_key_for_row(rows[0])
    index = mod.build_memory_index(rows)

    assert len(rows) == 6
    assert set(mod.MEMORY_KEY_FIELDS) <= key.keys()
    assert mod.memory_key_id(key) in index
    assert decisions["safe-accept"] == "suppress"
    assert decisions["safe-reject"] == "suppress"
    assert decisions["unsafe-reject"] == "escalate"
    assert decisions["unsafe-valid"] == "escalate"
    assert decisions["safe-variant-e"] == "suppress"
    assert routed["suppressed_check_count"] == 4
    assert routed["escalated_check_count"] == 2
    assert routed["estimated_check_savings_rate"] == pytest.approx(2 / 6)
    assert routed["residual_false_accept_risk"] == pytest.approx(0.0)
    assert routed["residual_false_reject_risk"] == pytest.approx(0.166667)
    assert routed["memory_key_summaries"][mod.memory_key_id(key)]["observation_count"] == 1


def test_scenario_learn_3143_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3143: artifact separates routing memory from weights."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=11.25,
        tests_run=["REQ-LEARN-3143 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_experience_verifier_memory_v1_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["memory_key_schema"]["fields"] == list(mod.MEMORY_KEY_FIELDS)
    assert artifact["replay_row_count"] == 6
    assert artifact["suppressed_check_count"] == 4
    assert artifact["escalated_check_count"] == 2
    assert artifact["estimated_check_savings_rate"] == pytest.approx(2 / 6)
    assert artifact["residual_false_accept_risk"] == pytest.approx(0.0)
    assert artifact["residual_false_reject_risk"] == pytest.approx(0.166667)
    assert artifact["ledger_consistency_rate"] == pytest.approx(0.5)
    assert artifact["no_weight_update_claim"] is True
    assert artifact["promotion_recommendation"].endswith(
        "block_model_weight_learning_until_ledger_consistency_is_1.0"
    )
    assert artifact["tests_run"] == ["REQ-LEARN-3143 focused"]
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["mode"] == "artifact_only_controller_routing_memory"
    assert artifact["inference_substrate"]["controller_routing_memory_only"] is True
    assert artifact["inference_substrate"]["model_weight_training"] is False
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3143_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3143-BLOCKED: missing experience sources fail closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_experience_verifier_memory_v1_ready"] is False
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["replay_row_count"] == 0
    assert artifact["suppressed_check_count"] == 0
    assert artifact["escalated_check_count"] == 0
    assert artifact["estimated_check_savings_rate"] == 0.0
    assert artifact["residual_false_accept_risk"] == 0.0
    assert artifact["residual_false_reject_risk"] == 0.0
    assert artifact["ledger_consistency_rate"] == 0.0
    assert artifact["no_weight_update_claim"] is True
    assert artifact["blocked_reason"] == "exp3136_false_accept_autopsy_missing_or_not_ready"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    assert (
        mod.precondition_blocker({"false_accept_autopsy_v1_ready": True}, {}, {})
        == "exp3142_vera_evoenv_missing_or_not_ready"
    )
    assert (
        mod.precondition_blocker(
            {"false_accept_autopsy_v1_ready": True},
            {"fr11_vera_evoenv_v2_ready": True},
            {},
        )
        == "exp3129_constraint_memory_audit_missing_or_not_ready"
    )
    mod.validate_artifact(artifact)


def test_req_learn_3143_validation_rejects_overclaims_and_unsafe_suppression(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3143-3/5/6: validation blocks unsafe routing and overclaims."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    mod.validate_artifact(artifact)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rate(1, 0) == 0.0
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.contract_decision_from_row({"contract_decision": "accept"}) == "accept"
    assert mod.contract_decision_from_row({"expected_action": "reject"}) == "reject"
    assert mod.contract_decision_from_row({}) == "unknown"
    assert mod.ledger_consistent_from_row({"ledger_consistent": False}) is False
    assert mod.ledger_consistent_from_row({"monitor_events": [None]}) is True
    assert mod.ledger_consistent_from_row({}) is True
    assert mod.verifier_replay_rows({"verifier_rows": [None]}) == []
    assert mod.variant_replay_rows({"variant_records": [None]}) == []
    assert mod.normalize_difficulty(123) == "unspecified"
    normal = mod.simulate_routing_policy(
        [
            {
                "row_id": "normal-false-reject",
                "fixture_family": "normal_family",
                "difficulty": "medium",
                "answer_format": "validity_token",
                "failure_mechanism": "no_failure",
                "contract_decision": "accept",
                "expected_action": "accept",
                "false_accept": False,
                "false_reject": True,
                "replay_error": False,
                "ledger_consistent": True,
            }
        ],
        ledger_consistency_rate=1.0,
    )
    assert normal["routing_rows"][0]["routing_decision"] == "normal"
    assert mod.promotion_recommendation(False, 0.0, 1.0).startswith("block_")
    assert mod.promotion_recommendation(True, 0.1, 1.0).startswith("block_")
    assert mod.promotion_recommendation(True, 0.0, 1.0) == "promote_controller_routing_memory"
    assert mod.honest_verdict(False, "blocked").startswith("blocked_precondition_failed")

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        mod.validate_artifact(artifact | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="model_weight_mutation"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            }
        )
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"fresh_live_inference_calls": 1}
            }
        )
    with pytest.raises(ValueError, match="residual_false_accept_risk"):
        mod.validate_artifact(artifact | {"residual_false_accept_risk": 0.1})
    with pytest.raises(ValueError, match="suppressed_check_count"):
        mod.validate_artifact(artifact | {"suppressed_check_count": 0})
    with pytest.raises(ValueError, match="escalated_check_count"):
        mod.validate_artifact(artifact | {"escalated_check_count": 0})
    with pytest.raises(ValueError, match="ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="replay_row_count"):
        mod.validate_artifact(artifact | {"replay_row_count": 0})
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})
