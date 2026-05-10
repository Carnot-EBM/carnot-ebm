"""Tests for Exp 1659 SMGI certified update gates.

Spec: REQ-LEARN-1659, SCENARIO-LEARN-1659, SCENARIO-LEARN-1660.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline import smgi_updates as mod


POLICY_ID = "daily_eval:dvi_v8:verified:exp1449:ltlzinc-always-power_ok-repair-hint-00"
CERTIFICATE_ID = "a815d75393b1671e2ae63262fa79e6f2c2d90e7bb1ea4da5a853566b26696a69"


def _ledger(*, unsafe: bool = False) -> dict[str, Any]:
    accepted = bool(unsafe)
    return {
        "status": "complete",
        "schema": "cerce_certificate_ledger_v1",
        "continuous_self_learning_task": True,
        "cerce_ledger_ready": not unsafe,
        "policy_certificates_evaluated": 1,
        "constraint_violation_records": 1,
        "accepted_violation_count": int(accepted),
        "false_accept_delta": int(accepted),
        "soundness_mistakes": int(accepted),
        "nonforgetting_certificate_rate": 0.0 if unsafe else 1.0,
        "promotion_safe_policy_updates": [] if unsafe else [POLICY_ID],
        "blocked_policy_updates": [POLICY_ID] if unsafe else [],
        "policy_certificates": [
            {
                "accepted_violation_count": int(accepted),
                "certificate_id": CERTIFICATE_ID,
                "constraint_count": 1,
                "constraint_ids": ["grammar_certificate:cctu-1486-arith-001:schema_only:1"],
                "false_accept_delta": int(accepted),
                "fr11_events_recorded": 0,
                "policy_update_id": POLICY_ID,
                "promotion_safe": not unsafe,
                "soundness_mistakes": int(accepted),
            }
        ],
        "ledger_rows": [
            {
                "accepted_violation": accepted,
                "baseline_violation": False,
                "certificate_id": "8d4f989726fd1baf1f90cf19f729459aaf89d64b0d3534e0f3aba4cc64fb2b58",
                "constraint_id": "grammar_certificate:cctu-1486-arith-001:schema_only:1",
                "constraint_type": "grammar_certificate",
                "false_accept_delta": int(accepted),
                "policy_update_id": POLICY_ID,
                "promoted_violation": accepted,
                "soundness_mistake": accepted,
                "source": "unit-test",
            }
        ],
        "blockers": ["accepted_constraint_violation"] if unsafe else [],
        "honest_verdict": (
            "complete: cerce_certificate_ledger_blocked"
            if unsafe
            else "complete: cerce_certificate_ledger_ready"
        ),
    }


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate = {
        "policy_update_id": POLICY_ID,
        "certificate_id": CERTIFICATE_ID,
        "prior_memory_hash": "a" * 64,
        "updated_memory_hash": "b" * 64,
        "replay_case_count": 3,
        "retained_case_count": 3,
        "replay_failure_count": 0,
        "utility_delta": 0.125,
        "no_model_weight_mutation": True,
        "provenance": ["results/experiment_1594_cerce_ledger.json"],
    }
    candidate.update(overrides)
    return candidate


def test_scenario_learn_1659_certifies_safe_cerce_update_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1659: safe CerCE and SMGI evidence produce a complete report."""

    ledger_path = tmp_path / "cerce.json"
    output_path = tmp_path / mod.OUTPUT_FILE
    ledger_path.write_text(json.dumps(_ledger(), sort_keys=True), encoding="utf-8")

    artifact = mod.certify_update_gates(
        ledger_path,
        [_candidate()],
        output_path=output_path,
        run_date="20260509",
        tests_run=["tests/python/test_smgi_updates.py"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["smgi_certified_update_ready"] is True
    assert artifact["certified_update_success"] is True
    assert artifact["cerce_ledger_ready"] is True
    assert artifact["policy_certificates_evaluated"] == 1
    assert artifact["accepted_violation_count"] == 0
    assert artifact["false_accept_delta"] == 0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["certified_update_count"] == 1
    assert artifact["rejected_update_count"] == 0
    assert artifact["promotion_safe_policy_updates"] == [POLICY_ID]
    assert artifact["blocked_policy_updates"] == []
    assert artifact["certified_updates"][0]["policy_update_id"] == POLICY_ID
    assert artifact["certified_updates"][0]["certified_update_success"] is True
    assert artifact["certified_updates"][0]["gates"]["cerce_certificate_match"] is True
    assert artifact["rejected_updates"] == []
    assert artifact["blockers"] == []
    assert artifact["tests_run"] == ["tests/python/test_smgi_updates.py"]
    mod.validate_artifact(artifact)


def test_req_learn_1659_derives_stable_candidates_from_safe_ledger() -> None:
    """REQ-LEARN-1659-4: promotion-safe CerCE rows produce deterministic candidates."""

    ledger = _ledger()

    first = mod.certify_update_gates(ledger, run_date="20260509")
    second = mod.certify_update_gates(ledger, run_date="20260509")

    assert first == second
    assert first["status"] == "complete"
    assert first["source_cerce_hash"] == mod.stable_hash(ledger)
    assert first["candidate_updates_evaluated"] == 1
    update = first["certified_updates"][0]
    assert update["policy_update_id"] == POLICY_ID
    assert update["prior_memory_hash"] != update["updated_memory_hash"]
    assert update["replay_case_count"] == update["retained_case_count"] == 1


@pytest.mark.parametrize(
    ("candidate_overrides", "expected_blocker"),
    [
        ({"replay_failure_count": 1, "retained_case_count": 2}, "smgi_candidate_rejected"),
        ({"policy_update_id": "missing-policy"}, "missing_cerce_certificate"),
        ({"certificate_id": "wrong"}, "smgi_candidate_rejected"),
        ({"prior_memory_hash": ""}, "smgi_candidate_rejected"),
        ({"updated_memory_hash": "a" * 64}, "unchanged_session_memory_hash"),
        ({"utility_delta": -0.01}, "smgi_candidate_rejected"),
        ({"no_model_weight_mutation": False}, "model_weight_mutation_detected"),
        ({"provenance": None}, "missing_update_provenance"),
    ],
)
def test_scenario_learn_1660_rejects_unsafe_candidate_evidence(
    candidate_overrides: dict[str, Any],
    expected_blocker: str,
) -> None:
    """SCENARIO-LEARN-1660: replay, hash, utility, and weight gates fail closed."""

    candidate = _candidate(**candidate_overrides)
    artifact = mod.certify_update_gates(_ledger(), [candidate])

    assert artifact["status"] == "blocked"
    assert artifact["smgi_certified_update_ready"] is False
    assert artifact["certified_update_success"] is False
    assert artifact["certified_update_count"] == 0
    assert artifact["rejected_update_count"] == 1
    assert expected_blocker in artifact["blockers"]
    assert artifact["rejected_updates"][0]["policy_update_id"] == candidate["policy_update_id"]
    assert artifact["rejected_updates"][0]["certified_update_success"] is False
    mod.validate_artifact(artifact)


def test_scenario_learn_1660_blocks_unsafe_cerce_ledger() -> None:
    """SCENARIO-LEARN-1660: unsafe CerCE evidence blocks every SMGI update."""

    artifact = mod.certify_update_gates(_ledger(unsafe=True), [_candidate()])

    assert artifact["status"] == "blocked"
    assert artifact["cerce_ledger_ready"] is False
    assert artifact["certified_update_success"] is False
    assert artifact["accepted_violation_count"] == 1
    assert artifact["false_accept_delta"] == 1
    assert artifact["soundness_mistakes"] == 1
    assert "accepted_constraint_violation" in artifact["blockers"]
    assert "positive_false_accept_delta" in artifact["blockers"]
    assert "soundness_mistake" in artifact["blockers"]
    assert "cerce_ledger_not_ready" in artifact["blockers"]
    assert artifact["rejected_updates"][0]["gate_failures"]
    mod.validate_artifact(artifact)


def test_req_learn_1659_empty_and_malformed_ledgers_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-1659-2/4/5: empty, incomplete, and malformed ledgers do not certify."""

    empty_ledger = dict(
        _ledger(),
        status="blocked",
        cerce_ledger_ready=False,
        policy_certificates=[],
        policy_certificates_evaluated=0,
        promotion_safe_policy_updates=[],
        blocked_policy_updates=0,
    )
    empty_artifact = mod.certify_update_gates(empty_ledger, [])
    malformed_cert_artifact = mod.certify_update_gates(
        dict(_ledger(), policy_certificates="not-a-list"),
        [_candidate()],
    )
    bad_json = tmp_path / "bad-ledger.json"
    bad_json.write_text("[]", encoding="utf-8")

    assert mod.derive_candidates_from_ledger(_ledger(unsafe=True)) == []
    assert empty_artifact["status"] == "blocked"
    assert "cerce_ledger_not_complete" in empty_artifact["blockers"]
    assert "no_policy_certificates" in empty_artifact["blockers"]
    assert "no_smgi_update_candidates" in empty_artifact["blockers"]
    assert malformed_cert_artifact["status"] == "blocked"
    assert "no_policy_certificates" in malformed_cert_artifact["blockers"]
    with pytest.raises(ValueError, match="JSON object"):
        mod.certify_update_gates(bad_json)


def test_req_learn_1659_validation_fails_closed_on_schema_edges() -> None:
    """REQ-LEARN-1659-1/5: report validation catches missing and impossible fields."""

    artifact = mod.certify_update_gates(_ledger(), [_candidate()])

    missing = dict(artifact)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(dict(artifact, schema="wrong"))

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(artifact, status="unknown"))

    with pytest.raises(AssertionError, match="continuous_self_learning_task"):
        mod.validate_artifact(dict(artifact, continuous_self_learning_task=False))

    with pytest.raises(AssertionError, match="candidate update counts"):
        mod.validate_artifact(dict(artifact, candidate_updates_evaluated=99))

    with pytest.raises(AssertionError, match="nonforgetting_certificate_rate"):
        mod.validate_artifact(dict(artifact, nonforgetting_certificate_rate=1.5))

    impossible_complete = dict(
        artifact,
        cerce_ledger_ready=False,
        certified_update_count=0,
        rejected_update_count=1,
        certified_update_success=False,
        smgi_certified_update_ready=False,
        accepted_violation_count=1,
        false_accept_delta=1,
        soundness_mistakes=1,
        nonforgetting_certificate_rate=0.5,
        no_model_weight_mutation=False,
        blockers=["x"],
        rejected_updates=[_candidate()],
    )
    with pytest.raises(AssertionError, match="complete artifact is invalid"):
        mod.validate_artifact(impossible_complete)
