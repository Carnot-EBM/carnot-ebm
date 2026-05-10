"""Tests for Exp 1668 pipeline CerCE promotion ledger.

Spec: REQ-LEARN-1668, SCENARIO-LEARN-1668, SCENARIO-LEARN-1669.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline import cerce_ledger as mod


POLICY_ID = "policy:fr11:simulated-memory:safe"


def _case(
    case_id: str,
    *,
    pre: float = 1.0,
    post: float = 0.0,
    retained: bool = True,
    replay_failed: bool = False,
) -> mod.ReplayCase:
    return mod.ReplayCase(
        case_id=case_id,
        pre_violation_bound=pre,
        post_violation_bound=post,
        retained=retained,
        replay_failed=replay_failed,
        source="unit-test",
    )


def _update(
    *,
    policy_id: str = POLICY_ID,
    cases: tuple[mod.ReplayCase, ...] | None = None,
    prior_hash: str = "a" * 64,
    updated_hash: str = "b" * 64,
    no_model_weight_mutation: bool = True,
) -> mod.MemoryPolicyUpdate:
    return mod.MemoryPolicyUpdate(
        policy_update_id=policy_id,
        prior_memory_hash=prior_hash,
        updated_memory_hash=updated_hash,
        replay_cases=cases if cases is not None else (_case("case-a"), _case("case-b")),
        utility_delta=0.25,
        no_model_weight_mutation=no_model_weight_mutation,
        provenance=("simulated-memory-update",),
    )


def test_scenario_learn_1668_safe_memory_update_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1668: retained replay with non-worsening bounds is promoted."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.evaluate_promotion_gate(
        [_update()],
        output_path=output,
        project_root=tmp_path,
        run_date="20260510",
        tests_run=["tests/python/test_cerce_ledger.py"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["cerce_ledger_ready"] is True
    assert artifact["promotion_gate_passed"] is True
    assert artifact["policy_certificates_evaluated"] == 1
    assert artifact["promotion_safe_policy_updates"] == [POLICY_ID]
    assert artifact["blocked_policy_updates"] == []
    assert artifact["accepted_violation_count"] == 0
    assert artifact["pre_violation_bound"] == pytest.approx(2.0)
    assert artifact["post_violation_bound"] == pytest.approx(0.0)
    assert artifact["violation_bound_delta"] == pytest.approx(-2.0)
    assert artifact["replay_retention_rate"] == pytest.approx(1.0)
    assert artifact["nonforgetting_rate"] == pytest.approx(1.0)
    assert artifact["nonforgetting_certificate_rate"] == pytest.approx(1.0)
    assert artifact["certificates"][0]["promotion_safe"] is True
    assert artifact["certificates"][0]["replay_retention_passed"] is True
    assert artifact["certificates"][0]["gate_failures"] == []
    assert artifact["blockers"] == []
    assert artifact["tests_run"] == ["tests/python/test_cerce_ledger.py"]
    mod.validate_artifact(artifact)


def test_scenario_learn_1669_worsened_bound_rejects_policy() -> None:
    """SCENARIO-LEARN-1669: a post-update bound increase blocks promotion."""

    update = _update(
        policy_id="policy:fr11:simulated-memory:worse",
        cases=(_case("old-case", pre=0.0, post=1.0),),
    )

    artifact = mod.evaluate_promotion_gate([update], run_date="20260510")

    assert artifact["status"] == "blocked"
    assert artifact["cerce_ledger_ready"] is False
    assert artifact["promotion_gate_passed"] is False
    assert artifact["accepted_violation_count"] == 1
    assert artifact["pre_violation_bound"] == pytest.approx(0.0)
    assert artifact["post_violation_bound"] == pytest.approx(1.0)
    assert artifact["violation_bound_delta"] == pytest.approx(1.0)
    assert artifact["nonforgetting_rate"] == pytest.approx(0.0)
    assert artifact["promotion_safe_policy_updates"] == []
    assert artifact["blocked_policy_updates"] == ["policy:fr11:simulated-memory:worse"]
    assert "bound_worsened" in artifact["blockers"]
    assert "case_bound_worsened" in artifact["certificates"][0]["gate_failures"]
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("candidate", "expected_failure"),
    [
        (_update(cases=()), "no_replay_cases"),
        (_update(cases=(_case("dropped", retained=False),)), "replay_retention_failed"),
        (_update(cases=(_case("failed", replay_failed=True),)), "replay_failed"),
        (_update(prior_hash=""), "missing_memory_hash"),
        (_update(updated_hash="a" * 64), "unchanged_memory_hash"),
        (_update(no_model_weight_mutation=False), "model_weight_mutation_detected"),
    ],
)
def test_req_learn_1668_replay_and_memory_gates_fail_closed(
    candidate: mod.MemoryPolicyUpdate,
    expected_failure: str,
) -> None:
    """REQ-LEARN-1668-3/5: retention, hash, and weight gates reject unsafe updates."""

    artifact = mod.evaluate_promotion_gate([candidate], run_date="20260510")

    assert artifact["status"] == "blocked"
    assert artifact["certificates"][0]["promotion_safe"] is False
    assert expected_failure in artifact["certificates"][0]["gate_failures"]
    assert expected_failure in artifact["blockers"]
    mod.validate_artifact(artifact)


def test_req_learn_1668_mapping_inputs_normalize_to_stable_certificates() -> None:
    """REQ-LEARN-1668-1/4: dict inputs produce deterministic JSON certificates."""

    candidate: dict[str, Any] = {
        "policy_update_id": "policy:dict",
        "prior_memory_hash": "0" * 64,
        "updated_memory_hash": "1" * 64,
        "utility_delta": 0.0,
        "no_model_weight_mutation": True,
        "provenance": ["dict-fixture"],
        "replay_cases": [
            {
                "case_id": "dict-case",
                "pre_violation_bound": 2,
                "post_violation_bound": 2,
                "retained": True,
                "replay_failed": False,
                "source": "dict",
            }
        ],
    }

    first = mod.evaluate_promotion_gate([candidate], run_date="20260510")
    second = mod.evaluate_promotion_gate([candidate], run_date="20260510")

    assert first == second
    assert first["status"] == "complete"
    certificate = first["certificates"][0]
    assert len(certificate["certificate_id"]) == 64
    assert certificate["policy_update_id"] == "policy:dict"
    assert certificate["replay_cases"][0]["bound_worsened"] is False
    assert certificate["replay_cases"][0]["bound_delta"] == pytest.approx(0.0)
    mod.validate_artifact(first)


def test_req_learn_1668_validation_rejects_malformed_artifacts() -> None:
    """REQ-LEARN-1668-4/5: artifact validation catches impossible reports."""

    valid = mod.evaluate_promotion_gate([_update()], run_date="20260510")
    empty = mod.evaluate_promotion_gate([], run_date="20260510")

    assert empty["status"] == "blocked"
    assert empty["blockers"] == ["no_policy_updates"]

    missing = dict(valid)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(dict(valid, schema="wrong"))

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(valid, status="unknown"))

    with pytest.raises(AssertionError, match="nonforgetting_rate"):
        mod.validate_artifact(dict(valid, nonforgetting_rate=1.5))

    with pytest.raises(AssertionError, match="nonforgetting_certificate_rate"):
        mod.validate_artifact(dict(valid, nonforgetting_certificate_rate=0.5))

    with pytest.raises(AssertionError, match="policy certificate counts"):
        mod.validate_artifact(dict(valid, policy_certificates_evaluated=99))

    impossible = dict(
        valid,
        cerce_ledger_ready=False,
        promotion_gate_passed=False,
        status="complete",
        blockers=["x"],
        blocked_policy_updates=["policy:x"],
        accepted_violation_count=1,
        nonforgetting_rate=0.0,
        nonforgetting_certificate_rate=0.0,
        policy_certificates_evaluated=0,
        certificates=[],
    )
    with pytest.raises(AssertionError, match="complete artifact is invalid"):
        mod.validate_artifact(impossible)
