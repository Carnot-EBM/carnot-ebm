"""RED-first tests for REQ-SELFLEARN-7069 and its registered scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_7069_v619_context_authorization_contract as exp7069
from carnot.learning.constraint_policy_store import PolicyStore
from carnot.learning.context_authorization import (
    Authorization,
    AuthorizationDecision,
    BoundedValidationEvent,
    ContextAuthorizationMachine,
    ContextBoundExperience,
    DecisionView,
    LifecycleState,
    NumericInterval,
    RetentionResult,
    SupportInterval,
    ValidationDisposition,
    ValidationPlan,
    settle_validation,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_HASH = "sha256:" + "a" * 64
SCHEMA_HASH = "sha256:" + "b" * 64
RECEIPT_HASH = "sha256:" + "c" * 64


def _record(
    *,
    source_group: str = "source-a",
    support: SupportInterval | None = None,
    effect: NumericInterval | None = None,
    retention: RetentionResult = RetentionResult.PASSED,
    conflicts: tuple[str, ...] = (),
    event_time: int = 0,
    policy_hash: str = POLICY_HASH,
    schema_hash: str = SCHEMA_HASH,
) -> ContextBoundExperience:
    return ContextBoundExperience(
        experience_id=f"experience:{source_group}:{event_time}",
        parent_policy_hash=policy_hash,
        source_group=source_group,
        constraint_schema_hash=schema_hash,
        support_interval=support or SupportInterval(1, 8),
        observed_effect_interval=effect or NumericInterval(0.10, 0.20),
        retention_result=retention,
        conflicts=conflicts,
        event_time=event_time,
        source_receipt_hash=RECEIPT_HASH,
    )


def _view(
    *,
    source_group: str = "source-a",
    event_time: int = 2,
    related_source_groups: tuple[str, ...] = (),
    active_constraints: tuple[str, ...] = (),
    policy_hash: str = POLICY_HASH,
    schema_hash: str = SCHEMA_HASH,
) -> DecisionView:
    return DecisionView(
        event_id=f"event-{event_time}",
        event_time=event_time,
        parent_policy_hash=policy_hash,
        source_group=source_group,
        related_source_groups=related_source_groups,
        constraint_schema_hash=schema_hash,
        active_constraints=active_constraints,
    )


def _store(path: Path, *, capacity: int = 4096) -> PolicyStore:
    return PolicyStore(path, initial_records=(), max_state_bytes=capacity)


def _validation(
    outcome: float,
    *,
    event_time: int = 3,
) -> BoundedValidationEvent:
    plan = ValidationPlan(
        validation_id=f"validation-{event_time}",
        cost=0.05,
        max_abs_outcome=1.0,
    )
    decision = AuthorizationDecision(
        authorization=Authorization.VALIDATE,
        reason="related_context_bounded_uncertainty",
        evidence_ids=("experience:source-a:0",),
        decided_at=event_time,
    )
    return BoundedValidationEvent.open_after_decision(
        plan=plan,
        authorization_decision=decision,
        exact_later_outcome=outcome,
        outcome_receipt_hash=sha256_json({"outcome": outcome, "event_time": event_time}),
        outcome_opened_at=event_time + 1,
    )


def test_req_selflearn_7069_spec_precedes_implementation() -> None:
    """REQ-SELFLEARN-7069 names every required contract scenario and field."""

    text = (REPO_ROOT / exp7069.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for marker in (
        "REQ-SELFLEARN-7069",
        "SCENARIO-SELFLEARN-7069-SCHEMA",
        "SCENARIO-SELFLEARN-7069-AUTHORIZATION",
        "SCENARIO-SELFLEARN-7069-TRANSITIONS",
        "SCENARIO-SELFLEARN-7069-TIME-FIREWALL",
        "SCENARIO-SELFLEARN-7069-VALIDATION",
        "SCENARIO-SELFLEARN-7069-NO-OP",
        "SCENARIO-SELFLEARN-7069-CAPACITY",
        "SCENARIO-SELFLEARN-7069-STREAM",
        "SCENARIO-SELFLEARN-7069-MUTATION",
    ):
        assert marker in text
    for field in exp7069.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in text


def test_scenario_7069_schema_is_immutable_validated_and_serializable() -> None:
    """SCENARIO-SELFLEARN-7069-SCHEMA preserves exact immutable evidence."""

    record = _record(conflicts=("constraint-z", "constraint-a", "constraint-a"))
    assert record.conflicts == ("constraint-a", "constraint-z")
    assert ContextBoundExperience.from_json(record.to_json()) == record
    assert ContextBoundExperience.from_dict(record.to_dict()) == record
    assert record.canonical_hash == sha256_json(record.to_dict())
    with pytest.raises(FrozenInstanceError):
        record.source_group = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="support interval"):
        _record(support=SupportInterval(9, 1))
    with pytest.raises(ValueError, match="effect interval"):
        _record(effect=NumericInterval(0.2, 0.1))
    with pytest.raises(ValueError, match="SHA-256"):
        _record(policy_hash="mutable-policy")
    with pytest.raises(ValueError, match="unknown experience fields"):
        ContextBoundExperience.from_dict({**record.to_dict(), "future_outcome": 1.0})


def test_scenario_7069_closed_schema_rejects_every_invalid_shape() -> None:
    """SCENARIO-SELFLEARN-7069-SCHEMA exercises each closed parsing boundary."""

    record = _record()
    payload = record.to_dict()
    invalid_constructors = (
        {"experience_id": ""},
        {"constraint_schema_hash": "invalid"},
        {"source_receipt_hash": "invalid"},
        {"event_time": -1},
        {"support_interval": "invalid"},
        {"observed_effect_interval": "invalid"},
        {"retention_result": "invalid"},
        {"parent_policy_hash": "sha256:" + "g" * 64},
    )
    for changes in invalid_constructors:
        values = {
            "experience_id": record.experience_id,
            "parent_policy_hash": record.parent_policy_hash,
            "source_group": record.source_group,
            "constraint_schema_hash": record.constraint_schema_hash,
            "support_interval": record.support_interval,
            "observed_effect_interval": record.observed_effect_interval,
            "retention_result": record.retention_result,
            "conflicts": record.conflicts,
            "event_time": record.event_time,
            "source_receipt_hash": record.source_receipt_hash,
            **changes,
        }
        with pytest.raises(ValueError):
            ContextBoundExperience(**values)  # type: ignore[arg-type]

    for changes in (
        {"schema": "wrong"},
        {"support_interval": {"start": 0}},
        {"observed_effect_interval": {"lower": 0}},
        {"conflicts": "constraint"},
        {"retention_result": "not-registered"},
    ):
        with pytest.raises(ValueError):
            ContextBoundExperience.from_dict({**payload, **changes})
    with pytest.raises(ValueError, match="one object"):
        ContextBoundExperience.from_json("[]")
    with pytest.raises(ValueError, match="no typed experience"):
        ContextBoundExperience.from_policy_record({})
    wrong_key = record.to_policy_record()
    wrong_key["policy_key"] = "changed"
    with pytest.raises(ValueError, match="key mismatch"):
        ContextBoundExperience.from_policy_record(wrong_key)


def test_scenario_7069_time_firewall_has_no_outcome_channel() -> None:
    """SCENARIO-SELFLEARN-7069-TIME-FIREWALL closes all outcome inputs."""

    assert set(inspect.signature(ContextAuthorizationMachine.authorize).parameters) == {
        "self",
        "records",
        "view",
    }
    assert set(DecisionView.schema()["fields"]) == {
        "event_id",
        "event_time",
        "parent_policy_hash",
        "source_group",
        "related_source_groups",
        "constraint_schema_hash",
        "active_constraints",
    }
    safe = _view()
    assert DecisionView.from_dict(safe.to_dict()) == safe
    for denied in (
        "exact_outcome",
        "future_events",
        "held_group_label",
        "post_event_aggregate",
        "outcome_provider",
    ):
        with pytest.raises(ValueError, match="forbidden decision fields"):
            DecisionView.from_dict({**safe.to_dict(), denied: "leak"})
    with pytest.raises(ValueError, match="forbidden decision fields"):
        DecisionView.from_dict({**safe.to_dict(), "metadata": {"later_outcome": 1}})
    with pytest.raises(ValueError, match="unknown decision fields"):
        DecisionView.from_dict({**safe.to_dict(), "metadata": {"safe": [1]}})
    missing = safe.to_dict()
    missing.pop("event_id")
    with pytest.raises(ValueError, match="missing decision fields"):
        DecisionView.from_dict(missing)
    malformed = safe.to_dict()
    malformed["related_source_groups"] = "source-a"
    with pytest.raises(ValueError, match="must be lists"):
        DecisionView.from_dict(malformed)
    for changes in (
        {"event_id": ""},
        {"event_time": -1},
        {"parent_policy_hash": "invalid"},
    ):
        values = {
            "event_id": safe.event_id,
            "event_time": safe.event_time,
            "parent_policy_hash": safe.parent_policy_hash,
            "source_group": safe.source_group,
            "related_source_groups": safe.related_source_groups,
            "constraint_schema_hash": safe.constraint_schema_hash,
            "active_constraints": safe.active_constraints,
            **changes,
        }
        with pytest.raises(ValueError):
            DecisionView(**values)  # type: ignore[arg-type]


def test_scenario_7069_direct_and_related_authorization() -> None:
    """SCENARIO-SELFLEARN-7069-AUTHORIZATION grants use or validation exactly."""

    machine = ContextAuthorizationMachine()
    direct = machine.authorize((_record(),), _view())
    related = machine.authorize(
        (_record(source_group="source-a"),),
        _view(source_group="source-b", related_source_groups=("source-a",)),
    )

    assert direct.authorization is Authorization.USE
    assert direct.reason == "direct_context_supported"
    assert related.authorization is Authorization.VALIDATE
    assert related.reason == "related_context_bounded_uncertainty"
    assert direct.evidence_ids == ("experience:source-a:0",)


@pytest.mark.parametrize(
    ("records", "view", "reason"),
    [
        ((), _view(), "missing_facts"),
        ((_record(),), _view(policy_hash="sha256:" + "d" * 64), "policy_mismatch"),
        ((_record(),), _view(schema_hash="sha256:" + "e" * 64), "schema_mismatch"),
        ((_record(),), _view(source_group="unknown"), "unknown_context"),
        (
            (_record(conflicts=("constraint-x",)),),
            _view(active_constraints=("constraint-x",)),
            "conflict_overlap",
        ),
        ((_record(support=SupportInterval(0, 1)),), _view(event_time=2), "stale_support"),
        (
            (_record(effect=NumericInterval(0.01, 0.50)),),
            _view(),
            "unsupported_uncertainty",
        ),
        (
            (_record(effect=NumericInterval(-0.20, -0.10)),),
            _view(),
            "contradicted_effect",
        ),
        (
            (_record(retention=RetentionResult.FAILED),),
            _view(),
            "contradicted_retention",
        ),
        ((_record(event_time=2),), _view(event_time=2), "non_predecessor_evidence"),
    ],
)
def test_scenario_7069_reject_rules(
    records: tuple[ContextBoundExperience, ...],
    view: DecisionView,
    reason: str,
) -> None:
    """SCENARIO-SELFLEARN-7069-AUTHORIZATION rejects each unsafe context."""

    decision = ContextAuthorizationMachine().authorize(records, view)
    assert decision.authorization is Authorization.REJECT
    assert decision.reason == reason


def test_scenario_7069_every_legal_transition_and_invalid_transition() -> None:
    """SCENARIO-SELFLEARN-7069-TRANSITIONS enumerates the complete table."""

    expected = {
        (LifecycleState.START, "use"): LifecycleState.USED,
        (LifecycleState.START, "validate"): LifecycleState.VALIDATING,
        (LifecycleState.START, "reject"): LifecycleState.REJECTED,
        (LifecycleState.VALIDATING, "commit"): LifecycleState.COMMITTED,
        (LifecycleState.VALIDATING, "rollback"): LifecycleState.ROLLED_BACK,
        (LifecycleState.VALIDATING, "no_op"): LifecycleState.NO_OP,
    }
    assert ContextAuthorizationMachine.transition_table() == expected
    for (source, action), destination in expected.items():
        machine = ContextAuthorizationMachine(state=source)
        assert machine.advance(action) is destination
        assert machine.state is destination

    machine = ContextAuthorizationMachine()
    with pytest.raises(ValueError, match="illegal authorization transition"):
        machine.advance("commit")
    assert machine.state is LifecycleState.START


def test_scenario_7069_validation_commit_is_post_decision_and_atomic(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-VALIDATION commits supported later evidence."""

    store = _store(tmp_path / "commit")
    parent = store.state_bytes
    record = _record(source_group="source-b", event_time=3)
    event = _validation(0.20)
    receipt = settle_validation(store, event, record)

    assert event.disposition is ValidationDisposition.COMMIT
    assert event.outcome_opened_at > event.authorization_decision.decided_at
    assert receipt["action"] == "commit"
    assert receipt["committed"] is True
    assert store.state_bytes != parent
    assert [row["phase"] for row in store.journal_rows()] == ["prepare", "commit"]
    assert ContextBoundExperience.from_policy_record(store.records()[0]) == record


def test_scenario_7069_validation_rollback_restores_exact_parent(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-VALIDATION restores parent bytes after harm."""

    store = _store(tmp_path / "rollback")
    parent = store.state_bytes
    receipt = settle_validation(store, _validation(-0.20), _record(event_time=3))

    assert receipt["action"] == "rollback"
    assert receipt["rolled_back"] is True
    assert receipt["parent_state_hash"] == receipt["restored_state_hash"]
    assert store.state_bytes == parent
    assert [row["phase"] for row in store.journal_rows()] == [
        "prepare",
        "commit",
        "rollback",
    ]
    assert store.replay_journal()["passed"] is True


def test_scenario_7069_no_op_is_terminal_and_nonmutating(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-NO-OP records no preference without a write."""

    store = _store(tmp_path / "no-op")
    parent = store.state_bytes
    event = _validation(0.01)
    receipt = settle_validation(store, event, _record(event_time=3))

    assert event.disposition is ValidationDisposition.NO_OP
    assert receipt == {
        "action": "no_op",
        "reason": "no_preference_supported",
        "state_hash": store.state_hash,
        "state_unchanged": True,
    }
    assert store.state_bytes == parent
    assert store.journal_rows() == []


def test_scenario_7069_validation_bounds_and_temporal_order_fail_closed() -> None:
    """SCENARIO-SELFLEARN-7069-VALIDATION enforces cost, bounds, and ordering."""

    with pytest.raises(ValueError, match="validation cost"):
        ValidationPlan(validation_id="invalid", cost=-0.1, max_abs_outcome=1.0)
    with pytest.raises(ValueError, match="validation bound"):
        ValidationPlan(validation_id="invalid", cost=0.2, max_abs_outcome=0.1)
    with pytest.raises(ValueError, match="outside the validation bound"):
        _validation(2.0)
    plan = ValidationPlan(validation_id="timing", cost=0.05, max_abs_outcome=1.0)
    decision = AuthorizationDecision(
        Authorization.VALIDATE,
        "related_context_bounded_uncertainty",
        ("experience",),
        4,
    )
    with pytest.raises(ValueError, match="after authorization"):
        BoundedValidationEvent.open_after_decision(
            plan=plan,
            authorization_decision=decision,
            exact_later_outcome=0.2,
            outcome_receipt_hash=RECEIPT_HASH,
            outcome_opened_at=4,
        )
    use_decision = AuthorizationDecision(
        Authorization.USE,
        "direct_context_supported",
        ("experience",),
        4,
    )
    with pytest.raises(ValueError, match="requires a validate decision"):
        BoundedValidationEvent.open_after_decision(
            plan=plan,
            authorization_decision=use_decision,
            exact_later_outcome=0.2,
            outcome_receipt_hash=RECEIPT_HASH,
            outcome_opened_at=5,
        )
    with pytest.raises(ValueError, match="outcome receipt"):
        BoundedValidationEvent.open_after_decision(
            plan=plan,
            authorization_decision=decision,
            exact_later_outcome=0.2,
            outcome_receipt_hash="invalid",
            outcome_opened_at=5,
        )
    with pytest.raises(ValueError, match="authorization and reason"):
        AuthorizationDecision("validate", "", (), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decision time"):
        AuthorizationDecision(Authorization.REJECT, "invalid", (), -1)
    assert decision.to_dict()["authorization"] == "validate"


def test_scenario_7069_capacity_rejection_preserves_parent_bytes(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-CAPACITY keeps bounded state exact."""

    store = _store(tmp_path / "capacity", capacity=256)
    parent = store.state_bytes
    oversized = _record(conflicts=tuple(f"constraint-{index}" for index in range(50)))
    with pytest.raises(ValueError, match="byte limit"):
        store.commit(oversized.to_policy_record())
    assert store.state_bytes == parent
    assert store.records() == []


def test_req_selflearn_7069_blocked_preconditions_are_exact(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7069 writes a schema-complete block with exact values."""

    checks = [
        exp7069.gate("transactional_memory_api", True, False),
        exp7069.gate("artifact_path_writable", True, True),
    ]
    artifact = exp7069.build_blocked_artifact(checks, duration_s=0.01)
    assert set(exp7069.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["context_authorization_contract_ready_score"] == 0
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_checks": [
            {
                "failed_check": "transactional_memory_api",
                "expected_value": True,
                "observed_value": False,
            }
        ],
    }
    assert exp7069.validate_artifact(artifact)

    built = exp7069.build_artifact(
        repo_root=REPO_ROOT,
        work_root=tmp_path / "blocked-work",
        preconditions_override=checks,
    )
    assert built["verdict_class"] == "blocked"


def test_req_selflearn_7069_precondition_file_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7069 preserves missing and unreadable source observations."""

    missing = tmp_path / "missing" / "nested" / "artifact.json"
    assert exp7069._sha256_path(missing) is None
    assert exp7069._read_object(missing) == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert exp7069._read_object(invalid) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp7069._read_object(array) == {}
    assert exp7069._nearest_existing_parent(missing) == tmp_path


def test_req_selflearn_7069_ready_artifact_stream_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-STREAM/MUTATION validates row-owned readiness."""

    artifact = exp7069.build_artifact(repo_root=REPO_ROOT, work_root=tmp_path / "work")
    assert exp7069.validate_artifact(artifact)
    assert artifact["context_authorization_contract_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate"] == "deterministic_verifier"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["chronological_stream_manifest_hash"] == sha256_json(
        artifact["chronological_stream_manifest"]
    )
    assert {row["case_kind"] for row in artifact["chronological_stream_manifest"]["events"]} >= {
        "positive_transfer",
        "conflict",
        "drift",
        "unknown",
        "no_op",
    }
    assert all(row["passed"] for row in artifact["mutation_rows"])
    assert all(
        not ({"held_group_label", "exact_outcome"} & set(row["decision_view"]))
        for row in artifact["temporal_firewall_rows"]
    )
    assert set(exp7069.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])

    for mutate in (
        lambda value: value.update(context_authorization_contract_ready_score=0),
        lambda value: value["chronological_stream_manifest"].update(events=[]),
        lambda value: value["authorization_rule_rows"][0].update(decision="reject"),
        lambda value: value.update(inference_substrate="mutable_outcome_oracle"),
        lambda value: value.update(verifier_is_oracle=True),
        lambda value: value.update(reproducibility_checksum="sha256:" + "0" * 64),
    ):
        changed = deepcopy(artifact)
        mutate(changed)
        with pytest.raises(ValueError):
            exp7069.validate_artifact(changed)


def _resign(artifact: dict) -> dict:
    artifact["reproducibility_checksum"] = exp7069.artifact_checksum(artifact)
    return artifact


def test_req_selflearn_7069_validator_recomputes_every_ready_boundary(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7069-MUTATION reaches every semantic artifact guard."""

    artifact = exp7069.build_artifact(repo_root=REPO_ROOT, work_root=tmp_path / "ready")
    mutations = []

    missing = deepcopy(artifact)
    missing.pop("rows")
    mutations.append(missing)

    for field, value in (
        ("field_principles", {}),
        ("context_authorization_contract_ready_score", True),
        ("chronological_stream_manifest_hash", "sha256:" + "0" * 64),
        ("verdict_class", "positive"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        mutations.append(_resign(changed))

    changed = deepcopy(artifact)
    changed["context_authorization_contract_ready_score"] = 0
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["rows"] = []
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["experience_schema"]["immutable"] = False
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["authorization_schema"]["mutable_outcome_source_allowed"] = True
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["authorization_rule_rows"][0]["decision"] = "reject"
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["state_transition_rows"][0]["destination"] = "rejected"
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["chronological_stream_manifest"]["events"] = []
    changed["chronological_stream_manifest_hash"] = sha256_json(
        changed["chronological_stream_manifest"]
    )
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["protected_retention_manifest"]["tuning_allowed"] = True
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["temporal_firewall_rows"][0]["current_outcome_hidden"] = False
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["conflict_rows"] = []
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["mutation_rows"] = []
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["transaction_rows"] = []
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["no_op_rows"][0]["action"] = "commit"
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["rollback_rows"][0]["parent_bytes_restored"] = False
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["transaction_rows"][0]["journal_phases"] = ["commit", "prepare"]
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["capacity_rows"][0]["state_unchanged"] = False
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["validation_event_schema"]["precommitted_before_outcome"] = False
    mutations.append(_resign(changed))

    for changed in mutations:
        with pytest.raises(ValueError):
            exp7069.validate_artifact(changed)


def test_req_selflearn_7069_validator_recomputes_blocked_boundaries() -> None:
    """REQ-SELFLEARN-7069 rejects forged blocked scores, prefixes, and failures."""

    checks = [exp7069.gate("required", True, False)]
    artifact = exp7069.build_blocked_artifact(checks, duration_s=0.01)
    mutations = []
    for field, value in (
        ("context_authorization_contract_ready_score", 1),
        ("honest_verdict", "complete_null_wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = {"passed": True, "failed_checks": []}
    mutations.append(_resign(changed))
    changed = deepcopy(artifact)
    changed["gate_check_summary"]["failed_checks"][0]["extra"] = True
    mutations.append(_resign(changed))
    for changed in mutations:
        with pytest.raises(ValueError):
            exp7069.validate_artifact(changed)


def test_req_selflearn_7069_atomic_write_cleans_failed_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SELFLEARN-7069 removes unpublished temporary bytes after failure."""

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("forced replacement failure")

    monkeypatch.setattr(exp7069.os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced replacement"):
        exp7069._atomic_write(tmp_path / "artifact.json", {"complete": True})
    assert list(tmp_path.iterdir()) == []


def test_req_selflearn_7069_serialized_artifact_and_cli(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7069 publishes one valid deterministic result document."""

    output = tmp_path / "artifact.json"
    artifact = exp7069.write_artifact(
        output_path=output,
        repo_root=REPO_ROOT,
        work_root=tmp_path / "work",
    )
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == artifact
    assert exp7069.validate_artifact(stored)

    cli_output = tmp_path / "cli.json"
    assert (
        exp7069.main(
            [
                "--date",
                "20260906",
                "--output",
                str(cli_output),
                "--work-root",
                str(tmp_path / "cli-work"),
            ]
        )
        == 0
    )
    assert exp7069.validate_artifact(json.loads(cli_output.read_text(encoding="utf-8")))
    with pytest.raises(SystemExit, match="execution date"):
        exp7069.main(["--date", "20260905", "--output", str(tmp_path / "wrong.json")])
