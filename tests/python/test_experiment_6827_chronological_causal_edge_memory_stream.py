"""Tests for the frozen chronological causal-edge memory stream.

Spec refs: REQ-CL-6827, SCENARIO-CL-6827-PRECONDITIONS,
SCENARIO-CL-6827-OPERATIONS, SCENARIO-CL-6827-VISIBILITY,
SCENARIO-CL-6827-SNAPSHOTS, SCENARIO-CL-6827-ORDERS,
SCENARIO-CL-6827-COUNTERFACTUALS, SCENARIO-CL-6827-SERIALIZATION,
and SCENARIO-CL-6827-READINESS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6827_chronological_causal_edge_memory_stream as stream
from scripts import adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = stream.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load immutable source evidence once to keep focused tests fast."""

    return stream.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict]) -> dict:
    """Build one complete stream for all row and readiness assertions."""

    return stream.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.25,
    )


def _operation(
    operation_id: str,
    kind: stream.OperationKind,
    key: str,
    *,
    value: dict | None = None,
    expected_revision: int | None = None,
    authority: str = "stream_builder",
) -> stream.MemoryOperation:
    """Create one typed operation with explicit optional fields."""

    return stream.MemoryOperation(
        operation_id=operation_id,
        kind=kind,
        key=key,
        value=value,
        expected_revision=expected_revision,
        authority=authority,
    )


def test_req_cl_6827_spec_precedes_implementation() -> None:
    """REQ-CL-6827 owns every public path, field, and scenario."""

    spec = (REPO_ROOT / stream.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6827", 1)[1]
    for requirement_id in stream.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in stream.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (
        stream.MODULE_RELATIVE_PATH,
        stream.SCRIPT_RELATIVE_PATH,
        stream.RESULT_RELATIVE_PATH,
    ):
        assert path.as_posix() in section


def test_scenario_cl_6827_preconditions_accept_frozen_sources(sources: dict[str, dict]) -> None:
    """SCENARIO-CL-6827-PRECONDITIONS accepts the complete sealed inputs."""

    summary = stream.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("audit", "selective_arbiter_audit_complete"),
        ("family", "three_source_families"),
        ("raw_hash", "raw_output_hashes"),
        ("decision", "terminal_component_decisions"),
        ("chronology", "chronological_keys"),
        ("alternative", "legal_alternatives"),
        ("future", "decision_snapshot_no_future_fields"),
    ],
)
def test_scenario_cl_6827_preconditions_fail_closed(
    sources: dict[str, dict], fault: str, failed_check: str
) -> None:
    """SCENARIO-CL-6827-PRECONDITIONS records each failed input gate."""

    changed = sources
    decision_fields = None
    if fault == "audit":
        changed = {**sources, "exp6826": {**sources["exp6826"]}}
        changed["exp6826"]["selective_arbiter_audit_complete"] = False
    elif fault == "family":
        changed = {**sources, "exp6812": deepcopy(sources["exp6812"])}
        changed["exp6812"]["rows"] = changed["exp6812"]["rows"][192:]
    elif fault == "raw_hash":
        changed = {**sources, "exp6812": deepcopy(sources["exp6812"])}
        changed["exp6812"]["rows"][0]["raw_output_sha256"] = ""
    elif fault == "decision":
        changed = {**sources, "exp6826": {**sources["exp6826"]}}
        changed["exp6826"]["utility_decision"] = "invented"
    elif fault == "chronology":
        changed = {**sources, "exp6812": deepcopy(sources["exp6812"])}
        changed["exp6812"]["rows"][2]["cell_id"] = changed["exp6812"]["rows"][0]["cell_id"]
    elif fault == "alternative":
        changed = {**sources, "exp6812": deepcopy(sources["exp6812"])}
        first_cell = changed["exp6812"]["rows"][0]["cell_id"]
        changed["exp6812"]["rows"] = [
            row
            for row in changed["exp6812"]["rows"]
            if not (row["cell_id"] == first_cell and row["candidate_index"] == 1)
        ]
    else:
        decision_fields = (*stream.FEATURE_ALLOWLIST, "future_outcome")

    summary = stream.check_preconditions(
        changed,
        SOURCE_PATHS,
        decision_snapshot_fields=decision_fields,
    )
    assert summary["passed"] is False
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert failed["passed"] is False
    assert "expected" in failed and "observed" in failed

    blocked = stream.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.25,
        decision_snapshot_fields=decision_fields,
    )
    assert blocked["status"] == stream.BLOCKED_STATUS
    assert blocked["rows"] == []
    assert blocked["verified_memory_stream_ready"] is False
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith(stream.BLOCKED_STATUS)
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]


def test_scenario_cl_6827_operations_have_exact_effects_and_receipts() -> None:
    """SCENARIO-CL-6827-OPERATIONS covers all six typed operations."""

    memory = stream.FixedCapacityMemory(capacity=2, scope="order_1::family_a")
    add = memory.apply(_operation("op1", stream.OperationKind.ADD, "b", value={"v": 1}))
    memory.apply(_operation("op2", stream.OperationKind.ADD, "a", value={"v": 2}))
    retrieve = memory.apply(_operation("op3", stream.OperationKind.RETRIEVE, "a"))
    filtered = memory.apply(_operation("op4", stream.OperationKind.FILTER, ""))
    revise = memory.apply(
        _operation(
            "op5",
            stream.OperationKind.REVISE,
            "a",
            value={"v": 3},
            expected_revision=1,
        )
    )
    deleted = memory.apply(
        _operation(
            "op6",
            stream.OperationKind.SOFT_DELETE,
            "a",
            expected_revision=2,
        )
    )
    restored = memory.apply(
        _operation(
            "op7",
            stream.OperationKind.RESTORE,
            "a",
            expected_revision=3,
            authority=stream.RESTORE_AUTHORITY,
        )
    )

    assert [row["key"] for row in memory.snapshot().records] == ["a", "b"]
    assert retrieve["effect"]["record"]["value"] == {"v": 2}
    assert [row["key"] for row in filtered["effect"]["records"]] == ["a", "b"]
    assert revise["effect"]["revision"] == 2
    assert deleted["inverse"]["kind"] == "restore"
    assert restored["inverse"]["kind"] == "soft_delete"
    for receipt in (add, retrieve, filtered, revise, deleted, restored):
        assert receipt["accepted"] is True
        assert receipt["receipt_sha256"] == stream.receipt_sha256(receipt)
        assert receipt["parent_state_sha256"].startswith("sha256:")
        assert receipt["new_state_sha256"].startswith("sha256:")
        assert receipt["operation_sha256"].startswith("sha256:")


def test_scenario_cl_6827_operations_reject_without_mutation() -> None:
    """SCENARIO-CL-6827-OPERATIONS rejects conflicts, stale writes, and pressure."""

    memory = stream.FixedCapacityMemory(capacity=1, scope="order_1::family_a")
    memory.apply(_operation("op1", stream.OperationKind.ADD, "a", value={"v": 1}))
    parent = memory.state_bytes()
    rejected = [
        memory.apply(_operation("op2", stream.OperationKind.ADD, "a", value={"v": 2})),
        memory.apply(_operation("op3", stream.OperationKind.ADD, "b", value={"v": 2})),
        memory.apply(
            _operation(
                "op4",
                stream.OperationKind.REVISE,
                "a",
                value={"v": 2},
                expected_revision=99,
            )
        ),
        memory.apply(
            _operation(
                "op5",
                stream.OperationKind.RESTORE,
                "a",
                expected_revision=1,
                authority="learner",
            )
        ),
    ]
    assert {row["reason"] for row in rejected} == {
        "key_conflict",
        "capacity_exceeded",
        "stale_revision",
        "restore_authority_required",
    }
    assert all(row["accepted"] is False for row in rejected)
    assert all(row["parent_state_bytes"] == row["new_state_bytes"] for row in rejected)
    assert memory.state_bytes() == parent


def test_scenario_cl_6827_all_operation_preconditions_are_closed() -> None:
    """SCENARIO-CL-6827-OPERATIONS gives every invalid form one exact reason."""

    with pytest.raises(ValueError, match="capacity must be positive"):
        stream.FixedCapacityMemory(capacity=0, scope="invalid")

    memory = stream.FixedCapacityMemory(capacity=2, scope="order_1::family_a")
    accepted = memory.apply(
        _operation("op1", stream.OperationKind.ADD, "a", value={"items": [1, 2]})
    )
    assert memory.snapshot().records[0]["value"]["items"] == (1, 2)
    duplicate = memory.apply(_operation("op1", stream.OperationKind.RETRIEVE, "a"))
    cases = [
        memory.apply(_operation("op2", stream.OperationKind.ADD, "b")),
        memory.apply(
            _operation(
                "op3",
                stream.OperationKind.REVISE,
                "missing",
                value={"v": 1},
                expected_revision=1,
            )
        ),
        memory.apply(
            _operation(
                "op4",
                stream.OperationKind.REVISE,
                "a",
                expected_revision=1,
            )
        ),
        memory.apply(
            _operation(
                "op5",
                stream.OperationKind.SOFT_DELETE,
                "missing",
                expected_revision=1,
            )
        ),
        memory.apply(
            _operation(
                "op6",
                stream.OperationKind.SOFT_DELETE,
                "a",
                expected_revision=99,
            )
        ),
        memory.apply(
            _operation(
                "op7",
                stream.OperationKind.RESTORE,
                "missing",
                expected_revision=1,
                authority=stream.RESTORE_AUTHORITY,
            )
        ),
    ]
    assert accepted["accepted"] is True
    assert duplicate["reason"] == "duplicate_operation"
    assert [row["reason"] for row in cases] == [
        "value_required",
        "active_record_required",
        "value_required",
        "active_record_required",
        "stale_revision",
        "deleted_record_required",
    ]

    deleted = memory.apply(
        _operation(
            "op8",
            stream.OperationKind.SOFT_DELETE,
            "a",
            expected_revision=1,
        )
    )
    deleted_revise = memory.apply(
        _operation(
            "op9",
            stream.OperationKind.REVISE,
            "a",
            value={"v": 2},
            expected_revision=2,
        )
    )
    deleted_delete = memory.apply(
        _operation(
            "op10",
            stream.OperationKind.SOFT_DELETE,
            "a",
            expected_revision=2,
        )
    )
    stale_restore = memory.apply(
        _operation(
            "op11",
            stream.OperationKind.RESTORE,
            "a",
            expected_revision=99,
            authority=stream.RESTORE_AUTHORITY,
        )
    )
    assert deleted["accepted"] is True
    assert deleted_revise["reason"] == "active_record_required"
    assert deleted_delete["reason"] == "active_record_required"
    assert stale_restore["reason"] == "stale_revision"


def test_scenario_cl_6827_visibility_and_stale_pressure_recovery() -> None:
    """SCENARIO-CL-6827-VISIBILITY hides deletes and permits exact recovery."""

    memory = stream.FixedCapacityMemory(capacity=1, scope="order_1::family_a")
    memory.apply(_operation("op1", stream.OperationKind.ADD, "stale", value={"v": 1}))
    memory.apply(
        _operation(
            "op2",
            stream.OperationKind.SOFT_DELETE,
            "stale",
            expected_revision=1,
        )
    )
    hidden = memory.apply(_operation("op3", stream.OperationKind.RETRIEVE, "stale"))
    assert hidden["effect"]["record"] is None
    assert memory.snapshot().retrieve("stale") is None
    assert memory.snapshot().retrieve("stale", include_deleted=True) is not None

    recovery = memory.apply(_operation("op4", stream.OperationKind.ADD, "fresh", value={"v": 2}))
    blocked_restore = memory.apply(
        _operation(
            "op5",
            stream.OperationKind.RESTORE,
            "stale",
            expected_revision=2,
            authority=stream.RESTORE_AUTHORITY,
        )
    )
    assert recovery["accepted"] is True
    assert blocked_restore["accepted"] is False
    assert blocked_restore["reason"] == "capacity_exceeded"


def test_scenario_cl_6827_snapshots_are_deeply_read_only() -> None:
    """SCENARIO-CL-6827-SNAPSHOTS rejects mutation at every nested level."""

    memory = stream.FixedCapacityMemory(capacity=2, scope="order_1::family_a")
    memory.apply(_operation("op1", stream.OperationKind.ADD, "a", value={"nested": {"value": 1}}))
    snapshot = memory.snapshot()
    with pytest.raises(TypeError):
        snapshot.records[0]["value"]["nested"]["value"] = 2
    with pytest.raises(TypeError):
        snapshot.records[0]["new"] = True
    with pytest.raises(AttributeError):
        snapshot.records.append({})
    assert snapshot.state_bytes == memory.state_bytes()
    assert tuple(snapshot.records[0]["value"]["nested"].values()) == (1,)


def test_scenario_cl_6827_source_failures_and_non_rows_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CL-6827-PRECONDITIONS keeps malformed input evidence."""

    list_path = tmp_path / "list.json"
    bad_path = tmp_path / "bad.json"
    missing_path = tmp_path / "missing.json"
    list_path.write_text("[]", encoding="utf-8")
    bad_path.write_text("{", encoding="utf-8")
    loaded = stream.load_sources({"list": list_path, "bad": bad_path, "missing": missing_path})
    assert loaded["list"] == {"_load_error": "not_object"}
    assert loaded["bad"] == {"_load_error": "JSONDecodeError"}
    assert loaded["missing"] == {"_load_error": "FileNotFoundError"}
    assert stream._group_source_rows({"rows": "not-a-list"}) == {}
    hashes = stream._source_hashes({"present": list_path, "missing": missing_path})
    assert hashes["present"]["sha256"].startswith("sha256:")
    assert hashes["missing"]["sha256"] == "missing"


def test_scenario_cl_6827_orders_are_complete_and_family_isolated(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6827-ORDERS freezes five complete isolated chronologies."""

    events = stream.build_events(sources["exp6812"])
    manifest = stream.freeze_chronology(events)
    assert len(events) == stream.EXPECTED_EVENT_COUNT
    assert len(manifest["orders"]) == stream.ORDER_COUNT
    assert len(manifest["rotations"]) == len(stream.MANDATED_FAMILIES)
    for order in manifest["orders"]:
        for family in stream.MANDATED_FAMILIES:
            ids = order["event_ids_by_family"][family]
            expected = {event["event_id"] for event in events if event["source_family"] == family}
            assert len(ids) == stream.EVENTS_PER_FAMILY
            assert set(ids) == expected
    assert all(
        set(rotation["development_families"]).isdisjoint({rotation["held_out_family"]})
        for rotation in manifest["rotations"]
    )


def test_scenario_cl_6827_counterfactual_rows_are_complete(artifact: dict) -> None:
    """SCENARIO-CL-6827-COUNTERFACTUALS emits every required unit exactly once."""

    expected = stream.EXPECTED_EVENT_COUNT * stream.ORDER_COUNT * len(stream.COUNTERFACTUAL_KINDS)
    rows = artifact["rows"]
    assert len(rows) == expected
    assert len({row["row_id"] for row in rows}) == expected
    assert {row["counterfactual_kind"] for row in rows} == set(stream.COUNTERFACTUAL_KINDS)
    applicable = [row for row in rows if row["counterfactual_applicable"]]
    assert applicable
    assert {row["counterfactual_kind"] for row in applicable} == set(stream.COUNTERFACTUAL_KINDS)
    assert all(row["write_operation_id"] and row["read_operation_id"] for row in applicable)
    assert all(row["memory_scope"].startswith(row["order_id"] + "::") for row in rows)


def test_scenario_cl_6827_edge_without_owned_writer_is_inapplicable() -> None:
    """SCENARIO-CL-6827-COUNTERFACTUALS rejects unowned and non-prior writes."""

    receipt = {
        "kind": stream.OperationKind.RETRIEVE.value,
        "operation_id": "read",
        "effect": {"record": {"key": "a", "value": {"v": 1}}},
    }
    event = {"hidden_outcome_sha256": "sha256:hidden"}
    assert stream._edge_for_receipt(receipt, event, 2, {}) is None
    writer = {"a": {"operation_id": "write", "position": 2}}
    assert stream._edge_for_receipt(receipt, event, 2, writer) is None


def test_scenario_cl_6827_sealed_fields_never_enter_decisions(artifact: dict) -> None:
    """SCENARIO-CL-6827-SNAPSHOTS keeps outcomes in the hidden harness view."""

    denied = set(artifact["feature_denylist"])
    assert set(artifact["feature_allowlist"]).isdisjoint(denied)
    assert all(set(row["decision_feature_keys"]).isdisjoint(denied) for row in artifact["rows"])
    assert all("final_acceptance" not in row for row in artifact["rows"])
    sealed = artifact["sealed_field_manifest"]
    assert sealed["final_acceptance_location"] == "hidden_harness_view"
    assert sealed["hidden_view_sha256"].startswith("sha256:")


def test_scenario_cl_6827_serialization_is_byte_stable(artifact: dict) -> None:
    """SCENARIO-CL-6827-SERIALIZATION binds canonical inputs but not wall time."""

    value = {"z": [2, 1], "a": {"b": True}}
    assert stream.canonical_json_bytes(value) == b'{"a":{"b":true},"z":[2,1]}\n'
    assert stream.sha256_json(value) == stream.sha256_json(deepcopy(value))
    changed_duration = {**artifact, "duration_s": 999.0}
    assert stream.reproducibility_checksum(changed_duration) == artifact["reproducibility_checksum"]
    assert stream.validate_artifact(artifact) == []


def test_scenario_cl_6827_validator_rejects_each_contract_fault(artifact: dict) -> None:
    """SCENARIO-CL-6827-SERIALIZATION fails closed on every artifact invariant."""

    faults = {
        "required field set mismatch": {
            key: value for key, value in artifact.items() if key != "title"
        },
        "field_principles coverage mismatch": {
            **artifact,
            "field_principles": {},
        },
        "inference_substrate mismatch": {**artifact, "inference_substrate": "LLM"},
        "verdict_class outside closed enum": {**artifact, "verdict_class": "invented"},
        "ready row count mismatch": {**artifact, "rows": artifact["rows"][:-1]},
        "ready artifact has failed gates": {
            **artifact,
            "gate_check_summary": {"failed_checks": ["fault"]},
        },
        "blocked artifact must not expose partial rows": {
            **artifact,
            "verified_memory_stream_ready": False,
        },
        "hidden final acceptance exposed": {
            **artifact,
            "rows": [{**artifact["rows"][0], "final_acceptance": True}, *artifact["rows"][1:]],
        },
    }
    for expected, changed in faults.items():
        assert expected in stream.validate_artifact(changed)


def test_req_cl_6827_build_and_writer_surface_validation_errors(
    sources: dict[str, dict], artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-6827 never publishes output after internal validation fails."""

    original_validate = stream.validate_artifact
    monkeypatch.setattr(stream, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        stream.build_artifact(
            sources,
            source_paths=SOURCE_PATHS,
            run_date="20260831",
            duration_s=0.25,
        )
    with pytest.raises(ValueError, match="forced"):
        stream.write_artifact(tmp_path / "blocked.json", artifact)
    monkeypatch.setattr(stream, "validate_artifact", original_validate)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(stream.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        stream.write_artifact(tmp_path / "replace-failure.json", artifact)
    assert not list(tmp_path.glob("*.tmp"))


def test_scenario_cl_6827_readiness_has_nonzero_headroom_and_ignores_adoption(
    sources: dict[str, dict], artifact: dict
) -> None:
    """SCENARIO-CL-6827-READINESS depends on stream evidence, not adoption sign."""

    assert artifact["verified_memory_stream_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["admissible_operation_count"] > 0
    assert artifact["rejected_operation_count"] > 0
    assert artifact["later_read_opportunity_count"] > 0
    assert all(value > 0 for value in artifact["headroom_metrics"].values())
    assert set(artifact["field_principles"]) == set(artifact)

    changed = {**sources, "exp6826": {**sources["exp6826"]}}
    changed["exp6826"]["deployment_adoption_decision"] = "retire"
    changed_artifact = stream.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.25,
    )
    assert changed_artifact["verified_memory_stream_ready"] is True
    assert changed_artifact["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_cl_6827_writer_and_validator_use_task_owned_path(
    sources: dict[str, dict], tmp_path: Path
) -> None:
    """REQ-CL-6827 writes canonical JSON and validates the same bytes."""

    output = tmp_path / "exp6827.json"
    exit_code = stream.main(["--date", "20260831", "--result-path", str(output)])
    assert exit_code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["verified_memory_stream_ready"] is True
    assert written["duration_s"] > 0
    assert stream.main(["--result-path", str(output), "--validate"]) == 0

    broken = {**written, "verified_memory_stream_ready": False}
    output.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        stream.main(["--result-path", str(output), "--validate"])


def test_req_cl_6827_task_owned_wrapper_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-6827 keeps the executable wrapper free of experiment logic."""

    monkeypatch.setattr(stream, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(REPO_ROOT / stream.SCRIPT_RELATIVE_PATH), run_name="__main__")
    assert raised.value.code == 0


def test_req_cl_6827_adversarial_audit_recognizes_cpu_source_transform(
    tmp_path: Path,
) -> None:
    """REQ-CL-6827 keeps quoted GGUF identities on the declared CPU floor."""

    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "6827",
                "honest_verdict": "complete: frozen stream ready",
                "inference_substrate": stream.INFERENCE_SUBSTRATE,
                "duration_s": 0.5,
                "random_seed": 6_827_001,
                "reproducibility_checksum": "sha256:control",
                "source_families": list(stream.MANDATED_FAMILIES),
            }
        ),
        encoding="utf-8",
    )
    floor = adversarial_verify.duration_floor_for_artifact(
        {"inference_substrate": stream.INFERENCE_SUBSTRATE}
    )
    assert floor == {
        "substrate": stream.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.DETERMINISTIC_VERIFIER_MIN_DURATION_S,
        "reason": "deterministic_verifier",
    }
    kinds = {flag["kind"] for flag in adversarial_verify.verify_artifact(path)["flags"]}
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds
