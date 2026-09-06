"""RED-first tests for REQ-SELFLEARN-7070 and its scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7070_v619_bcit_self_learning as exp7070


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture(count: int = 24, groups: int = 12) -> tuple[list[dict], list[dict]]:
    return exp7070.build_synthetic_fixture(event_count=count, source_group_count=groups)


def _comparison(tmp_path: Path, *, count: int = 24, capacity: int = 1_500) -> dict:
    events, outcomes = _fixture(count)
    return exp7070.run_comparison(
        events,
        outcomes,
        work_root=tmp_path / "arms",
        capacity_bytes=capacity,
        random_seed=exp7070.RANDOM_SEED,
    )


def _complete_comparison(*, positive: bool) -> dict:
    rows = []
    snapshots = []
    exact_rows = []
    capacity_rows = []
    for event_time in range(exp7070.MIN_EVENT_COUNT):
        event_id = f"complete-{event_time:04d}"
        for arm in exp7070.ARMS:
            quality = 0.6
            harmful = False
            useful = False
            if arm == "context_bound" and positive:
                quality = 0.8
                useful = True
            elif arm == "flat_reuse":
                harmful = True
            rows.append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "event_time": event_time,
                    "game_id": f"game-{event_time % 6}",
                    "source_group": f"source-{event_time % 12}",
                    "exact_quality": quality,
                    "harmful_update": harmful,
                    "useful_update": useful,
                    "validation_cost": float(arm == "validate_all"),
                    "abstained": arm == "no_reuse",
                    "protected_retention": 1.0,
                    "source_group_transfer": False,
                }
            )
            snapshots.append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "event_time": event_time,
                    "max_evidence_event_time": event_time - 1,
                }
            )
            exact_rows.append({"arm": arm, "event_id": event_id})
            capacity_rows.append(
                {
                    "arm": arm,
                    "event_id": event_id,
                    "within_capacity": True,
                    "protected_records_present": True,
                }
            )
    comparison = {
        "arm_definitions": [{"arm": arm, "store_path": f"/tmp/{arm}"} for arm in exp7070.ARMS],
        "chronological_event_rows": rows,
        "decision_snapshot_rows": snapshots,
        "authorization_rows": [],
        "validation_rows": [],
        "exact_outcome_rows": exact_rows,
        "journal_rows": [],
        "commit_rows": [],
        "rollback_rows": [],
        "capacity_rows": capacity_rows,
        "retention_rows": [],
    }
    comparison.update(exp7070.recompute_aggregates(rows))
    return comparison


def test_req_selflearn_7070_spec_precedes_implementation() -> None:
    """REQ-SELFLEARN-7070 names each required scenario and result field."""

    text = (REPO_ROOT / exp7070.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for marker in (
        "REQ-SELFLEARN-7070",
        "SCENARIO-SELFLEARN-7070-TIME",
        "SCENARIO-SELFLEARN-7070-ISOLATION",
        "SCENARIO-SELFLEARN-7070-ORDER",
        "SCENARIO-SELFLEARN-7070-TRANSACTION",
        "SCENARIO-SELFLEARN-7070-NO-OP",
        "SCENARIO-SELFLEARN-7070-CAPACITY",
        "SCENARIO-SELFLEARN-7070-METRICS",
        "SCENARIO-SELFLEARN-7070-BLOCKED",
    ):
        assert marker in text
    for field in exp7070.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in text


def test_scenario_7070_time_firewall_rejects_leaks_and_early_outcomes(
    tmp_path: Path,
) -> None:
    """SCENARIO-SELFLEARN-7070-TIME seals each outcome-free decision first."""

    events, outcomes = _fixture(16)
    safe = exp7070.DecisionSnapshot.from_event(events[3], evidence_event_times=(0, 1))
    assert exp7070.DecisionSnapshot.from_dict(safe.to_dict()) == safe
    for denied in ("exact_outcome", "self_score", "model_text", "future_event"):
        with pytest.raises(ValueError, match="forbidden decision fields"):
            exp7070.DecisionSnapshot.from_dict({**safe.to_dict(), denied: "leak"})

    leaked = deepcopy(events)
    leaked[2]["exact_outcome"] = outcomes[2]
    with pytest.raises(ValueError, match="outcome data before sealing"):
        exp7070.validate_stream(leaked, outcomes)

    result = exp7070.run_comparison(events, outcomes, work_root=tmp_path / "time")
    assert all(row["decision_sealed_before_outcome"] for row in result["exact_outcome_rows"])
    assert all(
        row["max_evidence_event_time"] < row["event_time"]
        for row in result["decision_snapshot_rows"]
    )


def test_scenario_7070_isolated_arms_share_only_initial_policy(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7070-ISOLATION keeps four stores independent."""

    result = _comparison(tmp_path)
    definitions = result["arm_definitions"]
    assert {row["arm"] for row in definitions} == set(exp7070.ARMS)
    assert len({row["store_path"] for row in definitions}) == 4
    assert len({row["initial_state_hash"] for row in definitions}) == 1
    assert all(row["isolated"] for row in definitions)
    assert all(row["arm_store_path"] != row["other_store_path"] for row in result["isolation_rows"])


def test_scenario_7070_order_changes_and_receipt_changes_fail_closed() -> None:
    """SCENARIO-SELFLEARN-7070-ORDER rejects order, identity, and receipt drift."""

    events, outcomes = _fixture(16)
    assert exp7070.validate_stream(events, outcomes)

    moved = deepcopy(events)
    moved[3], moved[4] = moved[4], moved[3]
    duplicated = deepcopy(events)
    duplicated[3]["event_id"] = duplicated[2]["event_id"]
    changed_outcomes = deepcopy(outcomes)
    changed_outcomes[4]["reuse_effect"] = -0.9
    missing = deepcopy(outcomes[:-1])

    for bad_events, bad_outcomes in (
        (moved, outcomes),
        (duplicated, outcomes),
        (events, changed_outcomes),
        (events, missing),
    ):
        with pytest.raises(ValueError):
            exp7070.validate_stream(bad_events, bad_outcomes)


def test_scenario_7070_failed_commit_recovers_exact_parent(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7070-TRANSACTION aborts a prepared failed write."""

    events, outcomes = _fixture(16)
    target = next(
        event["event_id"]
        for event, outcome in zip(events, outcomes, strict=True)
        if outcome["reuse_effect"] > exp7070.NO_PREFERENCE_BOUND
    )
    result = exp7070.run_comparison(
        events,
        outcomes,
        work_root=tmp_path / "failed",
        failed_commits={("context_bound", target)},
    )
    row = next(
        row
        for row in result["rollback_rows"]
        if row["arm"] == "context_bound" and row["event_id"] == target
    )
    assert row["reason"] == "forced_commit_failure"
    assert row["parent_state_hash"] == row["restored_state_hash"]
    assert row["parent_bytes_restored"] is True
    assert "abort_recovered" in row["journal_phases"]


def test_scenario_7070_no_op_and_harmful_rollback_are_exact(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7070-NO-OP keeps bytes; harm restores parent bytes."""

    result = _comparison(tmp_path)
    no_ops = [row for row in result["journal_rows"] if row["disposition"] == "no_op"]
    assert no_ops
    assert all(row["parent_state_hash"] == row["final_state_hash"] for row in no_ops)
    assert all(row["journal_phases"] == [] for row in no_ops)

    rollbacks = [row for row in result["rollback_rows"] if row["reason"] == "harmful_update"]
    assert rollbacks
    assert all(row["parent_bytes_restored"] for row in rollbacks)
    assert all(row["parent_state_hash"] == row["restored_state_hash"] for row in rollbacks)


def test_scenario_7070_capacity_evicts_old_learning_not_protected_policy(
    tmp_path: Path,
) -> None:
    """SCENARIO-SELFLEARN-7070-CAPACITY preserves protected initial records."""

    result = _comparison(tmp_path, count=48, capacity=1_150)
    evictions = [row for row in result["capacity_rows"] if row["evicted_policy_keys"]]
    assert evictions
    assert all(row["within_capacity"] for row in result["capacity_rows"])
    assert all(row["protected_records_present"] for row in result["capacity_rows"])
    assert all(
        not key.startswith("initial:") for row in evictions for key in row["evicted_policy_keys"]
    )


def test_scenario_7070_aggregates_recompute_from_event_rows(tmp_path: Path) -> None:
    """SCENARIO-SELFLEARN-7070-METRICS makes event rows own every headline."""

    result = _comparison(tmp_path)
    recomputed = exp7070.recompute_aggregates(result["chronological_event_rows"])
    for field in exp7070.AGGREGATE_FIELDS:
        assert result[field] == recomputed[field]
    assert result["per_game_results"] == recomputed["per_game_results"]
    assert result["per_source_group_rows"] == recomputed["per_source_group_rows"]
    assert result["paired_interval_rows"] == recomputed["paired_interval_rows"]

    artifact = exp7070.complete_artifact_from_comparison(
        result,
        run_date=exp7070.RUN_DATE,
        duration_s=0.01,
        source_hashes={},
        preconditions=[],
    )
    assert exp7070.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["harmful_update_rate_by_arm"]["flat_reuse"] = 0.0
    changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
    assert "aggregate_recomputation_mismatch" in exp7070.validate_artifact(changed)


def test_req_selflearn_7070_actual_upstream_shortfall_is_exactly_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-SELFLEARN-7070-BLOCKED reports the frozen 8-by-5 shortfall."""

    checks, source_hashes, _upstream = exp7070.collect_preconditions(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "artifact.json",
        work_root=tmp_path / "stores",
    )
    by_name = {row["check"]: row for row in checks}
    assert by_name["minimum_chronological_event_count"]["expected_value"] == ">=120"
    assert by_name["minimum_chronological_event_count"]["observed_value"] == 8
    assert by_name["minimum_source_group_count"]["expected_value"] == ">=12"
    assert by_name["minimum_source_group_count"]["observed_value"] == 5
    assert by_name["minimum_chronological_event_count"]["passed"] is False
    assert by_name["minimum_source_group_count"]["passed"] is False

    artifact = exp7070.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "artifact.json",
        work_root=tmp_path / "stores",
    )
    assert set(exp7070.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp7070.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["bcit_comparison_complete_score"] == 0
    assert artifact["bcit_self_learning_value_score"] == 0
    assert artifact["source_artifact_hashes"] == source_hashes
    assert artifact["gate_check_summary"]["failed_check"] == "minimum_chronological_event_count"
    assert exp7070.validate_artifact(artifact) == []


def test_req_selflearn_7070_validator_rejects_blocked_and_complete_mutations(
    tmp_path: Path,
) -> None:
    """REQ-SELFLEARN-7070 detects schema, score, verdict, and checksum changes."""

    blocked = exp7070.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "blocked.json",
        work_root=tmp_path / "blocked-stores",
    )
    for field, value, expected_error in (
        ("field_principles", {}, "field_principles_mismatch"),
        ("bcit_comparison_complete_score", True, "complete_score_must_be_bare_integer"),
        ("inference_substrate", "oracle", "inference_substrate_mismatch"),
        ("verifier_is_oracle", True, "verifier_is_oracle_mismatch"),
        ("honest_verdict", "complete_null_wrong", "verdict_prefix_mismatch"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "reproducibility_checksum_mismatch"),
    ):
        changed = deepcopy(blocked)
        changed[field] = value
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
        assert expected_error in exp7070.validate_artifact(changed)

    changed = deepcopy(blocked)
    changed["gate_check_summary"] = {"passed": True, "failed_check": None}
    changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
    assert "blocked_gate_summary_invalid" in exp7070.validate_artifact(changed)


def test_req_selflearn_7070_atomic_write_and_cli(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7070 writes one validated artifact through the command API."""

    output = tmp_path / "result.json"
    artifact = exp7070.write_artifact(
        output_path=output,
        repo_root=REPO_ROOT,
        work_root=tmp_path / "work",
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp7070.validate_artifact(artifact) == []
    assert (
        exp7070.main(
            [
                "--date",
                "20260906",
                "--output",
                str(tmp_path / "cli.json"),
                "--work-root",
                str(tmp_path / "cli-work"),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit, match="execution date"):
        exp7070.main(["--date", "20260905", "--output", str(tmp_path / "wrong.json")])


def test_req_selflearn_7070_defensive_input_boundaries(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7070 rejects malformed views, fixtures, files, and arms."""

    events, outcomes = _fixture(4)
    with pytest.raises(ValueError, match="forbidden decision fields"):
        exp7070.DecisionSnapshot.from_event(
            {**events[1], "metadata": {"self_score": 1}}, evidence_event_times=()
        )
    with pytest.raises(ValueError, match="strictly precede"):
        exp7070.DecisionSnapshot.from_event(events[1], evidence_event_times=(1,))
    snapshot = exp7070.DecisionSnapshot.from_event(events[1], evidence_event_times=())
    incomplete = snapshot.to_dict()
    incomplete.pop("read_state_hash")
    with pytest.raises(ValueError, match="closed schema"):
        exp7070.DecisionSnapshot.from_dict(incomplete)

    assert exp7070._sha256_path(tmp_path / "missing") is None
    assert exp7070._read_object(tmp_path / "missing") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert exp7070._read_object(invalid) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp7070._read_object(array) == {}
    with pytest.raises(ValueError, match="positive"):
        exp7070.build_synthetic_fixture(event_count=0, source_group_count=1)

    changed = deepcopy(outcomes)
    changed[1]["event_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        exp7070.validate_stream(events, changed)
    no_group = deepcopy(events)
    no_group[1]["source_group"] = ""
    with pytest.raises(ValueError, match="source group"):
        exp7070.validate_stream(no_group, outcomes)

    record = exp7070._experience_from_outcome(events[0], outcomes[0])
    with pytest.raises(ValueError, match="unknown arm"):
        exp7070._authorize("unknown", snapshot, (record,))
    assert exp7070._paired_interval([], seed=1)["unit_count"] == 0
    assert exp7070._comparison_complete({}) is False


def test_req_selflearn_7070_complete_positive_null_and_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SELFLEARN-7070 keeps positive and repeated-null terminals distinct."""

    passing = [exp7070._gate("ready", True, True)]
    positive_comparison = _complete_comparison(positive=True)
    positive = exp7070.complete_artifact_from_comparison(
        positive_comparison,
        run_date=exp7070.RUN_DATE,
        duration_s=0.01,
        source_hashes={},
        preconditions=passing,
    )
    assert positive["bcit_comparison_complete_score"] == 1
    assert positive["bcit_self_learning_value_score"] == 1
    assert positive["verdict_class"] == "positive"
    assert exp7070.validate_artifact(positive) == []

    null = exp7070.complete_artifact_from_comparison(
        _complete_comparison(positive=False),
        run_date=exp7070.RUN_DATE,
        duration_s=0.01,
        source_hashes={},
        preconditions=passing,
    )
    assert null["bcit_comparison_complete_score"] == 1
    assert null["bcit_self_learning_value_score"] == 0
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"].endswith("repeated_null_retire")

    monkeypatch.setattr(
        exp7070,
        "collect_preconditions",
        lambda **_kwargs: (
            passing,
            {},
            {"chronological_stream_manifest": {"events": []}, "exact_outcome_rows": []},
        ),
    )
    monkeypatch.setattr(exp7070, "run_comparison", lambda *_args, **_kwargs: positive_comparison)
    built = exp7070.build_artifact(
        repo_root=tmp_path,
        output_path=tmp_path / "success.json",
        work_root=tmp_path / "success-work",
    )
    assert built["verdict_class"] == "positive"


def test_req_selflearn_7070_validator_defensive_branches(tmp_path: Path) -> None:
    """REQ-SELFLEARN-7070 validates all blocked and complete terminal boundaries."""

    assert exp7070.validate_artifact(None) == ["artifact_object_required"]
    assert exp7070.validate_artifact({})[0].startswith("required_fields_missing:")
    blocked = exp7070.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "blocked.json",
        work_root=tmp_path / "blocked-work",
    )

    mutations = (
        ("verdict_class", "invented", "verdict_class_invalid"),
        ("rows", {}, "rows_not_list"),
        ("harmful_update_rate_by_arm", [], "harmful_update_rate_by_arm_not_mapping"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
        ("bcit_comparison_complete_score", 1, "blocked_scores_must_be_zero"),
        ("chronological_event_rows", [{"unexpected": True}], "blocked_event_rows_must_be_empty"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(blocked)
        changed[field] = value
        changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
        assert expected in exp7070.validate_artifact(changed)

    positive = exp7070.complete_artifact_from_comparison(
        _complete_comparison(positive=True),
        run_date=exp7070.RUN_DATE,
        duration_s=0.01,
        source_hashes={},
        preconditions=[],
    )
    changed = deepcopy(positive)
    changed["bcit_comparison_complete_score"] = 0
    changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
    errors = exp7070.validate_artifact(changed)
    assert "comparison_complete_score_mismatch" in errors
    assert "self_learning_value_score_mismatch" in errors

    changed = deepcopy(positive)
    changed["bcit_self_learning_value_score"] = 0
    changed["reproducibility_checksum"] = exp7070.artifact_checksum(changed)
    assert "self_learning_value_score_mismatch" in exp7070.validate_artifact(changed)


def test_req_selflearn_7070_storage_probe_and_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SELFLEARN-7070 reports storage failures without partial publication."""

    nested = tmp_path / "missing" / "deeper" / "state"
    assert exp7070._nearest_existing_parent(nested) == tmp_path
    monkeypatch.setattr(exp7070.os, "access", lambda *_args: False)
    assert exp7070._isolated_stores_writable(nested) is False
    monkeypatch.undo()

    def fail_store(*_args: object, **_kwargs: object) -> None:
        raise ValueError("forced store failure")

    monkeypatch.setattr(exp7070, "PolicyStore", fail_store)
    assert exp7070._isolated_stores_writable(tmp_path / "stores") is False
    monkeypatch.undo()

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("forced replacement failure")

    monkeypatch.setattr(exp7070.os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced replacement"):
        exp7070._atomic_write(tmp_path / "artifact.json", {"terminal": True})
    assert list(tmp_path.glob(".artifact.json.*")) == []
    monkeypatch.undo()

    monkeypatch.setattr(exp7070, "build_artifact", lambda **_kwargs: {"invalid": True})
    with pytest.raises(ValueError, match="artifact validation failed"):
        exp7070.write_artifact(output_path=tmp_path / "invalid-output.json")
