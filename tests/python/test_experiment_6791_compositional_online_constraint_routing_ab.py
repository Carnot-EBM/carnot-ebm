"""Tests for REQ-CL-6791 prospective compositional route learning."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6791_compositional_online_constraint_routing_ab as exp


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the frozen CPU comparison once for all row-level assertions."""

    state_root = tmp_path_factory.mktemp("exp6791-state")
    artifact = exp.run_experiment(state_root=state_root, duration_s=1.25)
    assert artifact["compositional_csl_completed"] is True
    return artifact


def _factor(event_id: str = "event-1", position: int = 0) -> dict:
    return {
        "factor_id": f"factor:order_1:{event_id}",
        "factor_type": "dependency_route_factor",
        "source_event_id": event_id,
        "source_position": position,
        "source_topology_family": "directed_implication_chain",
        "stratum": "easy",
        "motif_id": "motif:easy:balanced_odd:groups_1",
        "target_route": "dependency_suffix",
        "update_target_event_id": event_id,
        "update_target_position": position,
        "evidence_hash": "sha256:" + "1" * 64,
        "exact_provenance": True,
        "placebo_shuffled": False,
    }


def test_required_schema_and_frozen_protocol(complete_artifact: dict) -> None:
    """REQ-CL-6791 freezes the full protocol and principle coverage."""

    artifact = complete_artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert tuple(artifact["arm_definitions"]) == exp.ARMS
    assert artifact["frozen_manifest"]["order_hashes"] == exp.EXPECTED_ORDER_HASHES
    assert artifact["frozen_manifest"]["route_budget"] == 3
    assert artifact["frozen_manifest"]["stopping_rule"]["planned_rows"] == 4_800
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact) == []


def test_blocked_precondition_has_no_reduced_fallback(tmp_path: Path) -> None:
    """SCENARIO-CL-6791-BLOCKED keeps failed checks and emits no rows."""

    artifact = exp.run_experiment(
        state_root=tmp_path / "blocked",
        precondition_overrides={"constraint_routing_stream_ready": False},
        duration_s=0.5,
    )
    assert artifact["status"] == "complete_blocked_online_constraint_routing_ab"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["transaction_receipts"] == []
    assert artifact["frozen_manifest"]["stopping_rule"]["reduced_order_fallback"] is False
    failed = artifact["gate_check_summary"]["failures"]
    assert {row["check"] for row in failed} == {"constraint_routing_stream_ready"}
    assert failed[0]["observed"] is False
    assert exp.validate_artifact(artifact) == []


def test_active_events_are_read_only_and_stores_restart(tmp_path: Path) -> None:
    """SCENARIO-CL-6791-READ-ONLY rejects writes during an active action."""

    store = exp.IsolatedTransactionStore(tmp_path / "store", "compositional_online", "order_1")
    parent = store.state_bytes()
    snapshot = store.begin_event("event-1")
    assert snapshot["state_bytes"] == parent
    with pytest.raises(exp.ReadOnlyEventError):
        store.commit_factor(_factor(), transaction_id="tx-active")
    assert store.state_bytes() == parent
    assert store.active_event_write_violations[0]["rejected"] is True
    store.end_event()

    receipt = store.commit_factor(_factor(), transaction_id="tx-1")
    assert receipt["committed"] is True
    restart = store.restart_receipt()
    assert restart["bytes_match"] is True
    rollback = store.rollback(receipt)
    assert rollback["byte_identical"] is True
    store.reapply(receipt)
    assert store.state_hash() == receipt["new_state_hash"]


def test_arm_isolation_and_frozen_control(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-ARM-ISOLATION gives every arm an owned lineage."""

    rows = complete_artifact["rows"]
    paths_by_order: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        paths_by_order[row["order_id"]].add(row["snapshot"]["state_path"])
        assert row["snapshot"]["owner_arm"] == row["arm"]
        assert row["snapshot"]["owner_order"] == row["order_id"]
    assert all(len(paths) == len(exp.ARMS) for paths in paths_by_order.values())
    assert all(
        row["memory_read_count"] == 0
        and row["memory_write_count"] == 0
        and row["retrieved_factor_ids"] == []
        for row in rows
        if row["arm"] == "frozen_controller"
    )
    assert complete_artifact["active_event_write_violations"] == []


def test_commits_occur_between_events(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-BETWEEN-EVENT-COMMIT exposes writes only later."""

    rows = complete_artifact["rows"]
    by_key = {(row["order_id"], row["arm"], row["position"]): row for row in rows}
    committed = [row for row in rows if row["transaction"]["committed"] is True]
    assert committed
    for row in committed:
        assert row["transaction"]["phase"] == "after_exact_receipt"
        assert row["snapshot"]["version"] + 1 == row["transaction"]["state_version_after"]
        if row["position"] + 1 < 240:
            next_row = by_key[(row["order_id"], row["arm"], row["position"] + 1)]
            assert next_row["snapshot"]["version"] >= row["transaction"]["state_version_after"]


def test_placebo_matches_writes_and_uses_past_targets(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-PLACEBO matches activity with past-only shuffled targets."""

    writes = complete_artifact["writes_by_arm_order"]
    receipts = complete_artifact["transaction_receipts"]
    for order_id in exp.EXPECTED_ORDER_HASHES:
        assert writes["compositional_online"][order_id] == writes["random_update_placebo"][order_id]
        online_bytes = sum(
            row["logical_transaction_bytes"]
            for row in receipts
            if row["order_id"] == order_id and row["arm"] == "compositional_online"
        )
        placebo_bytes = sum(
            row["logical_transaction_bytes"]
            for row in receipts
            if row["order_id"] == order_id and row["arm"] == "random_update_placebo"
        )
        assert online_bytes == placebo_bytes
    placebo = [
        row
        for row in receipts
        if row["arm"] == "random_update_placebo" and row["committed"] is True
    ]
    assert placebo
    assert all(row["placebo_shuffled"] is True for row in placebo)
    assert all(row["update_target_position"] < row["position"] for row in placebo)
    assert all(row["update_target_stratum"] == row["stratum"] for row in placebo)


def test_retrieval_disabled_arm_keeps_exact_writes(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-RETRIEVAL-DISABLE removes reads, not factors."""

    rows = complete_artifact["rows"]
    assert all(
        row["retrieved_factor_ids"] == []
        and row["controller_state"]["retrieval"]["enabled"] is False
        for row in rows
        if row["arm"] == "retrieval_disabled_online"
    )
    receipts = complete_artifact["transaction_receipts"]
    factors: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in receipts:
        if row["committed"]:
            factors[(row["order_id"], row["event_id"])][row["arm"]].add(row["factor_id"])
    paired = [value for value in factors.values() if "compositional_online" in value]
    assert paired
    assert all(
        value["compositional_online"] == value["retrieval_disabled_online"] for value in paired
    )
    assert all(
        complete_artifact["later_reads_by_arm_order"]["retrieval_disabled_online"][order_id] == 0
        for order_id in exp.EXPECTED_ORDER_HASHES
    )


def test_future_receipts_and_held_family_do_not_leak(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-FUTURE-LEAKAGE keeps snapshots past-only."""

    assert complete_artifact["future_feature_violations"] == []
    for row in complete_artifact["rows"]:
        maximum = row["snapshot"]["max_source_position"]
        assert maximum is None or maximum < row["position"]
        assert row["chronology"]["receipt_visible_at_action"] is False
        assert row["chronology"]["actions_fixed_before_receipt"] is True
        if row["position"] == 160:
            assert row["snapshot"]["held_future_factor_count"] == 0


def test_every_order_event_arm_cell_has_one_paired_key(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-PAIRED-KEYS requires all 4,800 cells."""

    rows = complete_artifact["rows"]
    assert len(rows) == exp.PLANNED_ROW_COUNT == 4_800
    assert len({row["row_key"] for row in rows}) == len(rows)
    arms_by_pair: dict[str, set[str]] = defaultdict(set)
    receipt_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        arms_by_pair[row["pair_key"]].add(row["arm"])
        receipt_hashes[row["pair_key"]].add(row["hidden_receipt_hash"])
    assert len(arms_by_pair) == 1_200
    assert all(arms == set(exp.ARMS) for arms in arms_by_pair.values())
    assert all(len(hashes) == 1 for hashes in receipt_hashes.values())


def test_component_and_factor_counterfactuals_are_attributable(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-COMPONENT-ATTRIBUTION uses the same state bytes."""

    online_rows = [row for row in complete_artifact["rows"] if row["arm"] == "compositional_online"]
    assert all(
        set(row["controller_state"]) == {"factor_admission", "retrieval", "route_selection"}
        for row in online_rows
    )
    counterfactuals = [item for row in online_rows for item in row["factor_counterfactuals"]]
    assert counterfactuals
    assert all(item["same_snapshot_bytes"] is True for item in counterfactuals)
    assert any(item["action_changed"] is True for item in counterfactuals)
    assert any(
        row["component_counterfactual_actions"]["without_retrieval"] != row["selected_action"]
        for row in online_rows
    )
    reduced = exp.reduce_evidence(
        complete_artifact["rows"], complete_artifact["transaction_receipts"]
    )
    assert (
        reduced["component_action_attribution"] == complete_artifact["component_action_attribution"]
    )


def test_restart_and_rollback_cover_every_accepted_transaction(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-RESTART-ROLLBACK proves each commit is recoverable."""

    accepted = [row for row in complete_artifact["transaction_receipts"] if row["committed"]]
    assert accepted
    assert all(row["restart_bytes_match"] is True for row in accepted)
    assert all(row["rollback_byte_identical"] is True for row in accepted)
    counts = Counter((row["order_id"], row["arm"]) for row in accepted)
    assert all(
        counts[(order_id, "compositional_online")] > 0 for order_id in exp.EXPECTED_ORDER_HASHES
    )


def test_rows_rederive_metrics_completion_and_verdict(complete_artifact: dict) -> None:
    """SCENARIO-CL-6791-ROW-VERDICT makes rows own every conclusion."""

    artifact = complete_artifact
    reduced = exp.reduce_evidence(artifact["rows"], artifact["transaction_receipts"])
    for field in exp.ROW_DERIVED_FIELDS:
        assert artifact[field] == reduced[field]
    checks = exp.completion_checks(artifact)
    assert all(checks.values())
    positive = exp.positive_credit_checks(artifact)
    expected_class = "positive" if all(positive.values()) else "null"
    assert artifact["verdict_class"] == expected_class
    assert artifact["honest_verdict"].startswith("complete_")

    tampered = deepcopy(artifact)
    tampered["writes_by_arm_order"]["compositional_online"]["order_1"] += 1
    tampered["reproducibility_checksum"] = exp.reproducibility_checksum(tampered)
    assert "row-derived metrics mismatch" in exp.validate_artifact(tampered)


def test_artifact_writer_is_atomic_and_stable(complete_artifact: dict, tmp_path: Path) -> None:
    """REQ-CL-6791 writes the validated terminal artifact without state drift."""

    target = tmp_path / "artifact.json"
    receipt = exp.write_artifact(target, complete_artifact)
    assert receipt["atomic_rename"] is True
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored == complete_artifact
    assert exp.validate_artifact(stored) == []


def test_defensive_storage_and_source_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-6791 fails closed on corrupt ownership, capacity, and source data."""

    cleanup_target = tmp_path / "cleanup" / "state.json"
    real_replace = exp.os.replace

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("forced replace failure")

    monkeypatch.setattr(exp.os, "replace", fail_replace)
    with pytest.raises(OSError):
        exp._atomic_write(cleanup_target, b"value")
    assert list(cleanup_target.parent.glob("*.tmp")) == []
    monkeypatch.setattr(exp.os, "replace", real_replace)

    store = exp.IsolatedTransactionStore(tmp_path / "owned", exp.ONLINE_ARM, "order_1")
    with pytest.raises(ValueError, match="ownership"):
        exp.IsolatedTransactionStore(tmp_path / "owned", exp.PLACEBO_ARM, "order_1")
    store._state["records"] = [_factor(str(index), index) for index in range(exp.MAX_RECORDS)]
    with pytest.raises(ValueError, match="capacity"):
        store.commit_factor(_factor("overflow"), transaction_id="tx-overflow")

    corrupt = exp.IsolatedTransactionStore(tmp_path / "corrupt", exp.ONLINE_ARM, "order_1")
    corrupt.state_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid transaction state"):
        corrupt._read_state()
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact root"):
        exp._load_source(non_object)


def test_precondition_store_failure_and_malformed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-6791-BLOCKED records unavailable stores and bad source bytes."""

    source_path = exp.REPO_ROOT / exp.SOURCE_RELATIVE_PATH
    source = exp._load_source(source_path)
    real_store = exp.IsolatedTransactionStore

    class BrokenStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OSError("forced store failure")

    monkeypatch.setattr(exp, "IsolatedTransactionStore", BrokenStore)
    summary = exp.evaluate_preconditions(
        source,
        source_path=source_path,
        state_root=tmp_path / "broken-stores",
    )
    assert "writable_isolated_transaction_stores" in summary["failed_checks"]
    monkeypatch.setattr(exp, "IsolatedTransactionStore", real_store)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    artifact = exp.run_experiment(
        source_path=malformed,
        state_root=tmp_path / "malformed-state",
        duration_s=0.1,
    )
    assert artifact["verdict_class"] == "blocked"
    assert "source_artifact_sha256" in artifact["gate_check_summary"]["failed_checks"]

    artifact = exp.run_experiment(
        state_root=None,
        precondition_overrides={"constraint_routing_stream_ready": False},
        duration_s=0.1,
    )
    assert artifact["verdict_class"] == "blocked"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.run_experiment(state_root=tmp_path / "bad-date", run_date="2026-08-30")


def test_action_and_audit_defenses_report_exact_violations(
    complete_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-6791 names active writes, future factors, and shared arm state."""

    source = exp._load_source(exp.REPO_ROOT / exp.SOURCE_RELATIVE_PATH)
    order = deepcopy(source["order_definitions"][0])
    order["event_ids"] = order["event_ids"][:1]
    event_id = order["event_ids"][0]
    event_by_id = {
        event_id: next(
            row for row in source["frozen_manifest"]["events"] if row["event_id"] == event_id
        )
    }
    source_rows = {
        (order["order_id"], event_id): next(
            row
            for row in source["rows"]
            if row["order_id"] == order["order_id"] and row["event_id"] == event_id
        )
    }
    real_state_hash = exp.IsolatedTransactionStore.state_hash
    monkeypatch.setattr(exp.IsolatedTransactionStore, "state_hash", lambda self: "sha256:changed")
    _, _, active = exp._run_order(
        order=order,
        event_by_id=event_by_id,
        source_row_by_key=source_rows,
        state_root=tmp_path / "active-audit",
    )
    assert len(active) == len(exp.ARMS)
    monkeypatch.setattr(exp.IsolatedTransactionStore, "state_hash", real_state_hash)

    row = deepcopy(complete_artifact["rows"][0])
    row["position"] = 160
    row["snapshot"]["max_source_position"] = row["position"]
    row["snapshot"]["held_future_factor_count"] = 1
    row["factor_counterfactuals"] = [
        {"snapshot_state_hash": "sha256:wrong", "factor_id": "factor", "action_changed": False}
    ]
    online_receipt = next(
        deepcopy(item)
        for item in complete_artifact["transaction_receipts"]
        if item["committed"] and item["arm"] == exp.ONLINE_ARM
    )
    online_receipt["source_position"] += 1
    placebo_receipt = next(
        deepcopy(item)
        for item in complete_artifact["transaction_receipts"]
        if item["committed"] and item["arm"] == exp.PLACEBO_ARM
    )
    placebo_receipt["update_target_position"] = placebo_receipt["position"]
    future = exp.audit_future_features([row], [online_receipt, placebo_receipt])
    assert {item.split(":", 1)[0] for item in future} == {
        "counterfactual_snapshot",
        "early_held_factor",
        "future_snapshot",
        "placebo_future_target",
        "source_position",
    }

    pair_key = complete_artifact["rows"][0]["pair_key"]
    paired = [deepcopy(item) for item in complete_artifact["rows"] if item["pair_key"] == pair_key]
    paired[0]["snapshot"]["owner_arm"] = "wrong"
    paired[1]["snapshot"]["owner_order"] = "wrong"
    paired[2]["snapshot"]["state_path"] = paired[3]["snapshot"]["state_path"]
    paired[3]["hidden_receipt_hash"] = "sha256:wrong"
    cross_arm = exp.audit_cross_arm_state(paired)
    assert {item.split(":", 1)[0] for item in cross_arm} == {
        "owner_arm",
        "owner_order",
        "shared_state_path",
        "unpaired_receipt",
    }


def test_closed_validation_and_terminal_classes(
    complete_artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-6791-ROW-VERDICT covers every closed terminal outcome."""

    assert exp.bootstrap_lcb([]) == 0.0
    assert exp.terminal_verdict(completed=True, failed_checks=set(), positive=True)[0] == "positive"
    assert exp.terminal_verdict(completed=True, failed_checks=set(), positive=False)[0] == "null"
    assert (
        exp.terminal_verdict(
            completed=False, failed_checks={"future_features_absent"}, positive=False
        )[0]
        == "disqualified"
    )
    assert (
        exp.terminal_verdict(completed=False, failed_checks={"restart_complete"}, positive=False)[0]
        == "partial"
    )

    invalid = {}
    errors = exp.validate_artifact(invalid)
    assert "required field set mismatch" in errors
    assert "non-blocked artifact has no rows" in errors
    with pytest.raises(ValueError):
        exp.write_artifact(tmp_path / "invalid.json", invalid)

    blocked = exp.run_experiment(
        state_root=tmp_path / "blocked-validation",
        precondition_overrides={"constraint_routing_stream_ready": False},
        duration_s=0.1,
    )
    bad_general = dict(blocked)
    bad_general.update(
        {
            "field_principles": {},
            "inference_substrate": "wrong",
            "random_seed": -1,
            "verifier_is_oracle": True,
            "verdict_class": "wrong",
            "honest_verdict": "wrong",
            "reproducibility_checksum": "wrong",
        }
    )
    general_errors = exp.validate_artifact(bad_general)
    assert "field principle coverage mismatch" in general_errors
    assert "inference substrate mismatch" in general_errors
    assert "random seed mismatch" in general_errors
    assert "verifier_is_oracle must be false" in general_errors
    assert "verdict class is outside the closed enum" in general_errors
    assert "honest verdict lacks a terminal prefix" in general_errors
    assert "reproducibility checksum mismatch" in general_errors

    blocked_rows = dict(blocked)
    blocked_rows["rows"] = [{}]
    blocked_rows["transaction_receipts"] = [{}]
    blocked_rows["status"] = "wrong"
    blocked_rows["reproducibility_checksum"] = exp.reproducibility_checksum(blocked_rows)
    blocked_errors = exp.validate_artifact(blocked_rows)
    assert "blocked artifact contains prospective evidence" in blocked_errors
    assert "blocked artifact status mismatch" in blocked_errors

    no_rows = dict(blocked)
    no_rows["verdict_class"] = "null"
    no_rows["status"] = "complete_online_constraint_routing_ab"
    no_rows["reproducibility_checksum"] = exp.reproducibility_checksum(no_rows)
    assert "non-blocked artifact has no rows" in exp.validate_artifact(no_rows)

    wrong_complete = dict(complete_artifact)
    wrong_complete["verdict_class"] = "null"
    wrong_complete["compositional_csl_completed"] = False
    wrong_complete["reproducibility_checksum"] = exp.reproducibility_checksum(wrong_complete)
    wrong_errors = exp.validate_artifact(wrong_complete)
    assert "completion checks mismatch" in wrong_errors
    assert "row-derived verdict mismatch" in wrong_errors

    real_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        exp.run_experiment(
            state_root=tmp_path / "forced-invalid",
            precondition_overrides={"constraint_routing_stream_ready": False},
            duration_s=0.1,
        )
    monkeypatch.setattr(exp, "validate_artifact", real_validate)


def test_main_uses_owned_writer(
    complete_artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """REQ-CL-6791 keeps the CLI on the validated task-owned write path."""

    calls: dict[str, object] = {}
    monkeypatch.setattr(exp, "run_experiment", lambda **kwargs: complete_artifact)

    def fake_write(path: Path, artifact: dict) -> dict:
        calls["path"] = path
        calls["artifact"] = artifact
        return {"atomic_rename": True}

    monkeypatch.setattr(exp, "write_artifact", fake_write)
    absolute = tmp_path / "main.json"
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(absolute)]) == 0
    assert calls == {"path": absolute, "artifact": complete_artifact}
    assert complete_artifact["honest_verdict"] in capsys.readouterr().out
