"""Exp6149 certified strategy schema fixture tests.

Spec refs: REQ-LEARN-6149, REQ-LEARN-6149-1, REQ-LEARN-6149-2,
REQ-LEARN-6149-3, REQ-LEARN-6149-4, REQ-LEARN-6149-5,
REQ-LEARN-6149-6, SCENARIO-LEARN-6149-CALIBRATION-SCHEMA,
SCENARIO-LEARN-6149-SNAPSHOT-TRANSACTION,
SCENARIO-LEARN-6149-IDEMPOTENCE, SCENARIO-LEARN-6149-SAFETY,
SCENARIO-LEARN-6149-RETENTION-EVICTION-PARITY.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6149_certified_strategy_schema_fixture as mod
import carnot._rust as rust


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6149_spec_declares_strategy_schema_contract() -> None:
    """REQ-LEARN-6149: OpenSpec owns the certified strategy fixture contract."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6149") : text.index("## REQ-LEARN-6147")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6149-1",
        "REQ-LEARN-6149-2",
        "REQ-LEARN-6149-3",
        "REQ-LEARN-6149-4",
        "REQ-LEARN-6149-5",
        "REQ-LEARN-6149-6",
        "SCENARIO-LEARN-6149-CALIBRATION-SCHEMA",
        "SCENARIO-LEARN-6149-SNAPSHOT-TRANSACTION",
        "SCENARIO-LEARN-6149-IDEMPOTENCE",
        "SCENARIO-LEARN-6149-SAFETY",
        "SCENARIO-LEARN-6149-RETENTION-EVICTION-PARITY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6149_calibration_schema_certificate_binding_and_admission() -> None:
    """SCENARIO-LEARN-6149-CALIBRATION-SCHEMA: only certified calibration rows commit."""

    events = mod.load_calibration_events()
    replay = mod.replay_certified_strategy_fixture(events=events)
    schema = mod.strategy_schema_version_dimension_and_byte_budget(replay)
    contract = mod.certificate_and_applicability_contract(replay)

    assert events
    assert {event.row["partition"] for event in events} == {"calibration"}
    assert replay["input_partition_counts"] == {"calibration": len(events)}
    assert replay["committed_count"] > 0
    assert replay["rejected_count"] > 0
    assert replay["quarantined_count"] > 0
    assert schema["schema_version"] == mod.STRATEGY_SCHEMA_VERSION
    assert schema["fixed_width_record_bytes"] == mod.FIXED_WIDTH_RECORD_BYTES
    assert schema["record_dimension"] == mod.STRATEGY_RECORD_DIMENSION
    assert schema["max_runtime_state_bytes"] <= mod.STATE_BYTE_BUDGET
    assert schema["free_form_model_text_executable_without_certificate"] is False
    assert contract["all_committed_records_certificate_bound"] is True
    assert contract["all_committed_records_have_task_admission_metadata"] is True
    assert contract["non_calibration_input_count"] == 0
    assert contract["sample_committed_records"]
    for record in contract["sample_committed_records"]:
        assert record["certificate_hash"].startswith("sha256:")
        assert record["applicability_signature"].startswith("sha256:")
        assert record["fixed_width_bytes_len"] == mod.FIXED_WIDTH_RECORD_BYTES
        assert record["task_admission"]["partition"] == "calibration"
        assert record["task_admission"]["admitted"] is True


def test_scenario_6149_decision_snapshots_prepare_commit_abort_and_rollback() -> None:
    """SCENARIO-LEARN-6149-SNAPSHOT-TRANSACTION: writes are post-outcome only."""

    replay = mod.replay_certified_strategy_fixture(events=mod.load_calibration_events()[:24])
    snapshots = mod.decision_snapshot_and_no_same_decision_write_receipts(replay)
    transactions = mod.prepare_commit_abort_and_rollback_receipts(replay)

    assert snapshots["decision_count"] == 24
    assert snapshots["same_decision_read_after_write_count"] == 0
    assert snapshots["snapshot_mutation_count"] == 0
    assert snapshots["all_decisions_used_frozen_snapshot"] is True
    assert transactions["prepare_count"] == 24
    assert transactions["commit_count"] > 0
    assert transactions["abort_count"] > 0
    assert transactions["rollback_count"] >= 1
    assert transactions["rollback_exact"] is True
    assert transactions["all_commits_after_exact_certificate"] is True
    for receipt in transactions["sample_commit_receipts"]:
        assert receipt["before_state_hash"] != receipt["after_state_hash"]
        assert receipt["exact_certificate_visible_at_commit"] is True
        assert receipt["decision_snapshot_hash"].startswith("sha256:")


def test_scenario_6149_duplicate_reordered_restart_and_merge_are_idempotent() -> None:
    """SCENARIO-LEARN-6149-IDEMPOTENCE: stable event IDs make replay deterministic."""

    events = mod.load_calibration_events()
    replay = mod.replay_certified_strategy_fixture(events=events)
    receipt = mod.duplicate_reordered_restart_and_merge_idempotence(events)

    assert receipt["canonical_state_hash"] == replay["final_state_hash"]
    assert receipt["duplicate_delivery_state_hash"] == replay["final_state_hash"]
    assert receipt["reordered_delivery_state_hash"] == replay["final_state_hash"]
    assert receipt["restart_replay_state_hash"] == replay["final_state_hash"]
    assert receipt["merge_state_hash"] == replay["final_state_hash"]
    assert receipt["duplicate_future_retrieval"] == receipt["canonical_future_retrieval"]
    assert receipt["reordered_future_retrieval"] == receipt["canonical_future_retrieval"]
    assert receipt["restart_future_retrieval"] == receipt["canonical_future_retrieval"]
    assert receipt["merge_future_retrieval"] == receipt["canonical_future_retrieval"]
    assert receipt["idempotence_ready"] is True


def test_scenario_6149_safety_controls_fail_closed() -> None:
    """SCENARIO-LEARN-6149-SAFETY: unsafe strategy material never becomes policy."""

    events = mod.load_calibration_events()
    replay = mod.replay_certified_strategy_fixture(events=events)
    controls = mod.poison_invalid_alias_contradiction_and_corruption_controls(replay)

    assert controls["poison"]["quarantined"] > 0
    assert controls["invalid_certificate"]["rejected"] >= 1
    assert controls["malformed_proposal"]["rejected"] > 0
    assert controls["alias"]["counted_as_structural_shift"] == 0
    assert controls["alias"]["accepted"] > 0
    assert controls["contradiction"]["rejected"] >= 1
    assert controls["serialization_corruption"]["rejected"] is True
    assert controls["unsafe_executable_policy_count"] == 0

    state = mod.CertifiedStrategyState(active_capacity=6)
    valid = next(event for event in events if event.control_kind == "normal")
    snapshot = state.decision_snapshot(valid)
    prepared = state.prepare(valid, snapshot)
    with pytest.raises(mod.CertifiedStrategyError, match="certificate"):
        state.commit(valid.with_certificate_hash(mod.sha256_text("bad-certificate")), prepared)

    contradictory = valid.with_action_code(valid.action_code ^ 1)
    state.commit(valid, prepared)
    contrad_snapshot = state.decision_snapshot(contradictory)
    contrad_prepared = state.prepare(contradictory, contrad_snapshot)
    rejected = state.commit(contradictory, contrad_prepared)
    assert rejected["status"] == "aborted"
    assert rejected["reason"] == "contradictory_strategy"


def test_scenario_6149_retention_eviction_serialization_and_rollback() -> None:
    """SCENARIO-LEARN-6149-RETENTION-EVICTION-PARITY: bounded state keeps protected records."""

    replay = mod.replay_certified_strategy_fixture(
        events=mod.load_calibration_events(), active_capacity=5
    )
    metrics = mod.protected_retention_eviction_and_bounded_state_metrics(replay)
    state = mod.CertifiedStrategyState.recover(replay["serialized_state"].encode("utf-8"))
    before_hash = state.state_hash()
    restored = mod.CertifiedStrategyState.recover(state.serialize().encode("utf-8"))

    assert metrics["protected_prefix_retention"] == 1.0
    assert metrics["evicted_count"] > 0
    assert metrics["protected_eviction_count"] == 0
    assert metrics["runtime_state_bytes"] <= mod.STATE_BYTE_BUDGET
    assert restored.state_hash() == before_hash
    with pytest.raises(mod.CertifiedStrategyError, match="corrupt"):
        mod.CertifiedStrategyState.recover(b'{"schema":"wrong"}')

    rollback_hash = replay["rollback_target_hash"]
    state.rollback(rollback_hash)
    assert state.state_hash() == rollback_hash


def test_req_6149_transaction_and_serialization_defensive_branches() -> None:
    """REQ-LEARN-6149-3: transaction and recovery guards fail closed."""

    events = mod.load_calibration_events()
    valid = next(event for event in events if event.control_kind == "normal")

    with pytest.raises(ValueError, match="active_capacity"):
        mod.CertifiedStrategyState(active_capacity=0)
    with pytest.raises(ValueError, match="protected_prefix_count"):
        mod.CertifiedStrategyState(active_capacity=1, protected_prefix_count=2)

    state = mod.CertifiedStrategyState(active_capacity=4)
    snapshot = state.decision_snapshot(valid)
    stale_snapshot = dict(snapshot)
    stale_snapshot["state_hash"] = mod.sha256_text("stale")
    with pytest.raises(mod.CertifiedStrategyError, match="stale decision"):
        state.prepare(valid, stale_snapshot)

    prepared = state.prepare(valid, snapshot)
    stale_prepared = dict(prepared)
    stale_prepared["after_state_hash"] = mod.sha256_text("stale-prepare")
    with pytest.raises(mod.CertifiedStrategyError, match="stale prepare"):
        state.commit(valid, stale_prepared)

    commit_receipt = state.commit(valid, prepared)
    duplicate = state.commit(valid, prepared)
    assert commit_receipt["status"] == "committed"
    assert duplicate["status"] == "duplicate"
    assert duplicate["before_state_hash"] == duplicate["after_state_hash"]

    non_calibration_row = deepcopy(valid.row)
    non_calibration_row["partition"] = "validation"
    non_calibration = mod.StrategyEvent(row=non_calibration_row, outcome=deepcopy(valid.outcome))
    other_state = mod.CertifiedStrategyState(active_capacity=4)
    other_snapshot = other_state.decision_snapshot(non_calibration)
    other_prepared = other_state.prepare(non_calibration, other_snapshot)
    rejected = other_state.commit(non_calibration, other_prepared)
    assert rejected["reason"] == "non_calibration_partition"
    assert rejected["target_bucket"] == "rejected"

    protected_only = mod.CertifiedStrategyState(active_capacity=1, protected_prefix_count=1)
    protected_only.active = [
        {"fixed_width": {"freshness_event_index_u32": 1}, "protected": True, "record_hash": "a"},
        {"fixed_width": {"freshness_event_index_u32": 2}, "protected": True, "record_hash": "b"},
    ]
    assert protected_only._evict_if_needed() == []
    assert len(protected_only.active) == 2

    with pytest.raises(mod.CertifiedStrategyError, match="rollback target"):
        state.rollback(mod.sha256_text("missing"))
    with pytest.raises(mod.CertifiedStrategyError, match="corrupt serialization bytes"):
        mod.CertifiedStrategyState.recover(b"\xff")
    with pytest.raises(mod.CertifiedStrategyError, match="corrupt serialization payload"):
        mod.CertifiedStrategyState.recover(
            json.dumps(
                {
                    "schema": mod.CHECKPOINT_SCHEMA_VERSION,
                    "state": [],
                    "state_hash": mod.sha256_text("bad"),
                }
            ).encode("utf-8")
        )

    serialized = json.loads(state.serialize())
    serialized["history"] = []
    with pytest.raises(mod.CertifiedStrategyError, match="corrupt serialization history"):
        mod.CertifiedStrategyState.recover(json.dumps(serialized).encode("utf-8"))

    corrupted_hash = json.loads(state.serialize())
    corrupted_hash["state"]["version"] += 1
    with pytest.raises(mod.CertifiedStrategyError, match="corrupt serialization hash"):
        mod.CertifiedStrategyState.recover(json.dumps(corrupted_hash).encode("utf-8"))


def test_req_6149_python_rust_pyo3_fixed_width_parity() -> None:
    """REQ-LEARN-6149-6: Python, Rust, and PyO3 agree on strategy record bytes."""

    records = mod.golden_and_adversarial_fixed_width_records()
    assert records
    for record in records:
        py_result = mod.fixed_width_strategy_record_receipt(record)
        rust_result = rust.certified_strategy_schema_record(**record)
        assert rust_result == py_result
        assert bytes.fromhex(py_result["record_bytes_hex"]) == mod.pack_strategy_record(record)
        assert py_result["schema"] == mod.STRATEGY_SCHEMA_VERSION
        assert py_result["record_bytes_len"] == mod.FIXED_WIDTH_RECORD_BYTES
        assert isinstance(py_result["energy"], int)
        assert isinstance(py_result["action_code"], int)


def test_req_6149_artifact_schema_ready_score_and_nonreuse(tmp_path: Path) -> None:
    """REQ-LEARN-6149: terminal artifact is complete, ready, and checksummed."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        result_path=result_path,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
        write=True,
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["certified_strategy_fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["committed_rejected_and_quarantined_counts"]["committed"] > 0
    assert artifact["committed_rejected_and_quarantined_counts"]["rejected"] > 0
    assert artifact["committed_rejected_and_quarantined_counts"]["quarantined"] > 0
    assert artifact["retired_exp5895_scope_nonreuse_receipt"]["nonreuse_confirmed"] is True
    assert artifact["model_weight_immutability_receipt"]["all_unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    bad_score = deepcopy(artifact)
    bad_score["certified_strategy_fixture_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["certified_strategy_fixture_ready_score"] = mod.ready_score(bad_substrate)
    bad_substrate["status"] = mod.status(bad_substrate)
    bad_substrate["honest_verdict"] = mod.honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    assert bad_substrate["certified_strategy_fixture_ready_score"] == 0.0
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)


def test_req_6149_artifact_validation_and_parity_failure_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-LEARN-6149-4: validation, readiness, and parity failures are explicit."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["preconditions_ready"] = False
    blocked["certified_strategy_fixture_ready_score"] = mod.ready_score(blocked)
    blocked["status"] = mod.status(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["certified_strategy_fixture_ready_score"] = mod.ready_score(bad_oracle)
    bad_oracle["status"] = mod.status(bad_oracle)
    bad_oracle["honest_verdict"] = mod.honest_verdict(bad_oracle)
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_partial"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    original_rust_record = rust.certified_strategy_schema_record

    def mismatching_rust_record(**record: Any) -> dict[str, Any]:
        receipt = dict(original_rust_record(**record))
        receipt["energy"] += 1
        return receipt

    monkeypatch.setattr(rust, "certified_strategy_schema_record", mismatching_rust_record)
    mismatch = mod.python_rust_pyo3_fixed_width_parity()
    assert mismatch["all_byte_schema_energy_action_parity"] is False
    assert mismatch["parity_failures"]
    monkeypatch.setattr(rust, "certified_strategy_schema_record", original_rust_record)

    original_import = builtins.__import__

    def raising_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "carnot._rust":
            raise ImportError("forced")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)
    assert mod._rust_strategy_helper_available() is False
    import_failure = mod.python_rust_pyo3_fixed_width_parity()
    assert import_failure["all_byte_schema_energy_action_parity"] is False
    assert import_failure["parity_failures"][0]["error"] == "ImportError"


def test_req_6149_artifact_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-LEARN-6149-4: missing fields and failing commands cannot pass."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_provenance_type: dict[str, Any] = dict(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    failed_command = deepcopy(artifact)
    failed_command["test_exit_codes"][mod.FOCUSED_COMMAND] = 1
    failed_command["certified_strategy_fixture_ready_score"] = mod.ready_score(failed_command)
    failed_command["status"] = mod.status(failed_command)
    failed_command["honest_verdict"] = mod.honest_verdict(failed_command)
    failed_command["reproducibility_checksum"] = mod.reproducibility_checksum(failed_command)
    assert failed_command["certified_strategy_fixture_ready_score"] == 0.0
    assert failed_command["honest_verdict"].startswith("complete_partial:")
    assert mod.validate_artifact(failed_command) is True
