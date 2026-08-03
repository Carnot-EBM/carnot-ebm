"""Exp5967 delayed-commit memory fixture tests.

Spec refs: REQ-LEARN-5967, REQ-STORE-5967, REQ-HW-5967,
SCENARIO-LEARN-5967-FROZEN-SNAPSHOT,
SCENARIO-LEARN-5967-DELAYED-COMMIT,
SCENARIO-LEARN-5967-CONTROL,
SCENARIO-LEARN-5967-FAIL-CLOSED,
SCENARIO-LEARN-5967-PARITY,
SCENARIO-STORE-5967, SCENARIO-HW-5967.
"""

from __future__ import annotations

from pathlib import Path

from carnot import experiment_5967_delayed_commit_memory_fixture as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
HARDWARE_SPEC = REPO / "openspec/capabilities/hardware/spec.md"


def test_req_5967_specs_declare_delayed_commit_contract() -> None:
    """REQ-LEARN-5967/REQ-STORE-5967/REQ-HW-5967: specs anchor the fixture first."""

    learn = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    hardware = HARDWARE_SPEC.read_text(encoding="utf-8")
    learn_section = learn[learn.index("## REQ-LEARN-5967") : learn.index("## REQ-LEARN-5859")]
    store_section = store[store.index("### REQ-STORE-5967") :]
    hardware_section = hardware[hardware.index("### REQ-HW-5967") :]
    normalized = " ".join(learn_section.split())

    for marker in (
        "REQ-LEARN-5967",
        "SCENARIO-LEARN-5967-FROZEN-SNAPSHOT",
        "SCENARIO-LEARN-5967-DELAYED-COMMIT",
        "SCENARIO-LEARN-5967-CONTROL",
        "SCENARIO-LEARN-5967-FAIL-CLOSED",
        "SCENARIO-LEARN-5967-PARITY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.TRACE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in learn_section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in learn_section
        assert " ".join(principle.split()) in normalized
    assert "REQ-STORE-5967" in store_section
    assert "SCENARIO-STORE-5967" in store_section
    assert "REQ-HW-5967" in hardware_section
    assert "SCENARIO-HW-5967" in hardware_section


def test_scenario_5967_frozen_snapshot_and_delayed_commit_visibility() -> None:
    """SCENARIO-LEARN-5967-FROZEN-SNAPSHOT: base reads stay immutable."""

    memory = mod.DelayedCommitMemory(active_capacity=3, quarantine_capacity=3)
    base_version = memory.version
    event = mod.demo_events()[0]
    snapshot = memory.read_snapshot(base_version)
    proposal = memory.propose(base_version, event)

    assert proposal["accepted"] is True
    assert proposal["label_visible"] is False
    assert memory.lookup(snapshot["snapshot_id"], event["key"])["hit"] is False

    validation = memory.validate(proposal["proposal_id"], [event])
    assert validation["accepted"] is True
    assert validation["label_visible"] is True
    assert memory.lookup(snapshot["snapshot_id"], event["key"])["hit"] is False
    assert memory.lookup(memory.read_snapshot(memory.version)["snapshot_id"], event["key"])["hit"] is False

    commit = memory.commit(proposal["proposal_id"])
    assert commit["accepted"] is True
    assert memory.lookup(memory.read_snapshot(memory.version)["snapshot_id"], event["key"])["hit"] is True
    assert memory.lookup(snapshot["snapshot_id"], event["key"])["hit"] is False
    assert memory.read_snapshot(base_version)["state_hash"] == snapshot["state_hash"]


def test_scenario_5967_conflicts_quarantine_supersede_rollback_and_capacity() -> None:
    """SCENARIO-LEARN-5967-FAIL-CLOSED: unsafe lifecycle edges do not propagate."""

    memory = mod.DelayedCommitMemory(active_capacity=2, quarantine_capacity=2)
    events = mod.demo_events()
    first = memory.propose(memory.version, events[0])
    memory.validate(first["proposal_id"], [events[0]])
    memory.commit(first["proposal_id"])
    after_first = memory.state_hash()

    stale = memory.propose(0, events[3])
    memory.validate(stale["proposal_id"], [events[3]])
    stale_commit = memory.commit(stale["proposal_id"])
    assert stale_commit["code"] == "STALE_BASE_CONFLICT"
    assert memory.state_hash() == stale_commit["previous_state_hash"]

    replacement = memory.propose(memory.version, events[3])
    memory.validate(replacement["proposal_id"], [events[3]])
    supersede = memory.supersede(replacement["proposal_id"])
    assert supersede["accepted"] is True
    assert memory.active_keys() == [events[3]["key"]]

    poison = memory.propose(memory.version, events[2])
    memory.validate(poison["proposal_id"], [events[2]])
    assert memory.commit(poison["proposal_id"])["code"] == "COMMIT_ACTION_NOT_PROMOTE"
    assert memory.quarantine(poison["proposal_id"])["accepted"] is True
    assert memory.lookup(memory.read_snapshot(memory.version)["snapshot_id"], events[2]["key"])["hit"] is False

    no_target = memory.propose(memory.version, events[4])
    memory.validate(no_target["proposal_id"], [events[4]])
    assert memory.supersede(no_target["proposal_id"])["code"] == "NO_ACTIVE_TARGET"

    for event in events[4:7]:
        proposal = memory.propose(memory.version, event)
        memory.validate(proposal["proposal_id"], [event])
        memory.commit(proposal["proposal_id"])
    assert len(memory.active_keys()) <= 2
    rollback = memory.rollback(after_first)
    assert rollback["accepted"] is True
    assert memory.state_hash() == after_first


def test_scenario_5967_matched_write_through_control_contract() -> None:
    """SCENARIO-LEARN-5967-CONTROL: the coupled arm differs only in visibility timing."""

    receipt = mod.matched_write_through_control_receipt()
    delayed = receipt["production_delayed_commit"]
    control = receipt["same_event_write_through_control"]

    assert receipt["matched_capacity_retrieval_order_and_compute"] is True
    assert delayed["policy_label"] == "production_delayed_commit"
    assert control["policy_label"] == "coupled_same_event_write_through_control"
    assert delayed["same_event_visible_write_count"] == 0
    assert control["same_event_visible_write_count"] == delayed["event_count"]
    assert delayed["compute_accounting"] == control["compute_accounting"]


def test_scenario_5967_backend_trace_parity_and_fixed_width_trace(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5967-PARITY/SCENARIO-HW-5967: trace receipts match exactly."""

    trace_path = tmp_path / "trace.jsonl"
    parity = mod.python_rust_pyo3_trace_parity_receipt(trace_path=trace_path)

    assert parity["backends"] == ["python", "rust", "pyo3"]
    assert parity["all_operation_version_return_and_hash_parity"] is True
    assert parity["parity_failures"] == []
    assert trace_path.is_file()
    assert mod.sha256_file(trace_path) == parity["fixed_width_trace_hash"]
    assert parity["hardware_execution_claimed"] is False


def test_req_5967_artifact_schema_ready_score_and_reproducibility(tmp_path: Path) -> None:
    """REQ-LEARN-5967: artifact has every required field and a stable checksum."""

    result_path = tmp_path / "experiment_5967.json"
    trace_path = tmp_path / "experiment_5967.trace.jsonl"
    artifact = mod.run(
        result_path=result_path,
        trace_path=trace_path,
        duration_s=0.0,
        test_commands=mod.DEFAULT_TEST_COMMANDS,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )

    assert result_path.is_file()
    assert trace_path.is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["rejected_update_non_propagation_count"] == 0
    assert artifact["delayed_commit_fixture_ready_score"] == 1.0
    assert artifact["fixed_width_operation_trace_path_and_hash"]["sha256"] == mod.sha256_file(trace_path)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
