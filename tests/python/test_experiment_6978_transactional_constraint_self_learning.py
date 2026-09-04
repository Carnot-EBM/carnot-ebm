"""Tests for REQ-LEARN-6978 transactional constraint self-learning."""

from __future__ import annotations

from base64 import b64encode
import builtins
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6978_transactional_constraint_self_learning as exp6978
from carnot.experiment_6978_transactional_constraint_self_learning import (
    EXPECTED_EXP6967_SHA256,
    REQUIRED_ARTIFACT_FIELDS,
    build_blocked_artifact,
    build_plan,
    completion_score,
    collect_preconditions,
    main,
    positive_score,
    reduce_run,
    resolve_model_specs,
    run_plan,
    validate_artifact,
)
from carnot.learning import constraint_policy_store as policy_store_module
from carnot.learning.constraint_policy_store import (
    ForcedInterruption,
    JournalIntegrityError,
    PolicyStore,
    canonical_bytes,
    sha256_bytes,
)
from carnot.learning.continual_constraint_memory import (
    ARMS,
    build_prompt,
    build_update_proposal,
    build_writer_input,
    classify_error,
    find_forbidden_paths,
    initial_memory_records,
    lookup_memory,
    replay_safety_score,
    visible_predecessors,
)


MODEL_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "model_path": "/cache/qwen.gguf",
    "gpu_indices": [0, 1],
    "headline_eligible": True,
}


def _fixture() -> dict:
    events = []
    prompts = []
    witnesses = []
    predecessor_hash = "sha256:" + "0" * 64
    for ordinal in range(24):
        pair_id = f"pair-{ordinal}"
        event_hash = "sha256:" + f"{ordinal + 1:064x}"
        prompt_hash = "sha256:" + f"{ordinal + 101:064x}"
        family = ("boolean_cardinality", "bounded_integer_linear")[ordinal % 2]
        events.append(
            {
                "event_id": f"event-{ordinal}",
                "event_ordinal": ordinal,
                "event_hash": event_hash,
                "pair_id": pair_id,
                "formulation_family": family,
                "prompt_record_hash": prompt_hash,
                "predecessor_event_id": None if ordinal == 0 else f"event-{ordinal - 1}",
                "predecessor_record_hash": predecessor_hash,
                "later_outcome_exists": False,
            }
        )
        predecessor_hash = event_hash
        source = {
            "schema_version": "carnot.bounded_optimization_formulation.v1",
            "variables": [
                {
                    "name": "x",
                    "kind": "integer",
                    "domain": {"lower": "0", "upper": "1"},
                    "universe": [-1, 0, 1, 2],
                }
            ],
            "constraints": [{"terms": {"x": "1"}, "op": "<=", "rhs": "1"}],
            "objective": {
                "direction": "min",
                "expression": {"kind": "linear", "terms": {"x": "1"}, "constant": "0"},
            },
        }
        target = deepcopy(source)
        target["variables"][0]["name"] = "y"
        target["constraints"][0]["terms"] = {"y": "1"}
        target["objective"]["expression"]["terms"] = {"y": "1"}
        prompts.append(
            {
                "record_id": f"chronological:{pair_id}",
                "split": "chronological",
                "pair_id": pair_id,
                "formulation_family": family,
                "prompt_record_hash": prompt_hash,
                "source_formulation": source,
                "target_formulation": target,
            }
        )
        witnesses.append(
            {
                "subject_kind": "slice_pair",
                "split": "chronological",
                "pair_id": pair_id,
                "expected_label": "equivalent",
                "certificate_hash": "sha256:" + f"{ordinal + 201:064x}",
            }
        )
    return {
        "chronological_event_stream_ready_score": 1,
        "error_fixture_ready_score": 1,
        "chronological_event_rows": events,
        "prompt_visible_rows": prompts,
        "exact_witness_rows": witnesses,
        "split_hashes": {"chronological": "sha256:" + "a" * 64},
        "sealed_label_hashes": {"chronological": "sha256:" + "b" * 64},
    }


def _selection() -> dict:
    return {
        "selected_policy_ready_score": 1,
        "selected_policy_hash": "sha256:" + "c" * 64,
        "selected_policy": {
            "schedule_id": "direct",
            "selection_frozen": True,
            "selection_label_split": "calibration",
        },
    }


def _upstream(score_name: str) -> dict:
    return {score_name: 1}


def _proposal(ordinal: int = 0) -> dict:
    writer_input = build_writer_input(
        atomic_error_class="parse:malformed_json",
        exact_certificate_digest="sha256:" + "d" * 64,
        schedule_metadata={"schedule_id": "direct", "token_cap": 128},
        outcome="non_equivalent",
    )
    return build_update_proposal(
        writer_input,
        event_id=f"event-{ordinal}",
        event_ordinal=ordinal,
        formulation_family="boolean_cardinality",
    )


def _complete_null_run() -> dict:
    """Build a complete evidence row set for orchestration boundary tests."""

    rows = [
        {
            "event_id": f"event-{ordinal}",
            "event_ordinal": ordinal,
            "arm": arm,
            "terminal": True,
            "model_id": MODEL_SPEC["hf_id"],
            "random_seed": 6_978_202_609_04 + ordinal,
            "token_cap": 128,
            "exact_success": False,
            "parse_success": False,
            "error_class": "parse:malformed_json",
            "prompt_hash": f"prompt:{ordinal}:{arm}",
            "raw_completion_hash": f"raw:{ordinal}:{arm}",
            "state_hash_after": f"state:{arm}",
            "prompt_tokens": 10,
            "completion_tokens": 8,
        }
        for ordinal in range(24)
        for arm in ARMS
    ]
    return {
        "rows": rows,
        "prompt_visibility_rows": [
            {
                "event_id": row["event_id"],
                "arm": row["arm"],
                "passed": True,
                "forbidden_paths": [],
            }
            for row in rows
        ],
        "exact_outcome_rows": [],
        "memory_lookup_rows": [],
        "update_proposal_rows": [],
        "transaction_journal_rows": [{"phase": "prepare", "transaction_id": "fixture"}],
        "commit_rows": [],
        "rollback_rows": [{"fixture": "harmful_update", "passed": True}],
        "restart_recovery_rows": [{"fixture": "restart", "passed": True}],
        "checkpoint_rows": [],
        "live_duration_s": 0.5,
        "final_state_bytes": 512,
        "journal_replay_passed": True,
        "initial_state_hashes": {arm: f"initial:{arm}" for arm in ARMS},
        "store_paths": {arm: f"/private/{arm}" for arm in ARMS},
    }


def test_req_learn_6978_chronology_and_no_future_visibility() -> None:
    """SCENARIO-LEARN-6978-CHRONOLOGY and NO-FUTURE expose only t-1."""

    events = _fixture()["chronological_event_rows"]
    assert visible_predecessors(events, 0) == []
    assert [row["event_ordinal"] for row in visible_predecessors(events, 7)] == list(range(7))
    assert find_forbidden_paths({"safe": {"future_labels": [1]}}) == ["safe.future_labels"]
    assert find_forbidden_paths({"safe": {"event_id": "event-1"}}) == []
    with pytest.raises(ValueError, match="complete chronological"):
        visible_predecessors([events[1]], 0)
    with pytest.raises(ValueError, match="outside the frozen stream"):
        visible_predecessors(events, 24)


def test_req_learn_6978_writer_input_is_information_separated() -> None:
    """SCENARIO-LEARN-6978-NO-FUTURE limits post-outcome writer evidence."""

    writer_input = build_writer_input(
        atomic_error_class="objective:order",
        exact_certificate_digest="sha256:" + "e" * 64,
        schedule_metadata={"schedule_id": "direct", "token_cap": 128},
        outcome="equivalent",
    )
    assert set(writer_input) == {
        "atomic_error_class",
        "exact_certificate_digest",
        "schedule_metadata",
        "outcome",
    }
    assert find_forbidden_paths(writer_input) == []
    with pytest.raises(ValueError, match="forbidden writer fields"):
        build_writer_input(
            atomic_error_class="parse:malformed_json",
            exact_certificate_digest="sha256:" + "e" * 64,
            schedule_metadata={"schedule_id": "direct", "model_confidence": 0.9},
            outcome="non_equivalent",
        )
    with pytest.raises(ValueError, match="forbidden writer fields"):
        build_update_proposal(
            {**writer_input, "future_label": "equivalent"},
            event_id="event-1",
            event_ordinal=1,
            formulation_family="boolean_cardinality",
        )


def test_req_learn_6978_memory_lookup_and_error_classification() -> None:
    """REQ-LEARN-6978 uses bounded prior policies and atomic error classes."""

    records = initial_memory_records()
    hits = lookup_memory(records, formulation_family="boolean_cardinality", limit=8)
    assert hits
    assert all(row["scope"] in {"global", "boolean_cardinality"} for row in hits)
    assert classify_error({"parse_success": False, "parse_reason": "malformed_json"}) == (
        "parse:malformed_json"
    )
    assert classify_error({"parse_success": True, "schema_outcome": "rejected"}) == (
        "schema:rejected"
    )
    assert (
        classify_error(
            {
                "parse_success": True,
                "schema_outcome": "valid",
                "domain_correspondence_outcome": "failed",
            }
        )
        == "domain_correspondence"
    )
    assert (
        classify_error(
            {
                "parse_success": True,
                "schema_outcome": "valid",
                "domain_correspondence_outcome": "passed",
                "objective_direction_outcome": "failed",
            }
        )
        == "objective:direction"
    )
    assert (
        classify_error(
            {
                "parse_success": True,
                "schema_outcome": "valid",
                "domain_correspondence_outcome": "passed",
                "objective_direction_outcome": "passed",
                "objective_order_outcome": "failed",
            }
        )
        == "objective:order"
    )
    assert classify_error({"exact_success": False, "objective_order_outcome": "passed"}) == (
        "exact:relation"
    )
    assert classify_error({"exact_success": True}) == "none"


def test_req_learn_6978_prompt_uses_only_frozen_public_data() -> None:
    """SCENARIO-LEARN-6978-CHRONOLOGY binds memory without outcome labels."""

    fixture = _fixture()
    prompt, audit = build_prompt(
        event=fixture["chronological_event_rows"][2],
        pair=fixture["prompt_visible_rows"][2],
        memory_records=initial_memory_records(),
        schedule_id="direct",
    )
    assert "CONSTRAINT_IR_JSON_SCHEMA" in prompt
    assert "POLICY_MEMORY" in prompt
    assert audit["forbidden_paths"] == []
    assert audit["prompt_hash"] == sha256_bytes(prompt.encode())
    with pytest.raises(ValueError, match="selected direct schedule"):
        build_prompt(
            event=fixture["chronological_event_rows"][2],
            pair=fixture["prompt_visible_rows"][2],
            memory_records=[],
            schedule_id="legacy",
        )
    leaky_pair = deepcopy(fixture["prompt_visible_rows"][2])
    leaky_pair["source_formulation"]["future_labels"] = ["equivalent"]
    with pytest.raises(ValueError, match="forbidden prompt fields"):
        build_prompt(
            event=fixture["chronological_event_rows"][2],
            pair=leaky_pair,
            memory_records=[],
            schedule_id="direct",
        )


def test_req_learn_6978_journal_commit_is_durable_and_bounded(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-JOURNAL writes prepare before atomic commit."""

    store = PolicyStore(
        tmp_path / "write",
        initial_records=initial_memory_records(),
        max_state_bytes=8_192,
    )
    parent_hash = store.state_hash
    receipt = store.commit(_proposal())
    assert receipt["committed"] is True
    assert receipt["parent_state_hash"] == parent_hash
    assert store.state_hash == receipt["new_state_hash"]
    assert store.state_size <= 8_192
    assert [row["phase"] for row in store.journal_rows()] == ["prepare", "commit"]
    assert all(row["file_fsync"] is True for row in store.journal_rows())
    replay = store.replay_journal()
    assert replay["passed"] is True
    assert replay["final_state_hash"] == store.state_hash


def test_req_learn_6978_read_only_store_and_arm_isolation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-ARM-ISOLATION keeps three stores private."""

    initial = initial_memory_records()
    stores = {
        arm: PolicyStore(
            tmp_path / arm,
            initial_records=initial,
            max_state_bytes=8_192,
            read_only=arm != "transactional_write",
        )
        for arm in ARMS
    }
    before = {arm: store.state_hash for arm, store in stores.items()}
    with pytest.raises(PermissionError, match="read-only"):
        stores["read_only"].commit(_proposal())
    stores["transactional_write"].commit(_proposal())
    assert stores["frozen"].state_hash == before["frozen"]
    assert stores["read_only"].state_hash == before["read_only"]
    assert stores["transactional_write"].state_hash != before["transactional_write"]
    assert len({str(store.root) for store in stores.values()}) == 3


def test_req_learn_6978_rollback_restores_exact_parent_bytes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-ROLLBACK restores the byte-exact parent."""

    store = PolicyStore(
        tmp_path / "write",
        initial_records=initial_memory_records(),
        max_state_bytes=8_192,
    )
    parent = store.state_bytes
    receipt = store.commit(_proposal())
    rollback = store.rollback(parent, transaction_id=receipt["transaction_id"], reason="fixture")
    assert rollback["rolled_back"] is True
    assert store.state_bytes == parent
    assert rollback["restored_state_hash"] == sha256_bytes(parent)
    assert store.replay_journal()["passed"] is True


def test_req_learn_6978_forced_interruption_recovers_prepare(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-RESTART preserves the last committed state."""

    root = tmp_path / "write"
    store = PolicyStore(root, initial_records=initial_memory_records(), max_state_bytes=8_192)
    parent = store.state_bytes
    with pytest.raises(ForcedInterruption):
        store.commit(_proposal(), interrupt_after_prepare=True)
    restarted = PolicyStore(root, initial_records=initial_memory_records(), max_state_bytes=8_192)
    assert restarted.state_bytes == parent
    assert restarted.recovery_rows[-1]["incomplete_prepare_recovered"] is True
    assert restarted.journal_rows()[-1]["phase"] == "abort_recovered"


def test_req_learn_6978_journal_tamper_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-JOURNAL rejects a changed journal chain."""

    root = tmp_path / "write"
    store = PolicyStore(root, initial_records=initial_memory_records(), max_state_bytes=8_192)
    store.commit(_proposal())
    lines = store.journal_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["proposal"]["policy_text"] = "tampered"
    lines[0] = json.dumps(row, sort_keys=True)
    store.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="journal row hash"):
        store.replay_journal()


def test_req_learn_6978_store_rejects_invalid_state_and_storage_limits(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-JOURNAL rejects invalid or oversized state bytes."""

    with pytest.raises(ValueError, match="initial policy state exceeds"):
        PolicyStore(tmp_path / "tiny", initial_records=initial_memory_records(), max_state_bytes=1)

    bad_schema = tmp_path / "bad-schema"
    store = PolicyStore(bad_schema, initial_records=[], max_state_bytes=8_192)
    store.state_path.write_text('{"schema":"wrong","records":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid policy state schema"):
        PolicyStore(bad_schema, initial_records=[], max_state_bytes=8_192)

    bad_records = tmp_path / "bad-records"
    store = PolicyStore(bad_records, initial_records=[], max_state_bytes=8_192)
    store.state_path.write_text(
        json.dumps({"schema": policy_store_module.STATE_SCHEMA, "records": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid policy records"):
        PolicyStore(bad_records, initial_records=[], max_state_bytes=8_192)

    protected = initial_memory_records()
    initial_size = len(
        canonical_bytes(
            {"schema": policy_store_module.STATE_SCHEMA, "version": 0, "records": protected}
        )
    )
    bounded = PolicyStore(
        tmp_path / "bounded",
        initial_records=protected,
        max_state_bytes=initial_size + 16,
    )
    with pytest.raises(ValueError, match="proposal exceeds byte limit"):
        bounded.commit({**_proposal(), "policy_text": "x" * 1_000})


def test_req_learn_6978_store_evicts_only_older_learned_rows(tmp_path: Path) -> None:
    """REQ-LEARN-6978 bounds state by evicting old learned rows, not the proposal."""

    old = {**_proposal(1), "policy_key": "learned:old", "policy_text": "x" * 300}
    initial = [*initial_memory_records(), old]
    initial_size = len(
        canonical_bytes(
            {"schema": policy_store_module.STATE_SCHEMA, "version": 0, "records": initial}
        )
    )
    store = PolicyStore(
        tmp_path / "evict",
        initial_records=initial,
        max_state_bytes=initial_size + 40,
    )
    proposal = {**_proposal(2), "policy_key": "learned:new", "policy_text": "y" * 300}
    store.commit(proposal)
    keys = {row["policy_key"] for row in store.records()}
    assert "learned:old" not in keys
    assert "learned:new" in keys


def test_req_learn_6978_store_recovery_after_published_state(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-RESTART recovers state published before commit receipt."""

    root = tmp_path / "published"
    store = PolicyStore(root, initial_records=initial_memory_records(), max_state_bytes=8_192)
    original_journal = store._journal_row

    def fail_commit(phase: str, transaction_id: str, **fields: object) -> dict:
        if phase == "commit":
            raise RuntimeError("forced commit receipt interruption")
        return original_journal(phase, transaction_id, **fields)

    store._journal_row = fail_commit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="commit receipt interruption"):
        store.commit(_proposal())
    restarted = PolicyStore(root, initial_records=initial_memory_records(), max_state_bytes=8_192)
    assert restarted.recovery_rows[-1]["phase"] == "commit_recovered"
    assert restarted.recovery_rows[-1]["applied"] is True
    assert restarted.replay_journal()["passed"] is True


def test_req_learn_6978_store_rejects_bad_rollback_and_recovery(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-ROLLBACK fails closed on foreign state bytes."""

    readonly = PolicyStore(
        tmp_path / "readonly",
        initial_records=[],
        max_state_bytes=8_192,
        read_only=True,
    )
    with pytest.raises(PermissionError, match="read-only"):
        readonly.rollback(b"{}", transaction_id="x", reason="test")

    store = PolicyStore(tmp_path / "write", initial_records=[], max_state_bytes=8_192)
    with pytest.raises(ValueError, match="invalid schema"):
        store.rollback(b"{}", transaction_id="x", reason="test")

    with pytest.raises(ForcedInterruption):
        store.commit(_proposal(), interrupt_after_prepare=True)
    foreign = {
        "schema": policy_store_module.STATE_SCHEMA,
        "version": 99,
        "records": [],
    }
    store.state_path.write_bytes(canonical_bytes(foreign))
    with pytest.raises(JournalIntegrityError, match="incomplete transaction state mismatch"):
        PolicyStore(store.root, initial_records=[], max_state_bytes=8_192)


def test_req_learn_6978_store_replay_detects_semantic_journal_corruption(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-JOURNAL checks transaction meaning after hash integrity."""

    store = PolicyStore(tmp_path / "semantic", initial_records=[], max_state_bytes=8_192)
    state_hash = store.state_hash
    state_b64 = b64encode(store.state_bytes).decode("ascii")
    store._journal_row(
        "prepare",
        "first",
        parent_state_hash=state_hash,
        new_state_hash="sha256:new-first",
        parent_state_b64=state_b64,
        new_state_b64=state_b64,
        proposal={},
    )
    store._journal_row(
        "prepare",
        "second",
        parent_state_hash="sha256:wrong-parent",
        new_state_hash="sha256:new-second",
        parent_state_b64=state_b64,
        new_state_b64=state_b64,
        proposal={},
    )
    store._journal_row("commit", "missing", new_state_hash="sha256:none")
    store._journal_row("abort_recovered", "missing-abort")
    store._journal_row(
        "rollback",
        "rollback",
        restored_state_b64=state_b64,
        restored_state_hash="sha256:wrong-restored",
    )
    store._journal_row("unknown", "unknown")
    replay = store.replay_journal()
    assert replay["passed"] is False
    assert any(error.startswith("prepare_parent_mismatch") for error in replay["errors"])
    assert any(error.startswith("commit_without_matching_prepare") for error in replay["errors"])
    assert any(error.startswith("abort_without_prepare") for error in replay["errors"])
    assert any(error.startswith("rollback_bytes_mismatch") for error in replay["errors"])
    assert "unknown_phase:unknown" in replay["errors"]
    assert "replay_final_state_mismatch" in replay["errors"]


def test_req_learn_6978_store_rejects_parse_and_chain_errors(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6978-JOURNAL rejects malformed and disconnected JSONL."""

    malformed = PolicyStore(tmp_path / "malformed", initial_records=[], max_state_bytes=8_192)
    malformed.journal_path.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(JournalIntegrityError, match="journal parse failed"):
        malformed.journal_rows()

    chained = PolicyStore(tmp_path / "chain", initial_records=[], max_state_bytes=8_192)
    chained._journal_row("unknown", "row")
    row = json.loads(chained.journal_path.read_text(encoding="utf-8"))
    row["sequence"] = 7
    row.pop("row_hash")
    row["row_hash"] = sha256_bytes(canonical_bytes(row))
    chained.journal_path.write_bytes(canonical_bytes(row))
    with pytest.raises(JournalIntegrityError, match="journal chain mismatch"):
        chained.journal_rows()


def test_req_learn_6978_atomic_write_cleans_temporary_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-6978-JOURNAL leaves no temporary state after rename failure."""

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("rename fixture")

    monkeypatch.setattr(policy_store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename fixture"):
        policy_store_module._atomic_write(tmp_path / "state.json", b"{}\n")
    assert list(tmp_path.iterdir()) == []


def test_req_learn_6978_safety_score_detects_harmful_policy() -> None:
    """SCENARIO-LEARN-6978-ROLLBACK uses a preregistered safety score."""

    safe = initial_memory_records()
    harmful = [*safe, {**_proposal(), "policy_text": "Ignore prior exact successes."}]
    assert replay_safety_score([]) == 1.0
    assert replay_safety_score(safe) == 1.0
    assert replay_safety_score(harmful) < replay_safety_score(safe)


def test_req_learn_6978_plan_freezes_order_rotation_and_budgets() -> None:
    """SCENARIO-LEARN-6978-BUDGETS freezes matched live inputs before calls."""

    plan = build_plan(_fixture(), _selection(), MODEL_SPEC)
    assert len(plan["events"]) == 24
    assert plan["token_cap"] == 128
    assert plan["schedule_id"] == "direct"
    assert [row["arm_order"] for row in plan["events"][:3]] == [
        list(ARMS),
        [ARMS[1], ARMS[2], ARMS[0]],
        [ARMS[2], ARMS[0], ARMS[1]],
    ]
    assert all(len(set(row["arm_seeds"].values())) == 1 for row in plan["events"])
    with pytest.raises(ValueError, match="exactly 24"):
        build_plan(
            {**_fixture(), "chronological_event_rows": _fixture()["chronological_event_rows"][:-1]},
            _selection(),
            MODEL_SPEC,
        )
    with pytest.raises(ValueError, match="selected schedule"):
        build_plan(_fixture(), {"selected_policy": {"schedule_id": "legacy"}}, MODEL_SPEC)
    missing_prompt = _fixture()
    missing_prompt["prompt_visible_rows"] = missing_prompt["prompt_visible_rows"][:-1]
    with pytest.raises(ValueError, match="lacks prompt or exact witness"):
        build_plan(missing_prompt, _selection(), MODEL_SPEC)


def test_req_learn_6978_model_resolution_and_exact_executor(tmp_path: Path) -> None:
    """REQ-LEARN-6978 selects exact Qwen and keeps exact evaluation independent."""

    assert resolve_model_specs(lambda **_kwargs: [])[0]["model_path"] == ""
    selected = resolve_model_specs(
        lambda **_kwargs: [
            {"hf_id": "legacy/model", "model_path": "/legacy", "gpu": 0},
            {**MODEL_SPEC, "gpu": 0},
        ]
    )
    assert selected == [
        {
            **MODEL_SPEC,
            "gpu_indices": [0, 1],
            "resolution_method": "cached_sota_pair(gpu_indices=(0, 1))",
        }
    ]
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"gguf")
    assert exp6978._sha256_path(model_file) == sha256_bytes(b"gguf")
    assert exp6978._sha256_path(tmp_path / "absent") is None

    fixture = _fixture()
    raw = json.dumps(
        {
            "schema_version": "carnot.constraint_ir.mapping.v1",
            "variable_map": [{"source": "x", "target": "y", "scale": "1", "offset": "0"}],
            "objective_map": {"direction": "same", "scale": "1", "offset": "0"},
        }
    )
    outcome = exp6978.evaluate_completion(
        raw,
        fixture["chronological_event_rows"][0],
        fixture["prompt_visible_rows"][0],
        "equivalent",
    )
    assert outcome["terminal"] is True
    assert outcome["exact_success"] is True
    assert outcome["exact_certificate_digest"].startswith("sha256:")


def test_req_learn_6978_run_plan_produces_72_matched_terminal_rows(tmp_path: Path) -> None:
    """REQ-LEARN-6978 runs three matched arms and commits only after outcomes."""

    plan = build_plan(_fixture(), _selection(), MODEL_SPEC)

    def generate(row: dict, prompt: str) -> dict:
        return {
            "raw_completion": '{"schema_version":"carnot.constraint_ir.mapping.v1"}',
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 8,
            "context_id": f"context:{row['event_ordinal']}:{row['arm']}",
            "latency_ms": 2.0,
        }

    def evaluate(raw: str, event: dict, pair: dict, expected_label: str) -> dict:
        del raw, pair, expected_label
        ordinal = int(event["event_ordinal"])
        arm = str(event["active_arm"])
        successes = {
            "frozen": {6},
            "read_only": {6, 7},
            "transactional_write": {6, 7, 8, 9},
        }
        success = ordinal in successes[arm]
        return {
            "parse_success": success,
            "schema_outcome": "valid" if success else "rejected",
            "domain_correspondence_outcome": "passed" if success else "not_evaluated",
            "objective_direction_outcome": "passed" if success else "not_evaluated",
            "objective_order_outcome": "passed" if success else "not_evaluated",
            "certified_relation": "equivalent" if success else None,
            "exact_success": success,
            "exact_certificate_digest": "sha256:" + f"{ordinal + 401:064x}",
            "terminal": True,
        }

    run = run_plan(plan, state_root=tmp_path / "state", generate_fn=generate, evaluate_fn=evaluate)
    assert len(run["rows"]) == 72
    assert all(row["terminal"] is True for row in run["rows"])
    assert all(row["outcome_sequence"] > row["raw_sequence"] for row in run["rows"])
    assert all(
        row["commit_sequence"] is None or row["commit_sequence"] > row["outcome_sequence"]
        for row in run["rows"]
    )
    assert run["restart_recovery_rows"]
    assert any(row["fixture"] == "forced_interruption" for row in run["restart_recovery_rows"])
    assert any(row["fixture"] == "harmful_update" for row in run["rollback_rows"])
    artifact = reduce_run(
        plan,
        run,
        run_date="20260904",
        duration_s=1.0,
        live_duration_s=0.2,
        source_artifact_hashes={"fixture": "sha256:" + "1" * 64},
        model_file_hashes={MODEL_SPEC["hf_id"]: "sha256:" + "2" * 64},
        preconditions_checked=[],
    )
    assert artifact["self_learning_run_complete_score"] == 1
    assert artifact["transactional_learning_positive_score"] == 1
    assert artifact["chronological_gain_over_readonly"] == 2
    assert artifact["max_forgetting"] == 0
    assert validate_artifact(artifact) == []

    forgetting_run = deepcopy(run)
    for row in forgetting_run["rows"]:
        if row["event_ordinal"] == 6 and row["arm"] == "transactional_write":
            row["exact_success"] = False
    null_artifact = reduce_run(
        plan,
        forgetting_run,
        run_date="20260904",
        duration_s=1.0,
        live_duration_s=0.2,
        source_artifact_hashes={},
        model_file_hashes={},
        preconditions_checked=[],
    )
    assert null_artifact["verdict_class"] == "null"
    assert null_artifact["max_forgetting"] == 1


def test_req_learn_6978_run_plan_rolls_back_worse_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-6978-ROLLBACK checks every proposed live state."""

    plan = build_plan(_fixture(), _selection(), MODEL_SPEC)

    def generate(row: dict, _prompt: str) -> dict:
        return {
            "raw_completion": "{}",
            "context_id": f"context:{row['event_ordinal']}:{row['arm']}",
            "latency_ms": 0.0,
        }

    def evaluate(_raw: str, event: dict, _pair: dict, _label: str) -> dict:
        return {
            "parse_success": False,
            "schema_outcome": "rejected",
            "exact_success": False,
            "exact_certificate_digest": f"sha256:{int(event['event_ordinal']) + 1:064x}",
            "terminal": True,
        }

    monkeypatch.setattr(
        exp6978,
        "replay_safety_score",
        lambda records: 1.0 if len(records) <= len(initial_memory_records()) else 0.0,
    )
    state_root = tmp_path / "rollback-run"
    run = run_plan(plan, state_root=state_root, generate_fn=generate, evaluate_fn=evaluate)
    assert any(row["fixture"] == "live_commit" for row in run["rollback_rows"])
    assert all(row["passed"] is True for row in run["rollback_rows"])
    rerun = run_plan(plan, state_root=state_root, generate_fn=generate, evaluate_fn=evaluate)
    assert len(rerun["rows"]) == 72


def test_req_learn_6978_scores_reject_writes_without_utility() -> None:
    """SCENARIO-LEARN-6978-BARE keeps completion separate from utility."""

    rows = [
        {
            "arm": arm,
            "event_ordinal": ordinal,
            "terminal": True,
            "token_cap": 128,
            "random_seed": ordinal,
            "model_id": MODEL_SPEC["hf_id"],
            "exact_success": False,
        }
        for ordinal in range(24)
        for arm in ARMS
    ]
    assert completion_score(rows, budgets_match=True, journal_ok=True, leakage_ok=True) == 1
    assert (
        positive_score(
            chronological_gain=0,
            lost_read_only_successes=0,
            rollback_passed=True,
            memory_state_bytes=500,
            max_state_bytes=1_000,
        )
        == 0
    )
    assert isinstance(
        positive_score(
            chronological_gain=2,
            lost_read_only_successes=0,
            rollback_passed=True,
            memory_state_bytes=500,
            max_state_bytes=1_000,
        ),
        int,
    )


def test_req_learn_6978_preconditions_and_blocked_schema(tmp_path: Path) -> None:
    """REQ-LEARN-6978 preconditions fail closed with exact gate evidence."""

    fixture = _fixture()
    qwen = tmp_path / "qwen.gguf"
    qwen.write_bytes(b"gguf")
    transaction_root = tmp_path / "transactions"
    checks, hashes = collect_preconditions(
        lease_artifact=_upstream("lease_aware_runtime_ready_score"),
        admissibility_artifact=_upstream("fixture_admissibility_ready_score"),
        selection_artifact=_selection(),
        fixture_artifact=fixture,
        source_paths={
            "lease": tmp_path / "lease.json",
            "admissibility": tmp_path / "admissibility.json",
            "selection": tmp_path / "selection.json",
            "fixture": tmp_path / "fixture.json",
        },
        fixture_hash=EXPECTED_EXP6967_SHA256,
        model_spec={**MODEL_SPEC, "model_path": str(qwen)},
        transaction_root=transaction_root,
        z3_available=True,
    )
    assert all(row["passed"] is True for row in checks)
    assert set(hashes) == {"lease", "admissibility", "selection", "fixture"}
    failed = deepcopy(checks)
    failed[0]["observed_value"] = 0
    failed[0]["passed"] = False
    artifact = build_blocked_artifact(
        run_date="20260904",
        checks=failed,
        source_artifact_hashes=hashes,
        model_spec={**MODEL_SPEC, "model_path": str(qwen)},
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_transactional_constraint_self_learning"
    assert artifact["gate_check_summary"]["failed_check"]
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert validate_artifact(artifact) == []

    unwritable = tmp_path / "not-a-directory"
    unwritable.write_text("file", encoding="utf-8")
    failed_checks, _ = collect_preconditions(
        lease_artifact=_upstream("lease_aware_runtime_ready_score"),
        admissibility_artifact=_upstream("fixture_admissibility_ready_score"),
        selection_artifact=_selection(),
        fixture_artifact=fixture,
        source_paths={},
        fixture_hash=EXPECTED_EXP6967_SHA256,
        model_spec={**MODEL_SPEC, "model_path": str(qwen)},
        transaction_root=unwritable,
        z3_available=True,
    )
    assert (
        next(row for row in failed_checks if row["check"] == "writable_transaction_directory")[
            "passed"
        ]
        is False
    )


def test_req_learn_6978_validator_detects_aggregate_and_bare_field_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6978-BARE rejects wrapped or contradicted scores."""

    plan = build_plan(_fixture(), _selection(), MODEL_SPEC)
    empty_run = {
        "rows": [],
        "prompt_visibility_rows": [],
        "exact_outcome_rows": [],
        "memory_lookup_rows": [],
        "update_proposal_rows": [],
        "transaction_journal_rows": [],
        "commit_rows": [],
        "rollback_rows": [],
        "restart_recovery_rows": [],
        "checkpoint_rows": [],
        "live_duration_s": 0.0,
        "final_state_bytes": 0,
        "journal_replay_passed": False,
    }
    artifact = reduce_run(
        plan,
        empty_run,
        run_date="20260904",
        duration_s=0.1,
        live_duration_s=0.0,
        source_artifact_hashes={},
        model_file_hashes={},
        preconditions_checked=[],
    )
    artifact["self_learning_run_complete_score"] = {"value": 1}
    errors = validate_artifact(artifact)
    assert "self_learning_run_complete_score must be a bare integer" in errors
    artifact["self_learning_run_complete_score"] = 1
    assert "self_learning_run_complete_score disagrees with rows" in validate_artifact(artifact)

    missing = deepcopy(artifact)
    del missing["rows"]
    assert "missing required field:rows" in validate_artifact(missing)

    malformed = deepcopy(artifact)
    malformed["field_principles"].pop("rows")
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed["verdict_class"] = "unknown"
    malformed["honest_verdict"] = "wrong"
    errors = validate_artifact(malformed)
    assert "field_principles must cover every required field" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "verdict_class is invalid" in errors

    contradicted = reduce_run(
        plan,
        empty_run,
        run_date="20260904",
        duration_s=0.1,
        live_duration_s=0.0,
        source_artifact_hashes={},
        model_file_hashes={},
        preconditions_checked=[],
    )
    contradicted["honest_verdict"] = "blocked_wrong"
    contradicted["held_future_rows"] = [{"wrong": True}]
    contradicted["chronological_gain_over_readonly"] = 99
    contradicted["transactional_learning_positive_score"] = 1
    contradicted["reproducibility_checksum"] = "sha256:wrong"
    errors = validate_artifact(contradicted)
    assert "honest_verdict prefix disagrees with verdict_class" in errors
    assert "held_future_rows disagree with rows" in errors
    assert "chronological_gain_over_readonly disagrees with rows" in errors
    assert "transactional_learning_positive_score disagrees with rows" in errors
    assert "reproducibility_checksum mismatch" in errors

    blocked = build_blocked_artifact(
        run_date="20260904",
        checks=[{"check": "gate", "expected_value": 1, "observed_value": 0, "passed": False}],
        source_artifact_hashes={},
        model_spec=MODEL_SPEC,
    )
    blocked["rows"] = [{"unexpected": True}]
    blocked["gate_check_summary"]["failed_check"] = None
    errors = validate_artifact(blocked)
    assert "blocked artifact must not contain arm-event rows" in errors
    assert "blocked artifact must name a failed check" in errors


def test_req_learn_6978_cli_validate_and_blocked_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-6978 command writes a stable blocked artifact without live calls."""

    result = tmp_path / "result.json"
    monkeypatch.setattr(
        "carnot.experiment_6978_transactional_constraint_self_learning.run",
        lambda **kwargs: build_blocked_artifact(
            run_date=kwargs["run_date"],
            checks=[
                {
                    "check": "fixture",
                    "expected_value": 1,
                    "observed_value": 0,
                    "passed": False,
                }
            ],
            source_artifact_hashes={},
            model_spec=MODEL_SPEC,
        ),
    )
    assert main(["--date", "20260904", "--result-path", str(result)]) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith("blocked_")
    assert main(["--validate", "--result-path", str(result)]) == 0
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert main(["--validate", "--result-path", str(invalid)]) == 1


def test_req_learn_6978_run_writes_blocked_artifact_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6978 treats a missing external precondition as blocked."""

    fixture = _fixture()
    source_paths = {
        "lease": tmp_path / "lease.json",
        "admissibility": tmp_path / "admissibility.json",
        "selection": tmp_path / "selection.json",
        "fixture": tmp_path / "fixture.json",
    }
    payloads = {
        "lease": {"lease_aware_runtime_ready_score": 0},
        "admissibility": {"fixture_admissibility_ready_score": 1},
        "selection": _selection(),
        "fixture": fixture,
    }
    for name, path in source_paths.items():
        path.write_text(json.dumps(payloads[name]), encoding="utf-8")
    qwen = tmp_path / "qwen.gguf"
    qwen.write_bytes(b"qwen")
    monkeypatch.setattr(exp6978, "SOURCE_PATHS", source_paths)
    monkeypatch.setattr(exp6978, "MODEL_SPECS", [{**MODEL_SPEC, "model_path": str(qwen)}])
    monkeypatch.setattr(
        exp6978, "EXPECTED_EXP6967_SHA256", exp6978._sha256_path(source_paths["fixture"])
    )
    monkeypatch.setattr(
        exp6978,
        "_load_live_model",
        lambda _path: pytest.fail("a blocked run must not load a model"),
    )
    real_import = builtins.__import__

    def no_z3(name: str, *args: object, **kwargs: object) -> object:
        if name == "z3":
            raise ImportError("z3 fixture")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_z3)
    result_path = tmp_path / "blocked.json"
    artifact = exp6978.run(
        run_date="20260904",
        result_path=result_path,
        transaction_root=tmp_path / "transactions",
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "lease_aware_runtime_ready_score"
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    monkeypatch.setattr(exp6978, "validate_artifact", lambda _artifact: ["fixture invalid"])
    with pytest.raises(RuntimeError, match="blocked artifact validation failed"):
        exp6978.run(
            run_date="20260904",
            result_path=tmp_path / "invalid-blocked.json",
            transaction_root=tmp_path / "transactions-invalid",
        )


def test_req_learn_6978_run_reduces_complete_mocked_live_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6978 orchestration hashes the model and validates complete evidence."""

    fixture = _fixture()
    source_paths = {
        "lease": tmp_path / "lease.json",
        "admissibility": tmp_path / "admissibility.json",
        "selection": tmp_path / "selection.json",
        "fixture": tmp_path / "fixture.json",
    }
    payloads = {
        "lease": {"lease_aware_runtime_ready_score": 1},
        "admissibility": {"fixture_admissibility_ready_score": 1},
        "selection": _selection(),
        "fixture": fixture,
    }
    for name, path in source_paths.items():
        path.write_text(json.dumps(payloads[name]), encoding="utf-8")
    qwen = tmp_path / "qwen.gguf"
    qwen.write_bytes(b"qwen")
    monkeypatch.setattr(exp6978, "SOURCE_PATHS", source_paths)
    monkeypatch.setattr(exp6978, "MODEL_SPECS", [{**MODEL_SPEC, "model_path": str(qwen)}])
    monkeypatch.setattr(
        exp6978, "EXPECTED_EXP6967_SHA256", exp6978._sha256_path(source_paths["fixture"])
    )

    class Model:
        closed = False

        def close(self) -> None:
            self.closed = True

    model = Model()
    monkeypatch.setattr(exp6978, "_load_live_model", lambda _path: model)
    monkeypatch.setattr(exp6978, "_live_generator", lambda _model: lambda _row, _prompt: {})
    monkeypatch.setattr(exp6978, "run_plan", lambda *_args, **_kwargs: _complete_null_run())
    result_path = tmp_path / "complete.json"
    artifact = exp6978.run(
        run_date="20260904",
        result_path=result_path,
        transaction_root=tmp_path / "transactions",
    )
    assert model.closed is True
    assert artifact["self_learning_run_complete_score"] == 1
    assert artifact["transactional_learning_positive_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert validate_artifact(artifact) == []
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    monkeypatch.setattr(exp6978, "validate_artifact", lambda _artifact: ["fixture invalid"])
    with pytest.raises(RuntimeError, match="artifact validation failed"):
        exp6978.run(
            run_date="20260904",
            result_path=tmp_path / "invalid-complete.json",
            transaction_root=tmp_path / "transactions-invalid",
        )


def test_req_learn_6978_canonical_hash_is_stable() -> None:
    """REQ-LEARN-6978 hashes canonical state independent of mapping order."""

    assert canonical_bytes({"b": 2, "a": 1}) == canonical_bytes({"a": 1, "b": 2})
    assert sha256_bytes(b"same") == sha256_bytes(b"same")
