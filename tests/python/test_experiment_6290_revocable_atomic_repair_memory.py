"""Tests for Exp6290 revocable atomic repair memory.

Spec refs: REQ-LEARN-6290, SCENARIO-LEARN-6290-KEYS,
SCENARIO-LEARN-6290-TRANSACTION, SCENARIO-LEARN-6290-REVOCATION,
SCENARIO-LEARN-6290-RESTART, SCENARIO-LEARN-6290-STREAMS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6290_revocable_atomic_repair_memory as mod
from carnot.memory import revocable_atomic_repair as mem


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
GOOD_HASH = mod.sha256_text("current evidence")
NEW_HASH = mod.sha256_text("new evidence")
OLD_HASH = mod.sha256_text("old evidence")


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _item(
    suffix: str = "A",
    *,
    evidence_hash: str = GOOD_HASH,
    atomic: bool = True,
    poisoned: bool = False,
) -> mem.AtomicRepairItem:
    return mem.AtomicRepairItem(
        namespace="exp6290",
        model_family="model-a",
        task_family="credential_rotation",
        repair_atom=f"require-safe-evidence-{suffix}",
        scope="family",
        exact_evidence_key=f"sidecar-{suffix}",
        exact_evidence_hash=evidence_hash,
        correction_id=f"corr-{suffix}",
        source_event_id=f"event-{suffix}",
        atomic=atomic,
        poisoned=poisoned,
    )


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["revocable_memory_ready_score"] = mod.ready_score(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_6290_spec_declares_revocable_contract() -> None:
    """REQ-LEARN-6290: OpenSpec owns the artifact and memory contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6290") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6290-1",
        "REQ-LEARN-6290-9",
        "SCENARIO-LEARN-6290-KEYS",
        "SCENARIO-LEARN-6290-TRANSACTION",
        "SCENARIO-LEARN-6290-REVOCATION",
        "SCENARIO-LEARN-6290-RESTART",
        "SCENARIO-LEARN-6290-STREAMS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6290_keys_and_collisions_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6290-KEYS: key collisions abort the transaction."""

    left = _item(evidence_hash=GOOD_HASH)
    right = _item(evidence_hash=NEW_HASH)
    assert left.precedent_key == right.precedent_key
    assert mem.stable_precedent_key(left.key_parts) == left.precedent_key

    store = mem.TransactionalRevocableRepairMemory(audit_log_path=tmp_path / "audit.jsonl")
    checkpoint = store.snapshot_hash()
    receipt = store.commit_transaction(
        [left, right],
        exact_evidence={left.exact_evidence_key: GOOD_HASH},
        event_index=1,
        stream_id="clean",
    )

    assert receipt.committed is False
    assert receipt.rejected_count == 2
    assert "key_collision" in receipt.rejection_reasons
    assert store.snapshot_hash() == checkpoint
    assert store.audit_entries == []


def test_scenario_6290_partial_transaction_poison_and_time_reversal(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6290-TRANSACTION: unsupported batches do not publish."""

    store = mem.TransactionalRevocableRepairMemory(audit_log_path=tmp_path / "audit.jsonl")
    good = _item("A")
    bundled = _item("B", atomic=False)
    poisoned = _item("C", poisoned=True)
    checkpoint = store.snapshot_hash()

    receipt = store.commit_transaction(
        [good, bundled],
        exact_evidence={good.exact_evidence_key: GOOD_HASH, bundled.exact_evidence_key: GOOD_HASH},
        event_index=2,
        stream_id="clean",
    )
    assert receipt.committed is False
    assert "bundled_repair" in receipt.rejection_reasons
    assert store.snapshot_hash() == checkpoint

    poison_receipt = store.commit_transaction(
        [poisoned],
        exact_evidence={poisoned.exact_evidence_key: GOOD_HASH},
        event_index=3,
        stream_id="poison",
    )
    assert poison_receipt.committed is False
    assert "poison" in poison_receipt.rejection_reasons
    assert store.snapshot_hash() == checkpoint

    assert store.commit_transaction(
        [good],
        exact_evidence={good.exact_evidence_key: GOOD_HASH},
        event_index=4,
        stream_id="clean",
    ).committed
    reversed_time = store.commit_transaction(
        [_item("D")],
        exact_evidence={"sidecar-D": GOOD_HASH},
        event_index=3,
        stream_id="clean",
    )
    assert reversed_time.committed is False
    assert "time_reversal" in reversed_time.rejection_reasons

    missing_revoke = store.revoke(
        _item("missing").precedent_key,
        exact_evidence_hash=GOOD_HASH,
        event_index=5,
        stream_id="missing",
    )
    assert missing_revoke.committed is False
    assert "missing_active_precedent" in missing_revoke.rejection_reasons

    active_conflict = store.commit_transaction(
        [_item("A", evidence_hash=NEW_HASH)],
        exact_evidence={"sidecar-A": NEW_HASH},
        event_index=6,
        stream_id="active_conflict",
    )
    assert active_conflict.committed is False
    assert "active_conflict" in active_conflict.rejection_reasons

    time_reversed_revoke = store.revoke(
        good.precedent_key,
        exact_evidence_hash=NEW_HASH,
        event_index=4,
        stream_id="time_reversed_revoke",
    )
    assert time_reversed_revoke.committed is False
    assert "time_reversal" in time_reversed_revoke.rejection_reasons


def test_scenario_6290_revocation_stale_resurrection_and_evidence_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6290-REVOCATION: revoked and stale rows never retrieve."""

    store = mem.TransactionalRevocableRepairMemory(audit_log_path=tmp_path / "audit.jsonl")
    old = _item(evidence_hash=OLD_HASH)
    assert store.commit_transaction(
        [old],
        exact_evidence={old.exact_evidence_key: OLD_HASH},
        event_index=1,
        stream_id="clean",
    ).committed

    missing = store.retrieve(old.precedent_key, exact_evidence={})
    assert missing.items == []
    assert missing.exact_evidence_rejection_count == 1

    assert store.revoke(
        old.precedent_key,
        exact_evidence_hash=NEW_HASH,
        event_index=2,
        stream_id="full_reversal",
    ).committed
    revoked = store.retrieve(old.precedent_key, exact_evidence={old.exact_evidence_key: OLD_HASH})
    assert revoked.items == []
    assert revoked.revoked_retrieval_count == 1

    stale = store.commit_transaction(
        [old],
        exact_evidence={old.exact_evidence_key: OLD_HASH},
        event_index=3,
        stream_id="stale_resurrection",
    )
    assert stale.committed is False
    assert "stale_resurrection" in stale.rejection_reasons

    new = _item(evidence_hash=NEW_HASH)
    assert store.commit_transaction(
        [new],
        exact_evidence={new.exact_evidence_key: NEW_HASH},
        event_index=4,
        stream_id="repromotion",
    ).committed
    active = store.retrieve(new.precedent_key, exact_evidence={new.exact_evidence_key: NEW_HASH})
    assert [item.version for item in active.items] == [3]
    assert active.active_retrieval_count == 1
    assert store.state_counts() == {"active": 1, "revoked": 1, "repromoted": 1}


def test_scenario_6290_restart_and_byte_identical_rollback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6290-RESTART: restart and rollback hashes are stable."""

    audit_path = tmp_path / "audit.jsonl"
    store = mem.TransactionalRevocableRepairMemory(audit_log_path=audit_path)
    checkpoint = store.checkpoint()
    item = _item()

    assert store.commit_transaction(
        [item],
        exact_evidence={item.exact_evidence_key: GOOD_HASH},
        event_index=1,
        stream_id="clean",
    ).committed
    before_restart = store.snapshot_hash()
    restarted = mem.TransactionalRevocableRepairMemory.from_audit_log(audit_path)

    assert restarted.snapshot_hash() == before_restart
    assert restarted.audit_hash() == store.audit_hash()

    restarted.rollback(checkpoint)
    assert restarted.snapshot_hash() == checkpoint.snapshot_hash
    assert restarted.audit_entries == []
    assert restarted.audit_hash() == checkpoint.audit_hash

    assert (
        mem.TransactionalRevocableRepairMemory.from_audit_log(
            tmp_path / "missing.jsonl"
        ).audit_entries
        == []
    )

    blank_log = tmp_path / "blank.jsonl"
    blank_log.write_text("\n", encoding="utf-8")
    assert mem.TransactionalRevocableRepairMemory.from_audit_log(blank_log).audit_entries == []

    bad_log = tmp_path / "bad.jsonl"
    bad_log.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audit log entry"):
        mem.TransactionalRevocableRepairMemory.from_audit_log(bad_log)

    with pytest.raises(ValueError, match="precedent key missing"):
        mem.stable_precedent_key({"namespace": "only"})

    empty_store = mem.TransactionalRevocableRepairMemory()
    assert empty_store.retrieve(item.precedent_key, exact_evidence={}).items == []
    empty_store.rollback(empty_store.checkpoint())

    active_checkpoint_store = mem.TransactionalRevocableRepairMemory(
        audit_log_path=tmp_path / "active.jsonl"
    )
    assert active_checkpoint_store.commit_transaction(
        [item],
        exact_evidence={item.exact_evidence_key: GOOD_HASH},
        event_index=1,
        stream_id="clean",
    ).committed
    active_checkpoint = active_checkpoint_store.checkpoint()
    extra = _item("extra")
    assert active_checkpoint_store.commit_transaction(
        [extra],
        exact_evidence={extra.exact_evidence_key: GOOD_HASH},
        event_index=2,
        stream_id="clean",
    ).committed
    active_checkpoint_store.rollback(active_checkpoint)
    assert active_checkpoint_store.snapshot_hash() == active_checkpoint.snapshot_hash

    bad_entry = deepcopy(store.audit_entries[0])
    bad_entry["state"] = "unknown"
    with pytest.raises(ValueError, match="unknown audit state"):
        mem.TransactionalRevocableRepairMemory()._apply_audit_entry(bad_entry)


def test_scenario_6290_artifact_writes_required_receipts(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6290-STREAMS: all arms and receipts are reported."""

    artifact = _artifact(tmp_path, write=True)
    assert (
        _artifact(tmp_path, write=True)["append_only_audit_log_path_and_hash"]["entry_count"]
        == artifact["append_only_audit_log_path_and_hash"]["entry_count"]
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["revocable_memory_ready_score"] == 1.0
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    stream_manifest = artifact["sealed_stream_manifest_path_and_hash"]
    stream_path = Path(stream_manifest["path"])
    assert stream_path.exists()
    assert stream_manifest["sha256"] == mod.sha256_file(stream_path)

    audit = artifact["append_only_audit_log_path_and_hash"]
    audit_path = Path(audit["path"])
    assert audit_path.exists()
    assert audit["sha256"] == mod.sha256_file(audit_path)
    assert audit["separate_from_active_view"] is True

    counts = artifact["active_stale_and_revoked_retrieval_counts_by_arm"]
    assert counts[mod.REVOCABLE_ARM]["revoked_retrieval_count"] == 0
    assert counts[mod.APPEND_ONLY_ARM]["stale_retrieval_count"] > 0

    unsafe = artifact["unsafe_advice_counts_by_arm"]
    assert unsafe[mod.REVOCABLE_ARM]["all_streams"] == 0
    assert unsafe[mod.APPEND_ONLY_ARM]["all_streams"] > 0

    utility = artifact["utility_coverage_abstention_and_retention_by_arm"]
    assert utility[mod.REVOCABLE_ARM]["all_streams"]["noninferior_to_global_margin"] is True
    assert (
        utility[mod.REVOCABLE_ARM]["all_streams"]["utility_per_row"]
        >= utility[mod.GLOBAL_ARM]["all_streams"]["utility_per_row"]
        + mod.PREREGISTERED_GLOBAL_MARGIN
    )

    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_6290_validation_guards_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6290: validator rejects broken fields and receipts."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact(artifact) is True
    assert mod.sha256_file(tmp_path / "absent") is None
    timed = mod.run(
        result_path=tmp_path / "timed.json",
        test_exit_codes=_passing_exit_codes(),
    )
    assert timed["duration_s"] >= 0.001

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(non_object)

    failed_test = deepcopy(artifact)
    failed_test["test_exit_codes"][mod.FOCUSED_COMMAND] = 1
    _refresh(failed_test)
    assert failed_test["status"] == "complete_null"
    assert "test command failed" in failed_test["honest_verdict"]
    assert mod.validate_artifact(failed_test) is True

    readiness_null = deepcopy(artifact)
    readiness_null["rollback_and_restart_identity"]["exact_rollback"] = False
    _refresh(readiness_null)
    assert readiness_null["status"] == "complete_null"
    assert readiness_null["honest_verdict"].endswith("readiness gate")
    assert mod.validate_artifact(readiness_null) is True

    bad_weight = deepcopy(artifact)
    bad_weight["source_model_weight_mutation_count"] = 0.0
    bad_weight["reproducibility_checksum"] = mod.reproducibility_checksum(bad_weight)
    with pytest.raises(ValueError, match="source_model_weight_mutation_count"):
        mod.validate_artifact(bad_weight)

    bad_ready = deepcopy(artifact)
    bad_ready["unsafe_advice_counts_by_arm"][mod.REVOCABLE_ARM]["all_streams"] = 1
    bad_ready["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_ready)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    blocked = deepcopy(artifact)
    blocked["upstream_eligibility_path_hash_and_terminal_class"]["terminal_class"] = "blocked"
    _refresh(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete without prefix"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_verdict_mismatch = deepcopy(artifact)
    bad_verdict_mismatch["honest_verdict"] = "complete: wrong"
    bad_verdict_mismatch["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_verdict_mismatch
    )
    with pytest.raises(ValueError, match="honest_verdict mismatch"):
        mod.validate_artifact(bad_verdict_mismatch)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_null"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status mismatch"):
        mod.validate_artifact(bad_status)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = []
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    bad_provenance_field = deepcopy(artifact)
    bad_provenance_field["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance_field["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_field
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_field)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_arms = deepcopy(artifact)
    bad_arms["arm_definitions"]["arm_names"] = []
    bad_arms["reproducibility_checksum"] = mod.reproducibility_checksum(bad_arms)
    with pytest.raises(ValueError, match="arm definitions"):
        mod.validate_artifact(bad_arms)

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    events, _, _ = mod._load_events()
    fit6264 = mod.exp6264.fit_familiarity_thresholds(events)
    fit6276 = mod.exp6276.fit_certified_dual_cache(events)
    rows = mod._sealed_stream_rows(mod._base_events(events, fit6264))
    with pytest.raises(ValueError, match="unknown control arm"):
        mod._run_control_arm("bad_arm", rows[:1], fit6264, fit6276)
