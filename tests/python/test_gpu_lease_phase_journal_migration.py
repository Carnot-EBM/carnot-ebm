"""Fail-closed legacy GPU lease migration tests.

Spec refs: REQ-INFRA-7078, SCENARIO-INFRA-7078-LEGACY-FINGERPRINT,
SCENARIO-INFRA-7078-OWNER-AND-LOCK, SCENARIO-INFRA-7078-PID-REUSE,
SCENARIO-INFRA-7078-PRESERVE-AND-PUBLISH,
SCENARIO-INFRA-7078-IDEMPOTENT, and
SCENARIO-INFRA-7078-ATOMIC-INTERRUPTION.
"""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from carnot import gpu_lease_phase_journal as lease_api


REPO = Path(__file__).resolve().parents[2]


def _current_terminal(runtime_dir: Path, device_uuid: str = "GPU-legacy") -> dict:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir,
        task_id="legacy-task",
        device_uuid=device_uuid,
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
        ttl_s=30.0,
    )
    lease_api._complete_fixture(lease)
    lease.release()
    return json.loads(lease.journal_path.read_text(encoding="utf-8"))


def _legacy_document(runtime_dir: Path, device_uuid: str = "GPU-legacy") -> dict:
    document = _current_terminal(runtime_dir, device_uuid)
    document.pop("lease_id")
    document["checksum"] = lease_api.journal_checksum(document)
    return document


def _write_document(runtime_dir: Path, device_uuid: str, document: dict) -> Path:
    path = lease_api.journal_path_for(runtime_dir, device_uuid)
    lease_api.write_json_atomic(path, document)
    return path


def _migrate(runtime_dir: Path, device_uuid: str = "GPU-legacy", **kwargs: object) -> dict:
    return lease_api.migrate_legacy_journal(
        runtime_dir=runtime_dir,
        device_uuid=device_uuid,
        process_match=lambda _pid, _ticks: False,
        lease_id_factory=lambda: "lease:migrated-test-id",
        migration_monotonic_ns=9_000_000,
        migration_utc="2026-09-06T12:00:00Z",
        **kwargs,
    )


def test_req_infra_7078_spec_precedes_migration_implementation() -> None:
    """REQ-INFRA-7078: the migration contract exists before its code."""

    text = (REPO / "openspec/capabilities/research-harnesses/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-7078") :]
    for anchor in (
        "SCENARIO-INFRA-7078-LEGACY-FINGERPRINT",
        "SCENARIO-INFRA-7078-OWNER-AND-LOCK",
        "SCENARIO-INFRA-7078-PID-REUSE",
        "SCENARIO-INFRA-7078-PRESERVE-AND-PUBLISH",
        "SCENARIO-INFRA-7078-IDEMPOTENT",
        "SCENARIO-INFRA-7078-ATOMIC-INTERRUPTION",
        "SCENARIO-INFRA-7078-DUAL-DEVICE-PREFLIGHT",
    ):
        assert anchor in section


def test_scenario_infra_7078_valid_legacy_is_preserved_and_published(tmp_path: Path) -> None:
    """A valid released terminal legacy journal becomes strict current evidence."""

    document = _legacy_document(tmp_path)
    path = _write_document(tmp_path, "GPU-legacy", document)
    source_bytes = path.read_bytes()
    source_hash = lease_api.sha256_bytes(source_bytes)

    with pytest.raises(lease_api.JournalError, match="missing_field:lease_id"):
        lease_api.read_journal(path)
    assert lease_api.legacy_journal_errors(document, expected_device_uuid="GPU-legacy") == []

    receipt = _migrate(tmp_path)
    current = lease_api.read_journal(path)
    preserved = Path(receipt["preserved_path"])
    saved_receipt = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["action"] == "migrated"
    assert receipt["source_sha256"] == source_hash
    assert preserved.read_bytes() == source_bytes
    assert saved_receipt["target_sha256"] == lease_api.sha256_bytes(path.read_bytes())
    assert current["lease_id"] == "lease:migrated-test-id"
    assert current["recovery"]["legacy_migration"]["source_sha256"] == source_hash
    assert current["released"] is True
    assert current["phase"] == "terminal_complete"
    assert receipt["signals_sent"] == []
    assert receipt["files_removed"] == []


def test_scenario_infra_7078_bad_checksum_and_arbitrary_shape_block(tmp_path: Path) -> None:
    """The migration rejects changed bytes and arbitrary missing-field maps."""

    bad_checksum = _legacy_document(tmp_path / "bad-checksum")
    bad_checksum["task_id"] = "changed-without-rehash"
    _write_document(tmp_path / "bad-checksum", "GPU-legacy", bad_checksum)
    with pytest.raises(lease_api.MigrationBlocked, match="legacy_checksum_mismatch"):
        _migrate(tmp_path / "bad-checksum")

    extra = _legacy_document(tmp_path / "extra")
    extra["unrecognized"] = True
    extra["checksum"] = lease_api.journal_checksum(extra)
    _write_document(tmp_path / "extra", "GPU-legacy", extra)
    with pytest.raises(lease_api.MigrationBlocked, match="legacy_top_level_fingerprint_mismatch"):
        _migrate(tmp_path / "extra")


def test_scenario_infra_7078_missing_owner_live_pid_and_reused_pid(tmp_path: Path) -> None:
    """Missing or live ownership blocks, while a reused numeric PID can migrate."""

    missing = _legacy_document(tmp_path / "missing")
    missing["owner"].pop("pid_start_ticks")
    missing["checksum"] = lease_api.journal_checksum(missing)
    _write_document(tmp_path / "missing", "GPU-legacy", missing)
    with pytest.raises(lease_api.MigrationBlocked, match="legacy_owner_fingerprint_mismatch"):
        _migrate(tmp_path / "missing")

    live = _legacy_document(tmp_path / "live")
    _write_document(tmp_path / "live", "GPU-legacy", live)
    with pytest.raises(lease_api.MigrationBlocked, match="recorded_owner_still_live"):
        lease_api.migrate_legacy_journal(
            runtime_dir=tmp_path / "live",
            device_uuid="GPU-legacy",
            process_match=lambda _pid, _ticks: True,
        )

    reused = _legacy_document(tmp_path / "reused")
    _write_document(tmp_path / "reused", "GPU-legacy", reused)
    calls: list[tuple[int, int]] = []

    def reused_pid(pid: int, ticks: int) -> bool:
        calls.append((pid, ticks))
        return False

    receipt = lease_api.migrate_legacy_journal(
        runtime_dir=tmp_path / "reused",
        device_uuid="GPU-legacy",
        process_match=reused_pid,
        lease_id_factory=lambda: "lease:reused-pid",
    )
    assert receipt["action"] == "migrated"
    assert len(calls) == 2


def test_scenario_infra_7078_held_lock_and_wrong_uuid_block(tmp_path: Path) -> None:
    """A held device lock or a mismatched device identity preserves the source."""

    held_dir = tmp_path / "held"
    legacy = _legacy_document(held_dir)
    path = _write_document(held_dir, "GPU-legacy", legacy)
    original = path.read_bytes()
    lock_path = lease_api.lock_path_for(held_dir, "GPU-legacy")
    descriptor = lock_path.open("a+")
    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(lease_api.MigrationBlocked, match="device_lock_held"):
            _migrate(held_dir)
    finally:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
        descriptor.close()
    assert path.read_bytes() == original

    wrong_dir = tmp_path / "wrong"
    wrong = _legacy_document(wrong_dir, "GPU-other")
    _write_document(wrong_dir, "GPU-requested", wrong)
    with pytest.raises(lease_api.MigrationBlocked, match="legacy_device_mismatch"):
        _migrate(wrong_dir, "GPU-requested")


def test_scenario_infra_7078_nonterminal_and_current_schema_noop(tmp_path: Path) -> None:
    """A nonterminal legacy document blocks and a strict current document is unchanged."""

    nonterminal_dir = tmp_path / "nonterminal"
    lease = lease_api.GpuLease.acquire(
        runtime_dir=nonterminal_dir,
        task_id="legacy-task",
        device_uuid="GPU-legacy",
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
    )
    document = deepcopy(lease.document)
    lease.close()
    document.pop("lease_id")
    document["checksum"] = lease_api.journal_checksum(document)
    path = _write_document(nonterminal_dir, "GPU-legacy", document)
    original = path.read_bytes()
    with pytest.raises(lease_api.MigrationBlocked, match="legacy_not_released"):
        _migrate(nonterminal_dir)
    assert path.read_bytes() == original

    current_dir = tmp_path / "current"
    _current_terminal(current_dir)
    current_path = lease_api.journal_path_for(current_dir, "GPU-legacy")
    current_bytes = current_path.read_bytes()
    receipt = _migrate(current_dir)
    assert receipt["action"] == "current_noop"
    assert receipt["idempotent"] is True
    assert current_path.read_bytes() == current_bytes


def test_scenario_infra_7078_repeated_migration_is_byte_idempotent(tmp_path: Path) -> None:
    """A second migration validates the receipt and changes no durable byte."""

    _write_document(tmp_path, "GPU-legacy", _legacy_document(tmp_path))
    first = _migrate(tmp_path)
    journal_path = lease_api.journal_path_for(tmp_path, "GPU-legacy")
    before = {
        "journal": journal_path.read_bytes(),
        "preserved": Path(first["preserved_path"]).read_bytes(),
        "receipt": Path(first["receipt_path"]).read_bytes(),
    }
    second = _migrate(tmp_path)
    assert second["action"] == "idempotent_noop"
    assert second["idempotent"] is True
    assert journal_path.read_bytes() == before["journal"]
    assert Path(first["preserved_path"]).read_bytes() == before["preserved"]
    assert Path(first["receipt_path"]).read_bytes() == before["receipt"]


def test_scenario_infra_7078_atomic_interruption_preserves_final_path(tmp_path: Path) -> None:
    """A failed journal replace leaves the original legacy final path intact."""

    path = _write_document(tmp_path, "GPU-legacy", _legacy_document(tmp_path))
    original = path.read_bytes()

    def interrupt(_source: object, _target: object) -> None:
        raise OSError("simulated atomic interruption")

    with pytest.raises(lease_api.MigrationBlocked, match="atomic_publish_failed"):
        _migrate(tmp_path, journal_replace=interrupt)
    assert path.read_bytes() == original
    assert (
        lease_api.legacy_journal_errors(
            json.loads(path.read_text(encoding="utf-8")), expected_device_uuid="GPU-legacy"
        )
        == []
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_scenario_infra_7078_legacy_fingerprint_rejects_nested_mutations(tmp_path: Path) -> None:
    """Every nested legacy field set, identity, time, and event binding is exact."""

    base = _legacy_document(tmp_path)

    def errors_for(change) -> set[str]:
        document = deepcopy(base)
        change(document)
        document["checksum"] = lease_api.journal_checksum(document)
        return set(lease_api.legacy_journal_errors(document, expected_device_uuid="GPU-legacy"))

    assert "legacy_schema_mismatch" in errors_for(lambda row: row.update(schema="wrong"))
    assert "legacy_identity_field_invalid" in errors_for(lambda row: row.update(task_id=""))
    assert "legacy_vram_fingerprint_mismatch" in errors_for(
        lambda row: row["vram_mb"].update(extra=1)
    )
    assert "legacy_exit_fingerprint_mismatch" in errors_for(
        lambda row: row["exit_evidence"].update(extra=1)
    )
    assert "legacy_unload_fingerprint_mismatch" in errors_for(
        lambda row: row["unload_evidence"].update(extra=1)
    )
    assert "legacy_recovery_fingerprint_mismatch" in errors_for(
        lambda row: row["recovery"].update(extra=1)
    )
    assert "legacy_recovery_signal_evidence_invalid" in errors_for(
        lambda row: row["recovery"].update(signals_sent=["forbidden"])
    )
    assert "legacy_event_fingerprint_mismatch" in errors_for(
        lambda row: row["phase_history"][0].update(extra=1)
    )
    assert "legacy_event_details_invalid" in errors_for(
        lambda row: row["phase_history"][0].update(details=[])
    )
    assert "legacy_event_time_invalid" in errors_for(
        lambda row: row["phase_history"][0].update(monotonic_ns="bad")
    )
    assert "legacy_event_time_order_invalid" in errors_for(
        lambda row: row["phase_history"][1].update(
            monotonic_ns=row["phase_history"][0]["monotonic_ns"]
        )
    )
    assert "legacy_monotonic_time_invalid" in errors_for(
        lambda row: row.update(heartbeat_monotonic_ns="bad")
    )
    assert "legacy_monotonic_time_order_invalid" in errors_for(
        lambda row: row.update(expires_monotonic_ns=row["heartbeat_monotonic_ns"])
    )
    assert "legacy_acquisition_event_mismatch" in errors_for(
        lambda row: row.update(acquired_monotonic_ns=row["acquired_monotonic_ns"] - 1)
    )
    assert "legacy_current_phase_history_mismatch" in errors_for(
        lambda row: row.update(phase="terminal_blocked")
    )
    assert "legacy_generation_invalid" in errors_for(lambda row: row.update(lease_generation=0))

    performed = deepcopy(base)
    performed["recovery"] = {
        "performed": True,
        "reason": "old recovery",
        "previous_checksum": "sha256:old",
        "previous_task_id": "older-task",
        "previous_pid": 123,
        "previous_pid_start_ticks": 456,
        "signals_sent": [],
    }
    performed["checksum"] = lease_api.journal_checksum(performed)
    assert lease_api.legacy_journal_errors(performed, expected_device_uuid="GPU-legacy") == []


def test_scenario_infra_7078_byte_preservation_collision_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A content path is immutable and a failed byte replace leaves no temporary file."""

    target = tmp_path / "preserved.bin"
    lease_api.write_bytes_atomic(target, b"one")
    lease_api.write_bytes_atomic(target, b"one")
    with pytest.raises(lease_api.MigrationBlocked, match="preserved_source_hash_collision"):
        lease_api.write_bytes_atomic(target, b"two")

    failed = tmp_path / "failed.bin"
    monkeypatch.setattr(
        lease_api.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        lease_api.write_bytes_atomic(failed, b"payload")
    assert not failed.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_scenario_infra_7078_unreadable_and_invalid_current_inputs_block(tmp_path: Path) -> None:
    """Missing, malformed, non-object, and invalid current journals all fail closed."""

    with pytest.raises(lease_api.MigrationBlocked, match="journal_unreadable"):
        _migrate(tmp_path / "missing")

    for name, payload, reason in (
        ("malformed", b"{", "journal_unreadable"),
        ("list", b"[]", "journal_not_object"),
    ):
        root = tmp_path / name
        path = lease_api.journal_path_for(root, "GPU-legacy")
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        with pytest.raises(lease_api.MigrationBlocked, match=reason):
            _migrate(root)

    current_root = tmp_path / "invalid-current"
    current = _current_terminal(current_root)
    current["lease_id"] = "invalid-current-id"
    current["checksum"] = lease_api.journal_checksum(current)
    _write_document(current_root, "GPU-legacy", current)
    with pytest.raises(lease_api.MigrationBlocked, match="current_journal_invalid:lease_id_invalid"):
        _migrate(current_root)


def test_scenario_infra_7078_post_lock_races_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanished, changed, invalid, or newly live owner blocks after lock acquisition."""

    vanished_root = tmp_path / "vanished"
    vanished_path = _write_document(vanished_root, "GPU-legacy", _legacy_document(vanished_root))

    def vanish(_pid: int, _ticks: int) -> bool:
        vanished_path.unlink()
        return False

    with pytest.raises(lease_api.MigrationBlocked, match="journal_reread_failed"):
        lease_api.migrate_legacy_journal(
            runtime_dir=vanished_root, device_uuid="GPU-legacy", process_match=vanish
        )

    changed_root = tmp_path / "changed"
    changed_path = _write_document(changed_root, "GPU-legacy", _legacy_document(changed_root))

    def change(_pid: int, _ticks: int) -> bool:
        changed_path.write_bytes(changed_path.read_bytes() + b" ")
        return False

    with pytest.raises(lease_api.MigrationBlocked, match="journal_changed_after_precheck"):
        lease_api.migrate_legacy_journal(
            runtime_dir=changed_root, device_uuid="GPU-legacy", process_match=change
        )

    invalid_root = tmp_path / "invalid-after-lock"
    _write_document(invalid_root, "GPU-legacy", _legacy_document(invalid_root))
    calls = 0
    original_validator = lease_api.legacy_journal_errors

    def invalid_second(document: dict, **kwargs: object) -> list[str]:
        del document, kwargs
        nonlocal calls
        calls += 1
        return [] if calls == 1 else ["forced_post_lock_invalid"]

    monkeypatch.setattr(lease_api, "legacy_journal_errors", invalid_second)
    with pytest.raises(lease_api.MigrationBlocked, match="forced_post_lock_invalid"):
        _migrate(invalid_root)
    monkeypatch.setattr(lease_api, "legacy_journal_errors", original_validator)

    live_root = tmp_path / "became-live"
    _write_document(live_root, "GPU-legacy", _legacy_document(live_root))
    liveness = iter((False, True))
    with pytest.raises(lease_api.MigrationBlocked, match="recorded_owner_became_live"):
        lease_api.migrate_legacy_journal(
            runtime_dir=live_root,
            device_uuid="GPU-legacy",
            process_match=lambda _pid, _ticks: next(liveness),
        )


def test_scenario_infra_7078_target_and_receipt_corruption_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid generated targets and missing, malformed, or changed receipts fail closed."""

    invalid_id_root = tmp_path / "invalid-id"
    _write_document(invalid_id_root, "GPU-legacy", _legacy_document(invalid_id_root))
    with pytest.raises(lease_api.MigrationBlocked, match="generated_lease_id_invalid"):
        lease_api.migrate_legacy_journal(
            runtime_dir=invalid_id_root,
            device_uuid="GPU-legacy",
            process_match=lambda _pid, _ticks: False,
            lease_id_factory=lambda: "wrong",
        )

    target_root = tmp_path / "invalid-target"
    _write_document(target_root, "GPU-legacy", _legacy_document(target_root))
    original_validate = lease_api.validate_journal_document

    def reject_target(document: dict, **kwargs: object) -> list[str]:
        recovery = document.get("recovery", {})
        if isinstance(recovery, dict) and "legacy_migration" in recovery:
            return ["forced_target_invalid"]
        return original_validate(document, **kwargs)

    monkeypatch.setattr(lease_api, "validate_journal_document", reject_target)
    with pytest.raises(lease_api.MigrationBlocked, match="target_journal_invalid"):
        _migrate(target_root)
    monkeypatch.setattr(lease_api, "validate_journal_document", original_validate)

    mismatch_root = tmp_path / "hash-mismatch"
    mismatch_path = _write_document(mismatch_root, "GPU-legacy", _legacy_document(mismatch_root))

    def wrong_replace(source: object, target: object) -> None:
        del source
        Path(target).write_text("{}", encoding="utf-8")

    with pytest.raises(lease_api.MigrationBlocked, match="atomic_publish_hash_mismatch"):
        _migrate(mismatch_root, journal_replace=wrong_replace)
    assert mismatch_path.read_text(encoding="utf-8") == "{}"

    for case, receipt_payload, reason in (
        ("missing", None, "migration_receipt_unreadable"),
        ("nonobject", [], "migration_receipt_not_object"),
        ("invalid", {"receipt_checksum": "sha256:bad"}, "migration_receipt_invalid"),
    ):
        root = tmp_path / f"receipt-{case}"
        _write_document(root, "GPU-legacy", _legacy_document(root))
        first = _migrate(root)
        receipt_path = Path(first["receipt_path"])
        if receipt_payload is None:
            receipt_path.unlink()
        else:
            receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
        with pytest.raises(lease_api.MigrationBlocked, match=reason):
            _migrate(root)
