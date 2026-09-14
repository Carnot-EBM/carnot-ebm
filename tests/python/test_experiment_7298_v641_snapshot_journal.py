"""Tests for persistent SQLite full-snapshot acknowledgments.

Spec refs: REQ-CL-7298 and SCENARIO-CL-7298-*.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7298_v641_snapshot_journal as exp7298


@pytest.fixture(scope="session")
def native():
    """REQ-CL-7298: load the installed native controller named by Exp7256."""

    source = exp7298._read_json(exp7298.REPO_ROOT / exp7298.EXP7256_RELATIVE)
    return exp7230.load_native_extension(Path(source["native_binary_receipt"]["module_file"]))


def releases(count: int, *, offset: int = 0) -> list[dict[str, object]]:
    """Return valid releases with stable IDs and increasing chronology."""

    rows = []
    for index in range(offset, offset + count):
        row = exp7257._cost_release(500 + index, 0)
        row["event_id"] = f"journal-event-{index:03d}"
        row["request_index"] = 5_000 + index
        row["release_index"] = 5_000 + index
        rows.append(row)
    return rows


def make_adapter(native, tmp_path: Path, **kwargs):
    """Create an initialized adapter on caller-owned storage."""

    return exp7298.SQLiteSnapshotJournal.from_state(
        native,
        exp7257.seed_cost_state(4),
        tmp_path / "state.sqlite3",
        **kwargs,
    )


def test_req_cl_7298_preconditions_authenticate_native_storage_and_outputs(
    tmp_path: Path,
) -> None:
    """REQ-CL-7298: source, native, SQLite, VFS, and outputs authenticate."""

    paths = exp7298.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7298.collect_preconditions(exp7298.REPO_ROOT, paths)
    assert exp7298.gate_summary(checks)["passed"] is True
    assert evidence["native_module_sha256"].startswith("sha256:")
    assert evidence["sqlite_probe"]["journal_mode"] == "persist"
    assert evidence["sqlite_probe"]["synchronous"] == exp7298.SQLITE_FULL
    assert evidence["sqlite_probe"]["vfs_requested"] == exp7298.SQLITE_VFS
    assert all(paths.writable_targets())


def test_scenario_cl_7298_storage_is_one_complete_verified_row(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7298-STORAGE: initialization writes one full snapshot row."""

    adapter = make_adapter(native, tmp_path, max_group_size=4, max_wait_ms=10)
    receipt = adapter.storage_receipt()
    assert receipt["journal_mode"] == "persist"
    assert receipt["synchronous"] == exp7298.SQLITE_FULL
    assert receipt["snapshot_row_count"] == 1
    assert receipt["sequence"] == 0
    assert receipt["checksum_valid"] is True
    assert receipt["initialization_duration_ns"] > 0
    assert receipt["initialization_committed_before_acceptance"] is True
    assert receipt["journal_file_retained"] is True

    with sqlite3.connect(adapter.database_path) as connection:
        row = connection.execute(
            "SELECT schema_version, sequence, state, checksum FROM snapshot WHERE slot = 1"
        ).fetchone()
    assert row is not None
    assert row[0] == exp7298.STATE_SCHEMA
    assert row[1] == 0
    assert exp7298.snapshot_checksum(row[2]) == row[3]
    assert bytes(row[2]) == adapter.state_bytes
    adapter.close()


@pytest.mark.parametrize("group_size", exp7298.GROUP_SIZES)
def test_scenario_cl_7298_ack_after_commit_and_native_parity(
    native, tmp_path: Path, group_size: int
) -> None:
    """SCENARIO-CL-7298-ACK: commit returns before exact ordered acknowledgment."""

    stages: list[str] = []
    initial = exp7257.seed_cost_state(4)
    batch = releases(group_size)
    expected = exp7298.serial_reference(native, initial, batch)
    adapter = exp7298.SQLiteSnapshotJournal.from_state(
        native,
        initial,
        tmp_path / f"group-{group_size}.sqlite3",
        max_group_size=group_size,
        max_wait_ms=0 if group_size == 1 else exp7298.MAX_WAIT_MS,
        stage_hook=stages.append,
    )
    responses = [adapter.enqueue(row) for row in batch]
    receipt = responses[-1]["flush_receipt"]
    assert receipt["acknowledged_event_ids"] == [row["event_id"] for row in batch]
    assert receipt["linearization_point"] == "sqlite_commit_returned"
    assert stages.index("after_commit_before_acknowledgment") < stages.index("after_acknowledgment")
    assert adapter.state_bytes == expected["state_bytes"]
    assert adapter.state_hash == expected["state_hash"]
    recovered = exp7298.SQLiteSnapshotJournal.recover(
        native,
        adapter.database_path,
        max_group_size=group_size,
        max_wait_ms=0 if group_size == 1 else exp7298.MAX_WAIT_MS,
    )
    assert recovered.state_bytes == adapter.state_bytes
    assert recovered.sequence == adapter.sequence == 1
    recovered.close()
    adapter.close()


def test_scenario_cl_7298_failed_commit_has_no_ack_and_requires_restore(
    native, tmp_path: Path
) -> None:
    """SCENARIO-CL-7298-ACK: a failed transaction cannot acknowledge its group."""

    def fail(stage: str) -> None:
        if stage == "after_row_write_before_commit":
            raise sqlite3.OperationalError("injected commit failure")

    adapter = make_adapter(
        native,
        tmp_path,
        max_group_size=4,
        max_wait_ms=10,
        stage_hook=fail,
    )
    old = adapter.state_bytes
    assert adapter.enqueue(releases(1)[0])["acknowledged"] is False
    failed = adapter.flush(reason="injected")
    assert failed["disposition"] == "failed_restore_required"
    assert failed["acknowledged_event_ids"] == []
    assert adapter.state_bytes == old
    with pytest.raises(RuntimeError, match="fresh restore"):
        adapter.enqueue(releases(1, offset=1)[0])
    adapter.close()

    restored = exp7298.SQLiteSnapshotJournal.recover(
        native,
        tmp_path / "state.sqlite3",
        max_group_size=4,
        max_wait_ms=10,
    )
    assert restored.state_bytes == old
    assert restored.sequence == 0
    restored.close()


def test_scenario_cl_7298_corruption_sequence_and_queue_controls(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7298-CONTROLS: attacks fail closed and remain bounded."""

    controls = exp7298.run_failure_controls(native, tmp_path / "controls")
    assert controls["all_controls_passed"] is True
    assert controls["corrupt_checksum_rejected"] is True
    assert controls["duplicate_sequence_rejected"] is True
    assert controls["failed_commit_acknowledgment_count"] == 0
    assert controls["pending_duplicate_disposition"] == "duplicate_id"
    assert controls["event_overflow_disposition"] == "backpressure_events"
    assert controls["byte_overflow_disposition"] == "backpressure_bytes"
    assert controls["observed_peak_pending_events"] <= exp7298.MAX_PENDING_EVENTS
    assert controls["observed_peak_pending_bytes"] <= exp7298.MAX_PENDING_BYTES
    assert controls["observed_max_state_bytes"] <= exp7298.MAX_STATE_BYTES


def test_scenario_cl_7298_real_crash_matrix_and_idempotent_fresh_restore(
    native, tmp_path: Path
) -> None:
    """SCENARIO-CL-7298-CRASH: every seeded SIGKILL restores exact old or new state."""

    rows = exp7298.run_crash_matrix(
        Path(native.__file__),
        exp7257.seed_cost_state(4),
        tmp_path / "crash-matrix",
        seeds=exp7298.CRASH_SEEDS,
    )
    assert len(rows) == len(exp7298.CRASH_SEEDS) * len(exp7298.CRASH_BOUNDARIES)
    assert {row["crash_boundary"] for row in rows} == set(exp7298.CRASH_BOUNDARIES)
    assert all(row["process_death"] == "SIGKILL" for row in rows)
    assert all(row["valid_complete_state"] is True for row in rows)
    assert all(row["lost_acknowledged_event_count"] == 0 for row in rows)
    assert all(row["duplicate_apply_count"] == 0 for row in rows)
    assert all(row["idempotent_fresh_recovery"] is True for row in rows)
    assert all(row["recovered_state_kind"] in {"old", "new"} for row in rows)
    raced = [row for row in rows if row["crash_boundary"] == "kill_during_write_commit"]
    assert all(row["kill_timing_certain"] is False for row in raced)
    acknowledged = [row for row in rows if row["crash_boundary"] == "after_acknowledgment"]
    assert all(row["acknowledged_event_ids"] == row["issued_event_ids"] for row in acknowledged)


def test_scenario_cl_7298_e2e_fixture_reduces_to_circular_readiness(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7298-E2E: real native commits and restart evidence pass cold reduction."""

    rows, parity = exp7298.run_protocol_fixture(native, tmp_path / "fixture")
    assert len(rows) == len(exp7298.GROUP_SIZES) * exp7298.EVENTS_PER_GROUP_ARM
    summary = exp7298.reduce_rows(rows, parity)
    assert summary["parity_failure_count"] == 0
    assert summary["missing_acknowledged_event_count"] == 0
    assert summary["max_state_bytes"] <= exp7298.MAX_STATE_BYTES
    assert all(row["state_bytes_match"] for row in parity)

    artifact = exp7298.complete_artifact_fixture_for_test()
    assert exp7298.validate_artifact(artifact) == []
    assert artifact["snapshot_journal_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["speedup_claimed"] is False
    assert artifact["production_default_changed"] is False
    assert artifact["filesystem_receipt"]["physical_power_loss_proven"] is False

    changed = deepcopy(artifact)
    changed["crash_control_rows"][0]["lost_acknowledged_event_count"] = 1
    changed["reproducibility_checksum"] = exp7298.artifact_checksum(changed)
    assert "crash_control_rows" in exp7298.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["baseline_validation_failures"] = [
        exp7298.exp7284.validation_receipt("not-a-failure", ["true"], 0, "ok", 0.01)
    ]
    changed["reproducibility_checksum"] = exp7298.artifact_checksum(changed)
    assert "baseline_validation_failures" in exp7298.validate_artifact(changed)


def test_req_cl_7298_blocked_and_terminal_publish_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7298: external blocks are row-free and validation gates publication."""

    failed = exp7298.check("missing", "external", "path", "file", None, False)
    blocked = exp7298.blocked_artifact_for_test(failed)
    assert exp7298.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == blocked["crash_control_rows"] == []

    baseline = exp7298.exp7284.validation_receipt(
        "full_python_suite", ["python", "-m", "pytest", "tests/python"], 1, "failed", 1.0
    )
    checkpoint = tmp_path / exp7298.CHECKPOINT_RELATIVE
    exp7298.atomic_write(checkpoint, {"validation_receipts": [baseline]})
    assert exp7298._prior_full_suite_failures(checkpoint) == [baseline]
    assert exp7298._prior_full_suite_failures(tmp_path / "absent.json") == []

    artifact = exp7298.complete_artifact_fixture_for_test()
    passed = exp7298.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7298,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {"fixture": "sha256:fixture"}}),
    )
    monkeypatch.setattr(exp7298, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    commands: list[list[str]] = []

    def validate(command: list[str], **_kwargs):
        commands.append(command)
        return {"command": command, "exit_code": 0, "output": "ok\n"}

    monkeypatch.setattr(exp7298, "_stream_subprocess", validate)
    output = tmp_path / "terminal.json"
    result = exp7298.run_experiment(tmp_path, output, exp7298.RUN_DATE)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert len(commands) == len(exp7298._scoped_validation_commands(tmp_path)) + 3

    monkeypatch.setattr(
        exp7298,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "failed"},
    )
    output.unlink()
    with pytest.raises(RuntimeError, match="scoped validation failed"):
        exp7298.run_experiment(tmp_path, output, exp7298.RUN_DATE)
    assert not output.exists()


def test_req_cl_7298_thin_entrypoint_and_read_only_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7298: the thin wrapper delegates and validation preserves bytes."""

    script = exp7298.REPO_ROOT / "scripts/experiments/experiment_7298_v641_snapshot_journal.py"
    calls: list[object] = []
    original = exp7298.main
    monkeypatch.setattr(exp7298, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7298.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7298, "main", original)

    artifact = exp7298.complete_artifact_fixture_for_test()
    candidate = tmp_path / "candidate.json"
    exp7298.atomic_write(candidate, artifact)
    before = candidate.read_bytes()
    assert exp7298.main(["--validate", str(candidate)]) == 0
    assert candidate.read_bytes() == before
    candidate.write_text("{}", encoding="utf-8")
    assert exp7298.main(["--validate", str(candidate)]) == 2
    assert exp7298.main(["--date", "bad", "--output", str(candidate)]) == 2


def test_scenario_cl_7298_actual_builder_traces_and_cold_reducer(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7298-TERMINAL: actual raw evidence builds one valid candidate."""

    paths = exp7298.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7298.collect_preconditions(exp7298.REPO_ROOT, paths)
    artifact = exp7298.build_artifact(
        exp7298.REPO_ROOT,
        paths,
        validation_receipts=[
            exp7298.exp7284.validation_receipt("fixture", ["true"], 0, "ok", 0.01)
        ],
        precondition_bundle=(checks, evidence),
    )
    assert exp7298.validate_artifact(artifact) == []
    assert artifact["snapshot_journal_ready_score"] == 1
    assert artifact["cold_reducer_receipt"]["exit_code"] == 0
    assert artifact["filesystem_receipt"]["sync_trace"]["trace_available"] is True
    sqlite_trace = artifact["filesystem_receipt"]["sync_trace"]["arms"]["sqlite"]
    assert sqlite_trace["fsync_calls"] + sqlite_trace["fdatasync_calls"] > 0
    assert artifact["filesystem_receipt"]["sync_trace"]["arms"]["atomic"]["fsync_calls"] > 0
    assert paths.raw_rows.is_file()
    assert paths.raw_trace_sqlite.is_file()
    assert paths.raw_trace_atomic.is_file()


def test_scenario_cl_7298_worker_paths_execute_directly(
    native,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-7298-CRASH: direct workers expose real transaction markers."""

    class MarkerReachedError(RuntimeError):
        pass

    binding_path = Path(native.__file__)
    state = exp7257.seed_cost_state(4)
    batch = releases(1, offset=80)

    def prepare(name: str) -> tuple[Path, Path]:
        database = tmp_path / name / "state.sqlite3"
        release_path = tmp_path / name / "releases.json"
        adapter = exp7298.SQLiteSnapshotJournal.from_state(
            native,
            state,
            database,
            max_group_size=16,
            max_wait_ms=exp7298.MAX_WAIT_MS,
        )
        adapter.close()
        exp7298.atomic_write(release_path, {"releases": batch})
        return database, release_path

    monkeypatch.setattr(
        exp7298,
        "_pause_for_kill",
        lambda marker: (_ for _ in ()).throw(MarkerReachedError(marker["actual_kill_point"])),
    )
    database, release_path = prepare("before")
    with pytest.raises(MarkerReachedError, match="before_transaction"):
        exp7298._crash_worker(binding_path, database, release_path, "before_transaction")

    database, release_path = prepare("ack")
    with pytest.raises(MarkerReachedError, match="after_acknowledgment"):
        exp7298._crash_worker(binding_path, database, release_path, "after_acknowledgment")

    database, release_path = prepare("race")
    monkeypatch.setattr(
        exp7298.signal,
        "pause",
        lambda: (_ for _ in ()).throw(MarkerReachedError("race complete")),
    )
    with pytest.raises(MarkerReachedError, match="race complete"):
        exp7298._crash_worker(binding_path, database, release_path, "kill_during_write_commit")
    assert '"kill_timing_certain":false' in capsys.readouterr().out

    assert exp7298._restore_worker(binding_path, database) == 0
    assert "state_bytes_b64" in capsys.readouterr().out
    assert (
        exp7298._trace_worker(binding_path, tmp_path / "trace-sqlite/state.sqlite3", "sqlite") == 0
    )
    assert exp7298._trace_worker(binding_path, tmp_path / "trace-atomic/state.json", "atomic") == 0
    assert exp7298._trace_worker(binding_path, tmp_path / "bad", "bad") == 2

    monkeypatch.setattr(
        exp7298.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="restore failed", stdout=""),
    )
    with pytest.raises(RuntimeError, match="restore worker failed"):
        exp7298._fresh_restore(binding_path, database, {})
    monkeypatch.setattr(
        exp7298.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout="[]"),
    )
    with pytest.raises(RuntimeError, match="non-object"):
        exp7298._fresh_restore(binding_path, database, {})


def test_scenario_cl_7298_defensive_storage_failures(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7298-CONTROLS: malformed rows and storage errors fail closed."""

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        exp7298._read_json(sequence)

    artifact = exp7298.complete_artifact_fixture_for_test()
    original_checksum = exp7298.artifact_checksum
    monkeypatch.setattr(
        exp7298,
        "artifact_checksum",
        lambda _artifact: (_ for _ in ()).throw(TypeError("bad checksum")),
    )
    assert exp7298._checksum_valid({}) is False
    assert "reproducibility_checksum" in exp7298.validate_artifact(artifact)
    monkeypatch.setattr(exp7298, "artifact_checksum", original_checksum)

    empty = exp7298._connect(tmp_path / "empty.sqlite3")
    exp7298._create_schema(empty)
    with pytest.raises(exp7298.SnapshotCorruptionError, match="row count"):
        exp7298._read_snapshot(empty)
    empty.close()

    adapter = make_adapter(native, tmp_path / "schema", max_group_size=4, max_wait_ms=10)
    adapter._connection.execute("UPDATE snapshot SET schema_version = 'wrong'")
    with pytest.raises(exp7298.SnapshotCorruptionError, match="schema"):
        exp7298._read_snapshot(adapter._connection)
    adapter.close()

    class FakeRows:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _query: str):
            self.calls += 1
            row = (
                (1,)
                if self.calls == 1
                else (
                    exp7298.STATE_SCHEMA,
                    "bad",
                    b"state",
                    exp7298.snapshot_checksum(b"state"),
                )
            )
            return SimpleNamespace(fetchone=lambda: row)

    with pytest.raises(exp7298.SnapshotSequenceError, match="sequence"):
        exp7298._read_snapshot(FakeRows())

    class BadSettings:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, query: str):
            value = ("delete",) if "journal_mode" in query else (exp7298.SQLITE_FULL,)
            return SimpleNamespace(fetchone=lambda: value)

        def close(self) -> None:
            self.closed = True

    bad = BadSettings()
    monkeypatch.setattr(exp7298.sqlite3, "connect", lambda *_args, **_kwargs: bad)
    with pytest.raises(RuntimeError, match="settings unavailable"):
        exp7298._connect(tmp_path / "bad-settings.sqlite3")
    assert bad.closed is True
    monkeypatch.undo()

    controller = exp7298.exp7256.PersistentNativeArchiveController.from_state_with_binding(
        native, exp7257.seed_cost_state(4)
    )
    with pytest.raises(ValueError, match="group size"):
        exp7298.SQLiteSnapshotJournal(
            native,
            tmp_path / "unused",
            controller,
            0,
            max_group_size=2,
            max_wait_ms=10,
            initialization_duration_ns=1,
        )
    with pytest.raises(ValueError, match="queue bounds"):
        exp7298.SQLiteSnapshotJournal(
            native,
            tmp_path / "unused",
            controller,
            0,
            max_group_size=4,
            max_wait_ms=10,
            initialization_duration_ns=1,
            max_pending_events=0,
        )

    existing = tmp_path / "existing.sqlite3"
    existing.touch()
    with pytest.raises(FileExistsError, match="refusing"):
        exp7298.SQLiteSnapshotJournal.from_state(
            native,
            exp7257.seed_cost_state(4),
            existing,
            max_group_size=4,
            max_wait_ms=10,
        )

    class OversizedController:
        def state_bytes(self) -> bytes:
            return b"x" * (exp7298.MAX_STATE_BYTES + 1)

    monkeypatch.setattr(
        exp7298.exp7256.PersistentNativeArchiveController,
        "from_state_with_binding",
        lambda *_args, **_kwargs: OversizedController(),
    )
    with pytest.raises(ValueError, match="initial state"):
        exp7298.SQLiteSnapshotJournal.from_state(
            native,
            {},
            tmp_path / "oversized.sqlite3",
            max_group_size=4,
            max_wait_ms=10,
        )


def test_scenario_cl_7298_defensive_transaction_failures(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7298-ACK: transaction and transition failures never acknowledge."""

    adapter = make_adapter(native, tmp_path / "large", max_group_size=4, max_wait_ms=10)
    with pytest.raises(ValueError, match="snapshot exceeds"):
        adapter._commit_snapshot(b"x" * (exp7298.MAX_STATE_BYTES + 1), 1)
    adapter.close()

    mismatch = make_adapter(native, tmp_path / "constructor", max_group_size=4, max_wait_ms=10)
    mismatch.close()
    controller = exp7298.exp7256.PersistentNativeArchiveController.from_state_with_binding(
        native, exp7257.seed_cost_state(4)
    )
    with pytest.raises(exp7298.SnapshotCorruptionError, match="disagree"):
        exp7298.SQLiteSnapshotJournal(
            native,
            mismatch.database_path,
            controller,
            1,
            max_group_size=4,
            max_wait_ms=10,
            initialization_duration_ns=1,
        )

    parent = make_adapter(native, tmp_path / "parent", max_group_size=4, max_wait_ms=10)
    other = exp7298._connect(parent.database_path)
    other.execute("UPDATE snapshot SET sequence = 5")
    other.close()
    with pytest.raises(exp7298.SnapshotSequenceError, match="parent sequence"):
        parent._commit_snapshot(parent.state_bytes, 1)
    parent.close()

    pending = make_adapter(native, tmp_path / "pending", max_group_size=4, max_wait_ms=10)
    real_read = exp7298._read_snapshot
    monkeypatch.setattr(
        exp7298,
        "_read_snapshot",
        lambda connection: (
            exp7298.STATE_SCHEMA,
            99,
            pending.state_bytes,
            exp7298.snapshot_checksum(pending.state_bytes),
        ),
    )
    with pytest.raises(exp7298.SnapshotCorruptionError, match="transactional"):
        pending._commit_snapshot(pending.state_bytes, 1)
    pending.close()
    monkeypatch.setattr(exp7298, "_read_snapshot", real_read)

    committed = make_adapter(native, tmp_path / "committed", max_group_size=4, max_wait_ms=10)
    calls = 0

    def changed_after_commit(connection):
        nonlocal calls
        calls += 1
        result = real_read(connection)
        if calls == 2:
            return (result[0], 99, result[2], result[3])
        return result

    monkeypatch.setattr(exp7298, "_read_snapshot", changed_after_commit)
    with pytest.raises(exp7298.SnapshotCorruptionError, match="committed"):
        committed._commit_snapshot(committed.state_bytes, 1)
    committed.close()
    monkeypatch.setattr(exp7298, "_read_snapshot", real_read)

    transition = make_adapter(native, tmp_path / "transition", max_group_size=4, max_wait_ms=10)
    transition.enqueue(releases(1, offset=90)[0])
    monkeypatch.setattr(
        transition._controller,
        "commit_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            exp7298.exp7240.ArchiveCommitRejected("injected")
        ),
    )
    assert transition.flush(reason="failure")["disposition"] == "failed_rolled_back"
    assert transition.flush(reason="empty")["disposition"] == "flush_empty"
    transition.close()

    rollback = make_adapter(native, tmp_path / "rollback", max_group_size=4, max_wait_ms=10)
    rollback.enqueue(releases(1, offset=91)[0])
    real_connection = rollback._connection

    class BadRollback:
        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("write failed")

        def rollback(self) -> None:
            raise sqlite3.OperationalError("rollback failed")

    rollback._connection = BadRollback()
    assert rollback.flush(reason="failure")["acknowledged_event_ids"] == []
    with pytest.raises(RuntimeError, match="fresh restore"):
        rollback.flush(reason="again")
    rollback._connection = real_connection
    rollback.close()


def test_req_cl_7298_control_failures_and_cli_modes(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7298: negative controls and every CLI mode return explicit status."""

    real_commit = exp7298.SQLiteSnapshotJournal._commit_snapshot

    def allow_duplicate(adapter, state_bytes, sequence):
        if "sequence/state.sqlite3" in str(adapter.database_path):
            return {}
        return real_commit(adapter, state_bytes, sequence)

    monkeypatch.setattr(
        exp7298.SQLiteSnapshotJournal,
        "_commit_snapshot",
        allow_duplicate,
    )
    controls = exp7298.run_failure_controls(native, tmp_path / "sequence-control")
    assert controls["duplicate_sequence_rejected"] is False
    monkeypatch.setattr(exp7298.SQLiteSnapshotJournal, "_commit_snapshot", real_commit)

    real_recover = exp7298.SQLiteSnapshotJournal.recover
    monkeypatch.setattr(
        exp7298.SQLiteSnapshotJournal,
        "recover",
        lambda *_args, **_kwargs: None,
    )
    controls = exp7298.run_failure_controls(native, tmp_path / "corrupt-control")
    assert controls["corrupt_checksum_rejected"] is False
    monkeypatch.setattr(exp7298.SQLiteSnapshotJournal, "recover", real_recover)

    assert exp7298.main(["--crash-worker"]) == 2
    assert exp7298.main(["--restore-worker"]) == 2
    assert exp7298.main(["--trace-worker"]) == 2
    assert exp7298.main(["--reduce-raw", str(tmp_path / "missing")]) == 2

    rows, parity = exp7298._fixture_rows()
    raw = tmp_path / "raw.json"
    reduced = tmp_path / "reduced.json"
    exp7298.atomic_write(raw, {"rows": rows, "semantic_parity_rows": parity})
    assert exp7298.main(["--reduce-raw", str(raw), "--reduced-output", str(reduced)]) == 0
    assert exp7298._read_json(reduced)["completed_event_count"] == len(rows)
    raw.write_text("{}", encoding="utf-8")
    assert exp7298.main(["--reduce-raw", str(raw), "--reduced-output", str(reduced)]) == 2

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp7298.main(["--validate", str(malformed)]) == 2

    monkeypatch.setattr(exp7298, "_restore_worker", lambda *_args: 7)
    assert exp7298.main(["--restore-worker", "--binding", "b", "--database", "d"]) == 7
    monkeypatch.setattr(exp7298, "_trace_worker", lambda *_args: 8)
    assert (
        exp7298.main(
            [
                "--trace-worker",
                "--binding",
                "b",
                "--database",
                "d",
                "--trace-adapter",
                "sqlite",
            ]
        )
        == 8
    )
    monkeypatch.setattr(exp7298, "run_experiment", lambda *_args: {})
    assert exp7298.main(["--date", exp7298.RUN_DATE, "--output", str(reduced)]) == 0
    monkeypatch.setattr(
        exp7298,
        "run_experiment",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    assert exp7298.main(["--date", exp7298.RUN_DATE, "--output", str(reduced)]) == 2


def test_scenario_cl_7298_remaining_fail_closed_boundaries(
    native,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-7298-CONTROLS: every remaining error boundary is explicit."""

    binding_path = Path(native.__file__)
    state = exp7257.seed_cost_state(4)

    monkeypatch.setattr(
        exp7298,
        "_sqlite_probe",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    checks, _evidence = exp7298.collect_preconditions(
        exp7298.REPO_ROOT, exp7298.ExperimentPaths.under(tmp_path / "probe")
    )
    assert exp7298.gate_summary(checks)["passed"] is False
    monkeypatch.undo()

    real_read = exp7298._read_snapshot
    monkeypatch.setattr(
        exp7298,
        "_read_snapshot",
        lambda _connection: (
            exp7298.STATE_SCHEMA,
            0,
            b"different",
            exp7298.snapshot_checksum(b"different"),
        ),
    )
    with pytest.raises(exp7298.SnapshotCorruptionError, match="initial snapshot"):
        exp7298.SQLiteSnapshotJournal.from_state(
            native,
            state,
            tmp_path / "initial-mismatch/state.sqlite3",
            max_group_size=4,
            max_wait_ms=10,
        )
    monkeypatch.setattr(exp7298, "_read_snapshot", real_read)

    recovered_path = tmp_path / "recover-mismatch/state.sqlite3"
    adapter = exp7298.SQLiteSnapshotJournal.from_state(
        native,
        state,
        recovered_path,
        max_group_size=4,
        max_wait_ms=10,
    )
    adapter.close()

    class ChangedController:
        def state_bytes(self) -> bytes:
            return b"different"

    monkeypatch.setattr(
        exp7298.exp7256.PersistentNativeArchiveController,
        "from_snapshot",
        lambda *_args, **_kwargs: ChangedController(),
    )
    with pytest.raises(exp7298.SnapshotCorruptionError, match="canonical restore"):
        exp7298.SQLiteSnapshotJournal.recover(
            native,
            recovered_path,
            max_group_size=4,
            max_wait_ms=10,
        )
    monkeypatch.undo()

    with pytest.raises(ValueError, match="positive multiple"):
        exp7298.run_protocol_fixture(native, tmp_path / "bad-fixture", events_per_arm=3)
    rows, parity = exp7298._fixture_rows()
    with pytest.raises(ValueError, match="missing parity"):
        exp7298.reduce_rows(rows, parity[:-1])

    class PauseReachedError(RuntimeError):
        pass

    monkeypatch.setattr(
        exp7298.signal,
        "pause",
        lambda: (_ for _ in ()).throw(PauseReachedError("pause")),
    )
    with pytest.raises(PauseReachedError):
        exp7298._pause_for_kill({"stage": "test"})
    assert '"stage":"test"' in capsys.readouterr().out

    database = tmp_path / "unreached/state.sqlite3"
    releases_path = tmp_path / "unreached/releases.json"
    adapter = exp7298.SQLiteSnapshotJournal.from_state(
        native,
        state,
        database,
        max_group_size=16,
        max_wait_ms=10,
    )
    adapter.close()
    exp7298.atomic_write(releases_path, {"releases": releases(1, offset=120)})
    monkeypatch.setattr(
        exp7298.SQLiteSnapshotJournal,
        "flush",
        lambda *_args, **_kwargs: {"acknowledged_event_ids": []},
    )
    with pytest.raises(RuntimeError, match="unreached crash boundary"):
        exp7298._crash_worker(
            binding_path,
            database,
            releases_path,
            "after_commit_before_acknowledgment",
        )
    monkeypatch.undo()

    class TimeoutProcess:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("timeout stderr")
            self.pid = 123
            self.killed = False

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: int) -> int:
            assert timeout == 5
            return -9

    monkeypatch.setattr(exp7298.subprocess, "Popen", TimeoutProcess)
    monkeypatch.setattr(exp7298.select, "select", lambda *_args: ([], [], []))
    with pytest.raises(RuntimeError, match="crash worker timeout"):
        exp7298.run_crash_matrix(
            binding_path,
            state,
            tmp_path / "timeout",
            seeds=[exp7298.CRASH_SEEDS[0]],
        )
    monkeypatch.undo()

    parsed = exp7298._parse_mountinfo(
        "malformed\n1 2 3 - ext4\n1 2 3 4 /elsewhere rw - ext4 /dev/x rw\n",
        tmp_path.resolve(),
    )
    assert parsed == {}
    monkeypatch.setattr(exp7298.shutil, "which", lambda _name: None)
    missing_trace = exp7298.run_sync_traces(
        binding_path,
        tmp_path / "no-trace",
        tmp_path / "sqlite.trace",
        tmp_path / "atomic.trace",
    )
    assert missing_trace["trace_available"] is False


def test_req_cl_7298_remaining_builder_and_runner_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7298: failed reduction or terminal checks cannot publish success."""

    failed = exp7298.check("missing", "external", "path", "file", None, False)
    paths = exp7298.ExperimentPaths.under(tmp_path / "blocked-build")
    blocked = exp7298.build_artifact(
        exp7298.REPO_ROOT,
        paths,
        validation_receipts=[],
        precondition_bundle=([failed], {"hashes": {}}),
    )
    assert blocked["status"] == "blocked"

    monkeypatch.setattr(
        exp7298,
        "_stream_subprocess",
        lambda *_args, **_kwargs: {"exit_code": 1, "output": "failed"},
    )
    with pytest.raises(RuntimeError, match="cold raw-row reducer failed"):
        exp7298._reduce_raw_in_fresh_process(
            exp7298.REPO_ROOT, exp7298.ExperimentPaths.under(tmp_path / "reduce")
        )

    with pytest.raises(ValueError, match="run date"):
        exp7298.run_experiment(tmp_path, tmp_path / "bad-date.json", "bad")

    real_build = exp7298.build_artifact
    monkeypatch.setattr(exp7298, "collect_preconditions", lambda *_args: ([failed], {}))
    monkeypatch.setattr(exp7298, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid blocked"):
        exp7298.run_experiment(tmp_path, tmp_path / "invalid-blocked.json", exp7298.RUN_DATE)
    monkeypatch.setattr(exp7298, "build_artifact", real_build)
    output = tmp_path / "blocked.json"
    result = exp7298.run_experiment(tmp_path, output, exp7298.RUN_DATE)
    assert result["status"] == "blocked"
    assert output.is_file()

    passed = exp7298.check("fixture", "fixture", "field", True, True, True)
    artifact = exp7298.complete_artifact_fixture_for_test()
    monkeypatch.setattr(exp7298, "collect_preconditions", lambda *_args: ([passed], {}))
    monkeypatch.setattr(
        exp7298,
        "_stream_subprocess",
        lambda *_args, **_kwargs: {"exit_code": 0, "output": "ok"},
    )
    monkeypatch.setattr(exp7298, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid Exp7298 candidate"):
        exp7298.run_experiment(tmp_path, tmp_path / "invalid.json", exp7298.RUN_DATE)

    monkeypatch.setattr(exp7298, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    calls = 0
    scoped_count = len(exp7298._scoped_validation_commands(tmp_path))

    def fail_candidate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "exit_code": 1 if calls == scoped_count + 1 else 0,
            "output": "candidate failure",
        }

    monkeypatch.setattr(exp7298, "_stream_subprocess", fail_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        exp7298.run_experiment(tmp_path, tmp_path / "candidate-failure.json", exp7298.RUN_DATE)

    monkeypatch.setattr(exp7298, "_crash_worker", lambda *_args: None)
    monkeypatch.setattr(exp7298, "run_experiment", lambda *_args: {})
    assert (
        exp7298.main(
            [
                "--crash-worker",
                "--binding",
                "b",
                "--database",
                "d",
                "--releases",
                "r",
                "--boundary",
                "before_transaction",
            ]
        )
        == 0
    )
