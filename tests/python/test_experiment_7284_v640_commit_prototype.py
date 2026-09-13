"""Tests for the opt-in host group-commit prototype.

Spec refs: REQ-CL-7284 and SCENARIO-CL-7284-*.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7284_v640_commit_prototype as exp7284


@pytest.fixture(scope="session")
def native():
    """REQ-CL-7284: load the exact shipped controller named by Exp7270."""

    source = exp7284._read_json(exp7284.REPO_ROOT / exp7284.EXP7256_RELATIVE)
    module = Path(source["native_binary_receipt"]["module_file"])
    return exp7230.load_native_extension(module)


def releases(count: int, *, offset: int = 0) -> list[dict[str, object]]:
    """Return independent releases with stable IDs and increasing chronology."""

    rows = []
    for index in range(offset, offset + count):
        row = exp7257._cost_release(100 + index, 0)
        row["event_id"] = f"group-event-{index:03d}"
        row["request_index"] = 1000 + index
        row["release_index"] = 1000 + index
        rows.append(row)
    return rows


def make_wrapper(native, tmp_path: Path, **kwargs):
    """Create one wrapper from the shipped nonempty state on real disk."""

    return exp7284.HostGroupCommitController.from_state(
        native,
        exp7257.seed_cost_state(4),
        tmp_path / "state.json",
        **kwargs,
    )


def test_req_cl_7284_preconditions_authenticate_profile_and_outputs(tmp_path: Path) -> None:
    """REQ-CL-7284: exact sync-dominated evidence is required before work."""

    paths = exp7284.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7284.collect_preconditions(exp7284.REPO_ROOT, paths)
    assert exp7284.gate_summary(checks)["passed"] is True
    assert evidence["exp7270"]["declared_bottleneck"] == "durable_sync"
    assert evidence["exp7270"]["journal_optimization_warranted_score"] == 0
    assert all(paths.writable_targets())

    changed = tmp_path / "changed.json"
    changed.write_text('{"status":"complete"}\n', encoding="utf-8")
    failed, _ = exp7284.collect_preconditions(exp7284.REPO_ROOT, paths, exp7270_path=changed)
    assert exp7284.gate_summary(failed)["passed"] is False


def test_scenario_cl_7284_queue_bounds_duplicates_timeout_and_abort(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7284-QUEUE: every bounded queue disposition is explicit."""

    wrapper = make_wrapper(
        native,
        tmp_path,
        max_group_size=16,
        max_wait_ms=10,
        max_pending_events=2,
        max_pending_bytes=100_000,
    )
    first, second, third = releases(3)
    assert wrapper.enqueue(first, now_ns=0)["disposition"] == "accepted_pending"
    duplicate = wrapper.enqueue(first)
    assert duplicate["disposition"] == "duplicate_id"
    assert wrapper.enqueue(second, now_ns=1_000_000)["accepted"] is True
    assert wrapper.enqueue(third)["disposition"] == "backpressure_events"
    assert wrapper.flush_due(now_ns=11_000_000)["reason"] == "timeout"
    assert wrapper.pending_events == 0

    committed_duplicate = wrapper.enqueue(first)
    assert committed_duplicate["disposition"] == "duplicate_id"
    large = releases(1, offset=20)[0]
    byte_limited = make_wrapper(
        native,
        tmp_path / "bytes",
        max_group_size=16,
        max_wait_ms=10,
        max_pending_events=16,
        max_pending_bytes=1,
    )
    assert byte_limited.enqueue(large)["disposition"] == "backpressure_bytes"
    assert byte_limited.shutdown(abort=True)["disposition"] == "aborted_empty"

    aborting = make_wrapper(native, tmp_path / "abort", max_group_size=4, max_wait_ms=10)
    assert aborting.enqueue(releases(1, offset=30)[0])["accepted"] is True
    aborted = aborting.shutdown(abort=True)
    assert aborted["disposition"] == "aborted_unacknowledged"
    assert aborted["discarded_event_ids"] == ["group-event-030"]


def test_scenario_cl_7284_visibility_dependent_query_and_shutdown(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7284-VISIBILITY: only a durable flush changes query state."""

    wrapper = make_wrapper(native, tmp_path, max_group_size=4, max_wait_ms=10)
    event = {"event_id": "query", "family_id": "lower_bound", "numeric_value": 0}
    before = wrapper.query(event, dependent=False)
    accepted = wrapper.enqueue(releases(1)[0])
    assert accepted["acknowledged"] is False
    hidden = wrapper.query(event, dependent=False)
    assert hidden["state_hash"] == before["state_hash"]
    assert hidden["flush_forced"] is False

    visible = wrapper.query(event, dependent=True)
    assert visible["flush_forced"] is True
    assert visible["flush_receipt"]["acknowledged_event_ids"] == ["group-event-000"]
    assert visible["dependent_query_delay_ns"] >= visible["flush_receipt"]["commit_delay_ns"]
    assert visible["state_hash"] != before["state_hash"]
    assert wrapper.shutdown()["disposition"] == "shutdown_clean"


@pytest.mark.parametrize("group_size", [1, 4, 16])
def test_scenario_cl_7284_group_parity_matches_serial_reference(
    native, tmp_path: Path, group_size: int
) -> None:
    """SCENARIO-CL-7284-PARITY: each group endpoint equals serial transitions."""

    initial = exp7257.seed_cost_state(4)
    batch = releases(group_size)
    serial = exp7284.serial_reference(native, initial, batch)
    wrapper = exp7284.HostGroupCommitController.from_state(
        native,
        initial,
        tmp_path / f"state-{group_size}.json",
        max_group_size=group_size,
        max_wait_ms=0 if group_size == 1 else 10,
    )
    receipts = [wrapper.enqueue(row) for row in batch]
    if wrapper.pending_events:
        group = wrapper.flush(reason="shutdown")
    else:
        group = receipts[-1]["flush_receipt"]
    assert group["acknowledged_event_ids"] == [row["event_id"] for row in batch]
    assert wrapper.state_bytes == serial["state_bytes"]
    assert wrapper.state_hash == serial["state_hash"]
    assert group["file_fsync"] is group["directory_fsync"] is True
    assert group["linearization_point"] == "directory_fsync_complete"


def test_scenario_cl_7284_commit_failure_rolls_back_and_refuses_after_publish(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7284-FAILURE: no failed group leaks learned state."""

    wrapper = make_wrapper(native, tmp_path, max_group_size=4, max_wait_ms=10)
    parent = wrapper.state_bytes
    wrapper.enqueue(releases(1)[0])
    real_durable_replace = exp7284.durable_replace

    def fail_before(*_args, **_kwargs):
        raise exp7284.DurablePublicationError("after_file_sync", replaced=False)

    monkeypatch.setattr(exp7284, "durable_replace", fail_before)
    failed = wrapper.flush(reason="test")
    assert failed["disposition"] == "failed_rolled_back"
    assert failed["acknowledged_event_ids"] == []
    assert wrapper.state_bytes == parent

    monkeypatch.setattr(exp7284, "durable_replace", real_durable_replace)
    published = make_wrapper(native, tmp_path / "published", max_group_size=4, max_wait_ms=10)
    published.enqueue(releases(1, offset=20)[0])

    def fail_after(*_args, **_kwargs):
        raise exp7284.DurablePublicationError("directory_fsync", replaced=True)

    monkeypatch.setattr(exp7284, "durable_replace", fail_after)
    uncertain = published.flush(reason="test")
    assert uncertain["disposition"] == "failed_restore_required"
    with pytest.raises(RuntimeError, match="fresh restore"):
        published.query({"event_id": "q", "family_id": "lower_bound", "numeric_value": 0})
    with pytest.raises(RuntimeError, match="fresh restore"):
        published.enqueue(releases(1, offset=21)[0])
    with pytest.raises(RuntimeError, match="fresh restore"):
        published.flush(reason="failed")


def test_scenario_cl_7284_real_subprocess_crash_matrix(native, tmp_path: Path) -> None:
    """SCENARIO-CL-7284-CRASH: SIGKILL and fresh restore use real disk sync."""

    binding_path = Path(native.__file__)
    rows = exp7284.run_crash_matrix(
        binding_path,
        exp7257.seed_cost_state(4),
        releases(4),
        tmp_path / "crashes",
    )
    assert [row["crash_boundary"] for row in rows] == list(exp7284.CRASH_BOUNDARIES)
    assert all(row["process_death"] == "SIGKILL" for row in rows)
    assert all(row["fresh_process_restore"] is True for row in rows)
    assert all(row["valid_complete_state"] is True for row in rows)
    assert all(row["lost_acknowledged_event_count"] == 0 for row in rows)
    assert all(row["duplicate_apply_count"] == 0 for row in rows)
    acknowledged = rows[-1]
    assert acknowledged["acknowledged_event_ids"] == [
        "group-event-000",
        "group-event-001",
        "group-event-002",
        "group-event-003",
    ]
    assert acknowledged["recovered_event_ids"][-4:] == acknowledged["acknowledged_event_ids"]


def test_req_cl_7284_benchmark_reducer_and_terminal_artifacts(native, tmp_path: Path) -> None:
    """REQ-CL-7284: rows and gates cold-reduce to semantic readiness only."""

    rows, parity = exp7284.run_group_benchmark(native, tmp_path / "bench", events_per_arm=16)
    summary = exp7284.reduce_rows(rows, parity)
    assert len(rows) == 48
    assert summary["completed_event_count"] == 48
    assert summary["parity_failure_count"] == 0
    assert summary["acknowledgment_failure_count"] == 0
    assert {row["max_group_size"] for row in summary["arm_summaries"]} == {1, 4, 16}
    assert all(row["same_semantics_speed_claim"] is False for row in summary["arm_summaries"])

    artifact = exp7284.complete_artifact_fixture_for_test()
    assert exp7284.validate_artifact(artifact) == []
    assert artifact["commit_protocol_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["same_semantics_speed_claim"] is False
    assert artifact["production_default_changed"] is False

    changed = deepcopy(artifact)
    changed["crash_rows"][-1]["lost_acknowledged_event_count"] = 1
    changed["reproducibility_checksum"] = exp7284.artifact_checksum(changed)
    assert "crash_rows" in exp7284.validate_artifact(changed)

    failed = exp7284.check("missing", "upstream", "path", "file", None, False)
    blocked = exp7284.blocked_artifact_for_test(failed)
    assert exp7284.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == blocked["crash_rows"] == blocked["semantic_parity_rows"] == []


def test_req_cl_7284_thin_entrypoint_and_read_only_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7284: the wrapper delegates and validation does not mutate bytes."""

    script = exp7284.REPO_ROOT / "scripts/experiments/experiment_7284_v640_commit_prototype.py"
    calls: list[object] = []
    original = exp7284.main
    monkeypatch.setattr(exp7284, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7284.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7284, "main", original)

    artifact = exp7284.complete_artifact_fixture_for_test()
    path = tmp_path / "candidate.json"
    exp7284.atomic_write(path, artifact)
    before = path.read_bytes()
    assert exp7284.main(["--validate", str(path)]) == 0
    assert path.read_bytes() == before
    path.write_text("{}", encoding="utf-8")
    assert exp7284.main(["--validate", str(path)]) == 2
    assert exp7284.main(["--date", "bad", "--output", str(path)]) == 2


def test_scenario_cl_7284_terminal_publish_only_after_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7284-TERMINAL: a measured candidate publishes atomically."""

    artifact = exp7284.complete_artifact_fixture_for_test()
    passed = exp7284.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7284,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {"fixture": "sha256:fixture"}}),
    )
    monkeypatch.setattr(exp7284, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    commands: list[list[str]] = []

    def validate(command: list[str], **_kwargs):
        commands.append(command)
        return {"command": command, "exit_code": 0, "output": "ok\n"}

    monkeypatch.setattr(exp7284, "_stream_subprocess", validate)
    output = tmp_path / "terminal.json"
    result = exp7284.run_experiment(tmp_path, output, exp7284.RUN_DATE)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert len(commands) == len(exp7284._scoped_validation_commands(tmp_path)) + 3

    monkeypatch.setattr(
        exp7284,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "failed"},
    )
    output.unlink()
    with pytest.raises(RuntimeError, match="scoped validation failed"):
        exp7284.run_experiment(tmp_path, output, exp7284.RUN_DATE)
    assert not output.exists()


def test_scenario_cl_7284_defensive_queue_and_durable_paths(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7284-FAILURE: defensive paths remain bounded and observable."""

    stages: list[str] = []
    receipt = exp7284.durable_replace(
        tmp_path / "stages/state.json", b"state\n", stage_hook=stages.append
    )
    assert stages == [
        "before_write",
        "after_write",
        "after_file_sync",
        "after_rename",
        "after_directory_sync",
    ]
    assert receipt["linearization_point"] == "directory_fsync_complete"

    failing_path = tmp_path / "write-failure/state.json"
    real_write = exp7284.os.write
    monkeypatch.setattr(
        exp7284.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
    )
    with pytest.raises(exp7284.DurablePublicationError) as write_error:
        exp7284.durable_replace(failing_path, b"state\n")
    assert write_error.value.replaced is False
    assert not list(failing_path.parent.glob("*.tmp"))
    monkeypatch.setattr(exp7284.os, "write", real_write)

    replaced_path = tmp_path / "replace-failure/state.json"

    def fail_after_rename(stage: str) -> None:
        if stage == "after_rename":
            raise OSError("directory stage failed")

    with pytest.raises(exp7284.DurablePublicationError) as replace_error:
        exp7284.durable_replace(replaced_path, b"new\n", stage_hook=fail_after_rename)
    assert replace_error.value.replaced is True
    assert replaced_path.read_bytes() == b"new\n"

    state_path = tmp_path / "valid/state.json"
    exp7284.durable_replace(
        state_path,
        exp7284.transactional.canonical_json_bytes(exp7257.seed_cost_state(4)),
    )
    with pytest.raises(ValueError, match="group size"):
        exp7284.HostGroupCommitController(native, state_path, max_group_size=0, max_wait_ms=0)
    with pytest.raises(ValueError, match="queue bounds"):
        exp7284.HostGroupCommitController(
            native,
            state_path,
            max_group_size=1,
            max_wait_ms=0,
            max_pending_events=0,
        )

    wrapper = exp7284.HostGroupCommitController(
        native, state_path, max_group_size=4, max_wait_ms=10
    )
    assert wrapper.pending_bytes == 0
    assert wrapper.flush_due() is None
    assert wrapper.flush(reason="empty")["disposition"] == "flush_empty"
    assert wrapper.enqueue({})["disposition"] == "invalid_release"
    assert wrapper.enqueue({"event_id": "bad"})["disposition"] == "invalid_release"
    wrapper.enqueue(releases(1)[0], now_ns=0)
    assert wrapper.flush_due(now_ns=1) is None
    assert wrapper.shutdown()["disposition"] == "committed_acknowledged"

    transition_failure = make_wrapper(
        native, tmp_path / "transition", max_group_size=4, max_wait_ms=10
    )
    transition_failure.enqueue(releases(1, offset=40)[0])
    monkeypatch.setattr(
        transition_failure._controller,
        "commit_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            exp7284.exp7240.ArchiveCommitRejected("transition")
        ),
    )
    assert transition_failure.flush(reason="test")["disposition"] == "failed_rolled_back"


def test_scenario_cl_7284_worker_and_builder_paths(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7284-CRASH: worker and measured builder paths are directly checked."""

    class PauseReachedError(RuntimeError):
        pass

    monkeypatch.setattr(
        exp7284.signal,
        "pause",
        lambda: (_ for _ in ()).throw(PauseReachedError("pause")),
    )
    with pytest.raises(PauseReachedError):
        exp7284._pause_for_kill({"stage": "test"})
    assert '"stage":"test"' in capsys.readouterr().out

    state = exp7257.seed_cost_state(4)
    state_path = tmp_path / "worker/state.json"
    release_path = tmp_path / "worker/releases.json"
    exp7284.durable_replace(state_path, exp7284.transactional.canonical_json_bytes(state))
    batch = releases(1, offset=50)
    exp7284.atomic_write(release_path, {"releases": batch})
    binding_path = Path(native.__file__)
    markers: list[dict[str, object]] = []

    def stop(marker):
        markers.append(dict(marker))
        raise PauseReachedError(str(marker["crash_boundary"]))

    monkeypatch.setattr(exp7284, "_pause_for_kill", stop)
    with pytest.raises(PauseReachedError, match="before_write"):
        exp7284._crash_worker(binding_path, state_path, release_path, "before_write")
    exp7284.durable_replace(state_path, exp7284.transactional.canonical_json_bytes(state))
    with pytest.raises(PauseReachedError, match="after_acknowledgment"):
        exp7284._crash_worker(binding_path, state_path, release_path, "after_acknowledgment")
    assert markers[-1]["acknowledged_event_ids"] == ["group-event-050"]
    assert exp7284._restore_worker(binding_path, state_path) == 0
    assert "state_bytes_b64" in capsys.readouterr().out

    exp7284.durable_replace(state_path, exp7284.transactional.canonical_json_bytes(state))
    monkeypatch.setattr(exp7284, "durable_replace", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="unreached crash boundary"):
        exp7284._crash_worker(binding_path, state_path, release_path, "before_write")
    monkeypatch.undo()

    queue = exp7284.run_queue_controls(native, tmp_path / "queue-controls")
    assert queue["all_controls_passed"] is True
    with pytest.raises(ValueError, match="positive multiple"):
        exp7284.run_group_benchmark(native, tmp_path / "bad-bench", events_per_arm=3)
    rows, parity = exp7284._fixture_rows()
    with pytest.raises(ValueError, match="missing benchmark arm"):
        exp7284.reduce_rows(rows[:-1], parity[:-1])

    checks, evidence = exp7284.collect_preconditions(
        exp7284.REPO_ROOT, exp7284.ExperimentPaths.under(tmp_path / "build")
    )
    artifact = exp7284.build_artifact(
        tmp_path / "build",
        exp7284.ExperimentPaths.under(tmp_path / "build"),
        validation_receipts=[exp7284.validation_receipt("fixture", ["true"], 0, "ok", 0.1)],
        precondition_bundle=(checks, evidence),
        events_per_arm=16,
    )
    assert artifact["commit_protocol_ready_score"] == 1
    assert exp7284.validate_artifact(artifact) == []
    assert (tmp_path / "build" / exp7284.RAW_ROWS_RELATIVE).is_file()

    failed = exp7284.check("missing", "external", "path", "file", None, False)
    blocked = exp7284.build_artifact(
        tmp_path,
        exp7284.ExperimentPaths.under(tmp_path / "blocked"),
        validation_receipts=[],
        precondition_bundle=([failed], {"hashes": {}}),
    )
    assert blocked["status"] == "blocked"


def test_req_cl_7284_validator_subprocess_and_runner_failures(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7284: malformed evidence and failed subprocesses cannot publish."""

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        exp7284._read_json(sequence)
    artifact = exp7284.complete_artifact_fixture_for_test()
    original_checksum = exp7284.artifact_checksum
    monkeypatch.setattr(
        exp7284,
        "artifact_checksum",
        lambda _artifact: (_ for _ in ()).throw(TypeError("bad checksum")),
    )
    assert exp7284._checksum_valid({}) is False
    assert "reproducibility_checksum" in exp7284.validate_artifact(artifact)
    monkeypatch.setattr(exp7284, "artifact_checksum", original_checksum)

    monkeypatch.setattr(
        exp7284.exp7270,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    assert exp7284._stream_subprocess(["true"], root=tmp_path, operation="test")["exit_code"] == 0

    binding_path = Path(native.__file__)
    state = exp7257.seed_cost_state(4)
    batch = releases(1, offset=60)

    class FakeProcess:
        def __init__(self, output: str = "") -> None:
            self.stdout = io.StringIO(output)
            self.stderr = io.StringIO("worker stderr")
            self.pid = 12345

        def kill(self) -> None:
            return None

        def wait(self, timeout: int) -> int:
            assert timeout == 5
            return -9

    monkeypatch.setattr(exp7284.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(exp7284.select, "select", lambda *_args, **_kwargs: ([], [], []))
    with pytest.raises(RuntimeError, match="crash worker timeout"):
        exp7284.run_crash_matrix(binding_path, state, batch, tmp_path / "timeout")

    marker = exp7284.canonical_json(
        {
            "crash_boundary": "before_write",
            "issued_event_ids": ["group-event-060"],
            "acknowledged_event_ids": [],
        }
    )
    monkeypatch.setattr(
        exp7284.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(marker + "\n"),
    )
    monkeypatch.setattr(
        exp7284.select,
        "select",
        lambda readers, *_args, **_kwargs: (readers, [], []),
    )
    monkeypatch.setattr(exp7284.os, "kill", lambda *_args: None)
    monkeypatch.setattr(
        exp7284.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="restore failed"),
    )
    with pytest.raises(RuntimeError, match="restore worker failed"):
        exp7284.run_crash_matrix(binding_path, state, batch, tmp_path / "restore-fail")

    with pytest.raises(ValueError, match="run date"):
        exp7284.run_experiment(tmp_path, tmp_path / "bad.json", "bad")
    failed = exp7284.check("missing", "external", "path", "file", None, False)
    monkeypatch.setattr(
        exp7284,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([failed], {"hashes": {}}),
    )
    blocked_output = tmp_path / "blocked.json"
    assert exp7284.run_experiment(tmp_path, blocked_output, exp7284.RUN_DATE)["status"] == "blocked"
    monkeypatch.setattr(exp7284, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid blocked"):
        exp7284.run_experiment(tmp_path, tmp_path / "invalid-blocked.json", exp7284.RUN_DATE)


def test_scenario_cl_7284_candidate_and_main_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-CL-7284-TERMINAL: candidate and CLI failures remain nonzero."""

    passed = exp7284.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7284,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed], {"hashes": {}}),
    )
    monkeypatch.setattr(
        exp7284,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    monkeypatch.setattr(exp7284, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid Exp7284 candidate"):
        exp7284.run_experiment(tmp_path, tmp_path / "invalid.json", exp7284.RUN_DATE)

    complete = exp7284.complete_artifact_fixture_for_test()
    monkeypatch.setattr(exp7284, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    call_count = 0

    def fail_candidate(command: list[str], **_kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "command": command,
            "exit_code": int(call_count == len(exp7284._scoped_validation_commands(tmp_path)) + 1),
            "output": "candidate failed",
        }

    monkeypatch.setattr(exp7284, "_stream_subprocess", fail_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        exp7284.run_experiment(tmp_path, tmp_path / "candidate-fail.json", exp7284.RUN_DATE)

    assert exp7284.main(["--crash-worker"]) == 2
    assert exp7284.main(["--restore-worker"]) == 2

    class WorkerCalledError(RuntimeError):
        pass

    monkeypatch.setattr(
        exp7284,
        "_crash_worker",
        lambda *_args: (_ for _ in ()).throw(WorkerCalledError("crash worker")),
    )
    with pytest.raises(WorkerCalledError):
        exp7284.main(
            [
                "--crash-worker",
                "--binding",
                "binding.so",
                "--state",
                "state.json",
                "--releases",
                "releases.json",
                "--boundary",
                "before_write",
            ]
        )
    monkeypatch.setattr(exp7284, "_restore_worker", lambda *_args: 7)
    assert exp7284.main(["--restore-worker", "--binding", "x", "--state", "y"]) == 7

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp7284.main(["--validate", str(invalid)]) == 2
    outputs: list[Path] = []
    monkeypatch.setattr(
        exp7284,
        "run_experiment",
        lambda _root, output, _date: outputs.append(output) or complete,
    )
    absolute = tmp_path / "absolute.json"
    assert exp7284.main(["--output", str(absolute)]) == 0
    assert exp7284.main(["--output", "relative.json"]) == 0
    assert outputs == [absolute, exp7284.REPO_ROOT / "relative.json"]
    monkeypatch.setattr(
        exp7284,
        "run_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("execution")),
    )
    assert exp7284.main(["--output", str(absolute)]) == 2
