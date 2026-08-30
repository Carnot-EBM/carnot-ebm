"""Tests for the parent-owned durable row checkpoint contract.

Spec refs: REQ-INFRA-6785 and SCENARIO-INFRA-6785-*.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
import os
from pathlib import Path

import pytest

from carnot import durable_row_checkpoint as checkpointing
from carnot import experiment_6785_durable_row_checkpoint_contract as exp


def _manifest(count: int = 3) -> dict:
    return {
        "schema": "probe-manifest-v1",
        "random_seed": 6785,
        "row_ids": [f"probe-{index:02d}" for index in range(1, count + 1)],
    }


def _envelope(store: checkpointing.DurableRowCheckpoint, index: int, attempt: int = 1) -> dict:
    return checkpointing.complete_row_envelope(
        row_id=f"probe-{index:02d}",
        manifest_hash=store.manifest_hash,
        payload={"unit_index": index, "value": index * index},
        attempt=attempt,
        start_receipt={"phase": "start", "unit_index": index},
        end_receipt={"phase": "complete", "unit_index": index},
    )


def _fake_repo(root: Path) -> Path:
    for relative in exp.REQUIRED_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            exp.KNOWN_ISSUE_MARKER if relative == Path("ops/known-issues.md") else "fixture\n"
        )
    return root


def test_scenario_infra_6785_durable_publish_uses_parent_path_replace_and_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6785-DURABLE-PUBLISH syncs both sides of replacement."""
    parent_path = tmp_path / "parent" / "rows.json"
    replace_calls: list[tuple[Path, Path]] = []
    fsync_calls: list[int] = []
    real_replace = checkpointing.os.replace
    real_fsync = checkpointing.os.fsync

    def observed_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        replace_calls.append((Path(source), Path(target)))
        real_replace(source, target)

    def observed_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(checkpointing.os, "replace", observed_replace)
    monkeypatch.setattr(checkpointing.os, "fsync", observed_fsync)

    store = checkpointing.DurableRowCheckpoint(parent_path, _manifest())
    receipt = store.append(_envelope(store, 1))

    assert store.path == parent_path
    assert [target for _, target in replace_calls] == [parent_path, parent_path]
    assert all(source.parent == parent_path.parent for source, _ in replace_calls)
    assert len(fsync_calls) == 4
    assert receipt["accepted"] is True
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert not list(parent_path.parent.glob(f".{parent_path.name}.*.tmp"))


def test_req_infra_6785_envelope_requires_complete_hash_bound_receipts(tmp_path: Path) -> None:
    """REQ-INFRA-6785 refuses incomplete or payload-tampered envelopes."""
    store = checkpointing.DurableRowCheckpoint(tmp_path / "rows.json", _manifest())
    valid = _envelope(store, 1)
    assert valid["payload_hash"] == checkpointing.sha256_json(valid["payload"])

    mutations = []
    for field in checkpointing.ENVELOPE_FIELDS:
        changed = deepcopy(valid)
        del changed[field]
        mutations.append(changed)
    mutations.extend(
        [
            {**valid, "status": "started"},
            {**valid, "attempt": 0},
            {**valid, "row_id": ""},
            {**valid, "manifest_hash": "sha256:wrong"},
            {**valid, "payload_hash": "sha256:wrong"},
            {**valid, "start_receipt": []},
            {**valid, "end_receipt": []},
            {**valid, "unexpected": True},
        ]
    )
    for changed in mutations:
        with pytest.raises(checkpointing.InvalidEnvelopeError):
            store.append(changed)
    assert store.rows == []


def test_scenario_infra_6785_conflicts_refuse_without_byte_change(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6785-CONFLICTS-REFUSE preserves good bytes."""
    path = tmp_path / "rows.json"
    store = checkpointing.DurableRowCheckpoint(path, _manifest())
    original = _envelope(store, 1)
    store.append(original)
    good_bytes = path.read_bytes()

    duplicate = checkpointing.complete_row_envelope(
        row_id=original["row_id"],
        manifest_hash=store.manifest_hash,
        payload=original["payload"],
        attempt=2,
        start_receipt={"phase": "retry"},
        end_receipt={"phase": "complete"},
    )
    duplicate_receipt = store.append(duplicate)
    assert duplicate_receipt["accepted"] is False
    assert duplicate_receipt["duplicate_suppressed"] is True
    assert path.read_bytes() == good_bytes

    conflict = checkpointing.complete_row_envelope(
        row_id=original["row_id"],
        manifest_hash=store.manifest_hash,
        payload={"unit_index": 1, "value": -1},
        attempt=2,
        start_receipt={"phase": "retry"},
        end_receipt={"phase": "complete"},
    )
    with pytest.raises(checkpointing.RowConflictError, match="probe-01"):
        store.append(conflict)
    assert path.read_bytes() == good_bytes

    with pytest.raises(checkpointing.ManifestMismatchError):
        checkpointing.DurableRowCheckpoint(path, {**_manifest(), "random_seed": 99})
    assert path.read_bytes() == good_bytes


def test_req_infra_6785_load_refuses_corrupt_checkpoint(tmp_path: Path) -> None:
    """REQ-INFRA-6785 fails closed when stored state is not valid."""
    path = tmp_path / "rows.json"
    store = checkpointing.DurableRowCheckpoint(path, _manifest())
    store.append(_envelope(store, 1))
    state = json.loads(path.read_text())

    corruptions = [
        {**state, "schema": "wrong"},
        {**state, "manifest_hash": "sha256:wrong"},
        {**state, "revision": 0},
        {**state, "rows": [state["rows"][0], state["rows"][0]], "revision": 2},
        {**state, "extra": True},
    ]
    non_object_manifest = {**state, "manifest": []}
    non_object_manifest["manifest_hash"] = checkpointing.sha256_json([])
    non_list_rows = {**state, "rows": "bad", "revision": 3}
    non_object_row = {**state, "rows": ["bad"], "revision": 1}
    invalid_row = deepcopy(state)
    invalid_row["rows"][0]["status"] = "started"
    corruptions.extend([non_object_manifest, non_list_rows, non_object_row, invalid_row])
    for index, corrupt in enumerate(corruptions):
        corrupt_path = tmp_path / f"corrupt-{index}.json"
        corrupt_path.write_text(json.dumps(corrupt))
        with pytest.raises(checkpointing.CorruptCheckpointError):
            checkpointing.DurableRowCheckpoint(corrupt_path, _manifest())

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json")
    with pytest.raises(checkpointing.CorruptCheckpointError):
        checkpointing.DurableRowCheckpoint(malformed, _manifest())


def test_req_infra_6785_worker_protocol_and_parent_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-6785 sends complete rows and rejects invalid worker boundaries."""
    job = tmp_path / "job.json"
    worker_dir = tmp_path / "worker"
    manifest_hash = checkpointing.sha256_json(exp.frozen_manifest("20260830"))
    job.write_text(
        json.dumps(
            {
                "row_ids": ["probe-01"],
                "manifest_hash": manifest_hash,
                "attempt": 1,
                "worker_directory": str(worker_dir),
            }
        )
    )
    monkeypatch.setattr(exp.sys, "stdin", io.StringIO("ack\n"))
    assert exp._worker_main(job) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["payload"] == {"unit_index": 1, "value": 6786, "random_seed": 6785}
    assert (worker_dir / "worker-started.txt").read_text() == "cpu-only\n"

    monkeypatch.setattr(exp.sys, "stdin", io.StringIO(""))
    assert exp._worker_main(job) == 3
    capsys.readouterr()

    invalid = tmp_path / "invalid-job.json"
    invalid.write_text(
        json.dumps(
            {
                "row_ids": [1],
                "manifest_hash": manifest_hash,
                "attempt": 1,
                "worker_directory": str(worker_dir),
            }
        )
    )
    assert exp._worker_main(invalid) == 2
    assert "worker row_ids" in capsys.readouterr().err


def test_req_infra_6785_worker_parent_refuses_inside_path_and_bad_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6785 keeps the checkpoint outside all worker-owned paths."""
    worker_root = tmp_path / "fixed-worker"
    worker_root.mkdir()
    inside_store = checkpointing.DurableRowCheckpoint(worker_root / "rows.json", _manifest())

    class FixedTemporaryDirectory:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> str:
            return str(worker_root)

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(exp.tempfile, "TemporaryDirectory", FixedTemporaryDirectory)
    with pytest.raises(ValueError, match="inside the worker"):
        exp._run_worker_process(
            checkpoint=inside_store,
            row_ids=[],
            attempt=1,
            interrupt_after=None,
        )

    outside_store = checkpointing.DurableRowCheckpoint(tmp_path / "outside.json", _manifest())

    class MissingPipeProcess:
        stdin = None
        stdout = io.StringIO()
        stderr = io.StringIO()

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *_args, **_kwargs: MissingPipeProcess())
    with pytest.raises(RuntimeError, match="pipes"):
        exp._run_worker_process(
            checkpoint=outside_store,
            row_ids=[],
            attempt=1,
            interrupt_after=None,
        )


def test_req_infra_6785_worker_parent_refuses_early_stop_and_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6785 accepts only the requested complete row sequence."""
    store = checkpointing.DurableRowCheckpoint(tmp_path / "rows.json", _manifest())

    class FakeProcess:
        stdin = io.StringIO()
        stderr = io.StringIO("child error")
        pid = 123

        def __init__(self, output: str) -> None:
            self.stdout = io.StringIO(output)

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess(""))
    with pytest.raises(RuntimeError, match="stopped before"):
        exp._run_worker_process(
            checkpoint=store,
            row_ids=["probe-01"],
            attempt=1,
            interrupt_after=None,
        )

    monkeypatch.setattr(
        exp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess('{"row_id":"probe-02"}\n'),
    )
    with pytest.raises(RuntimeError, match="order changed"):
        exp._run_worker_process(
            checkpoint=store,
            row_ids=["probe-01"],
            attempt=1,
            interrupt_after=None,
        )


def test_scenario_infra_6785_prefix_resume_aggregation_and_cleanup_scope(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6785-PREFIX-SURVIVES and resume run in fresh processes."""
    sentinel = tmp_path / "unrelated-user-file.txt"
    sentinel.write_text("keep me")
    artifact_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "checkpoint" / "rows.json"

    artifact = exp.run_probe(
        run_date="20260830",
        checkpoint_path=checkpoint_path,
        artifact_path=artifact_path,
    )

    assert artifact["durable_checkpoint_ready"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["prefix_rows_preserved"]["count"] == 9
    assert artifact["prefix_rows_preserved"]["worker_temporary_directory_removed"] is True
    assert artifact["fresh_process_resume_rows"]["row_ids"] == [
        f"probe-{index:02d}" for index in range(10, 25)
    ]
    assert artifact["fresh_process_resume_rows"]["idempotent_resume_row_ids"] == []
    assert artifact["duplicate_rows"]["suppressed"] == 1
    assert artifact["conflicting_rows_refused"]["refused"] is True
    assert artifact["changed_manifest_refused"]["refused"] is True
    assert len(artifact["atomic_replace_receipts"]) == 25
    assert len(artifact["fsync_receipts"]) == 25
    assert len(artifact["rows"]) == 25
    unit_rows = [row for row in artifact["rows"] if row["row_kind"] == "probe_unit"]
    assert len(unit_rows) == 24
    assert len({row["row_id"] for row in unit_rows}) == 24
    assert checkpoint_path.is_file()
    assert not Path(artifact["prefix_rows_preserved"]["worker_temporary_path"]).exists()
    assert artifact["cleanup_receipt"]["action"] == "retained_task_checkpoint"
    assert artifact["cleanup_receipt"]["broad_delete_performed"] is False
    assert sentinel.read_text() == "keep me"
    assert exp.validate_artifact(json.loads(artifact_path.read_text())) == []
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)

    stable_bytes = artifact_path.read_bytes()
    assert (
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )
        == artifact
    )
    assert artifact_path.read_bytes() == stable_bytes


def test_req_infra_6785_artifact_validator_names_contract_failures(tmp_path: Path) -> None:
    """REQ-INFRA-6785 validates each load-bearing final contract."""
    artifact = exp.run_probe(
        run_date="20260830",
        checkpoint_path=tmp_path / "rows.json",
        artifact_path=tmp_path / "artifact.json",
    )
    cases = [
        ("field", {key: value for key, value in artifact.items() if key != "status"}),
        ("field", {**artifact, "duration_s": -1.0}),
        ("principle", {**artifact, "field_principles": {}}),
        ("substrate", {**artifact, "inference_substrate": "GPU"}),
        ("verdict", {**artifact, "honest_verdict": "partial"}),
        ("class", {**artifact, "verdict_class": "unknown"}),
        ("rows", {**artifact, "rows": artifact["rows"][:-1]}),
        ("ready", {**artifact, "durable_checkpoint_ready": False}),
        ("gates", {**artifact, "gate_check_summary": {"failed_checks": ["bad"]}}),
        ("checksum", {**artifact, "reproducibility_checksum": "sha256:wrong"}),
    ]
    for expected, changed in cases:
        assert any(expected in error for error in exp.validate_artifact(changed))

    duplicate_rows = deepcopy(artifact)
    duplicate_rows["rows"][1]["row_id"] = duplicate_rows["rows"][0]["row_id"]
    assert any("duplicate" in error for error in exp.validate_artifact(duplicate_rows))
    blocked_wrong_class = {**artifact, "durable_checkpoint_ready": False, "verdict_class": "null"}
    assert any("blocked class" in error for error in exp.validate_artifact(blocked_wrong_class))
    blocked_with_rows = {
        **artifact,
        "durable_checkpoint_ready": False,
        "verdict_class": "blocked",
    }
    assert any(
        "blocked artifact rows" in error for error in exp.validate_artifact(blocked_with_rows)
    )


def test_req_infra_6785_precondition_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-INFRA-6785 stops before the probe when required sources are absent."""
    output = tmp_path / "results" / "blocked.json"
    artifact = exp.run(
        run_date="20260830",
        repo_root=tmp_path / "missing-repository",
        artifact_path=output,
        checkpoint_path=tmp_path / "checkpoint" / "rows.json",
    )
    assert artifact["status"] == "complete_blocked_durable_checkpoint"
    assert artifact["honest_verdict"].startswith("complete_blocked_durable_checkpoint")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["durable_checkpoint_ready"] is False
    assert artifact["gate_check_summary"]["failed_checks"]
    assert artifact["gate_check_summary"]["first_failure"]["observed"] is not True
    assert output.is_file()
    assert not (tmp_path / "checkpoint" / "rows.json").exists()


def test_req_infra_6785_precondition_os_failures_are_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6785 reports storage failures instead of starting a worker."""
    checkpoint = tmp_path / "results/.checkpoints/task/rows.json"
    monkeypatch.setattr(
        exp.Path, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    checks = exp.check_preconditions(tmp_path, checkpoint)
    assert (
        next(item for item in checks if item["check"] == "checkpoint_directory_writable")[
            "observed"
        ]
        is False
    )

    monkeypatch.undo()
    checkpoint.parent.mkdir(parents=True)
    monkeypatch.setattr(exp.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError()))
    checks = exp.check_preconditions(tmp_path, checkpoint)
    assert (
        next(item for item in checks if item["check"] == "atomic_rename_same_filesystem")[
            "observed"
        ]
        is False
    )


def test_req_infra_6785_existing_or_failed_publish_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6785 never overwrites unexplained state or trusts a bad final write."""
    checkpoint_path = tmp_path / "existing.json"
    artifact_path = tmp_path / "artifact.json"
    store = checkpointing.DurableRowCheckpoint(checkpoint_path, exp.frozen_manifest("20260830"))
    store.append(
        checkpointing.complete_row_envelope(
            row_id="probe-01",
            manifest_hash=store.manifest_hash,
            payload=exp._worker_payload("probe-01"),
            attempt=1,
            start_receipt={},
            end_receipt={},
        )
    )
    with pytest.raises(ValueError, match="without a valid final artifact"):
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )

    empty_path = tmp_path / "existing-empty.json"
    checkpointing.DurableRowCheckpoint(empty_path, exp.frozen_manifest("20260830"))
    with pytest.raises(RuntimeError, match="initialization"):
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=empty_path,
            artifact_path=artifact_path,
        )


def test_req_infra_6785_final_validation_and_retention_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6785 keeps the checkpoint until a valid final artifact exists."""
    real_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation"):
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=tmp_path / "validation-rows.json",
            artifact_path=tmp_path / "validation-artifact.json",
        )

    monkeypatch.setattr(exp, "validate_artifact", real_validate)
    real_write = exp.atomic_write_json
    corrupt_artifact_path = tmp_path / "corrupt-artifact.json"

    def corrupt_final(path: Path, value: object) -> dict:
        receipt = real_write(path, value)
        if Path(path) == corrupt_artifact_path:
            changed = deepcopy(value)
            changed["reproducibility_checksum"] = "sha256:wrong"
            real_write(path, changed)
        return receipt

    monkeypatch.setattr(exp, "atomic_write_json", corrupt_final)
    with pytest.raises(RuntimeError, match="hash verification"):
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=tmp_path / "corrupt-rows.json",
            artifact_path=corrupt_artifact_path,
        )

    disappearing_checkpoint = tmp_path / "disappearing-rows.json"
    disappearing_artifact = tmp_path / "disappearing-artifact.json"

    def remove_checkpoint_after_publish(path: Path, value: object) -> dict:
        receipt = real_write(path, value)
        if Path(path) == disappearing_artifact:
            disappearing_checkpoint.unlink()
        return receipt

    monkeypatch.setattr(exp, "atomic_write_json", remove_checkpoint_after_publish)
    with pytest.raises(RuntimeError, match="checkpoint disappeared"):
        exp.run_probe(
            run_date="20260830",
            checkpoint_path=disappearing_checkpoint,
            artifact_path=disappearing_artifact,
        )


def test_req_infra_6785_run_and_parent_main_success_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-6785 uses the gated parent paths through both public entry points."""
    repo = _fake_repo(tmp_path / "repo")
    checkpoint = repo / exp.CHECKPOINT_RELATIVE_PATH
    artifact_path = repo / exp.ARTIFACT_RELATIVE_PATH
    artifact = exp.run(run_date="20260830", repo_root=repo)
    assert artifact["durable_checkpoint_ready"] is True
    assert checkpoint.is_file()
    assert artifact_path.is_file()

    outside_artifact = tmp_path / "main-blocked.json"
    assert (
        exp.main(
            [
                "--date",
                "20260830",
                "--artifact-path",
                str(outside_artifact),
                "--checkpoint-path",
                str(tmp_path / "outside-checkpoint.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verdict_class"] == "blocked"


def test_req_infra_6785_date_and_worker_arguments_fail_closed(tmp_path: Path) -> None:
    """REQ-INFRA-6785 rejects malformed dates and worker jobs."""
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.run_probe(
            run_date="2026-08-30",
            checkpoint_path=tmp_path / "rows.json",
            artifact_path=tmp_path / "artifact.json",
        )
    bad_job = tmp_path / "bad-job.json"
    bad_job.write_text("{}")
    assert exp.main(["--worker-job", str(bad_job)]) == 2
