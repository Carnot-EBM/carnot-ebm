"""Tests for the persistent native FIFO archive controller.

Spec refs: REQ-CL-7256, SCENARIO-CL-7256-*, REQ-RUSTPY-7256, and
SCENARIO-RUSTPY-7256-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256


@pytest.fixture(scope="session")
def native_extension() -> Path:
    """REQ-RUSTPY-7256: build isolated bytes for the running interpreter."""

    extension, _ = exp7256.build_native_extension(exp7256.REPO_ROOT, show_progress=False)
    return extension


@pytest.fixture(scope="session")
def native(native_extension: Path):
    """SCENARIO-RUSTPY-7256-TYPED: import the selected compiled module."""

    return exp7230.load_native_extension(native_extension)


def _release(index: int, label: str, family: str = "lower_bound") -> dict[str, object]:
    """Make one due support release with the complete typed contract."""

    return {
        "event_id": f"release-{index}",
        "family_id": family,
        "numeric_value": (index * 7) % 33,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def test_scenario_cl_7256_typed_controller_matches_python(native) -> None:
    """SCENARIO-CL-7256-PARITY: typed calls preserve exact FIFO state."""

    reference = exp7240.ArchivedBeliefController()
    controller = exp7256.PersistentNativeArchiveController(native)
    for index in range(40):
        event = {
            "event_id": f"probe-{index}",
            "family_id": exp7240.FAMILIES[index % 4],
            "numeric_value": index * 11,
        }
        assert controller.predict(event) == reference.predict(event)
        assert controller.energy("accept", event) == reference.energy("accept", event)
        block = [event, {**event, "event_id": f"alternate-{index}", "numeric_value": index + 3}]
        ranks = {str(row["event_id"]): rank for rank, row in enumerate(reversed(block))}
        assert controller.select_request(block, ranks) == reference.select_request(block, ranks)
        release = _release(index, "accept" if index % 2 else "reject", exp7240.FAMILIES[index % 4])
        expected = reference.commit_batch(
            [release], current_cycle=index, expected_parent_hash=reference.state_hash()
        )
        actual = controller.commit_batch(
            [release], current_cycle=index, expected_parent_hash=controller.state_hash()
        )
        assert actual["operations"] == expected["operations"]
        assert controller.state_bytes() == reference.state_bytes()

    counters = controller.conversion_counts()
    assert counters["typed_event_calls"] > 0
    assert counters["hot_path_json_parse_count"] == 0
    assert counters["active_reconstruction_count"] == 0
    restored = exp7256.PersistentNativeArchiveController.from_state_with_binding(
        native, controller.state_dict()
    )
    assert restored.state_bytes() == controller.state_bytes()


def test_scenario_rustpy_7256_snapshot_transaction_and_rollback(native, tmp_path: Path) -> None:
    """SCENARIO-RUSTPY-7256-SNAPSHOT: rejected changes preserve parent bytes."""

    controller = exp7256.PersistentNativeArchiveController(native)
    parent = controller.state_bytes()
    with pytest.raises(ValueError):
        exp7256.PersistentNativeArchiveController.from_snapshot(native, "{}")
    assert controller.state_bytes() == parent
    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_parent"):
        controller.commit_batch(
            [_release(0, "accept")], current_cycle=0, expected_parent_hash="bad"
        )
    assert controller.state_bytes() == parent

    durable = tmp_path / "controller.json"
    controller.save(durable)
    receipt = controller.commit_batch(
        [_release(0, "reject")],
        current_cycle=0,
        expected_parent_hash=controller.state_hash(),
        state_path=durable,
    )
    child = controller.state_bytes()
    restored = exp7256.PersistentNativeArchiveController.load(native, durable)
    assert restored.state_bytes() == child
    rollback = controller.rollback(receipt, state_path=durable)
    assert rollback["byte_identical"] is True
    assert controller.state_bytes() == parent == durable.read_bytes()
    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_rollback"):
        controller.rollback(receipt)

    before_failure = controller.state_bytes()
    with (
        patch.object(exp7256.transactional, "_atomic_write", side_effect=OSError("interrupted")),
        pytest.raises(OSError, match="interrupted"),
    ):
        controller.commit_batch(
            [_release(1, "accept")],
            current_cycle=1,
            expected_parent_hash=controller.state_hash(),
            state_path=durable,
        )
    assert controller.state_bytes() == before_failure
    assert durable.read_bytes() == before_failure


def test_scenario_cl_7256_eight_stream_replay_and_restart(native, native_extension: Path) -> None:
    """SCENARIO-CL-7256-RESTORE: eight streams and a fresh process stay exact."""

    parity_rows, checkpoints = exp7256.run_differential_replay(
        native, stream_count=8, event_limit=192
    )
    assert len(checkpoints) == 8
    assert {row["arm"] for row in parity_rows} == {
        "python_reference",
        "old_native_wrapper",
        "persistent_native_controller",
    }
    assert all(row["mismatch_count"] == 0 for row in parity_rows)
    persistent = [row for row in parity_rows if row["arm"] == "persistent_native_controller"]
    assert all(row["completed_events"] == 192 for row in persistent)
    assert all(row["archive_slots_configured"] == 4 for row in persistent)
    assert max(row["maximum_archive_slots"] for row in persistent) <= 4
    receipt = exp7256.run_fresh_process_continuation(native_extension, checkpoints[-1])
    assert receipt["passed"] is True
    assert receipt["same_next_decisions"] is True
    assert receipt["module_file"] == str(native_extension.resolve())


def test_req_cl_7256_artifact_schema_and_circular_verdict() -> None:
    """SCENARIO-CL-7256-TERMINAL: readiness stays circular and unscored for speed."""

    artifact = exp7256.complete_artifact_fixture_for_test()
    assert exp7256.validate_artifact(artifact) == []
    assert artifact["native_controller_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["throughput_value_score"] is None
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False

    for field, value, error in (
        ("verdict_class", "positive", "verdict_class"),
        ("native_controller_ready_score", 0, "ready_score"),
        ("model_invoked", True, "model_invoked"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp7256.artifact_checksum(changed)
        assert error in exp7256.validate_artifact(changed)

    failed = exp7256.check("missing", "upstream", "field", 1, None, False)
    blocked = exp7256.blocked_artifact_for_test(failed)
    assert exp7256.validate_artifact(blocked) == []
    assert blocked["rows"] == []
    assert blocked["verdict_class"] == "blocked"


def test_req_cl_7256_thin_entrypoint_and_validation_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7256: the runnable entrypoint delegates and validation is read-only."""

    script = exp7256.REPO_ROOT / "scripts/experiments/experiment_7256_v638_native_controller.py"
    calls: list[list[str] | None] = []
    original_main = exp7256.main
    monkeypatch.setattr(exp7256, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7256.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7256, "main", original_main)

    artifact = exp7256.complete_artifact_fixture_for_test()
    output = tmp_path / "artifact.json"
    exp7256.atomic_write(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp7256.main(["--validate", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp7256.main(["--validate", str(output)]) == 2


def test_req_cl_7256_native_boundary_defensive_paths(
    native, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-RUSTPY-7256: malformed typed, durable, and rollback inputs fail closed."""

    assert exp7256.ExperimentPaths.under(tmp_path).artifact.parent == tmp_path
    assert (
        exp7256.ExperimentPaths.defaults().artifact == exp7256.REPO_ROOT / exp7256.RESULT_RELATIVE
    )
    assert exp7256._writable(tmp_path / "missing/child/result.json") is True
    exp7256.progress(9, "test", "visible")
    assert "[phase 9 test] visible" in capsys.readouterr().out

    controller = exp7256.PersistentNativeArchiveController(native)
    invalid = {"event_id": "bad", "family_id": "bad", "numeric_value": 0}
    assert controller.predict(invalid) == ("abstain", 0.0)
    assert controller.energy("bad", invalid)["status"] == "unknown_label"
    assert controller.energy("accept", invalid)["status"] == "unknown_input"
    assert controller.select_request([invalid], {"bad": 0}) == invalid
    controller._durable_parent_matches(tmp_path / "absent.json", controller.state_hash())

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(exp7240.ArchiveCommitRejected, match="corrupt_durable_state"):
        controller._durable_parent_matches(malformed, controller.state_hash())
    stale = exp7256.PersistentNativeArchiveController(native)
    stale.commit_batch(
        [_release(5, "accept")], current_cycle=5, expected_parent_hash=stale.state_hash()
    )
    stale.save(malformed)
    with pytest.raises(exp7240.ArchiveCommitRejected, match="stale_durable_parent"):
        controller._durable_parent_matches(malformed, controller.state_hash())

    with pytest.raises(exp7240.ArchiveCommitRejected, match="future_release"):
        controller.commit_batch(
            [_release(10, "accept")], current_cycle=0, expected_parent_hash=controller.state_hash()
        )
    real_native = controller._native
    controller._native = SimpleNamespace(
        commit_batch=lambda *_args: (_ for _ in ()).throw(ValueError("native rejected"))
    )
    with pytest.raises(exp7240.ArchiveCommitRejected, match="native rejected"):
        controller.commit_batch(
            [_release(0, "accept")], current_cycle=0, expected_parent_hash="ignored"
        )
    controller._native = real_native

    receipt = controller.commit_batch(
        [_release(0, "reject")], current_cycle=0, expected_parent_hash=controller.state_hash()
    )
    corrupt = dict(receipt, parent_bytes_b64="not-base64")
    with pytest.raises(exp7240.ArchiveCommitRejected, match="invalid_rollback_receipt"):
        controller.rollback(corrupt)
    wrong_parent = dict(receipt, parent_hash="sha256:" + "0" * 64)
    with pytest.raises(exp7240.ArchiveCommitRejected, match="rollback_parent_hash"):
        controller.rollback(wrong_parent)
    controller._native = SimpleNamespace(
        state_hash=lambda: receipt["new_state_hash"],
        rollback=lambda *_args: (_ for _ in ()).throw(ValueError("native rollback rejected")),
    )
    with pytest.raises(exp7240.ArchiveCommitRejected, match="native rollback rejected"):
        controller.rollback(receipt)

    malformed_state = exp7256.PersistentNativeArchiveController(native)
    malformed_state._native = SimpleNamespace(snapshot_state=lambda: "[]")
    with pytest.raises(ValueError, match="invalid_archive_state_object"):
        malformed_state.state_dict()


def test_req_rustpy_7256_build_receipts_and_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7256: isolated build, wheel, suffix, and cleanup paths are explicit."""

    release = tmp_path / exp7256.TARGET_RELATIVE / "release"
    release.mkdir(parents=True)
    library = release / "libcarnot_python.so"
    library.write_bytes(b"native")
    environment = {
        "PYO3_PYTHON": sys.executable,
        "CARGO_TARGET_DIR": str(tmp_path / exp7256.TARGET_RELATIVE),
    }
    monkeypatch.setattr(exp7256.exp7217, "interpreter_build_environment", lambda *_: environment)
    monkeypatch.setattr(
        exp7256.exp7217,
        "_stream_process",
        lambda *_args, **_kwargs: {"command": ["cargo", "build"], "exit_code": 0, "output": "ok"},
    )
    extension, receipt = exp7256.build_native_extension(tmp_path)
    assert extension.read_bytes() == b"native"
    assert Path(receipt["wheel_path"]).is_file()
    assert receipt["wheel_sha256"].startswith("sha256:")

    library.unlink()
    with pytest.raises(RuntimeError, match="native build output missing"):
        exp7256.build_native_extension(tmp_path, show_progress=False)
    library.write_bytes(b"native")
    with (
        patch.object(exp7256.sysconfig, "get_config_var", return_value=None),
        pytest.raises(RuntimeError, match="extension suffix unavailable"),
    ):
        exp7256.build_native_extension(tmp_path, show_progress=False)
    with (
        patch.object(exp7256.os, "replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        exp7256.build_native_extension(tmp_path, show_progress=False)
    assert not list((tmp_path / exp7256.LOAD_RELATIVE).glob("*.tmp"))


def test_req_cl_7256_preconditions_negative_controls_and_heartbeat(
    native, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7256: preflight and negative evidence execute before terminal output."""

    paths = exp7256.ExperimentPaths.under(tmp_path)
    checks, sources = exp7256.collect_preconditions(exp7256.REPO_ROOT, paths)
    assert exp7256.gate_summary(checks)["passed"] is True
    assert sources["upstream"]["native_archive_ready_score"] == 1
    with patch.object(exp7256.exp7243, "artifact_checksum", side_effect=TypeError("bad")):
        defensive, _ = exp7256.collect_preconditions(exp7256.REPO_ROOT, paths)
    assert any(
        row["check"] == "exp7243_ready_and_complete" and not row["passed"] for row in defensive
    )

    rows = exp7256.run_negative_controls(native, tmp_path / "negative-durable.json")
    assert len(rows) == 3
    assert all(row["passed"] for row in rows)

    clock = iter(float(index * 61) for index in range(100_000))
    with patch.object(exp7256.time, "monotonic", side_effect=lambda: next(clock)):
        parity, _ = exp7256.run_differential_replay(native, stream_count=1, event_limit=8)
    assert len(parity) == 3
    assert "completed_events=" in capsys.readouterr().out


def test_req_cl_7256_validator_file_checks_and_exception_paths(tmp_path: Path) -> None:
    """REQ-CL-7256: cold validation rejects corrupt hashes and malformed rows."""

    artifact = exp7256.complete_artifact_fixture_for_test()
    with patch.object(exp7256, "artifact_checksum", side_effect=TypeError("bad")):
        assert "reproducibility_checksum" in exp7256.validate_artifact(artifact)

    malformed = deepcopy(artifact)
    malformed["parity_rows"] = [{"arm": "python_reference", "event_rows": [{"expected": {}}]}] * 24
    malformed["reproducibility_checksum"] = exp7256.artifact_checksum(malformed)
    assert "raw_row_reducer" in exp7256.validate_artifact(malformed)

    module = tmp_path / f"_rust{exp7256.sysconfig.get_config_var('EXT_SUFFIX')}"
    wheel = tmp_path / "native.whl"
    source = tmp_path / "source.txt"
    module.write_bytes(b"module")
    wheel.write_bytes(b"wheel")
    source.write_text("source", encoding="utf-8")
    artifact["native_binary_receipt"].update(
        {
            "module_file": str(module),
            "module_sha256": exp7256.sha256_file(module),
            "wheel_path": str(wheel),
            "wheel_sha256": exp7256.sha256_file(wheel),
        }
    )
    artifact["source_artifact_hashes"] = {str(source): exp7256.sha256_file(source)}
    artifact["reproducibility_checksum"] = exp7256.artifact_checksum(artifact)
    assert exp7256.validate_artifact(artifact, check_files=True, root=tmp_path) == []
    source.write_text("changed", encoding="utf-8")
    module.write_bytes(b"changed")
    wheel.write_bytes(b"changed")
    errors = exp7256.validate_artifact(artifact, check_files=True, root=tmp_path)
    assert {"source_artifact_hashes", "native_module_hash", "native_wheel_hash"} <= set(errors)


def test_req_cl_7256_build_artifact_and_dispatch_paths(
    native, native_extension: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7256: blocked, complete, failed-gate, and CLI dispatch paths stay terminal."""

    paths = exp7256.ExperimentPaths.under(tmp_path)
    upstream = json.loads((exp7256.REPO_ROOT / exp7256.UPSTREAM_RELATIVE).read_text())
    passed = exp7256.check("fixture", "fixture", "value", 1, 1, True)
    failed = exp7256.check("missing", "upstream", "value", 1, None, False)
    sources = {
        "hashes": {str(exp7256.UPSTREAM_RELATIVE): exp7256.EXPECTED_EXP7243_SHA256},
        "upstream": upstream,
        "quarantine": {"quarantined": False},
    }
    parity = exp7256._fixture_parity_rows()
    checkpoint = exp7256.exp7243._continuation_fixture(
        exp7240.ArchivedBeliefController().state_dict(), exp7256.STREAM_SEEDS[-1]
    )
    fixture = exp7256.complete_artifact_fixture_for_test()
    fresh = deepcopy(fixture["fresh_process_receipt"])
    fresh.update(
        {
            "command": ["fresh"],
            "exit_code": 0,
            "stdout_sha256": "sha256:a",
            "stderr_sha256": "sha256:b",
        }
    )
    negatives = deepcopy(fixture["negative_control_rows"])
    wheel = tmp_path / "native.whl"
    wheel.write_bytes(b"wheel")
    build_receipt = {
        "command": ["cargo", "build"],
        "exit_code": 0,
        "output": "ok",
        "built_library": str(native_extension),
        "built_library_sha256": exp7256.sha256_file(native_extension),
        "wheel_path": str(wheel),
        "wheel_sha256": exp7256.sha256_file(wheel),
        "compiler_command": ["rustc", "--version"],
        "compiler_exit_code": 0,
        "compiler_output": "rustc fixture",
        "PYO3_PYTHON": sys.executable,
        "CARGO_TARGET_DIR": str(tmp_path / "target"),
    }
    identity = exp7256._native_identity(native_extension, native, build_receipt)
    receipt = exp7256._validation_receipt("test", ["test"], 0, "ok")
    exp7256.atomic_write(paths.validation_sidecar, {"validation_receipts": [receipt]})

    common = (
        patch.object(exp7256, "collect_preconditions", return_value=([passed], sources)),
        patch.object(
            exp7256, "build_native_extension", return_value=(native_extension, build_receipt)
        ),
        patch.object(exp7256.exp7230, "load_native_extension", return_value=native),
        patch.object(exp7256, "_native_identity", return_value=identity),
        patch.object(exp7256, "run_differential_replay", return_value=(parity, [checkpoint])),
        patch.object(exp7256, "run_fresh_process_continuation", return_value=fresh),
        patch.object(exp7256, "run_negative_controls", return_value=negatives),
    )
    with common[0], common[1], common[2], common[3], common[4], common[5], common[6]:
        artifact = exp7256.build_artifact(exp7256.REPO_ROOT, paths)
    assert artifact["status"] == "complete"
    assert artifact["native_controller_ready_score"] == 1
    assert receipt in artifact["validation_receipts"]

    with patch.object(exp7256, "collect_preconditions", return_value=([failed], sources)):
        blocked = exp7256.build_artifact(exp7256.REPO_ROOT, paths)
    assert blocked["status"] == "blocked"

    broken = deepcopy(fresh)
    broken["passed"] = False
    common = (
        patch.object(exp7256, "collect_preconditions", return_value=([passed], sources)),
        patch.object(
            exp7256, "build_native_extension", return_value=(native_extension, build_receipt)
        ),
        patch.object(exp7256.exp7230, "load_native_extension", return_value=native),
        patch.object(exp7256, "_native_identity", return_value=identity),
        patch.object(exp7256, "run_differential_replay", return_value=(parity, [checkpoint])),
        patch.object(exp7256, "run_fresh_process_continuation", return_value=broken),
        patch.object(exp7256, "run_negative_controls", return_value=negatives),
    )
    with common[0], common[1], common[2], common[3], common[4], common[5], common[6]:
        with pytest.raises(RuntimeError, match="semantic parity"):
            exp7256.build_artifact(exp7256.REPO_ROOT, paths)

    with pytest.raises(ValueError, match="run date"):
        exp7256.run_experiment(tmp_path, paths.artifact, "bad")
    with (
        patch.object(exp7256, "build_artifact", return_value=fixture),
        patch.object(exp7256, "validate_artifact", return_value=[]),
    ):
        assert exp7256.run_experiment(tmp_path, paths.artifact, exp7256.RUN_DATE) == fixture
    with (
        patch.object(exp7256, "build_artifact", return_value=fixture),
        patch.object(exp7256, "validate_artifact", return_value=["bad"]),
        pytest.raises(ValueError, match="invalid Exp7256"),
    ):
        exp7256.run_experiment(tmp_path, paths.artifact, exp7256.RUN_DATE)

    missing = tmp_path / "missing.json"
    assert exp7256.main(["--validate", str(missing)]) == 2
    assert exp7256.main(["--date", "bad"]) == 2
    with patch.object(exp7256, "run_experiment", return_value=fixture):
        assert exp7256.main(["--date", exp7256.RUN_DATE, "--output", "relative.json"]) == 0
    with patch.object(exp7256, "run_experiment", side_effect=RuntimeError("owned")):
        assert exp7256.main(["--date", exp7256.RUN_DATE, "--output", str(paths.artifact)]) == 2

    monkeypatch.setattr(sys, "argv", ["experiment_7256", "--validate", str(paths.artifact)])
    exp7256.atomic_write(paths.artifact, fixture)
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit) as exit_info:
        runpy.run_module("carnot.experiment_7256_v638_native_controller", run_name="__main__")
    assert exit_info.value.code == 0
