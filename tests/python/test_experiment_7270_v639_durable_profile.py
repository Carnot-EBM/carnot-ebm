"""Tests for the durable archive-event cost profile.

Spec refs: REQ-RUSTPY-7270 and SCENARIO-RUSTPY-7270-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
import sys
import time

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7270_v639_durable_profile as exp7270


@pytest.fixture(scope="session")
def native():
    """SCENARIO-RUSTPY-7270-BOUNDED-REPLAY: load authenticated shipped bytes."""

    source = exp7257._read_upstream_summary(
        exp7257.UPSTREAM_PATH, exp7257.REPO_ROOT / exp7257.EXCLUSION_RELATIVE
    )
    return exp7230.load_native_extension(Path(source["identity"]["module_file"]))


def test_req_rustpy_7270_preconditions_authenticate_both_inputs(tmp_path: Path) -> None:
    """REQ-RUSTPY-7270: changed upstream bytes block before measurement."""

    paths = exp7270.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7270.collect_preconditions(exp7270.REPO_ROOT, paths)
    assert exp7270.gate_summary(checks)["passed"] is True
    assert evidence["exp7257"]["status"] == "complete"
    assert evidence["exp7256"]["native_controller_ready_score"] == 1
    assert evidence["quarantine"]["exp7257"]["quarantined"] is False
    assert all(paths.writable_targets())

    changed = tmp_path / "changed.json"
    changed.write_text('{"status":"complete"}\n', encoding="utf-8")
    failed, _ = exp7270.collect_preconditions(exp7270.REPO_ROOT, paths, exp7257_path=changed)
    assert exp7270.gate_summary(failed)["passed"] is False


def test_scenario_rustpy_7270_cold_reducer_retains_outlier_and_diagnostics() -> None:
    """SCENARIO-RUSTPY-7270-COLD-REDUCTION: all paired rows remain visible."""

    rows = exp7257.synthetic_cost_rows(python_ns=300, old_native_ns=200, persistent_ns=100)
    outlier = next(
        row
        for row in rows
        if row["archive_capacity"] == 4
        and row["batch_size"] == 1
        and row["block"] == 0
        and row["arm"] == "persistent_native_controller"
    )
    outlier["total_block_ns"] = 10_000
    outlier["total_event_ns"] = 10_000.0
    outlier["metric"] = 10_000.0
    outlier["component_ns"]["residual_unaccounted_ns"] += 9_900

    summary = exp7270.reduce_saved_cost_rows(rows)
    cell = next(
        row
        for row in summary["paired_ratio_diagnostics"]
        if row["archive_capacity"] == 4
        and row["batch_size"] == 1
        and row["comparison"] == "python_reference_over_persistent_native_controller"
    )
    expected = (29 * 3.0 + 0.03) / 30
    assert cell["primary_arithmetic_mean_of_paired_ratios"] == pytest.approx(expected)
    assert cell["secondary_ratio_of_total_time"] == pytest.approx(9_000 / 12_900)
    assert cell["secondary_mean_paired_log_ratio"] == pytest.approx(
        (29 * math.log(3) + math.log(0.03)) / 30
    )
    assert cell["paired_block_count"] == 30
    assert cell["outlier_rows_removed"] == 0
    assert summary["saved_row_count"] == 540
    assert summary["censored_block_arm_count"] == 0
    assert len(summary["component_profiles"]) == 18
    profile = next(
        row
        for row in summary["component_profiles"]
        if row["archive_capacity"] == 4
        and row["batch_size"] == 1
        and row["arm"] == "persistent_native_controller"
    )
    assert profile["p95_total_block_ns"] >= profile["p50_total_block_ns"]
    assert sum(profile["component_fraction_of_total"].values()) == pytest.approx(1.0)
    assert profile["component_sum_matches_total"] is True
    assert profile["storage_devices"] == [1]
    assert len(profile["arm_order_by_block"]) == 30

    with pytest.raises(ValueError, match="saved row roster"):
        exp7270.reduce_saved_cost_rows(rows[:-1])


def test_scenario_rustpy_7270_bounded_replay_splits_durable_operations(
    native, tmp_path: Path
) -> None:
    """SCENARIO-RUSTPY-7270-BOUNDED-REPLAY: each exclusive operation is timed."""

    rows = exp7270.instrument_durable_blocks(
        native, tmp_path / "profile", blocks=2, max_duration_s=30.0
    )
    assert len(rows) == 2
    assert {row["block"] for row in rows} == {0, 1}
    required = {
        "lookup_ns",
        "query_ns",
        "update_ns",
        "snapshot_construction_ns",
        "encoding_ns",
        "validation_ns",
        "metadata_ns",
        "data_write_ns",
        "file_sync_ns",
        "rename_ns",
        "directory_sync_ns",
        "parent_reload_ns",
        "restore_ns",
        "instrumentation_overhead_ns",
    }
    assert all(set(row["exclusive_component_ns"]) == required for row in rows)
    assert all(sum(row["exclusive_component_ns"].values()) == row["total_event_ns"] for row in rows)
    assert all(row["parity_passed"] and row["durability_passed"] for row in rows)
    assert all(row["durable_bytes"] == row["snapshot_bytes"] for row in rows)
    assert all(row["transferred_bytes"] >= row["snapshot_bytes"] * 3 for row in rows)
    assert all(row["comparison_count"] > 0 and row["bit_operation_count"] > 0 for row in rows)
    reduced = exp7270.reduce_component_rows(rows, planned_blocks=2)
    assert reduced["completed_blocks"] == 2
    assert reduced["component_sum_failure_count"] == 0
    assert reduced["parity_failure_count"] == 0
    assert reduced["durability_failure_count"] == 0

    with pytest.raises(ValueError, match="between 1 and 12"):
        exp7270.instrument_durable_blocks(native, tmp_path, blocks=13)


def test_scenario_rustpy_7270_amdahl_and_journal_warrant_are_fail_closed() -> None:
    """SCENARIO-RUSTPY-7270-BOUND: fixed costs cap speed and sync can veto."""

    envelope = exp7270.acceleration_envelope(total_ns=100.0, fixed_ns=60.0)
    assert envelope["unaccelerated_fraction"] == pytest.approx(0.6)
    assert envelope["best_case_speedup"] == pytest.approx(5 / 3)
    assert envelope["targets"]["10x"]["feasible"] is False
    assert envelope["targets"]["100x"]["feasible"] is False

    feasible = exp7270.acceleration_envelope(total_ns=100.0, fixed_ns=5.0)
    assert feasible["targets"]["10x"]["required_replaceable_speedup"] == 19.0
    assert feasible["targets"]["100x"]["feasible"] is False

    supported = exp7270.journal_warrant(
        replaceable_fraction=0.60,
        conservative_delta_log_speedup=1.7,
        sync_fraction=0.20,
    )
    assert supported["journal_optimization_warranted_score"] == 1
    vetoed = exp7270.journal_warrant(
        replaceable_fraction=0.70,
        conservative_delta_log_speedup=2.0,
        sync_fraction=0.51,
    )
    assert vetoed["journal_optimization_warranted_score"] == 0
    assert vetoed["sync_dominates"] is True


def test_req_rustpy_7270_complete_and_blocked_artifacts_fail_closed() -> None:
    """REQ-RUSTPY-7270: terminal schema and recomputed findings must agree."""

    artifact = exp7270.complete_artifact_fixture_for_test()
    assert exp7270.validate_artifact(artifact) == []
    assert artifact["durable_profile_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == {
        "attempted_model_loads": 0,
        "completed_model_loads": 0,
        "attempted_generation_calls": 0,
        "completed_generation_calls": 0,
        "usable_answers": 0,
    }

    changed = deepcopy(artifact)
    changed["journal_optimization_warranted_score"] = 1
    changed["reproducibility_checksum"] = exp7270.artifact_checksum(changed)
    assert "journal_warrant" in exp7270.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["component_rows"][0]["total_event_ns"] += 1
    changed["rows"] = [*changed["saved_cost_rows"], *changed["component_rows"]]
    changed["reproducibility_checksum"] = exp7270.artifact_checksum(changed)
    assert "component_reducer" in exp7270.validate_artifact(changed)

    failed = exp7270.check("missing", "exp7257", "path", "file", None, False)
    blocked = exp7270.blocked_artifact_for_test(failed)
    assert exp7270.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == blocked["component_rows"] == []
    assert blocked["gate_check_summary"]["failed_check"] == "missing"


def test_req_rustpy_7270_thin_entrypoint_and_read_only_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7270: the script delegates and validation never rewrites bytes."""

    script = exp7270.REPO_ROOT / "scripts/experiments/experiment_7270_v639_durable_profile.py"
    calls: list[object] = []
    original_main = exp7270.main
    monkeypatch.setattr(exp7270, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7270.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7270, "main", original_main)

    artifact = exp7270.complete_artifact_fixture_for_test()
    path = tmp_path / "candidate.json"
    exp7270.atomic_write(path, artifact)
    before = path.read_bytes()
    assert exp7270.main(["--validate", str(path)]) == 0
    assert path.read_bytes() == before
    path.write_text("{}", encoding="utf-8")
    assert exp7270.main(["--validate", str(path)]) == 2
    assert exp7270.main(["--date", "bad", "--output", str(path)]) == 2


def test_scenario_rustpy_7270_e2e_candidate_publish_is_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-RUSTPY-7270-E2E: only the validated raw candidate is published."""

    artifact = exp7270.complete_artifact_fixture_for_test()
    passed_check = exp7270.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7270,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([passed_check], {"fixture": True}),
    )
    monkeypatch.setattr(exp7270, "build_artifact", lambda *_args, **_kwargs: deepcopy(artifact))
    commands: list[list[str]] = []

    def validate(command: list[str], **_kwargs):
        commands.append(command)
        return {"command": command, "exit_code": 0, "output": "ok\n"}

    monkeypatch.setattr(exp7270, "_stream_subprocess", validate)
    output = tmp_path / "terminal.json"
    result = exp7270.run_experiment(tmp_path, output, exp7270.RUN_DATE)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert len(commands) == 9
    candidate_commands = commands[-3:]
    assert all(str(tmp_path / exp7270.RAW_CANDIDATE_RELATIVE) in row for row in candidate_commands)

    monkeypatch.setattr(
        exp7270,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 1, "output": "failed"},
    )
    output.unlink()
    with pytest.raises(RuntimeError, match="scoped validation failed"):
        exp7270.run_experiment(tmp_path, output, exp7270.RUN_DATE)
    assert not output.exists()


def test_req_rustpy_7270_stream_wrapper_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7270: subprocess failures remain explicit validation evidence."""

    monkeypatch.setattr(
        exp7270.exp7217,
        "_stream_process",
        lambda *_args, **_kwargs: {"command": ["ok"], "exit_code": 0, "output": "ok"},
    )
    assert exp7270._stream_subprocess(["ok"], root=tmp_path, operation="ok")["exit_code"] == 0

    def fail(*_args, **_kwargs):
        raise RuntimeError("bad command")

    monkeypatch.setattr(exp7270.exp7217, "_stream_process", fail)
    receipt = exp7270._stream_subprocess(["bad"], root=tmp_path, operation="bad")
    assert receipt["exit_code"] == 1
    assert "bad command" in receipt["output"]


def test_req_rustpy_7270_build_reconciles_reducers_without_rebenchmarking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7270: the builder joins saved rows to bounded replay evidence."""

    saved = exp7257.synthetic_cost_rows(python_ns=100, old_native_ns=90, persistent_ns=120)
    component_rows = exp7270.synthetic_component_rows(blocks=2, sync_ns=600)
    checks = [exp7270.check("fixture", "fixture", "value", True, True, True)]
    evidence = {
        "hashes": {"fixture": "sha256:fixture"},
        "exp7257": {"status": "complete"},
        "exp7256": {
            "native_controller_ready_score": 1,
            "native_binary_receipt": {"module_file": "/tmp/native.so"},
        },
        "quarantine": {
            "exp7257": {"quarantined": False},
            "exp7256": {"quarantined": False},
        },
    }
    monkeypatch.setattr(
        exp7270, "collect_preconditions", lambda *_args, **_kwargs: (checks, evidence)
    )
    monkeypatch.setattr(exp7270, "_read_json", lambda path: {"cost_rows": saved})
    monkeypatch.setattr(exp7270.exp7230, "load_native_extension", lambda path: object())
    monkeypatch.setattr(
        exp7270,
        "instrument_durable_blocks",
        lambda *_args, **_kwargs: deepcopy(component_rows),
    )
    paths = exp7270.ExperimentPaths.under(tmp_path)
    artifact = exp7270.build_artifact(
        tmp_path,
        paths,
        validation_receipts=[exp7270.validation_receipt("fixture", ["true"], 0, "ok")],
        instrumented_blocks=2,
    )
    assert artifact["status"] == "complete"
    assert artifact["durable_profile_complete_score"] == 1
    assert artifact["journal_optimization_warranted_score"] == 0
    assert artifact["declared_bottleneck"] == "durable_sync"
    assert len(artifact["rows"]) == 542
    assert paths.raw_rows.is_file()
    assert paths.checkpoint.is_file()


def test_req_rustpy_7270_defensive_helpers_and_cleanup(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-RUSTPY-7270: malformed inputs, time bounds, and cleanup fail closed."""

    real_monotonic = time.monotonic
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        exp7270._read_json(sequence)
    assert exp7270._quantile([7.0], 0.95) == 7.0
    with pytest.raises(ValueError, match="invalid Amdahl"):
        exp7270.acceleration_envelope(total_ns=0.0, fixed_ns=0.0)

    changed = exp7270.complete_artifact_fixture_for_test()
    original_checksum = exp7270.artifact_checksum
    monkeypatch.setattr(
        exp7270, "artifact_checksum", lambda _artifact: (_ for _ in ()).throw(TypeError("bad"))
    )
    assert exp7270._checksum_valid({}) is False
    assert "reproducibility_checksum" in exp7270.validate_artifact(changed)
    monkeypatch.setattr(exp7270, "artifact_checksum", original_checksum)

    clock = iter([0.0, 0.0, 301.0, 302.0])
    monkeypatch.setattr(exp7270.time, "monotonic", lambda: next(clock))
    assert exp7270.instrument_durable_blocks(native, tmp_path / "timeout", blocks=1) == []

    clock = iter([0.0, 1.0, 2.0, 63.0, 64.0])
    monkeypatch.setattr(exp7270.time, "monotonic", lambda: next(clock))
    rows = exp7270.instrument_durable_blocks(
        native, tmp_path / "heartbeat", blocks=1, max_duration_s=300.0
    )
    assert len(rows) == 1
    assert "completed_blocks=1/1" in capsys.readouterr().out

    monkeypatch.setattr(exp7270.time, "monotonic", real_monotonic)
    monkeypatch.setattr(exp7270.transactional, "_atomic_write", lambda *_args: {})
    monkeypatch.setattr(
        exp7270.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(OSError("sync"))
    )
    with pytest.raises(OSError, match="sync"):
        exp7270.instrument_durable_blocks(native, tmp_path / "cleanup", blocks=1)
    assert not list((tmp_path / "cleanup").glob("*.tmp"))


def test_req_rustpy_7270_validator_and_builder_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7270: malformed rows and incomplete replay cannot look terminal."""

    artifact = exp7270.complete_artifact_fixture_for_test()
    artifact["component_rows"] = [{}]
    artifact["rows"] = [*artifact["saved_cost_rows"], {}]
    artifact["reproducibility_checksum"] = exp7270.artifact_checksum(artifact)
    assert "component_reducer" in exp7270.validate_artifact(artifact)

    failed = exp7270.check("missing", "external", "path", "file", None, False)
    paths = exp7270.ExperimentPaths.under(tmp_path)
    blocked = exp7270.build_artifact(
        tmp_path,
        paths,
        validation_receipts=[],
        precondition_bundle=([failed], {"hashes": {}}),
    )
    assert blocked["status"] == "blocked"

    saved = exp7257.synthetic_cost_rows(python_ns=100, old_native_ns=90, persistent_ns=120)
    passed = exp7270.check("fixture", "fixture", "field", True, True, True)
    evidence = {
        "hashes": {},
        "exp7256": {"native_binary_receipt": {"module_file": "/tmp/native.so"}},
    }
    monkeypatch.setattr(exp7270, "_read_json", lambda _path: {"cost_rows": saved})
    monkeypatch.setattr(exp7270.exp7230, "load_native_extension", lambda _path: object())
    monkeypatch.setattr(
        exp7270,
        "instrument_durable_blocks",
        lambda *_args, **_kwargs: exp7270.synthetic_component_rows(1),
    )
    with pytest.raises(RuntimeError, match="replay incomplete"):
        exp7270.build_artifact(
            tmp_path,
            paths,
            validation_receipts=[exp7270.validation_receipt("fixture", ["true"], 0, "ok")],
            instrumented_blocks=2,
            precondition_bundle=([passed], evidence),
        )


def test_scenario_rustpy_7270_terminal_runner_failure_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-RUSTPY-7270-E2E: blocked and invalid candidates never bypass checks."""

    with pytest.raises(ValueError, match="run date"):
        exp7270.run_experiment(tmp_path, tmp_path / "bad.json", "bad")

    failed = exp7270.check("missing", "external", "path", "file", None, False)
    monkeypatch.setattr(
        exp7270, "collect_preconditions", lambda *_args, **_kwargs: ([failed], {"hashes": {}})
    )
    blocked = exp7270.blocked_artifact_for_test(failed)
    monkeypatch.setattr(exp7270, "build_artifact", lambda *_args, **_kwargs: deepcopy(blocked))
    output = tmp_path / "blocked.json"
    assert exp7270.run_experiment(tmp_path, output, exp7270.RUN_DATE)["status"] == "blocked"

    monkeypatch.setattr(exp7270, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="invalid blocked"):
        exp7270.run_experiment(tmp_path, tmp_path / "invalid-blocked.json", exp7270.RUN_DATE)

    passed = exp7270.check("fixture", "fixture", "field", True, True, True)
    monkeypatch.setattr(
        exp7270, "collect_preconditions", lambda *_args, **_kwargs: ([passed], {"hashes": {}})
    )
    monkeypatch.setattr(
        exp7270,
        "_stream_subprocess",
        lambda command, **_kwargs: {"command": command, "exit_code": 0, "output": "ok"},
    )
    with pytest.raises(ValueError, match="invalid Exp7270 candidate"):
        exp7270.run_experiment(tmp_path, tmp_path / "invalid.json", exp7270.RUN_DATE)

    complete = exp7270.complete_artifact_fixture_for_test()
    monkeypatch.setattr(exp7270, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    call_count = 0

    def fail_candidate(command: list[str], **_kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "command": command,
            "exit_code": int(call_count == 7),
            "output": "candidate failed" if call_count == 7 else "ok",
        }

    monkeypatch.setattr(exp7270, "_stream_subprocess", fail_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        exp7270.run_experiment(tmp_path, tmp_path / "candidate-failed.json", exp7270.RUN_DATE)


def test_req_rustpy_7270_main_validation_and_execution_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7270: CLI parse and execution failures return a nonzero code."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp7270.main(["--validate", str(invalid)]) == 2

    calls: list[Path] = []
    monkeypatch.setattr(
        exp7270,
        "run_experiment",
        lambda _root, output, _date: (
            calls.append(output) or exp7270.complete_artifact_fixture_for_test()
        ),
    )
    absolute = tmp_path / "absolute.json"
    assert exp7270.main(["--date", exp7270.RUN_DATE, "--output", str(absolute)]) == 0
    assert exp7270.main(["--date", exp7270.RUN_DATE, "--output", "relative.json"]) == 0
    assert calls == [absolute, exp7270.REPO_ROOT / "relative.json"]

    monkeypatch.setattr(
        exp7270,
        "run_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("execution")),
    )
    assert exp7270.main(["--date", exp7270.RUN_DATE, "--output", str(absolute)]) == 2
