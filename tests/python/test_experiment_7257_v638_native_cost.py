"""Tests for the complete three-arm native archive event-cost study.

Spec refs: REQ-RUSTPY-7257 and SCENARIO-RUSTPY-7257-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import sysconfig
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7257_v638_native_cost as exp7257


@pytest.fixture(scope="session")
def native_extension() -> Path:
    """REQ-RUSTPY-7257: use the exact binary authenticated by Exp7256."""

    source = exp7257._read_upstream_summary(
        exp7257.UPSTREAM_PATH, exp7257.REPO_ROOT / exp7257.EXCLUSION_RELATIVE
    )
    return Path(source["identity"]["module_file"])


@pytest.fixture(scope="session")
def native(native_extension: Path):
    """SCENARIO-RUSTPY-7257-COST: load the real persistent controller binary."""

    return exp7230.load_native_extension(native_extension)


def test_req_rustpy_7257_preconditions_authenticate_ready_binary(tmp_path: Path) -> None:
    """REQ-RUSTPY-7257: readiness alone cannot replace exact source bytes."""

    paths = exp7257.ExperimentPaths.under(tmp_path)
    checks, evidence = exp7257.collect_preconditions(exp7257.REPO_ROOT, paths)
    assert exp7257.gate_summary(checks)["passed"] is True
    assert evidence["upstream"]["native_controller_ready_score"] == 1
    assert evidence["upstream_binary_hash"] == evidence["declared_binary_hash"]
    assert evidence["quarantine"]["quarantined"] is False
    assert all(paths.writable_targets())

    changed_path = tmp_path / "changed.json"
    changed_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "native_controller_ready_score": 0,
                "reproducibility_checksum": "sha256:changed",
            }
        ),
        encoding="utf-8",
    )
    failed, _ = exp7257.collect_preconditions(exp7257.REPO_ROOT, paths, upstream_path=changed_path)
    assert exp7257.gate_summary(failed)["passed"] is False


def test_scenario_rustpy_7257_full_event_cost_has_equal_durability(native, tmp_path: Path) -> None:
    """SCENARIO-RUSTPY-7257-COST: all arms pay for the full durable boundary."""

    rows = exp7257.run_cost_benchmark(
        native,
        storage_dir=tmp_path / "cost",
        capacities=(1, 4),
        batch_sizes=(1, 2),
        blocks=2,
        seed=7_257_100,
    )
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == set(exp7257.ARMS)
    assert {row["archive_capacity"] for row in rows} == {1, 4}
    assert all(row["durability_policy"] == exp7257.DURABILITY_POLICY for row in rows)
    assert all(row["durable_commit_count"] == 1 for row in rows)
    assert all(row["safety_checks_passed"] is True for row in rows)
    assert all(row["parity_mismatch_count"] == 0 for row in rows)
    assert all(row["warmup_included"] is False for row in rows)
    required = {
        "lookup_ns",
        "query_ns",
        "delayed_release_update_ns",
        "validation_ns",
        "serialization_ns",
        "durable_commit_ns",
        "restore_ns",
        "residual_unaccounted_ns",
    }
    assert all(required <= set(row["component_ns"]) for row in rows)
    assert all(row["total_block_ns"] > 0 and row["total_event_ns"] > 0 for row in rows)
    for batch in (1, 2):
        assert (
            len(
                {
                    row["trace_sha256"]
                    for row in rows
                    if row["block"] == 0 and row["batch_size"] == batch
                }
            )
            == 1
        )


def test_scenario_rustpy_7257_interactive_gate_and_independent_reducer() -> None:
    """SCENARIO-RUSTPY-7257-GATE: only both batch-one capacity cells gate value."""

    rows = exp7257.synthetic_cost_rows(python_ns=300, old_native_ns=200, persistent_ns=100)
    summary = exp7257.reduce_cost_rows(rows)
    assert summary["cost_row_count"] == 540
    assert len(summary["cells"]) == 12
    assert summary["native_event_cost_value_score"] == 1
    assert summary["batch_one_capacity_gates"] == {"1": True, "4": True}
    assert summary["nfr_01_10x_met"] is False
    assert summary["research_program_100x_met"] is False

    slow = exp7257.synthetic_cost_rows(python_ns=100, old_native_ns=80, persistent_ns=120)
    slow_summary = exp7257.reduce_cost_rows(slow)
    assert slow_summary["native_event_cost_value_score"] == 0
    assert all(value is False for value in slow_summary["batch_one_capacity_gates"].values())
    with pytest.raises(ValueError, match="incomplete cost roster"):
        exp7257.reduce_cost_rows(rows[:-1])


def test_scenario_rustpy_7257_fresh_process_replays_timed_snapshot(
    native_extension: Path, tmp_path: Path
) -> None:
    """SCENARIO-RUSTPY-7257-RESTART: private snapshot replay uses exact bytes."""

    state = exp7257.seed_cost_state(4)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(exp7257.canonical_json(state), encoding="utf-8")
    receipt = exp7257.run_fresh_process_replay(native_extension, snapshot)
    assert receipt["passed"] is True
    assert receipt["same_decisions"] is True
    assert receipt["same_state_hash"] is True
    assert receipt["module_sha256"] == exp7257.sha256_file(native_extension)


def test_scenario_rustpy_7257_terminal_validator_separates_cost_and_oracle() -> None:
    """SCENARIO-RUSTPY-7257-TERMINAL: oracle evidence cannot become positive."""

    passed = exp7257.complete_artifact_fixture_for_test(cost_pass=True)
    assert exp7257.validate_artifact(passed) == []
    assert passed["verdict_class"] == "circular_positive"
    assert passed["native_cost_complete_score"] == 1
    assert passed["native_event_cost_value_score"] == 1

    null = exp7257.complete_artifact_fixture_for_test(cost_pass=False)
    assert exp7257.validate_artifact(null) == []
    assert null["verdict_class"] == "null"
    assert null["native_cost_complete_score"] == 1

    for field, value, error in (
        ("verdict_class", "positive", "verdict_class"),
        ("native_cost_complete_score", 0, "complete_score"),
        ("model_invoked", True, "model_invocation"),
    ):
        changed = deepcopy(passed)
        changed[field] = value
        changed["reproducibility_checksum"] = exp7257.artifact_checksum(changed)
        assert error in exp7257.validate_artifact(changed)

    changed = deepcopy(passed)
    changed["cost_rows"][0]["safety_checks_passed"] = False
    changed["rows"] = [*changed["parity_rows"], *changed["cost_rows"]]
    changed["reproducibility_checksum"] = exp7257.artifact_checksum(changed)
    assert "cost_rows" in exp7257.validate_artifact(changed)


def test_req_rustpy_7257_blocked_artifact_is_terminal_and_row_free() -> None:
    """REQ-RUSTPY-7257: an external precondition failure is blocked, not partial."""

    failed = exp7257.check("upstream_missing", "exp7256", "path", "file", None, False)
    artifact = exp7257.blocked_artifact_for_test(failed)
    assert exp7257.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["rows"] == artifact["cost_rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "upstream_missing"


def test_req_rustpy_7257_amortization_keeps_setup_outside_rows() -> None:
    """REQ-RUSTPY-7257: build, import, transfer, and break-even stay separate."""

    rows = exp7257.synthetic_cost_rows(python_ns=300, old_native_ns=200, persistent_ns=100)
    amortization = exp7257.amortization_rows(
        rows,
        {
            "startup_duration_s": 0.01,
            "build_duration_s": 1.0,
            "import_duration_s": 0.02,
            "state_transfer_duration_s": 0.003,
        },
    )
    assert {row["term"] for row in amortization} == {
        "process_startup",
        "isolated_build",
        "native_import",
        "initial_state_transfer",
        "persistent_native_break_even",
    }
    assert all(row["included_in_steady_state"] is False for row in amortization)
    assert (
        next(row for row in amortization if row["term"] == "persistent_native_break_even")[
            "break_even_calls"
        ]
        > 0
    )


def test_req_rustpy_7257_thin_entrypoint_and_validation_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: the runnable file delegates and validation is read-only."""

    script = exp7257.REPO_ROOT / "scripts/experiments/experiment_7257_v638_native_cost.py"
    calls: list[list[str] | None] = []
    original_main = exp7257.main
    monkeypatch.setattr(exp7257, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7257.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(exp7257, "main", original_main)

    artifact = exp7257.complete_artifact_fixture_for_test(cost_pass=False)
    output = tmp_path / "artifact.json"
    exp7257.atomic_write(output, artifact)
    original = output.read_bytes()
    assert exp7257.main(["--validate", str(output)]) == 0
    assert output.read_bytes() == original
    output.write_text("{}", encoding="utf-8")
    assert exp7257.main(["--validate", str(output)]) == 2
    assert exp7257.main(["--date", "bad", "--output", str(output)]) == 2


def test_req_rustpy_7257_defensive_measurement_and_reducer_paths(
    native, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-RUSTPY-7257: malformed rosters and unknown arms fail closed."""

    assert (
        exp7257.ExperimentPaths.defaults().artifact == exp7257.REPO_ROOT / exp7257.RESULT_RELATIVE
    )
    with pytest.raises(ValueError, match="unknown cost arm"):
        exp7257._controller_from_state(native, "unknown", exp7257.seed_cost_state(1))
    with pytest.raises(ValueError, match="nonempty"):
        exp7257.run_cost_benchmark(native, storage_dir=tmp_path, capacities=())
    with pytest.raises(ValueError, match="empty"):
        exp7257._bootstrap_interval([], exp7257.RANDOM_SEED)

    rows = exp7257.synthetic_cost_rows(python_ns=3, old_native_ns=2, persistent_ns=1)
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="incomplete cost roster"):
        exp7257.reduce_cost_rows(rows)
    assert len(exp7257._fixture_parity_rows()) == 3

    clock = iter(float(index * 61) for index in range(100))
    monkeypatch.setattr(exp7257.time, "monotonic", lambda: next(clock))
    measured = exp7257.run_cost_benchmark(
        native, storage_dir=tmp_path / "progress", capacities=(1,), batch_sizes=(1,), blocks=1
    )
    assert len(measured) == 3
    assert "completed_blocks=1/1" in capsys.readouterr().out


def test_req_rustpy_7257_build_helper_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: the isolated builder selects only real output bytes."""

    target = tmp_path / exp7257.TARGET_RELATIVE / "release"
    target.mkdir(parents=True)
    library = target / "libcarnot_python.so"
    library.write_bytes(b"native")
    environment = {
        "PYO3_PYTHON": str(Path(sys.executable).absolute()),
        "CARGO_TARGET_DIR": str((tmp_path / exp7257.TARGET_RELATIVE).resolve()),
    }
    build = {"command": ["cargo", "build"], "exit_code": 0, "output": "ok"}
    monkeypatch.setattr(exp7257.exp7217, "interpreter_build_environment", lambda *_: environment)
    monkeypatch.setattr(exp7257.exp7217, "_stream_process", lambda *_args, **_kwargs: dict(build))
    extension, receipt = exp7257._build_timed_extension(tmp_path)
    actual_suffix = extension.name.removeprefix("_rust")
    assert extension.read_bytes() == b"native"
    assert receipt["module_sha256"] == exp7257.sha256_file(extension)

    library.unlink()
    with pytest.raises(RuntimeError, match="output missing"):
        exp7257._build_timed_extension(tmp_path)
    library.write_bytes(b"native")
    monkeypatch.setattr(exp7257.sysconfig, "get_config_var", lambda _name: None)
    with pytest.raises(RuntimeError, match="suffix unavailable"):
        exp7257._build_timed_extension(tmp_path)

    monkeypatch.setattr(exp7257.sysconfig, "get_config_var", lambda _name: actual_suffix)
    monkeypatch.setattr(
        exp7257.shutil,
        "copyfile",
        lambda *_args: (_ for _ in ()).throw(OSError("copy failed")),
    )
    with pytest.raises(OSError, match="copy failed"):
        exp7257._build_timed_extension(tmp_path)


def test_req_rustpy_7257_build_artifact_blocked_and_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: orchestration stops on external blocks and classifies measured rows."""

    paths = exp7257.ExperimentPaths.under(tmp_path)
    failed = exp7257.check("missing", "exp7256", "field", 1, None, False)
    evidence = {
        "hashes": {},
        "upstream": {},
        "upstream_binary_path": "missing",
        "upstream_binary_hash": None,
    }
    monkeypatch.setattr(
        exp7257, "collect_preconditions", lambda *_args, **_kwargs: ([failed], evidence)
    )
    blocked = exp7257.build_artifact(tmp_path, paths)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []

    extension = tmp_path / f"_rust{sysconfig.get_config_var('EXT_SUFFIX')}"
    extension.write_bytes(b"compiled")
    binary_hash = exp7257.sha256_file(extension)
    passed = exp7257.check("ready", "exp7256", "field", 1, 1, True)
    evidence = {
        "hashes": {str(exp7257.UPSTREAM_RELATIVE): "sha256:fixture"},
        "upstream": {"native_controller_ready_score": 1},
        "upstream_binary_path": str(extension),
        "upstream_binary_hash": binary_hash,
    }
    build_receipt = {
        "command": ["cargo", "build"],
        "exit_code": 0,
        "output": "ok",
        "build_duration_s": 0.1,
        "PYO3_PYTHON": str(Path(sys.executable).absolute()),
        "CARGO_TARGET_DIR": str(tmp_path / "target"),
    }
    costs = exp7257.synthetic_cost_rows(python_ns=300, old_native_ns=200, persistent_ns=100)
    fresh = {
        "passed": True,
        "same_decisions": True,
        "same_state_hash": True,
        "command": ["python", "fresh"],
        "exit_code": 0,
        "stdout_sha256": "sha256:out",
        "stderr_sha256": "sha256:err",
    }
    sidecar = {
        "validation_receipts": [
            {"name": "focused", "command": ["pytest"], "exit_code": 0, "log_sha256": "sha256:test"}
        ],
        "baseline_validation_failures": [{"name": "broad", "exit_code": 1}],
    }
    paths.validation_sidecar.parent.mkdir(parents=True, exist_ok=True)
    paths.validation_sidecar.write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(
        exp7257, "collect_preconditions", lambda *_args, **_kwargs: ([passed], evidence)
    )
    monkeypatch.setattr(exp7257, "_build_timed_extension", lambda _root: (extension, build_receipt))
    monkeypatch.setattr(
        exp7257.exp7230,
        "load_native_extension",
        lambda _path: SimpleNamespace(__file__=str(extension)),
    )
    monkeypatch.setattr(
        exp7257.exp7256.PersistentNativeArchiveController,
        "from_state_with_binding",
        lambda *_args: object(),
    )
    monkeypatch.setattr(exp7257, "run_cost_benchmark", lambda *_args, **_kwargs: costs)
    monkeypatch.setattr(exp7257, "run_fresh_process_replay", lambda *_args, **_kwargs: fresh)
    complete = exp7257.build_artifact(tmp_path, paths)
    assert complete["status"] == "complete"
    assert complete["native_event_cost_value_score"] == 1
    assert complete["baseline_validation_failures"] == sidecar["baseline_validation_failures"]
    assert any(row["name"] == "focused" for row in complete["validation_receipts"])


def test_req_rustpy_7257_validator_file_checks_and_error_reduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: cold file checks and reducer failures cannot pass."""

    artifact = exp7257.complete_artifact_fixture_for_test(cost_pass=True)
    module = tmp_path / f"_rust{sysconfig.get_config_var('EXT_SUFFIX')}"
    module.write_bytes(b"native")
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    artifact["native_binary_receipt"]["module_file"] = str(module)
    artifact["native_binary_receipt"]["module_sha256"] = exp7257.sha256_file(module)
    artifact["source_artifact_hashes"] = {str(source): exp7257.sha256_file(source)}
    artifact["reproducibility_checksum"] = exp7257.artifact_checksum(artifact)
    assert exp7257.validate_artifact(artifact, check_files=True, root=tmp_path) == []

    source.write_text("changed", encoding="utf-8")
    assert "source_artifact_hashes" in exp7257.validate_artifact(
        artifact, check_files=True, root=tmp_path
    )
    monkeypatch.setattr(
        exp7257, "reduce_cost_rows", lambda _rows: (_ for _ in ()).throw(ValueError("bad"))
    )
    errors = exp7257.validate_artifact(artifact)
    assert "raw_row_reducer" in errors
    with patch.object(exp7257, "artifact_checksum", side_effect=TypeError("bad")):
        assert "reproducibility_checksum" in exp7257.validate_artifact(artifact)


def test_req_rustpy_7257_run_experiment_and_main_defensive_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: terminal publication follows cold validation only."""

    artifact = exp7257.complete_artifact_fixture_for_test(cost_pass=False)
    monkeypatch.setattr(exp7257, "build_artifact", lambda *_args: artifact)
    monkeypatch.setattr(exp7257, "validate_artifact", lambda *_args, **_kwargs: [])
    written: list[Path] = []
    monkeypatch.setattr(
        exp7257,
        "atomic_write",
        lambda path, _artifact: (
            written.append(path) or {"path": str(path), "sha256": "sha256:test", "bytes": 1}
        ),
    )
    output = tmp_path / "result.json"
    assert exp7257.run_experiment(tmp_path, output, exp7257.RUN_DATE) == artifact
    assert written == [output]
    with pytest.raises(ValueError, match="run date"):
        exp7257.run_experiment(tmp_path, output, "bad")

    monkeypatch.setattr(exp7257, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7257"):
        exp7257.run_experiment(tmp_path, output, exp7257.RUN_DATE)
    assert exp7257.main(["--validate", str(tmp_path / "missing.json")]) == 2

    monkeypatch.setattr(exp7257, "run_experiment", lambda *_args: artifact)
    assert exp7257.main(["--date", exp7257.RUN_DATE, "--output", str(output)]) == 0
    monkeypatch.setattr(
        exp7257, "run_experiment", lambda *_args: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert exp7257.main(["--date", exp7257.RUN_DATE, "--output", str(output)]) == 2

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(
        exp7257.argparse.ArgumentParser,
        "parse_args",
        lambda _self, _argv=None: SimpleNamespace(
            date=exp7257.RUN_DATE, output=output, validate=fixture_path
        ),
    )
    monkeypatch.setattr(exp7257, "validate_artifact", lambda *_args, **_kwargs: [])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(exp7257.__file__)), run_name="__main__")
    assert exit_info.value.code == 0


def test_req_rustpy_7257_precondition_checksum_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: malformed upstream checksum material remains a failed gate."""

    paths = exp7257.ExperimentPaths.under(tmp_path)
    failed_summary = {
        "upstream": {"status": "complete", "native_controller_ready_score": 1},
        "checksum_valid": False,
        "identity": {},
        "quarantine": {"quarantined": False},
    }
    with patch.object(exp7257, "_read_upstream_summary", return_value=failed_summary):
        checks, _ = exp7257.collect_preconditions(exp7257.REPO_ROOT, paths)
    gate = next(row for row in checks if row["check"] == "exp7256_ready_complete_checksum")
    assert gate["passed"] is False
    assert exp7257._accept_validation_sidecar(tmp_path / "missing.json") == ([], [])


def test_req_rustpy_7257_upstream_subprocess_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-RUSTPY-7257: missing or failed authentication output blocks the study."""

    failed = SimpleNamespace(returncode=2, stdout="", stderr="failed")
    monkeypatch.setattr(exp7257.subprocess, "run", lambda *_args, **_kwargs: failed)
    summary = exp7257._read_upstream_summary(tmp_path / "source.json", tmp_path / "exclude")
    assert summary["quarantine"]["quarantined"] is True
    assert summary["quarantine"]["authentication_error"] == "failed"

    missing = SimpleNamespace(returncode=0, stdout="unrelated\n", stderr="")
    monkeypatch.setattr(exp7257.subprocess, "run", lambda *_args, **_kwargs: missing)
    summary = exp7257._read_upstream_summary(tmp_path / "source.json", tmp_path / "exclude")
    assert summary["quarantine"]["authentication_error"] == "missing_summary"
