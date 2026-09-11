"""Tests for the interpreter-bound ABI and board-readiness receipt.

Spec: REQ-ISING-7217 and SCENARIO-ISING-7217-PREFLIGHT through
SCENARIO-ISING-7217-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7217_v635_abi_board_readiness as experiment


ROOT = Path(__file__).resolve().parents[2]


def test_principled_values_are_narrow_and_quarantine_precedes_gate() -> None:
    """REQ-ISING-7217: arbitrary mappings cannot become producer gate values."""

    exact = {"principle": "why", "value": 1}
    extra = {"principle": "why", "value": 1, "evidence": "untrusted"}
    malformed = {"principle": 7, "value": 1}
    assert experiment.unwrap_principled_value(exact) == 1
    assert experiment.unwrap_principled_value(extra) is extra
    assert experiment.unwrap_principled_value(malformed) is malformed

    upstream = {"native_gate": exact, "flagged_adversarial": True}
    quarantine = experiment.upstream_quarantine_observation(upstream, manifest_match=False)
    assert quarantine["quarantined"] is True
    assert experiment.gated_upstream_value(upstream, quarantine, "native_gate") == (
        "not_consumed_due_to_quarantine"
    )


def test_interpreter_build_environment_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ISING-7217-REBUILD: bind PyO3 without the abi3 workaround."""

    monkeypatch.setenv("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
    target = ROOT / "target" / "experiment-7217-test"
    environment = experiment.interpreter_build_environment(Path(sys.executable), target)
    assert environment["PYO3_PYTHON"] == str(Path(sys.executable).absolute())
    assert environment["CARGO_TARGET_DIR"] == str(target.resolve())
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in environment


def test_native_fixture_uses_exp7187_authority() -> None:
    """SCENARIO-ISING-7217-ABI: expected transitions use the Python authority."""

    fixture = experiment.native_fixture()
    assert fixture["authority"] == "experiment_7187_v633_slice_sampler"
    assert fixture["cardinality"] == 2
    assert len(fixture["expected_replay"]["steps"]) == len(fixture["tape"])
    assert all(step["cardinality"] == 2 for step in fixture["expected_replay"]["steps"])
    assert fixture["expected_replay"]["final_state"].count(1) == 2


def test_board_receipts_remain_independent_and_read_only() -> None:
    """SCENARIO-ISING-7217-BOARDS: preserve all three terminal dispositions."""

    rows, operator = experiment.load_board_evidence(ROOT)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["disposition"] == "graduated_preserved"
    assert by_board["GateMate"]["disposition"] == "blocked_inherited_no_new_physical_state"
    assert by_board["PolarFire"]["disposition"] == "blocked_missing_raw_dispatch_transcript"
    assert all(row["hardware_command_count"] == 0 for row in rows)
    assert operator["newer_than_exp6559"] is False
    assert operator["hardware_operations_issued"] == []


def test_preconditions_authenticate_producers_and_preserve_failed_value(tmp_path: Path) -> None:
    """SCENARIO-ISING-7217-PREFLIGHT: use exact gates and shipped validators."""

    checks, hashes = experiment.collect_preconditions(
        ROOT,
        result_path=tmp_path / "result.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    by_name = {row["check"]: row for row in checks}
    assert all(row["passed"] for row in checks)
    assert by_name["exp7201_producer_authentication"]["observed_value"] == []
    assert by_name["exp7201_native_gate"]["observed_value"] == 1
    assert by_name["exp7203_board_producer_authentication"]["observed_value"] == []
    assert by_name["exp7202_failed_value_preserved"]["observed_value"] is False
    assert len(hashes) == len(experiment.REQUIRED_SOURCE_PATHS)


def test_complete_artifact_validator_separates_host_and_boards() -> None:
    """SCENARIO-ISING-7217-ARTIFACT: board blocks cannot erase host readiness."""

    artifact = experiment.complete_artifact_fixture_for_test(ROOT)
    assert experiment.validate_artifact(artifact) == []
    assert artifact["native_abi_ready_score"] == 1
    assert artifact["abi_board_receipt_complete_score"] == 1
    assert any(row["abstention"] for row in artifact["board_rows"])

    promoted = deepcopy(artifact)
    promoted["upstream_failed_value"]["promoted"] = True
    promoted["reproducibility_checksum"] = experiment.artifact_checksum(promoted)
    assert "upstream_failed_value_promoted" in experiment.validate_artifact(promoted)


def test_blocked_artifact_has_exact_gate_and_no_computation_rows() -> None:
    """SCENARIO-ISING-7217-PREFLIGHT: external blocks are terminal and explicit."""

    failed = {
        "check": "missing_tool",
        "upstream": "host",
        "field": "cargo",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    artifact = experiment.blocked_artifact_for_test(ROOT, failed)
    assert experiment.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "missing_tool"


def test_atomic_write_and_validation_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7217: terminal bytes are atomic and support read-only validation."""

    artifact = experiment.complete_artifact_fixture_for_test(ROOT)
    output = tmp_path / "artifact.json"
    receipt = experiment.atomic_write(output, artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(output)]) == 0
    assert '"valid":true' in capsys.readouterr().out

    assert experiment.main(["--date", "20260910", "--output", str(output)]) == 2
    assert "run date must be 20260911" in capsys.readouterr().out


def test_entrypoint_exists_and_does_not_mutate_board_or_prior_results() -> None:
    """REQ-ISING-7217: the executable is scoped to its own terminal artifact."""

    entrypoint = ROOT / "scripts/experiments/experiment_7217_v635_abi_board_readiness.py"
    source = entrypoint.read_text(encoding="utf-8")
    assert "experiment_7217_v635_abi_board_readiness" in source
    assert "openFPGALoader" not in source
    assert "experiment_7201_v634_slice_pyo3.json" not in source
    assert "experiment_7202_v634_slice_cost_quality.json" not in source
    assert os.access(entrypoint.parent, os.W_OK)


def test_manifest_json_contract_and_cargo_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ISING-7217-PREFLIGHT: malformed local contracts fail closed."""

    assert experiment._manifest_mentions_experiment(
        {"nested": [{"experiment_ids": ["exp7201", "exp7"]}]}, "7201"
    )
    assert experiment._manifest_mentions_experiment(
        {"nested": {"experiment_id": "exp7203"}}, "7203"
    )
    assert not experiment._manifest_mentions_experiment({"note": "7201"}, "7201")
    assert experiment.upstream_quarantine_observation({}, manifest_match=True)["quarantined"]

    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment._read_json(missing) == {}
    assert experiment._read_json(invalid) == {}
    assert experiment._read_json(scalar) == {}
    assert experiment._task_contract(tmp_path) is None

    (tmp_path / "research-roadmap.yaml").write_text("tasks: not-a-list\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    assert experiment._cargo_configuration(tmp_path) == {}


def test_heartbeat_timeout_and_streaming_subprocess_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7217: bounded native and build calls report observed waiting state."""

    with experiment._Heartbeat("short-test", interval_s=0.001):
        time.sleep(0.005)
    assert "state=waiting" in capsys.readouterr().out

    timeout = subprocess.TimeoutExpired(["python"], 1, output=b"out", stderr=b"undefined symbol: X")

    def expire(*_: Any, **__: Any) -> Any:
        raise timeout

    monkeypatch.setattr(experiment.subprocess, "run", expire)
    receipt = experiment._bounded_native_process(tmp_path / "missing.so", "import", timeout_s=1)
    assert receipt["timed_out"] is True
    assert receipt["stdout"] == "out"
    assert receipt["undefined_symbol"] == "X"
    assert experiment._undefined_symbol("ordinary error") is None

    monkeypatch.undo()
    environment = experiment.interpreter_build_environment(Path(sys.executable), tmp_path)
    streamed = experiment._stream_process(
        [sys.executable, "-u", "-c", "print('observed-line')"],
        root=ROOT,
        environment=environment,
        operation="short stream",
        timeout_s=5,
    )
    assert streamed["exit_code"] == 0
    with pytest.raises(RuntimeError, match="failed with exit 3"):
        experiment._stream_process(
            [sys.executable, "-c", "raise SystemExit(3)"],
            root=ROOT,
            environment=environment,
            operation="failed stream",
            timeout_s=5,
        )
    with pytest.raises(subprocess.TimeoutExpired):
        experiment._stream_process(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            root=ROOT,
            environment=environment,
            operation="timed stream",
            timeout_s=0,
        )


def test_extension_suffix_and_build_copy_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-REBUILD: missing ABI metadata or output cannot pass."""

    monkeypatch.setattr(experiment.sysconfig, "get_config_var", lambda _name: None)
    with pytest.raises(RuntimeError, match="extension suffix"):
        experiment._extension_suffix()
    monkeypatch.undo()

    monkeypatch.setattr(experiment, "_stream_process", lambda *args, **kwargs: {"exit_code": 0})
    monkeypatch.setattr(experiment, "TASK_TARGET_DIR", Path("target-test"))
    with pytest.raises(RuntimeError, match="build output is missing"):
        experiment.build_interpreter_bound_extension(tmp_path)

    library = tmp_path / "target-test/release/libcarnot_python.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"native")
    monkeypatch.setattr(experiment.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("copy")))
    with pytest.raises(OSError, match="copy"):
        experiment.build_interpreter_bound_extension(tmp_path)
    assert not list((tmp_path / experiment.TASK_LOAD_DIR).glob("*.tmp"))


def test_native_e2e_rejects_malformed_process_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-ABI: every missing native result blocks readiness."""

    extension = tmp_path / "native.so"
    extension.write_bytes(b"binary")

    first_failures = [
        {"exit_code": 1, "stderr": "load failed", "result": None},
        {"exit_code": 0, "stderr": "", "result": {}},
        {"exit_code": 0, "stderr": "", "result": {"replay": {}}},
        {
            "exit_code": 0,
            "stderr": "",
            "result": {"replay": {}, "serialized_state": "{}", "seeded": {}},
        },
    ]
    expected = [
        "fresh native replay failed",
        "transition output",
        "restart state",
        "malformed restart state",
    ]
    for response, message in zip(first_failures, expected, strict=True):
        monkeypatch.setattr(
            experiment, "_bounded_native_process", lambda *_a, _r=response, **_k: _r
        )
        with pytest.raises(RuntimeError, match=message):
            experiment.run_native_e2e(extension)

    fixture = experiment.native_fixture()
    valid_first = {
        "exit_code": 0,
        "stderr": "",
        "result": {
            "replay": fixture["expected_replay"],
            "serialized_state": '{"spins":[1,1,-1,-1,-1,-1,-1,-1],"rng_state":1,"transition":1}',
            "seeded": {
                "final_state": {
                    "spins": [1, 1, -1, -1, -1, -1, -1, -1],
                    "rng_state": 1,
                    "transition": 1,
                }
            },
        },
    }
    for second, message in (
        ({"exit_code": 1, "stderr": "restore failed", "result": None}, "fresh native restore"),
        ({"exit_code": 0, "stderr": "", "result": {}}, "did not return state"),
    ):
        responses = iter([valid_first, second])
        monkeypatch.setattr(
            experiment, "_bounded_native_process", lambda *_a, **_k: next(responses)
        )
        with pytest.raises(RuntimeError, match=message):
            experiment.run_native_e2e(extension)


def test_board_loader_rejects_bad_receipt_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-BOARDS: each board source and disposition authenticates."""

    cases: list[tuple[dict[str, Any], str]] = [
        ({"board_rows": {}}, "must be a list"),
        ({"board_rows": [7]}, "row is malformed"),
        ({"board_rows": [{"board": "KV260"}]}, "evidence hash mismatch"),
    ]
    for payload, message in cases:
        monkeypatch.setattr(experiment, "_read_json", lambda *_a, _p=payload: _p)
        with pytest.raises(ValueError, match=message):
            experiment.load_board_evidence(tmp_path)

    evidence = tmp_path / "receipt.json"
    evidence.write_text("{}", encoding="utf-8")
    rows = [
        {
            "board": board,
            "evidence_path": "receipt.json",
            "source_hash": "sha256:test",
            "recorded_date": "20260911",
            "terminal_criterion": "criterion",
            "unresolved_prerequisite": "next",
            "disposition": disposition,
        }
        for board, disposition in (
            ("KV260", "graduated_preserved"),
            ("GateMate", "blocked_inherited_no_new_physical_state"),
            ("PolarFire", "blocked_missing_raw_dispatch_transcript"),
        )
    ]
    monkeypatch.setattr(experiment, "sha256_file", lambda _path: "sha256:test")
    monkeypatch.setattr(experiment, "_read_json", lambda *_: {"board_rows": rows[:2]})
    with pytest.raises(ValueError, match="exactly the three"):
        experiment.load_board_evidence(tmp_path)

    changed = deepcopy(rows)
    changed[0]["disposition"] = "changed"
    monkeypatch.setattr(
        experiment, "_read_json", lambda *_: {"board_rows": changed, "operator_state_receipt": {}}
    )
    with pytest.raises(ValueError, match="disposition changed"):
        experiment.load_board_evidence(tmp_path)

    monkeypatch.setattr(
        experiment, "_read_json", lambda *_: {"board_rows": rows, "operator_state_receipt": []}
    )
    with pytest.raises(ValueError, match="operator_state_receipt"):
        experiment.load_board_evidence(tmp_path)


def test_build_artifact_block_fast_path_and_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7217: preflight, build, and execution failures terminate exactly."""

    failed = {
        "check": "source",
        "upstream": "repo",
        "field": "bytes",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    monkeypatch.setattr(experiment, "collect_preconditions", lambda _root: ([failed], {}))
    assert experiment.build_artifact(tmp_path)["verdict_class"] == "blocked"

    monkeypatch.setattr(experiment, "collect_preconditions", lambda _root: ([], {}))
    monkeypatch.setattr(experiment, "interpreter_metadata", lambda _root: {})
    missing = tmp_path / "missing.so"
    monkeypatch.setattr(experiment, "historical_extension_path", lambda _root: missing)
    monkeypatch.setattr(
        experiment,
        "build_interpreter_bound_extension",
        lambda _root: (_ for _ in ()).throw(RuntimeError("build unavailable")),
    )
    blocked = experiment.build_artifact(tmp_path)
    assert blocked["gate_check_summary"]["failed_check"] == "interpreter_bound_native_build"

    selected = tmp_path / "working.so"
    selected.write_bytes(b"native")
    monkeypatch.setattr(experiment, "historical_extension_path", lambda _root: selected)
    monkeypatch.setattr(
        experiment,
        "_bounded_native_process",
        lambda *_a, **_k: {
            "exit_code": 0,
            "result": {"class_present": True},
            "undefined_symbol": None,
        },
    )
    monkeypatch.setattr(
        experiment,
        "run_native_e2e",
        lambda _path: (_ for _ in ()).throw(RuntimeError("execution unavailable")),
    )
    blocked = experiment.build_artifact(tmp_path)
    assert blocked["gate_check_summary"]["failed_check"] == "fresh_process_native_execution"


def test_validator_reports_each_invalid_terminal_category() -> None:
    """REQ-ISING-7217: the validator recomputes all readiness and claim gates."""

    base = experiment.complete_artifact_fixture_for_test(ROOT)

    def errors_for(change: Any, *, checksum: bool = True) -> list[str]:
        candidate = deepcopy(base)
        change(candidate)
        if checksum:
            candidate["reproducibility_checksum"] = experiment.artifact_checksum(candidate)
        return experiment.validate_artifact(candidate)

    mutations = [
        (lambda a: a.pop("task_id"), "missing_required_fields"),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles_invalid"),
        (lambda a: a.__setitem__("run_date", "bad"), "run_date_invalid"),
        (lambda a: a.__setitem__("MODEL_SPECS", [{}]), "model_declaration_invalid"),
        (lambda a: a.__setitem__("execution_venue", "host:x"), "execution_identity_invalid"),
        (lambda a: a.__setitem__("verifier_is_oracle", False), "verifier_authority_invalid"),
        (
            lambda a: a.__setitem__("hardware_operations_issued", ["probe"]),
            "hardware_operations_invalid",
        ),
        (lambda a: a.__setitem__("topology_fit", "fits"), "topology_fit_invalid"),
        (lambda a: a.__setitem__("hardware_performance_claimed", True), "hardware_claim_invalid"),
        (lambda a: a.__setitem__("throughput_sweep_rerun", True), "throughput_rerun_invalid"),
        (lambda a: a.__setitem__("rows", []), "rows_invalid"),
        (lambda a: a.__setitem__("e2e_receipts", []), "e2e_receipts_invalid"),
        (
            lambda a: a.__setitem__("native_execution_receipt", {}),
            "native_execution_receipt_invalid",
        ),
        (lambda a: a.__setitem__("board_rows", []), "board_rows_invalid"),
        (lambda a: a.__setitem__("operator_state_receipt", {}), "operator_state_receipt_invalid"),
        (lambda a: a.__setitem__("sample_size_budget", {}), "sample_size_budget_invalid"),
        (lambda a: a.__setitem__("native_abi_ready_score", 0), "native_readiness_invalid"),
        (
            lambda a: a.__setitem__("abi_board_receipt_complete_score", 0),
            "complete_receipt_score_invalid",
        ),
        (lambda a: a.__setitem__("status", "running"), "terminal_state_invalid"),
        (lambda a: a.__setitem__("abi_rows", []), "abi_rows_invalid"),
    ]
    for mutation, expected in mutations:
        assert expected in errors_for(mutation)

    assert "reproducibility_checksum_mismatch" in errors_for(
        lambda a: a.__setitem__("duration_s", object()), checksum=False
    )
    assert "source_artifact_hashes_invalid" in experiment.validate_artifact(base, root=ROOT)
    assert "native_binary_hash_invalid" in experiment.validate_artifact(base, root=ROOT)

    invalid_block = experiment.blocked_artifact_for_test(
        ROOT,
        {
            "check": "x",
            "upstream": "y",
            "field": "z",
            "expected_value": True,
            "observed_value": False,
        },
    )
    invalid_block["rows"] = [{}]
    invalid_block["reproducibility_checksum"] = experiment.artifact_checksum(invalid_block)
    assert "blocked_state_invalid" in experiment.validate_artifact(invalid_block)


def test_run_and_cli_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ISING-7217: invalid terminal bytes and CLI failures return nonzero."""

    invalid = experiment.complete_artifact_fixture_for_test(ROOT)
    invalid["status"] = "running"
    invalid["reproducibility_checksum"] = experiment.artifact_checksum(invalid)
    monkeypatch.setattr(experiment, "build_artifact", lambda *_a, **_k: invalid)
    with pytest.raises(ValueError, match="invalid Exp7217 artifact"):
        experiment.run_experiment(ROOT, tmp_path / "out.json", experiment.RUN_DATE)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert experiment.main(["--validate", str(bad_json)]) == 2
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("run failed")),
    )
    assert experiment.main(["--output", "relative.json"]) == 2


def test_collect_preconditions_handles_unreadable_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-PREFLIGHT: exclusion parse failure remains observable."""

    original = experiment.yaml.safe_load
    calls = 0

    def load_then_fail(value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise experiment.yaml.YAMLError("invalid exclusion")
        return original(value)

    monkeypatch.setattr(experiment.yaml, "safe_load", load_then_fail)
    checks, _ = experiment.collect_preconditions(
        ROOT,
        result_path=tmp_path / "result.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert all(row["passed"] for row in checks)


def test_native_process_build_and_linkage_success_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7217: successful native probes retain ABI and linker provenance."""

    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            "diagnostic\n"
            "__CARNOT_JSON__[]\n"
            '__CARNOT_JSON__{"class_present":true}\n'
            "libpython3.12.so => /usr/lib/libpython3.12.so\n"
        ),
        stderr="undefined symbol: Py_TestSymbol",
    )
    monkeypatch.setattr(experiment.subprocess, "run", lambda *_a, **_k: completed)
    probe = experiment._bounded_native_process(tmp_path / "native.so", "import")
    assert probe["exit_code"] == 0
    assert probe["timed_out"] is False
    assert probe["result"] == {"class_present": True}
    assert probe["undefined_symbol"] == "Py_TestSymbol"

    linkage = experiment._linkage_receipt(tmp_path / "native.so")
    assert linkage["exit_code"] == 0
    assert linkage["libpython_lines"] == ["libpython3.12.so => /usr/lib/libpython3.12.so"]

    metadata = experiment.interpreter_metadata(ROOT)
    assert metadata["interpreter"] == str(Path(sys.executable).absolute())
    assert metadata["cargo"]["extension_module_enabled"] is True
    assert experiment.historical_extension_path(tmp_path).parent.name == "exp7201-pyo3-load"

    target = Path("target-success")
    load = Path("load-success")
    monkeypatch.setattr(experiment, "TASK_TARGET_DIR", target)
    monkeypatch.setattr(experiment, "TASK_LOAD_DIR", load)
    monkeypatch.setattr(
        experiment,
        "_stream_process",
        lambda *_a, **_k: {"exit_code": 0, "duration_s": 0.01, "output": "built"},
    )
    library = tmp_path / target / "release/libcarnot_python.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"compiled-extension")
    destination, receipt = experiment.build_interpreter_bound_extension(tmp_path)
    assert destination.read_bytes() == b"compiled-extension"
    assert receipt["loaded_copy"] == str(destination.resolve())
    assert receipt["binary_sha256"] == experiment.sha256_file(destination)
    assert receipt["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] is None


def test_stream_process_defensive_and_observed_waiting_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7217: stream setup and silent waits remain explicit and bounded."""

    killed: list[bool] = []
    no_stdout = SimpleNamespace(stdout=None, kill=lambda: killed.append(True))
    with monkeypatch.context() as patch:
        patch.setattr(experiment.subprocess, "Popen", lambda *_a, **_k: no_stdout)
        with pytest.raises(RuntimeError, match="did not expose build output"):
            experiment._stream_process(
                [sys.executable, "-c", "pass"],
                root=tmp_path,
                environment=os.environ,
                operation="missing stream",
                timeout_s=5,
            )
    assert killed == [True]

    calls = 0

    def elapsed_clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 31.0 + calls / 100.0

    with monkeypatch.context() as patch:
        patch.setattr(experiment.time, "monotonic", elapsed_clock)
        receipt = experiment._stream_process(
            [sys.executable, "-u", "-c", "import time; time.sleep(1.1); print('done')"],
            root=tmp_path,
            environment=os.environ,
            operation="silent then complete",
            timeout_s=100,
        )
    assert receipt["exit_code"] == 0
    assert receipt["output"] == "done\n"
    assert "state=process_running" in capsys.readouterr().out


def test_native_e2e_success_retains_exact_cross_process_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-ABI: a valid replay and restart pass every native gate."""

    extension = tmp_path / "native.so"
    extension.write_bytes(b"compiled")
    fixture = experiment.native_fixture()
    final_state = {
        "spins": [1, 1, -1, -1, -1, -1, -1, -1],
        "rng_state": 1,
        "transition": 1,
    }
    serialized = json.dumps(final_state, sort_keys=True)
    continuation_tape = experiment.exp7189.make_replay_tape(
        8,
        2,
        seed=experiment.REPLAY_SEED + 2,
        steps=experiment.REPLAY_STEPS,
    )
    continued = experiment.exp7189.python_replay(
        experiment.slices.make_frustrated_instance(8, 718701),
        2,
        2.0,
        final_state["spins"],
        continuation_tape,
    )
    first = {
        "exit_code": 0,
        "stderr": "",
        "result": {
            "class_present": True,
            "module_file": str(extension.resolve()),
            "interpreter": str(Path(sys.executable).absolute()),
            "replay": fixture["expected_replay"],
            "serialized_state": serialized,
            "seeded": {"final_state": final_state},
        },
    }
    second = {
        "exit_code": 0,
        "stderr": "",
        "result": {
            "module_file": str(extension.resolve()),
            "restored_state": final_state,
            "reserialized_state": serialized,
            "continued_replay": continued,
        },
    }
    responses = iter([first, second])
    monkeypatch.setattr(experiment, "_bounded_native_process", lambda *_a, **_k: next(responses))
    monkeypatch.setattr(
        experiment,
        "_linkage_receipt",
        lambda path: {"exit_code": 0, "path": str(path.resolve())},
    )

    rows, receipts, native = experiment.run_native_e2e(extension)
    assert len(rows) == experiment.REPLAY_STEPS + 1
    assert all(row["passed"] for row in rows)
    assert len(receipts) == 3
    assert all(receipt["passed"] for receipt in receipts)
    assert native["binary_sha256"] == experiment.sha256_file(extension)
    assert native["exact_outputs"]["second_process"]["restored_state"] == final_state


def test_successful_rebuild_artifact_run_and_module_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7217-ARTIFACT: successful rebuild and CLI paths publish valid bytes."""

    fixture = experiment.complete_artifact_fixture_for_test(ROOT)
    selected = tmp_path / "selected.so"
    selected.write_bytes(b"selected compiled extension")
    native = deepcopy(fixture["native_execution_receipt"])
    native.update(
        {
            "binary_sha256": experiment.sha256_file(selected),
            "module_file": str(selected.resolve()),
            "linked_libraries": {"exit_code": 0},
        }
    )
    monkeypatch.setattr(experiment, "collect_preconditions", lambda _root: ([], {}))
    monkeypatch.setattr(experiment, "interpreter_metadata", lambda _root: {})
    monkeypatch.setattr(
        experiment, "historical_extension_path", lambda _root: tmp_path / "missing.so"
    )
    monkeypatch.setattr(
        experiment,
        "build_interpreter_bound_extension",
        lambda _root: (selected, {"build": "successful"}),
    )
    monkeypatch.setattr(
        experiment,
        "run_native_e2e",
        lambda _path: (fixture["rows"], fixture["e2e_receipts"], native),
    )
    monkeypatch.setattr(
        experiment,
        "load_board_evidence",
        lambda _root: (fixture["board_rows"], fixture["operator_state_receipt"]),
    )
    artifact = experiment.build_artifact(tmp_path)
    assert artifact["status"] == "complete"
    assert artifact["native_abi_ready_score"] == 1
    assert artifact["abi_board_receipt_complete_score"] == 1
    assert artifact["abi_rows"][-1]["path"] == str(selected.resolve())
    assert experiment.validate_artifact(artifact) == []

    monkeypatch.setattr(experiment, "build_artifact", lambda *_a, **_k: artifact)
    monkeypatch.setattr(experiment, "validate_artifact", lambda *_a, **_k: [])
    output = tmp_path / "published.json"
    assert experiment.run_experiment(tmp_path, output, experiment.RUN_DATE) is artifact
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(experiment, "run_experiment", lambda *_a, **_k: artifact)
    assert experiment.main(["--output", str(tmp_path / "main.json")]) == 0

    direct_artifact = experiment.complete_artifact_fixture_for_test(ROOT)
    direct_path = tmp_path / "direct.json"
    direct_path.write_text(json.dumps(direct_artifact), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", [str(experiment.__file__), "--validate", str(direct_path)])
        with pytest.raises(SystemExit) as raised:
            runpy.run_path(str(experiment.__file__), run_name="__main__")
    assert raised.value.code == 0


def test_atomic_write_removes_temporary_file_after_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7217: a failed atomic replace leaves no partial artifact behind."""

    output = tmp_path / "artifact.json"
    with monkeypatch.context() as patch:
        patch.setattr(
            experiment.os,
            "replace",
            lambda *_a: (_ for _ in ()).throw(OSError("replace failed")),
        )
        with pytest.raises(OSError, match="replace failed"):
            experiment.atomic_write(output, {"terminal": True})
    assert not output.exists()
    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []
