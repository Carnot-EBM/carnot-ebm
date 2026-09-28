"""Task-owned fixture and terminal checks for REQ-VERIFY-7797."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_7797_v678_training_runtime as exp
from carnot import experiment_7760_v675_online_runner as online
from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import training_qualification as qualification
from carnot.verify import training_runtime as runtime


def test_frozen_protocol_and_roles() -> None:
    """SCENARIO-VERIFY-7797-NUMERICAL: six independent roles and exact recipe."""
    records = exp.fixture_records()
    assert len(records) == 6
    assert {row["role"] for row in records} == {"fit", "tune", "evaluation"}
    assert {row["label"] for row in records} == {0, 1}
    assert len({row["family"] for row in records}) == 6
    exp.validate_roles(records)
    with pytest.raises(ValueError, match="role_poisoning"):
        exp.validate_roles([*records, {**records[0], "role": "evaluation"}])
    with pytest.raises(ValueError, match="feature_label_poisoning"):
        exp.validate_roles([{**records[0], "view_a": {"label": 1}}, *records[1:]])
    protocol = exp.training_protocol([f"p{i}" for i in range(16)])
    assert protocol["natural_recipe"]["epochs"] == 16
    assert protocol["natural_recipe"]["learning_rate"] == 0.01
    assert protocol["miniature_epochs"] == 2
    assert protocol["seeds"] == [67801, 67802, 67803]
    assert len(protocol["arms"]) == 9
    assert protocol["parameter_max"] == 4096


def test_numerical_fit_save_load_and_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-NUMERICAL: actual gradients and cold head parity."""
    names = [f"p{i}" for i in range(16)]
    result = exp.fit_one("constrained_set", 67801, exp.fixture_records(), names, tmp_path)
    assert result["gradient_norm"] > 0
    assert result["initial_hash"] != result["final_hash"]
    assert result["parameter_count"] <= 4096
    assert result["gradient_error"] < 1e-4
    assert result["normalization_error"] < 1e-8
    assert result["reload_decision_equal"] is True
    assert 0 < result["temperature"]
    assert np.isfinite(result["final_loss"])
    assert result["head_hash"] == sha256_file(Path(result["head_path"]))
    assert result["dual_effect_nonzero"] is True
    with pytest.raises(ValueError, match="nonfinite"):
        exp.reject_nonfinite(np.asarray([0.0, float("nan")]))


def test_online_query_modes_and_restart(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-ONLINE: real bank versions and durable queue."""
    checks, _, names = online.preflight(exp.ROOT)
    assert all(row["passed"] for row in checks)
    result = exp.exercise_online(tmp_path, names)
    assert result["valid"] is True
    assert {row["mode"] for row in result["query_rows"]} == {
        "next_query",
        "next_block",
    }
    assert all(row["bank_before"] == row["bank_after"] for row in result["query_rows"])
    assert all(row["model_before"] == row["model_after"] for row in result["query_rows"])
    assert result["restart_parity"] is True
    assert result["overflow_rejected"] is True
    assert result["duplicate_rejected"] is True
    assert result["false_admission_rolled_back"] is True


@pytest.mark.parametrize("failure", ["coverage", "consumer", "stale_log", "broad_child"])
def test_gate_fails_closed(failure: str, tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: four invalid receipts close both scores."""
    scope = {"direct_tests": ["task.py"], "transitive_tests": ["consumer.py"]}
    log = tmp_path / "check.log"
    log.write_text("passed\n")
    receipts = [
        {
            "name": "affected_pytest",
            "command_argv": ["pytest", "task.py", "consumer.py"],
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": sha256_file(log),
        },
        {
            "name": "coverage_report",
            "command_argv": ["coverage", "report", "--fail-under=100"],
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": sha256_file(log),
            "coverage_percent": 100,
        },
    ]
    if failure == "coverage":
        receipts[1]["exit_code"] = 1
    elif failure == "consumer":
        receipts[0]["command_argv"].remove("consumer.py")
    elif failure == "stale_log":
        log.write_text("changed\n")
    else:
        receipts.append({**receipts[0], "name": "full_python_suite"})
    verdict = exp.reduce_validation(scope, receipts, {"affected_pytest", "coverage_report"})
    assert verdict["training_runtime_ready_score"] == 0
    assert verdict["online_runtime_ready_score"] == 0
    assert verdict["verdict_class"] == "disqualified"
    assert verdict["gate_check_summary"]


def test_cold_rows_and_external_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: raw bytes and missing producer stay visible."""
    rows = [{"family": "alpha", "arm": "response_set", "seed": 67801}]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows, "raw_paths": {"rows": str(raw)}}))
    assert exp.cold_reduce(candidate)["row_count"] == 1
    raw.write_text("[]")
    with pytest.raises(ValueError, match="raw_rows_invalid"):
        exp.cold_reduce(candidate)
    blocked = exp.blocked_record(
        "20260928",
        [
            {
                "upstream_id": "producer",
                "artifact_path": "absent.json",
                "field": "exists",
                "operator": "==",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        ],
        0.1,
    )
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["artifact_path"] == "absent.json"
    assert blocked["training_runtime_ready_score"] == blocked["online_runtime_ready_score"] == 0


def test_measure_blocks_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: a missing producer prevents compute."""
    check = {
        "upstream_id": "science",
        "artifact_path": "missing.json",
        "field": "exists",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    monkeypatch.setattr(exp, "preflight", lambda root: ([check], [], []))
    monkeypatch.setattr(exp, "fit_one", lambda *args: pytest.fail("fit was reached"))
    record = exp.measure("20260928", tmp_path)
    assert record["verdict_class"] == "blocked"
    assert record["gate_check_summary"][0]["artifact_path"] == "missing.json"
    assert len(record["rows"]) == 6 * 9 * 3


def test_real_miniature_all_arms(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-NUMERICAL: all nine shapes and three seeds fit."""
    record = exp.measure("20260928", tmp_path)
    assert record["acceptance_gate_results"]["validity"] is True
    assert len(record["fixture_training_rows"]) == 27
    assert len(record["rows"]) == 162
    assert {row["arm"] for row in record["fixture_training_rows"]} == set(exp.ARMS)
    assert {row["seed"] for row in record["fixture_training_rows"]} == set(exp.SEEDS)
    assert {row["role"] for row in record["rows"]} == {"fit", "tune", "evaluation"}
    assert record["online_fixture"]["valid"] is True
    assert record["training_runtime_ready_score"] == 0
    assert record["acceptance_gate_results"]["probability_quality"] is None
    assert exp.cold_reduce(_write_candidate(tmp_path, record))["row_count"] == 162


def _write_candidate(folder: Path, record: dict) -> Path:
    """Save one test candidate for the independent raw-row reader."""
    candidate = folder / "candidate.json"
    candidate.write_text(json.dumps(record))
    return candidate


def test_cli_cold_dispatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: task entrypoint reaches cold reader."""
    path = exp.ROOT / "scripts/experiments/experiment_7797_v678_training_runtime.py"
    spec = importlib.util.spec_from_file_location("exp7797_cli_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"family": "alpha", "arm": "response_set", "seed": 67801}]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows, "raw_paths": {"rows": str(raw)}}))
    assert module.main(["--cold-reduce", str(candidate)]) == 0
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("[exp7797] start")
    assert json.loads(output[-1])["row_count"] == 1


def test_reject_bad_numeric_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-NUMERICAL: invalid shape and custody fail before fit."""
    records = exp.fixture_records()
    with pytest.raises(ValueError, match="role_poisoning"):
        exp.validate_roles(records[:2])
    with pytest.raises(ValueError, match="sixteen_predicates"):
        exp.training_protocol(["one"])
    with pytest.raises(ValueError, match="unregistered"):
        exp.fit_one("missing", 67801, records, [f"p{i}" for i in range(16)], tmp_path)
    monkeypatch.setattr(exp.qualification, "make_batch", lambda *args: ({}, [{"id": "bad"}]))
    with pytest.raises(ValueError, match="unexpected_fixture_exclusion"):
        exp.fit_one("local_set", 67801, records, [f"p{i}" for i in range(16)], tmp_path)


def test_runtime_numeric_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-NUMERICAL: inherited optimizer guards are executable."""
    import jax.numpy as jnp

    names = [f"p{i}" for i in range(16)]
    fit_rows = [row for row in exp.fixture_records() if row["role"] == "fit"]
    batch, excluded = qualification.make_batch(fit_rows, "local_set", names)
    assert excluded == []
    with pytest.raises(ValueError, match="empty fixture"):
        runtime.prepare_batch([])
    bad_view = {"view_a": {"abstention": "empty", "answer_units": []}}
    with pytest.raises(ValueError, match="abstained fixture"):
        runtime._one_view([bad_view], "view_a")
    original = runtime._one_view
    monkeypatch.setattr(
        runtime,
        "_one_view",
        lambda rows, key: {"unit_mask": jnp.zeros((2 if key == "view_a" else 1, 1))},
    )
    with pytest.raises(ValueError, match="view sentence mismatch"):
        runtime.prepare_batch([{}])
    monkeypatch.setattr(runtime, "_one_view", original)
    with pytest.raises(ValueError, match="unregistered arm or seed"):
        runtime.init_params("local_set", 67801)
    params = runtime.init_params("energy_local", 67801)
    assert len(runtime.energy_parameters(params)["weights"]) == 2
    with pytest.raises(ValueError, match="unknown arm"):
        runtime.sentence_support(params, batch["a"], "unknown")
    with pytest.raises(ValueError, match="unknown training mode"):
        runtime.loss(params, batch, "energy_local", "unknown", (0.0, 0.0))
    with pytest.raises(ValueError, match="unregistered training budget"):
        runtime.fit("energy_local", batch, batch, 1, 0.01, 2, "canonical")
    huge = {"w": jnp.zeros((4097, 2)), "b": jnp.zeros(2)}
    with pytest.raises(ValueError, match="parameter budget"):
        runtime.fit("energy_local", batch, batch, 67801, 0.01, 2, "canonical", initial=huge)
    poisoned = {**batch, "label": jnp.asarray([float("nan"), 1.0])}
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.fit("energy_local", poisoned, batch, 67801, 0.01, 2, "canonical")
    assert runtime.temperature_risk(0.0, 1.0) == 0.0


def test_validation_missing_command_and_coverage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: omission and short coverage disqualify."""
    log = tmp_path / "receipt.log"
    log.write_text("ok")
    row = {
        "name": "coverage_report",
        "command_argv": ["coverage", "report"],
        "exit_code": 0,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
        "coverage_percent": 99,
    }
    scope = {"direct_tests": [], "transitive_tests": []}
    result = exp.reduce_validation(scope, [row], {"coverage_report", "affected_pytest"})
    assert {item["field"] for item in result["gate_check_summary"]} == {
        "affected_pytest.count",
        "coverage_percent",
    }
    assert result["training_runtime_ready_score"] == 0


def test_measure_retains_owned_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: failed numeric and online checks stay rows."""
    names = [f"p{i}" for i in range(16)]
    monkeypatch.setattr(exp, "preflight", lambda root: ([], [], names))

    def failed_fit(arm: str, seed: int, records: list, names: list, folder: Path) -> dict:
        return {
            "arm": arm,
            "seed": seed,
            "initial_hash": "same",
            "final_hash": "same",
            "gradient_norm": 0.0,
            "gradient_error": 0.0,
            "normalization_error": 0.0,
            "reload_decision_equal": True,
            "parameter_count": 266,
            "final_loss": 1.0,
            "mode": "canonical",
            "dual_effect_nonzero": None,
            "rows": [{"family": records[0]["family"], "arm": arm, "seed": seed}],
        }

    monkeypatch.setattr(exp, "fit_one", failed_fit)
    monkeypatch.setattr(
        exp, "exercise_online", lambda folder, names: {"valid": False, "query_rows": []}
    )
    record = exp.measure("20260928", tmp_path)
    assert {item["field"] for item in record["gate_check_summary"]} == {
        "numerical_valid",
        "online_valid",
    }
    assert len(record["rows"]) == 27
    assert record["acceptance_gate_results"]["validity"] is False


def _load_cli() -> object:
    """Load the task CLI as code so its orchestration can be checked directly."""
    path = exp.ROOT / "scripts/experiments/experiment_7797_v678_training_runtime.py"
    spec = importlib.util.spec_from_file_location("exp7797_cli_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_frozen_supervisor_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: shard bytes and coverage total are retained."""
    cli = _load_cli()
    data = tmp_path / ".coverage.shard0"
    data.write_bytes(b"private-shard")
    calls = []

    def fake_commands(root: Path, commands: list, **kwargs: object) -> list[dict]:
        calls.append((commands[0].name, commands[0].timeout_s, kwargs["extra_env"]))
        return [{"name": commands[0].name, "output_tail": "TOTAL 10 0 100%"}]

    monkeypatch.setattr(cli, "run_commands", fake_commands)
    shard = cli._one_command({"name": "coverage_shard_0", "argv": [f"--data-file={data}"]})
    assert shard["data_hash"] == sha256_file(data)
    report = cli._one_command({"name": "coverage_report", "argv": ["coverage", "report"]})
    assert report["coverage_percent"] == 100
    cli._one_command({"name": "full_python_suite", "argv": ["pytest"]}, child=True)
    assert calls[-1][1] == 3600
    assert calls[-1][2]["CARNOT_VALIDATION_CHILD"] == "1"
    assert calls[0][2]["COVERAGE_CORE"] == "sysmon"


def test_cli_gate_and_reader_boolean(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: real receipt fields control promotion."""
    cli = _load_cli()
    record = {"gate_check_summary": [], "acceptance_gate_results": {"validity": True}}
    cli._apply_gate(record, {"direct_tests": [], "transitive_tests": []}, [], set())
    assert record["training_runtime_ready_score"] == 1
    assert record["verdict_class"] == "circular_positive"
    record["gate_check_summary"] = [{"field": "failed"}]
    cli._apply_gate(record, {"direct_tests": [], "transitive_tests": []}, [], set())
    assert record["online_runtime_ready_score"] == 0
    path = tmp_path / "adversarial.log"
    path.write_text('{"flagged_count": 0}')
    assert cli._reader_flag({"exit_code": 0, "log_path": str(path)}) is False
    path.write_text('{"flagged_count": 2}')
    assert cli._reader_flag({"exit_code": 0, "log_path": str(path)}) is True
    path.write_text("invalid")
    assert cli._reader_flag({"exit_code": 0, "log_path": str(path)}) is True
    assert cli._reader_flag({"exit_code": 1, "log_path": str(path)}) is True


def test_cli_blocked_publishes_exact_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: blocked external input publishes atomically."""
    cli = _load_cli()
    check = {
        "upstream_id": "science",
        "artifact_path": "missing.json",
        "field": "exists",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    blocked = exp.blocked_record("20260928", [check], 0.01)
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {"commands": [{"name": "adversarial_verify"}, {"name": "strict_row_consistency"}]}
        )
    )
    log = tmp_path / "reader.log"
    log.write_text('{"flagged_count": 0}')
    monkeypatch.setattr(cli.exp, "measure", lambda date: blocked)
    monkeypatch.setattr(cli.exp, "SCOPE", scope)
    monkeypatch.setattr(cli.exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(cli, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(
        cli,
        "_one_command",
        lambda spec: {"name": spec["name"], "exit_code": 0, "log_path": str(log)},
    )
    value = cli.run("20260928")
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["artifact_path"] == "missing.json"
    assert json.loads(cli.exp.OUTPUT.read_text())["verdict_class"] == "blocked"


@pytest.mark.parametrize("failure", ["none", "strict", "adversarial"])
def test_cli_validation_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: exact readers decide final readiness."""
    cli = _load_cli()
    scope = {
        "direct_tests": ["task.py"],
        "transitive_tests": ["consumer.py"],
        "commands": [
            {"name": name}
            for name in (
                "coverage_shard_0",
                "coverage_report",
                "affected_pytest",
                "task_e2e",
                "cold_replay",
                "adversarial_verify",
                "strict_row_consistency",
            )
        ],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    record = {
        "experiment_id": 7797,
        "verdict_class": "disqualified",
        "gate_check_summary": [],
        "rows": [{"family": "alpha"}],
        "acceptance_gate_results": {"validity": True, "readiness": 0},
        "phase_spans": [],
        "validation_receipts": {},
    }
    monkeypatch.setattr(cli.exp, "measure", lambda date: record)
    monkeypatch.setattr(cli.exp, "SCOPE", scope_path)
    monkeypatch.setattr(cli.exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(cli, "PRIVATE", tmp_path / "private")

    def fake_command(spec: dict, *, child: bool = False) -> dict:
        name = spec["name"]
        path = tmp_path / f"{name}.log"
        path.write_text(
            json.dumps({"flagged_count": int(failure == "adversarial")})
            if name == "adversarial_verify"
            else "required check passed"
        )
        return {
            "name": name,
            "command_argv": ["pytest", "task.py", "consumer.py"],
            "exit_code": int(failure == "strict" and name == "strict_row_consistency"),
            "log_path": str(path),
            "log_sha256": sha256_file(path),
            "coverage_percent": 100,
            "timed_out": False,
        }

    monkeypatch.setattr(cli, "_one_command", fake_command)
    value = cli.run("20260928")
    assert value["verdict_class"] == ("disqualified" if failure != "none" else "circular_positive")
    assert value["training_runtime_ready_score"] == int(failure == "none")
    assert value["online_runtime_ready_score"] == int(failure == "none")
    assert len(value["coverage_shard_rows"]) == 1
    assert json.loads(cli.exp.OUTPUT.read_text())["verdict_class"] == value["verdict_class"]


def test_cli_child_and_main_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: child checks raw rows without publishing."""
    cli = _load_cli()
    monkeypatch.setattr(cli, "PRIVATE", tmp_path)
    monkeypatch.setenv("CARNOT_VALIDATION_CHILD", "1")
    measured = {
        "verdict_class": "disqualified",
        "online_fixture": {"valid": True},
        "fixture_training_rows": [{"reload_decision_equal": True}],
    }
    monkeypatch.setattr(cli.exp, "measure", lambda date, raw: measured)
    monkeypatch.setattr(cli.exp, "cold_reduce", lambda path: {"valid": True, "row_count": 162})
    assert cli.main(["--date", "20260928"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["row_count"] == 162
    measured["verdict_class"] = "blocked"
    assert cli.main(["--date", "20260928"]) == 1
    measured["verdict_class"] = "disqualified"
    measured["online_fixture"]["valid"] = False
    assert cli.main(["--date", "20260928"]) == 1
    monkeypatch.delenv("CARNOT_VALIDATION_CHILD")
    monkeypatch.setattr(
        cli, "run", lambda date: {"experiment_id": 7797, "honest_verdict": "complete_disqualified"}
    )
    assert cli.main(["--date", "20260928"]) == 0


def test_cli_main_guard_cold_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7797-TERMINAL: the executable guard reaches cold replay."""
    path = exp.ROOT / "scripts/experiments/experiment_7797_v678_training_runtime.py"
    rows = [{"family": "alpha", "seed": 67801, "arm": "local_set"}]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows, "raw_paths": {"rows": str(raw)}}))
    monkeypatch.setattr(sys, "argv", [str(path), "--cold-reduce", str(candidate)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(path), run_name="__main__")
    assert result.value.code == 0
