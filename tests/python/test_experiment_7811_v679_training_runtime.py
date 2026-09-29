"""Regression tests for REQ-VERIFY-7811 and its sealed attempt contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot import experiment_7811_v679_training_runtime as exp
from carnot.reporting.current_work_receipt import sha256_file


def load_cli():
    """Load the actual CLI file so dispatch tests cover its production functions."""
    path = exp.ROOT / "scripts/experiments/experiment_7811_v679_training_runtime.py"
    spec = importlib.util.spec_from_file_location("experiment_7811_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_and_fixture_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-MECHANICS: recipe and source roles stay fixed."""
    records = exp.fixture_records()
    assert len(records) == 6
    assert len({row["family"] for row in records}) == 6
    assert {row["label"] for row in records} == {0, 1}
    names = [f"p{i}" for i in range(16)]
    training, online = exp.protocols(names)
    assert training["seeds"] == [67815, 67816, 67817]
    assert training["miniature_epochs"] == 2
    assert training["natural_recipe"] == {
        "epochs": 16,
        "learning_rate": 0.01,
        "optimizer": "full_batch_gradient_descent",
    }
    assert training["parameter_max"] == 4096
    assert len(training["arms"]) == 9
    assert online["commit_modes"] == ["next_query", "next_block"]
    poisoned = [{**records[0], "role": "evaluation"}, *records[1:]]
    with pytest.raises(ValueError, match="role_poisoning"):
        exp.validate_roles(poisoned)
    with pytest.raises(ValueError, match="nonfinite"):
        exp.reject_nonfinite([float("nan")])


def test_sealed_logs_survive_retry_and_reject_mutation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-RECEIPT: a retry cannot overwrite prior bytes."""
    receipts = []
    for attempt in ("first", "second"):
        private = tmp_path / attempt / "private"
        private.mkdir(parents=True)
        log = private / "unit.log"
        log.write_bytes(b"passed\n")
        row = exp.seal_log(log, tmp_path / "durable" / attempt, "unit")
        assert Path(row["log_path"]).read_bytes() == b"passed\n"
        assert row["log_sha256"] == sha256_file(Path(row["log_path"]))
        receipts.append(row)
    assert receipts[0]["log_path"] != receipts[1]["log_path"]
    assert exp.read_sealed_log(receipts[0]) == b"passed\n"
    Path(receipts[0]["log_path"]).write_bytes(b"qassed\n")
    with pytest.raises(ValueError, match="log_sha256"):
        exp.read_sealed_log(receipts[0])


def test_dispatch_matches_frozen_manifest_including_tail(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: the real CLI runs every frozen child."""
    cli = load_cli()
    scope = json.loads(exp.SCOPE.read_text())
    private = tmp_path / "attempt"
    manifest = cli.materialize(scope, private)
    observed = []

    def recording_child(command, durable):
        assert private.is_dir()
        for arg in command["argv"]:
            if arg.startswith("--basetemp="):
                assert Path(arg.split("=", 1)[1]).parent.is_dir()
        observed.append({key: command[key] for key in ("name", "argv", "classification")})
        log = private / command["name"] / "child.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"ok\n")
        return {
            **exp.seal_log(log, durable, command["name"]),
            **observed[-1],
            "exit_code": 0,
            "timed_out": False,
        }

    rows = cli.dispatch(manifest, private, tmp_path / "durable", recording_child)
    expected = [
        {key: row[key] for key in ("name", "argv", "classification")}
        for row in manifest["commands"]
    ]
    assert observed == expected
    assert len(rows) == len(expected)
    assert expected[-1]["name"] == "repository_health"
    assert expected[-2]["name"] == "strict_row_consistency"


@pytest.mark.parametrize("failure", ["coverage", "consumer", "stale_log", "undeclared"])
def test_failed_required_receipt_closes_both_gates(failure: str, tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: changed evidence cannot open readiness."""
    scope = {"direct_tests": ["direct.py"], "transitive_tests": ["consumer.py"]}
    log = tmp_path / "original.log"
    log.write_bytes(b"ok")
    sealed = exp.seal_log(log, tmp_path / "durable", "affected_pytest")
    manifest = {
        "commands": [
            {
                "name": "affected_pytest",
                "argv": ["pytest", "direct.py", "consumer.py"],
                "classification": "required",
            },
            {
                "name": "coverage_report",
                "argv": ["coverage", "report"],
                "classification": "required",
            },
        ]
    }
    receipts = [
        {**sealed, **manifest["commands"][0], "exit_code": 0, "timed_out": False},
        {
            **sealed,
            **manifest["commands"][1],
            "exit_code": 0,
            "timed_out": False,
            "coverage_percent": 100,
        },
    ]
    if failure == "coverage":
        receipts[1]["exit_code"] = 1
    elif failure == "consumer":
        receipts[0]["argv"] = ["pytest", "direct.py"]
    elif failure == "stale_log":
        Path(sealed["log_path"]).write_bytes(b"no")
    else:
        receipts.append({**receipts[0], "name": "full_python_suite"})
    gate = exp.reduce_validation(scope, manifest, receipts)
    assert gate["training_runtime_ready_score"] == 0
    assert gate["online_runtime_ready_score"] == 0
    assert gate["verdict_class"] == "disqualified"
    assert gate["gate_check_summary"]


def test_missing_science_source_blocks_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7811: an absent producer is an external blocked operand."""
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "fit_one", lambda *args: pytest.fail("fit started"))
    record = exp.measure("20260928", tmp_path / "raw")
    assert record["honest_verdict"].startswith("complete_blocked_")
    assert record["verdict_class"] == "blocked"
    assert record["gate_check_summary"]
    assert record["sample_size_budget"]["started"] == 0
    assert record["training_runtime_ready_score"] == 0
    assert record["online_runtime_ready_score"] == 0
    assert len(record["rows"]) == 6 * 9 * 3


def test_single_head_warms_real_optimizer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-MECHANICS: one seeded head has a real update."""
    result = exp.fit_one(
        "response_set", exp.SEEDS[0], exp.fixture_records(), [f"p{i}" for i in range(16)], tmp_path
    )
    assert result["gradient_norm"] > 0
    assert result["initial_hash"] != result["final_hash"]
    assert result["reload_decision_equal"] is True


def test_miniature_fit_bank_and_cold_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-MECHANICS: real heads and bank produce replayable rows."""
    record = exp.measure("20260928", tmp_path / "raw")
    assert record["acceptance_gate_results"]["validity"] is True
    assert len(record["fixture_training_rows"]) == 27
    assert len(record["rows"]) == 162
    assert {row["arm"] for row in record["fixture_training_rows"]} == set(exp.ARMS)
    assert {row["seed"] for row in record["fixture_training_rows"]} == set(exp.SEEDS)
    assert all(row["gradient_norm"] > 0 for row in record["fixture_training_rows"])
    assert all(row["initial_hash"] != row["final_hash"] for row in record["fixture_training_rows"])
    assert all(row["normalization_error"] < 1e-8 for row in record["fixture_training_rows"])
    assert all(row["reload_decision_equal"] for row in record["fixture_training_rows"])
    assert all(row["parameter_count"] <= 4096 for row in record["fixture_training_rows"])
    assert record["online_fixture"]["valid"] is True
    assert record["acceptance_gate_results"]["probability_quality"] is None
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(record))
    assert exp.cold_reduce(candidate)["row_count"] == 162
    Path(record["raw_paths"]["rows"]).write_text("[]")
    with pytest.raises(ValueError, match="raw_rows_invalid"):
        exp.cold_reduce(candidate)


def test_historical_bytes_and_failure_are_preserved() -> None:
    """SCENARIO-VERIFY-7811-RECEIPT: Exp7797 stays disqualified."""
    old = exp.ROOT / "results/experiment_7797_v678_training_runtime.json"
    value = json.loads(old.read_text())
    assert value["verdict_class"] == "disqualified"
    assert any(row["field"] == "ruff_format.exit_code" for row in value["gate_check_summary"])
    assert sum(row["field"].endswith("log_sha256") for row in value["gate_check_summary"]) == 5
    assert sha256_file(old) == exp.HISTORICAL_7797_SHA256
    audit = exp.historical_receipt_byte_audit(exp.ROOT)
    assert len(audit) == 5
    assert {row["name"] for row in audit} == {
        "coverage_shard_0",
        "coverage_shard_1",
        "coverage_shard_2",
        "coverage_report",
        "affected_pytest",
    }
    assert all(row["original_bytes_available"] is False for row in audit)
    assert all(bytes.fromhex(row["current_bytes_hex"]) for row in audit)
    assert all("run_commands" in row["writer_code_path"] for row in audit)


def test_cli_executes_and_seals_real_children(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-RECEIPT: closed subprocess logs get their own paths."""
    cli = load_cli()
    private, durable = tmp_path / "private", tmp_path / "durable"
    commands = [
        {
            "name": "unit",
            "argv": [sys.executable, "-c", "print('ok')"],
            "classification": "required",
            "timeout_s": 3,
        },
        {
            "name": "coverage_report",
            "argv": [sys.executable, "-c", "print('TOTAL 10 0 100%')"],
            "classification": "required",
            "timeout_s": 3,
        },
        {
            "name": "coverage_shard_0",
            "argv": [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1].split('=',1)[1]).write_bytes(b'shard')",
                f"--data-file={tmp_path / 'shard'}",
            ],
            "classification": "required",
            "timeout_s": 3,
        },
    ]
    for command in commands:
        row = cli.execute(command, durable, private)
        assert row["exit_code"] == 0
        assert isinstance(exp.read_sealed_log(row), bytes)
        assert Path(row["log_path"]).is_relative_to(durable)
        if command["name"] == "coverage_report":
            assert row["coverage_percent"] == 100
        if command["name"].startswith("coverage_shard_"):
            assert row["data_sha256"] == sha256_file(tmp_path / "shard")
    timeout = {
        "name": "timeout",
        "argv": [sys.executable, "-c", "import time; time.sleep(5)"],
        "classification": "diagnostic",
        "timeout_s": 0.1,
    }
    row = cli.execute(timeout, durable, private)
    assert row["timed_out"] is True
    assert row["exit_code"] != 0


def test_cli_hard_kill_of_its_owned_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: a stuck owned child is reaped."""
    cli = load_cli()
    events = []

    class Child:
        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if len([event for event in events if event[0] == "wait"]) < 3:
                raise subprocess.TimeoutExpired("owned", timeout)
            return -9

        def terminate(self):
            events.append(("terminate", None))

        def kill(self):
            events.append(("kill", None))

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: Child())
    command = {
        "name": "owned",
        "argv": [sys.executable],
        "classification": "required",
        "timeout_s": 0,
    }
    row = cli.execute(command, tmp_path / "durable", tmp_path / "private")
    assert row["timed_out"] and row["exit_code"] == -9
    assert [event[0] for event in events] == ["wait", "terminate", "wait", "kill", "wait"]


def test_core_rejects_unregistered_and_bad_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7811-RECEIPT: numerical and byte guards fail closed."""
    with pytest.raises(ValueError, match="unregistered"):
        exp._init_params("response_set", 1)
    with pytest.raises(ValueError, match="unregistered"):
        exp.fit_one("unknown", exp.SEEDS[0], exp.fixture_records(), [], tmp_path)
    with pytest.raises(ValueError, match="unregistered"):
        exp._fit_head(
            "response_set",
            {},
            {},
            1,
            "canonical",
            {"w": exp._init_params("response_set", exp.SEEDS[0])["w"]},
        )
    log = tmp_path / "private.log"
    log.write_bytes(b"ok")
    real_hash = exp.sha256_file
    monkeypatch.setattr(
        exp, "sha256_file", lambda path: real_hash(path) if path == log else "sha256:bad"
    )
    with pytest.raises(ValueError, match="log_sha256_mismatch_after_copy"):
        exp.seal_log(log, tmp_path / "durable", "unit")


def test_nonfinite_gradient_and_excluded_fixture_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7811-MECHANICS: invalid optimizer inputs cannot qualify."""
    import jax.numpy as jnp

    monkeypatch.setattr(
        exp.jax,
        "value_and_grad",
        lambda fn: lambda p, d: (jnp.nan, {key: jnp.zeros_like(value) for key, value in p.items()}),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        exp._fit_head(
            "response_set",
            {},
            {},
            exp.SEEDS[0],
            "canonical",
            exp._init_params("response_set", exp.SEEDS[0]),
        )
    monkeypatch.setattr(exp.qualification, "make_batch", lambda *args: ({}, ["excluded"]))
    with pytest.raises(ValueError, match="unexpected_fixture_exclusion"):
        exp.fit_one(
            "response_set",
            exp.SEEDS[0],
            exp.fixture_records(),
            [f"p{i}" for i in range(16)],
            tmp_path,
        )


def test_invalid_fixture_keeps_failure_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7811: a failed measurement cannot acquire readiness."""

    def invalid_fit(arm, seed, records, names, folder):
        return {
            "arm": arm,
            "seed": seed,
            "initial_hash": "same",
            "final_hash": "same",
            "gradient_norm": 0,
            "gradient_error": 0,
            "normalization_error": 0,
            "reload_decision_equal": True,
            "parameter_count": 1,
            "final_loss": 0,
            "mode": "canonical",
            "rows": [],
        }

    monkeypatch.setattr(exp, "fit_one", invalid_fit)
    monkeypatch.setattr(
        exp.prior, "exercise_online", lambda *args: {"valid": False, "query_rows": []}
    )
    record = exp.measure("20260928", tmp_path / "raw")
    assert record["acceptance_gate_results"]["validity"] is False
    assert record["gate_check_summary"][0]["field"] == "fixture_valid"
    assert record["training_runtime_ready_score"] == 0


def test_missing_order_coverage_and_cold_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: every declared child and sealed byte matters."""
    log = tmp_path / "child.log"
    log.write_bytes(b"ok")
    seal = exp.seal_log(log, tmp_path / "durable", "unit")
    commands = [
        {"name": "affected_pytest", "argv": ["pytest", "a.py"], "classification": "required"},
        {"name": "coverage_report", "argv": ["coverage", "report"], "classification": "required"},
    ]
    rows = [
        {**seal, **command, "exit_code": 0, "timed_out": False, "coverage_percent": 100}
        for command in commands
    ]
    scope = {"direct_tests": ["a.py"], "transitive_tests": []}
    assert (
        exp.reduce_validation(scope, {"commands": commands}, rows)["training_runtime_ready_score"]
        == 1
    )
    reversed_rows = list(reversed(rows))
    failure = exp.reduce_validation(scope, {"commands": commands}, reversed_rows)
    assert any(item["field"] == "command_order" for item in failure["gate_check_summary"])
    missing = exp.reduce_validation(scope, {"commands": commands}, rows[:1])
    assert any(item["field"] == "coverage_report.count" for item in missing["gate_check_summary"])
    rows[1]["coverage_percent"] = 99
    coverage = exp.reduce_validation(scope, {"commands": commands}, rows)
    assert any(item["field"] == "coverage_percent" for item in coverage["gate_check_summary"])
    raw = tmp_path / "rows.json"
    raw.write_text("[]")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "rows": [],
                "raw_paths": {"rows": str(raw)},
                "validation_receipts": {"commands": [rows[0]]},
            }
        )
    )
    assert exp.cold_reduce(candidate)["row_count"] == 0


@pytest.mark.parametrize("flagged", [False, True])
def test_real_cli_run_with_recorded_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flagged: bool
) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: the CLI publishes only after its tail."""
    cli = load_cli()
    scope = {
        "requirement": "REQ-VERIFY-7811",
        "direct_tests": ["a.py"],
        "transitive_tests": [],
        "changed_modules_and_cli": [],
        "dependency_modules": [],
        "commands": [
            {
                "name": "affected_pytest",
                "argv": ["pytest", "a.py"],
                "classification": "required",
                "timeout_s": 2,
            },
            {
                "name": "coverage_report",
                "argv": ["coverage", "report"],
                "classification": "required",
                "timeout_s": 2,
            },
            {
                "name": "cold_replay",
                "argv": ["python", "{private}/candidate.json"],
                "classification": "required",
                "timeout_s": 2,
            },
            {
                "name": "adversarial_verify",
                "argv": ["python", "verify"],
                "classification": "required",
                "timeout_s": 2,
            },
            {
                "name": "strict_row_consistency",
                "argv": ["python", "strict"],
                "classification": "required",
                "timeout_s": 2,
            },
            {
                "name": "repository_health",
                "argv": ["pytest", "tests/python"],
                "classification": "diagnostic",
                "timeout_s": 2,
            },
        ],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(exp, "SCOPE", scope_path)
    record = exp._base_record("20260928", [], [], 0.01)
    record["acceptance_gate_results"]["validity"] = True
    monkeypatch.setattr(exp, "measure", lambda *args: json.loads(json.dumps(record)))

    def recorded(command, durable, private):
        log = private / command["name"] / "child.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps({"flagged_count": int(flagged)})
            if command["name"] == "adversarial_verify"
            else "ok"
        )
        receipt = exp.seal_log(log, durable, command["name"])
        return {
            **receipt,
            "name": command["name"],
            "argv": command["argv"],
            "classification": command["classification"],
            "exit_code": 1 if command["name"] == "repository_health" else 0,
            "timed_out": False,
            "coverage_percent": 100,
        }

    monkeypatch.setattr(cli, "execute", recorded)
    result = cli.run("20260928", tmp_path / "attempt")
    assert result["flagged_adversarial"] is flagged
    assert result["repository_health"]["exit_code"] == 1
    assert result["training_runtime_ready_score"] == (0 if flagged else 1)
    assert result["online_runtime_ready_score"] == (0 if flagged else 1)
    assert result["observed_child_commands"][-1]["name"] == "repository_health"
    assert json.loads(exp.OUTPUT.read_text())["verdict_class"] == result["verdict_class"]
    assert Path(result["validation_receipts"]["candidate_path"]).is_file()


def test_real_cli_blocked_and_main_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7811: external blocks and CLI modes retain their true state."""
    cli = load_cli()
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps({"requirement": "REQ-VERIFY-7811", "commands": []}))
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(exp, "SCOPE", scope_path)
    check = {
        "upstream_id": "producer",
        "artifact_path": "missing.json",
        "artifact_sha256": None,
        "field": "exists",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    blocked = exp._base_record("20260928", [check], [], 0.01)
    monkeypatch.setattr(exp, "measure", lambda *args: blocked)
    result = cli.run("20260928", tmp_path / "blocked-attempt")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"][0]["artifact_path"] == "missing.json"
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "e2e")]) == 1
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"valid": True, "row_count": 2})
    assert cli.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 0
    monkeypatch.setattr(
        cli,
        "run",
        lambda *args: {
            "experiment_id": 7811,
            "honest_verdict": "complete_blocked_external_precondition",
        },
    )
    assert cli.main(["--date", "20260928"]) == 0


def test_cli_miniature_routes_and_terminal_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: miniature and reader flags use actual values."""
    cli = load_cli()
    record = exp._base_record("20260928", [], [], 0.01)
    record["online_fixture"] = {"valid": True}
    record["fixture_training_rows"] = [{"reload_decision_equal": True}]
    monkeypatch.setattr(exp, "measure", lambda *args: record)
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"valid": True, "row_count": 6})
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "mini")]) == 0
    assert (tmp_path / "mini/candidate.json").is_file()
    record["fixture_training_rows"][0]["reload_decision_equal"] = False
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "mini-bad")]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--mini-e2e"])
    log = tmp_path / "flag.log"
    log.write_text("not-json")
    receipt = {
        **exp.seal_log(log, tmp_path / "durable", "flag"),
        "exit_code": 0,
        "timed_out": False,
    }
    assert cli._adversarial_flag(receipt) is True
    receipt["exit_code"] = 1
    assert cli._adversarial_flag(receipt) is True
    log.write_text('{"flagged_count": 0}')
    clean = {
        **exp.seal_log(log, tmp_path / "durable2", "clean"),
        "exit_code": 0,
        "timed_out": False,
    }
    assert cli._adversarial_flag(clean) is False


def test_cli_main_guard_cold_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7811-DISPATCH: the file entrypoint honors cold mode."""
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"valid": True, "row_count": 0})
    path = exp.ROOT / "scripts/experiments/experiment_7811_v679_training_runtime.py"
    monkeypatch.setattr(sys, "argv", [str(path), "--cold-reduce", str(candidate)])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(path), run_name="__main__")
    assert done.value.code == 0
