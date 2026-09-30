"""Private subprocess regressions for REQ-REPORT-7941 and REQ-VERIFY-7941."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import training_publication_7941 as run


def child(tmp_path: Path, name: str, *args: str, expected: int = 0) -> dict:
    """Retain real child exits and logs so missing inputs cannot stand in for a fit."""
    prefix = [sys.executable]
    if os.environ.get("CARNOT7941_COVERAGE"):
        prefix += ["-m", "coverage", "run", "--parallel-mode", f"--include={run.INCLUDES}"]
    command = CommandSpec(
        name, tuple(prefix + [run.CLI, "--date", "20260930", *args]), "required", 90
    )
    row = run_commands(run.ROOT, [command], log_dir=tmp_path / "logs" / name, heartbeat_s=30)[0]
    assert row["exit_code"] == expected, row["output_tail"]
    return row


@pytest.mark.parametrize("mode", ["missing", "malformed", "hash"])
def test_blocked_routes(tmp_path: Path, mode: str) -> None:
    """SCENARIO-REPORT-7941-ROUTES: blocked success and required failure are separate."""
    runtime = tmp_path / "runtime.json"
    if mode == "malformed":
        runtime.write_text("{")
    elif mode == "hash":
        value = json.loads(run.RUNTIME.read_text())
        value["changed"] = True
        atomic_json(runtime, value)
    output = tmp_path / "blocked" / "experiment_7941_blocked.json"
    child(tmp_path, mode, "--runtime", str(runtime), "--output", str(output), "--fixture")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["runtime_ready_score"] == value["training_publication_ready_score"] == 0
    assert value["gate_check_summary"] and not value["trained_head_specs"]
    child(tmp_path, mode + "_replay", "--cold-replay", str(output))
    negative = tmp_path / "negative" / "experiment_7941_negative.json"
    row = child(
        tmp_path,
        mode + "_negative",
        "--runtime",
        str(runtime),
        "--output",
        str(negative),
        "--fixture",
        "--assert-ready",
        expected=2,
    )
    assert "source evidence blocked" in row["output_tail"]


def test_fixture_replay_and_final_recheck(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7941-CUSTODY: real fit, selected hash and cold predictions agree."""
    output = tmp_path / "fixture" / "experiment_7941_fixture.json"
    child(tmp_path, "fixture", "--fixture", "--output", str(output), "--assert-ready")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["runtime_ready_score"] == 1
    assert value["trained_head_specs"][0]["fresh_current_fit"]
    assert len(value["rows"]) == 3 and value["sample_size_budget"]["independent"] == 0
    child(tmp_path, "replay", "--cold-replay", str(output))
    child(tmp_path, "recheck", "--terminal-recheck", str(output))
    receipt = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(output)
    assert receipt["gate_path"] == str(output)
    value["rows"][0]["label"] = 0
    atomic_json(output, value)
    child(tmp_path, "tamper", "--cold-replay", str(output), expected=2)
    child(tmp_path, "missing_replay", "--cold-replay", str(tmp_path / "absent.json"), expected=2)


@pytest.mark.parametrize(
    "fault,reason", [("conflict", "conflicting_primary"), ("reject", "candidate_rejected")]
)
def test_publication_faults(tmp_path: Path, fault: str, reason: str) -> None:
    """SCENARIO-REPORT-7941-ROUTES: publication faults stay owned failures."""
    output = tmp_path / fault / "experiment_7941_fixture.json"
    row = child(
        tmp_path,
        fault,
        "--fixture",
        "--output",
        str(output),
        "--publication-fault",
        fault,
        expected=2,
    )
    assert reason in row["output_tail"]
    assert not output.exists()


def test_same_id_different_primary_names(tmp_path: Path) -> None:
    """REQ-REPORT-7941: same numeric identity cannot publish differently named siblings."""
    value = {
        "experiment_id": 7941,
        "task_id": "exp7941-training-publication",
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    first = tmp_path / "experiment_7941_negative.json"
    publish_primary(first, value, lambda p: {"passed": True})
    with pytest.raises(ValueError, match="conflicting_primary"):
        publish_primary(
            tmp_path / "experiment_7941_blocked.json", value, lambda p: {"passed": True}
        )
    assert json.loads(first.read_text()) == value


def test_frozen_routes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7941-TERMINAL: scope and private roots precede results."""
    path, commands = run.freeze(tmp_path)
    frozen = json.loads(path.read_text())
    assert frozen["coverage_includes"] == run.INCLUDES
    roots = [
        Path(c.argv[c.argv.index("--output") + 1]).parent for c in commands if "--output" in c.argv
    ]
    assert len(roots) == len(set(roots))
    assert frozen["training_dependency_hashes"]
    assert not Path(frozen["coverage_file"]).resolve().is_relative_to(run.ROOT / "results")
    for command in commands:
        for arg in command.argv:
            if arg.startswith("--basetemp="):
                assert not Path(arg.split("=", 1)[1]).resolve().is_relative_to(run.ROOT / "results")
    for command in commands:
        if command.name.startswith("e2e_016"):
            assert command.argv[command.argv.index("--date") + 1] == "20260929"


def test_repository_health_runs_once(tmp_path: Path) -> None:
    """REQ-REPORT-7941: retained bounded health is reported without a second full run."""
    from carnot.verify import training_publication_7941_run as driver

    log = tmp_path / "old_health.log"
    log.write_text("interrupted owned health\n")
    health = {
        "name": "full_pytest",
        "scope": "repository_health",
        "argv": [sys.executable, "-c", "raise RuntimeError('must not execute')"],
        "passed": False,
        "exit_code": None,
        "timed_out": False,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
        "duration_s": 1,
    }
    atomic_json(tmp_path / "attempts/initial_private_scope/repository_health.json", health)
    _, commands = driver.freeze(tmp_path)
    observed = driver.execute([commands[-1]], tmp_path)
    assert observed[0]["actual_exit"] is None and not observed[0]["passed"]
    assert observed[0]["reused_single_repository_invocation"] is True


@pytest.mark.parametrize(
    "outcome",
    ["ready", "failure", "coverage", "fixture", "blocked", "no_coverage", "missing_receipt"],
)
def test_qualification_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """SCENARIO-REPORT-7941-TERMINAL: only complete owned receipts open readiness."""
    from carnot.verify import training_publication_7941_run as driver

    output = tmp_path / "experiment_7941_qualification.json"
    raw = tmp_path / "raw" / output.stem
    monkeypatch.setattr(run, "seal", lambda value, path: atomic_json(path, value))
    monkeypatch.setattr(run, "replay", lambda *a, **kw: None)
    if outcome == "blocked":

        def blocked(*a):
            raise run.custody.custody.InputBlocked([{"upstream_id": "exp7916", "observed": None}])

        monkeypatch.setattr(run, "authenticate", blocked)

    def execute(commands, raw):
        counts = {
            name: {"summary": {"num_statements": 1, "missing_lines": int(outcome == "coverage")}}
            for name in run.OWNED
        }
        if outcome != "no_coverage":
            atomic_json(raw / "coverage.json", {"files": counts})
        for name in (
            "expected_failure",
            "blocked",
            "malformed",
            "hash_mismatch",
            "success",
            "terminal",
        ):
            value = {
                "experiment_id": 7941,
                "task_id": "exp7941-training-publication",
                "honest_verdict": "complete_circular_positive_fixture",
                "verdict_class": "circular_positive",
                "flagged_adversarial": False,
                "runtime_ready_score": int(name in ("success", "terminal")),
                "trained_head_specs": [{"fresh_current_fit": True}],
                "rows": [{}, {}, {}],
                "checkpoint_manifest_path": str(tmp_path / "checkpoint.json"),
                "checkpoint_manifest_sha256": "unit-fixture-only",
            }
            if outcome == "fixture" and name == "success":
                continue
            atomic_json(driver.route(raw, name), value)
        receipts = [
            {
                "name": c.name,
                "scope": c.scope,
                "passed": outcome != "failure",
                "expected_exit": 0,
                "actual_exit": 0,
                "argv": list(c.argv),
            }
            for c in commands
        ]
        return receipts[:-1] if outcome == "missing_receipt" else receipts

    monkeypatch.setattr(driver, "execute", execute)
    assert driver.qualify(output) == 0
    value = json.loads(output.read_text())
    assert value["training_publication_ready_score"] == int(outcome == "ready")
    assert value["verdict_class"] == (
        "blocked"
        if outcome == "blocked"
        else "circular_positive"
        if outcome == "ready"
        else "disqualified"
    )


def test_guard_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7941-CUSTODY: malformed receipts, reader drift and replay drift fail closed."""
    runtime = tmp_path / "runtime.json"
    atomic_json(
        runtime,
        {
            "experiment_id": 7916,
            "training_runtime_ready_score": 1,
            "flagged_adversarial": False,
            "verdict_class": "null",
            "honest_verdict": "complete_null_fixture",
        },
    )
    with pytest.raises(run.custody.custody.InputBlocked, match="runtime.valid_structure"):
        run.authenticate(runtime)
    monkeypatch.setattr(
        run, "terminal_checks", lambda *a: {"passed": False, "flagged_adversarial": True}
    )
    value = run.base([], [])
    output = tmp_path / "experiment_7941_fixture.json"
    atomic_json(tmp_path / "sidecar.json", {})
    monkeypatch.setattr(
        run,
        "publish_primary",
        lambda p, v, cb: (
            atomic_json(p, v)
            or {"primary_sha256": sha256_file(p), "sidecar_path": str(tmp_path / "sidecar.json")}
        ),
    )
    monkeypatch.setattr(run, "reader_receipt", lambda *a, **kw: {"passed": False})
    with pytest.raises(ValueError, match="consumer mismatch"):
        run.seal(value, output)
    assert json.loads(output.read_text())["training_publication_ready_score"] == 0
    monkeypatch.setattr(run, "read_bound_sidecar", lambda *a: {})
    for mode in ("blocked", "rows", "recheck"):
        value = run.base([], [{"upstream_id": "exp7916"}] if mode == "blocked" else [])
        value["terminal_validation_sidecar_path"] = str(tmp_path / "terminal.json")
        atomic_json(
            tmp_path / "terminal.json", {"validator_sidecar_path": str(tmp_path / "sidecar.json")}
        )
        if mode == "blocked":
            value["runtime_ready_score"] = 1
        elif mode == "recheck":
            from types import SimpleNamespace

            value["fixture_prediction_rows"] = [{}]
            monkeypatch.setattr(run, "library", lambda: SimpleNamespace(replay=lambda *a: None))
        atomic_json(output, value)
        with pytest.raises(ValueError):
            run.replay(output, recheck=mode == "recheck")


def test_real_command_reducer(tmp_path: Path) -> None:
    """REQ-REPORT-7941: expected exits need their reasons and health stays separate."""
    from carnot.verify import training_publication_7941_run as driver

    commands = [
        CommandSpec(
            "expected_failure",
            (sys.executable, "-c", "print('source evidence blocked');raise SystemExit(2)"),
            "route",
            30,
        ),
        CommandSpec("reject", (sys.executable, "-c", "raise SystemExit(2)"), "route", 30),
        CommandSpec("normal", (sys.executable, "-c", "print('passed')"), "required", 30),
    ]
    rows = driver.execute(commands, tmp_path)
    assert [row["passed"] for row in rows] == [True, False, True]
    assert not driver.reduce_readiness(rows, {})


def test_default_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7941-TERMINAL: default dispatch executes qualification once."""
    from scripts.experiments import experiment_7941_v689_training_publication as cli

    output = tmp_path / "experiment_7941_fixture.json"
    calls = []

    def qualify(path):
        calls.append(path)
        atomic_json(path, {"runtime_ready_score": 1})
        return 0

    monkeypatch.setattr(cli, "qualify", qualify)
    assert cli.main(["--date", "20260930", "--output", str(output)]) == 0
    assert calls == [output]


def test_cold_reduction_checks_primitive_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7941-TERMINAL: frozen bytes do not waive missing primitive receipts."""
    from carnot.verify import training_publication_7941_run as driver

    monkeypatch.setattr(run, "read_bound_sidecar", lambda *a: {})
    output = tmp_path / "experiment_7941_fixture.json"
    raw = tmp_path / "raw" / output.stem
    manifest, commands = driver.freeze(raw)
    value = run.base([], [])
    value.update(
        terminal_validation_sidecar_path=str(raw / "terminal.json"),
        validation_command_manifest_path=str(manifest),
        isolated_route_rows=[{"passed": True}],
        training_publication_ready_score=1,
    )
    atomic_json(raw / "terminal.json", {"validator_sidecar_path": str(raw / "sidecar.json")})
    atomic_json(output, value)
    with pytest.raises(ValueError, match="primitive receipt"):
        run.replay(output)


@pytest.mark.parametrize(
    "mode", ["good", "relative", "log", "coverage", "score", "denominator", "hash", "source"]
)
def test_primitive_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7941-TERMINAL: cold reduction rejects each primitive claim drift."""
    monkeypatch.setattr(run, "read_bound_sidecar", lambda *a: {})
    output = tmp_path / "experiment_7941_fixture.json"
    raw = tmp_path / "raw" / output.stem
    counts = {name: {"num_statements": 1, "missing_lines": 0} for name in run.OWNED}
    atomic_json(
        raw / "coverage.json", {"files": {name: {"summary": item} for name, item in counts.items()}}
    )
    atomic_json(raw / "manifest.json", {"commands": [{"name": "done"}]})
    log = tmp_path / "log.txt"
    log.write_text("passed\n")
    receipt = {
        "name": "done",
        "scope": "route",
        "passed": True,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
    }
    value = run.base([], [])
    value.update(
        terminal_validation_sidecar_path=str(raw / "terminal.json"),
        validation_command_manifest_path=str(raw / "manifest.json"),
        isolated_route_rows=[receipt],
        validation_receipts=[receipt],
        training_publication_ready_score=1,
        coverage_statement_counts=counts,
    )
    value["sample_size_budget"]["completed"] = 1
    atomic_json(raw / "terminal.json", {"validator_sidecar_path": str(raw / "sidecar.json")})
    if mode in ("good", "relative"):
        blocked = tmp_path / "blocked" / "experiment_7941_fixture.json"
        blocked_value = run.base([], [{"upstream_id": "exp7916", "observed": None}])
        blocked_value["terminal_validation_sidecar_path"] = str(raw / "terminal.json")
        atomic_json(blocked, blocked_value)
        value["consumer_selection_rows"] = [
            {"gate_path": str(blocked), "gate_sha256": sha256_file(blocked)}
        ]
        if mode == "relative":
            receipt["log_path"] = os.path.relpath(log, run.ROOT)
    elif mode == "log":
        log.write_text("changed")
    elif mode == "coverage":
        atomic_json(raw / "coverage.json", {"files": {}})
    elif mode == "score":
        value["training_publication_ready_score"] = 0
    elif mode == "denominator":
        value["sample_size_budget"]["completed"] = 2
    elif mode == "hash":
        value["consumer_selection_rows"] = [{"gate_path": str(log), "gate_sha256": "wrong"}]
    elif mode == "source":
        value["source_artifact_hashes"] = [{"path": str(log), "sha256": "wrong"}]
    atomic_json(output, value)
    if mode in ("good", "relative"):
        run.replay(output)
    else:
        with pytest.raises(ValueError):
            run.replay(output)
