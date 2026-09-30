"""Current custody and isolated publication checks for REQ-VERIFY-7943."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import energy_fit_7943 as core


def test_current_primary() -> None:
    """SCENARIO-VERIFY-7943-CUSTODY: use actual current upstream byte custody."""
    sources, dependencies, value = core.authenticate(core.PUBLICATION, core.RUNTIME, core.UPSTREAM)
    assert sources[0]["sha256"] == sha256_file(core.PUBLICATION)
    assert dependencies and value["training_publication_ready_score"] == 1


@pytest.mark.parametrize("mode", ["missing", "malformed", "score", "hash", "callable"])
def test_external_blocks(tmp_path: Path, mode: str) -> None:
    """SCENARIO-VERIFY-7943-CUSTODY: external failure never starts a head fit."""
    path = tmp_path / "publication.json"
    if mode == "malformed":
        path.write_text("{")
    elif mode != "missing":
        value = json.loads(core.PUBLICATION.read_text())
        if mode == "score":
            value["training_publication_ready_score"] = 0
        elif mode == "callable":
            value["training_entrypoint"]["callable"] = "unqualified"
        else:
            value["training_dependency_hashes"]["python/carnot/verify/natural_training.py"] = (
                "changed"
            )
        atomic_json(path, value)
    with pytest.raises(core.old.custody.InputBlocked) as error:
        core.authenticate(path, core.RUNTIME, core.UPSTREAM)
    operand = error.value.operands[0]
    assert operand["upstream_id"] == "exp7941-training-publication"
    assert all(
        key in operand
        for key in (
            "artifact_path",
            "artifact_sha256",
            "artifact_field",
            "op",
            "expected",
            "observed",
        )
    )


def test_frozen_scope(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: route directories and includes are frozen."""
    driver = core.driver(core.PUBLICATION)
    path, commands = driver.freeze(tmp_path / "private", tmp_path / "raw")
    value = json.loads(path.read_text())
    assert value["coverage_includes"] == core.INCLUDES
    roots = [
        Path(c.argv[c.argv.index("--output") + 1]).parent for c in commands if "--output" in c.argv
    ]
    assert len(roots) == len(set(roots))
    assert all(
        c.argv[c.argv.index("--date") + 1] == "20260929"
        for c in commands
        if c.name.startswith("e2e_016")
    )
    assert len(core.old.ARMS) * len(core.old.SEEDS) == 27
    assert next(c for c in commands if c.name == "full_pytest").scope == "repository_health"


def test_fresh_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7943-CUSTODY: fitted metadata belongs to the new producer."""
    monkeypatch.setattr(core.old.natural_training, "fit", lambda *a: {"params": {}, "seed": 67801})
    value = core.current_fit([], [], "local_set", 67801, 0.01, 16)
    assert value["experiment_id"] == 7943 and value["task_id"] == core.TASK


def test_publication_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: newer nested sidecars keep both readers bound."""
    driver = core.driver(core.PUBLICATION)
    output = tmp_path / "experiment_7943_fixture.json"
    value = driver.base(
        core.old.library(core.UPSTREAM, output), core.UPSTREAM, [], [{"external": True}]
    )
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    core.seal(value, output)
    receipt = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(output)
    core.replay(output, True)
    value["rows"] = [{"changed": True}]
    atomic_json(output, value)
    with pytest.raises(ValueError):
        core.replay(output)


def test_terminal_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: owned validation closes readiness before sealing."""
    output = tmp_path / "experiment_7943_fixture.json"
    driver = core.driver(core.PUBLICATION)
    value = driver.base(core.old.library(core.UPSTREAM, output), core.UPSTREAM, [], [])
    calls = []

    def check(*args: object) -> dict:
        calls.append(args)
        return {"passed": True, "flagged_adversarial": len(calls) == 1}

    monkeypatch.setattr(core, "terminal_checks", check)
    core.seal(value, output)
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert len(calls) >= 2


def test_isolated_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7943: reuse callable math without changing historical globals."""
    driver = core.driver(core.PUBLICATION)
    assert driver.core is not core.old
    assert driver.core.OWNED == core.OWNED
    assert core.old.OWNED[0].endswith("7930.py")
    q = driver.core.library(core.UPSTREAM, tmp_path / "experiment_7943_fixture.json")
    assert q.OWNED == list(core.OWNED)
    assert q.START == core.START


def child(tmp_path: Path, name: str, *args: str, expected: int = 0) -> dict:
    """Save real exits and logs so fixture success requires actual numerical fitting."""
    prefix = [sys.executable]
    if os.environ.get("CARNOT7943_COVERAGE_FILE"):
        prefix += [
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            f"--data-file={os.environ['CARNOT7943_COVERAGE_FILE']}",
            f"--include={core.INCLUDES}",
        ]
    command = CommandSpec(
        name, tuple(prefix + [core.CLI, "--date", "20260930", *args]), "required", 90
    )
    row = run_commands(core.ROOT, [command], log_dir=tmp_path / "logs" / name, heartbeat_s=30)[0]
    assert row["exit_code"] == expected, row["output_tail"]
    return row


def test_real_fixture_routes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7943-REPLAY: fresh fit, cold reconstruction and tamper rejection."""
    output = tmp_path / "success/experiment_7943_fixture.json"
    child(tmp_path, "success", "--fixture", "--output", str(output), "--assert-ready")
    value = json.loads(output.read_text())
    assert value["fixture_ready_score"] == 1 and value["energy_fit_ready_score"] == 0
    assert value["verdict_class"] == "circular_positive" and value["verifier_is_oracle"]
    assert value["checkpoint_identity"]["experiment_id"] == 7943
    assert all(r["experiment_id"] == 7943 for r in value["rows"])
    child(tmp_path, "replay", "--cold-replay", str(output))
    child(tmp_path, "terminal", "--terminal-recheck", str(output))
    value["rows"][0]["label"] = 1 - value["rows"][0]["label"]
    atomic_json(output, value)
    child(tmp_path, "tamper", "--cold-replay", str(output), expected=2)
    child(tmp_path, "absent", "--cold-replay", str(tmp_path / "absent.json"), expected=2)


def test_real_blocked_routes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: every publication scenario uses its own directory."""
    output = tmp_path / "blocked/experiment_7943_fixture.json"
    child(
        tmp_path,
        "blocked",
        "--fixture",
        "--publication",
        str(tmp_path / "missing.json"),
        "--output",
        str(output),
    )
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert not value["trained_head_specs"]
    child(tmp_path, "blocked_replay", "--cold-replay", str(output))
    negative = tmp_path / "negative/experiment_7943_fixture.json"
    row = child(
        tmp_path,
        "negative",
        "--fixture",
        "--publication",
        str(tmp_path / "missing.json"),
        "--output",
        str(negative),
        "--assert-ready",
        expected=2,
    )
    assert "source evidence blocked" in row["output_tail"]


@pytest.mark.parametrize("passed", [True, False])
def test_prefit_and_receipt_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: owned route failure prevents natural fitting."""
    original = core.isolated
    called = []

    def isolate(path: Path) -> ModuleType:
        module = original(path)
        if path == Path(core.old_run.__file__):

            def execute(commands: list, raw: Path) -> list:
                called.extend(c.name for c in commands)
                return [{"name": c.name, "passed": passed} for c in commands]

            module.execute = execute
        return module

    monkeypatch.setattr(core, "isolated", isolate)
    run = core.driver(core.PUBLICATION)
    _, commands = run.freeze(tmp_path / "private", tmp_path / "raw")
    if passed:
        refs, deps, value = run.core.authenticate(core.RUNTIME, core.UPSTREAM)
        assert refs and deps and value["runtime_ready_score"] == 1
        receipts = run.execute(commands, tmp_path / "raw")
        assert all(r["passed"] for r in receipts)
        late = [c for c in commands if c.scope == "late"]
        assert run.execute(late, tmp_path / "raw")
    else:
        with pytest.raises(ValueError, match="qualification failed"):
            run.core.authenticate(core.RUNTIME, core.UPSTREAM)
        value = run.base(
            core.old.library(core.UPSTREAM, tmp_path / "experiment_7943_fixture.json"),
            core.UPSTREAM,
            [],
            [],
        )
        assert value["validation_receipts"] and not value["validation_receipts"][0]["passed"]
    from carnot.verify.energy_fit_7943_validation import PREFIT

    assert PREFIT <= set(called)


def test_current_score_and_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7943-REPLAY: identity and class diagnostics derive from primitive rows."""
    rows = [
        {
            "family_id": "one",
            "role": "fit",
            "arm": "local_set",
            "seed": 67801,
            "probability": 0.2,
            "raw_risk": 0.3,
            "label": 0,
            "known_mask": [1, 0, -1],
        }
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    original = core.isolated
    value = {
        "rows": rows,
        "prediction_rows_path": str(path),
        "trained_head_specs": [{"arm": "local_set"}],
        "sample_size_budget": {"exclusion_rows": []},
        "preconditions_checked": {},
    }

    def isolate(source: Path) -> ModuleType:
        module = original(source)
        if source == Path(core.old.__file__):
            module.library = lambda *a: SimpleNamespace(score=lambda *a: (path, []))
        elif source == Path(core.old_run.__file__):
            module.enrich = lambda *a: value
        return module

    monkeypatch.setattr(core, "isolated", isolate)
    run = core.driver(core.PUBLICATION)
    q = run.core.library(core.UPSTREAM, tmp_path / "experiment_7943_fixture.json")
    result, controls = q.score()
    assert json.loads(result.read_text())["experiment_id"] == 7943 and not controls
    actual = run.enrich()
    assert actual["class_counts"]["fit"] == {0: 1}
    assert actual["label_mask_checks"]["unknown"] == 1
    assert actual["calibration_bins"][0]["observed_frequency"] == 0
    assert Path(actual["calibration_plot_path"]).is_file()
    assert actual["trained_head_specs"][0]["experiment_id"] == 7943


def test_owned_reader_and_recheck_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7943-TERMINAL: failed readers and rechecks cannot pass silently."""
    output = tmp_path / "experiment_7943_fixture.json"
    run = core.driver(core.PUBLICATION)
    value = run.base(
        core.old.library(core.UPSTREAM, output), core.UPSTREAM, [], [{"external": True}]
    )
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    core.seal(value, output)
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": False, "flagged_adversarial": False}
    )
    with pytest.raises(ValueError, match="terminal recheck"):
        core.replay(output, True)
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    monkeypatch.setattr(core, "reader_receipt", lambda *a, **kw: {"passed": False})
    with pytest.raises(ValueError, match="consumer mismatch"):
        core.seal(value, output)


def test_main_natural_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7943: the script dispatches explicit natural paths to the qualified driver."""
    from scripts.experiments import experiment_7943_v689_energy_fit as cli

    output = tmp_path / "experiment_7943_fixture.json"
    called = []

    def produce(*args: Path) -> int:
        called.append(args)
        atomic_json(output, {"energy_fit_ready_score": 1})
        return 0

    monkeypatch.setattr(core, "driver", lambda *a: SimpleNamespace(produce=produce))
    assert cli.main(["--date", "20260930", "--output", str(output), "--assert-ready"]) == 0
    assert called == [(core.UPSTREAM, core.RUNTIME, output)]
