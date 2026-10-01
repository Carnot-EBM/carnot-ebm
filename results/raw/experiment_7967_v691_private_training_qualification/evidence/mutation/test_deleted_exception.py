"""Qualify real fitting exceptions and receipt reductions for REQ-VERIFY-7954."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import energy_fit_7943 as fit
from carnot.verify import training_coverage_7954 as core




def test_main_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-7954: dispatch coverage preserves numerical and readiness branches."""
    from scripts.experiments import experiment_7943_v689_energy_fit as cli

    output = tmp_path / "experiment_7943_fixture.json"

    def produce(*args: object) -> int:
        atomic_json(output, {"energy_fit_ready_score": 1, "fixture_ready_score": 1})
        return 0

    monkeypatch.setattr(fit, "driver", lambda *a: SimpleNamespace(produce=produce))
    monkeypatch.setattr(fit, "fixture", produce)
    monkeypatch.setattr(fit, "replay", lambda *a: None)
    base = ["--date", "20260930", "--output", str(output)]
    assert cli.main(base) == 0
    assert cli.main(base + ["--assert-ready"]) == 0
    assert cli.main(base + ["--fixture", "--assert-ready"]) == 0
    assert cli.main(base + ["--cold-replay", str(output)]) == 0
    assert cli.main(base + ["--terminal-recheck", str(output)]) == 0
    monkeypatch.setattr(fit, "fixture", lambda *a: 0)
    atomic_json(output, {"fixture_ready_score": 0})
    assert cli.main(base + ["--fixture", "--assert-ready"]) == 2


def test_freeze(tmp_path: Path) -> None:
    """REQ-REPORT-7954: explicit scope and historical E2E dates precede execution."""
    manifest, commands = core.freeze(tmp_path)
    frozen = json.loads(manifest.read_text())
    assert frozen["coverage_includes"] == core.INCLUDES
    assert all(frozen["dependencies"].values())
    assert (tmp_path / "attempts").is_dir()
    assert Path(frozen["coverage_file"]).is_relative_to(Path("/tmp"))
    roots = [
        Path(c.argv[c.argv.index("--output") + 1]).parent for c in commands if "--output" in c.argv
    ]
    assert len(roots) == len(set(roots))
    assert all(
        c.argv[c.argv.index("--date") + 1] == "20260929"
        for c in commands
        if c.name.startswith("e2e_016")
    )
    assert not any(c.scope == "repository_health" for c in commands)


def test_execute_and_reduce(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7954-READINESS: nonzero coverage and required reasons are mandatory."""
    commands = [
        CommandSpec(
            "malformed_replay",
            (
                str(core.ROOT / ".venv/bin/python"),
                "-c",
                "print('phase=failure Expecting property name');raise SystemExit(2)",
            ),
            "exception",
            30,
        )
    ]
    rows = core.execute(commands, tmp_path)
    assert rows[0]["passed"] and rows[0]["expected_exit"] == 2
    counts = {name: {"num_statements": 1, "missing_lines": 0} for name in core.COVERED}
    assert core.coverage_ready(counts)
    for change in (
        {},
        {core.COVERED[0]: {"num_statements": 0, "missing_lines": 0}},
        {core.COVERED[0]: {"num_statements": 1, "missing_lines": 1}},
    ):
        assert not core.coverage_ready(change)
    assert not core.ready([], counts, [])


def test_seal_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7954-TERMINAL: final bytes and newer sidecars bind both readers."""
    value = core.base([], [], {})
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    output = tmp_path / "experiment_7954_fixture.json"
    core.seal(value, output)
    core.replay(output)
    selected = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert selected["gate_sha256"] == selected["document_sha256"] == sha256_file(output)
    atomic_json(output.with_name("experiment_7954_conflict.json"), value)
    with pytest.raises(ValueError, match="conflicting_primary"):
        core.seal(value, output)
    output.with_name("experiment_7954_conflict.json").unlink()
    value["rows"] = [{"passed": False}]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="stale_primary_hash"):
        core.replay(output)


def test_terminal_and_reader_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7954: owned terminal failures close all current readiness fields."""
    calls = []

    def check(*args: object) -> dict:
        calls.append(args)
        return {"passed": True, "flagged_adversarial": len(calls) == 1}

    monkeypatch.setattr(core, "terminal_checks", check)
    value = core.base([], [], {})
    value.update(
        training_coverage_ready_score=1, prefit_coverage_ready_score=1, runtime_ready_score=1
    )
    output = tmp_path / "experiment_7954_fixture.json"
    core.seal(value, output)
    assert value["verdict_class"] == "disqualified"
    assert all(value[k] == 0 for k in core.SCORES)
    monkeypatch.setattr(core, "reader_receipt", lambda *a, **k: {"passed": False})
    with pytest.raises(ValueError, match="primary consumer mismatch"):
        core.seal(value, output)


@pytest.mark.parametrize("mode", ["ready", "failed", "blocked", "missing_outputs"])
def test_qualify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-VERIFY-7954-READINESS: cold evidence controls every readiness field."""
    output = tmp_path / "experiment_7954_fixture.json"
    raw = output.parent / "raw" / output.stem
    if mode == "ready":
        atomic_json(
            raw / "attempts/initial_private_scope/experiment_7954_private.json",
            {
                "honest_verdict": "complete_disqualified_required_checks",
                "validation_receipts": [
                    {"name": "affected_unit", "scope": "required", "passed": False}
                ],
            },
        )
    if mode == "blocked":

        def authenticate(*args: object) -> None:
            raise fit.old.custody.InputBlocked(
                [{"artifact_field": "artifact.exists", "observed": False}]
            )

        monkeypatch.setattr(fit, "authenticate", authenticate)

    def execute(commands: list, root: Path) -> list:
        frozen = json.loads((root / "validation_command_manifest.json").read_text())
        Path(frozen["coverage_file"]).write_bytes(b"private measured data")
        counts = {name: {"num_statements": 1, "missing_lines": 0} for name in core.COVERED}
        if mode != "missing_outputs":
            atomic_json(
                root / "coverage.json", {"files": {n: {"summary": v} for n, v in counts.items()}}
            )
        fixture = {
            "fixture_ready_score": 1,
            "energy_fit_ready_score": 0,
            "trained_head_specs": [{"arm": "local_set"}],
            "rows": [{}, {}, {}],
            "checkpoint_manifest_path": str(root / "checkpoint.json"),
            "checkpoint_manifest_sha256": "sha256:fixture",
            "source_artifact_hashes": [],
        }
        if mode != "missing_outputs":
            atomic_json(core.route(root, "success"), fixture)
        return [
            {
                "name": c.name,
                "passed": mode == "ready",
                "scope": c.scope,
                "argv": list(c.argv),
                "actual_exit": 0,
                "expected_exit": 0,
                "log_path": str(root / "log"),
                "log_sha256": "sha256:test",
            }
            for c in commands
        ]

    monkeypatch.setattr(core, "execute", execute)
    monkeypatch.setattr(
        core, "negative_mutation", lambda *a: {"passed": mode == "ready", "owned_control": True}
    )
    monkeypatch.setattr(core, "selections", lambda *a: [{"passed": True}] * 3)
    monkeypatch.setattr(core, "seal", lambda v, p: atomic_json(p, v))
    monkeypatch.setattr(core, "replay", lambda *a: None)
    assert core.qualify(output) == 0
    value = json.loads(output.read_text())
    assert value["runtime_ready_score"] == int(mode == "ready")
    assert value["training_coverage_ready_score"] == value["prefit_coverage_ready_score"]
    assert (
        value["verdict_class"]
        == {
            "ready": "circular_positive",
            "failed": "disqualified",
            "blocked": "blocked",
            "missing_outputs": "disqualified",
        }[mode]
    )
    assert value["MODEL_SPECS"] == [] and value["run_date"] == "20260930"
    if mode != "blocked":
        assert value["cited_upstream_artifacts"]
        assert bool(value["coverage_statement_counts"]) == (mode != "missing_outputs")


def test_selection_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7954: success and blocked primaries require actual reader checks."""
    monkeypatch.setattr(core, "reader_receipt", lambda *a, **k: {"passed": True})
    assert not core.selections(tmp_path)
    for name in ("success", "blocked", "expected_failure"):
        atomic_json(core.route(tmp_path, name), {})
    assert len(core.selections(tmp_path)) == 3


def test_negative_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7954-EXCEPTIONS: deleting the boundary test exposes 51-53."""
    from coverage import CoverageData

    data = CoverageData(basename=str(tmp_path / ".coverage"))
    data.add_lines({str(core.ROOT / name): {1} for name in core.COVERED})
    data.write()

    def execute(root: Path, commands: list, **kwargs: object) -> list:
        assert "--append" in next(c.argv for c in commands if c.name == "mutation_combine")
        atomic_json(
            tmp_path / "mutation/coverage.json",
            {"files": {fit.CLI: {"missing_lines": [51, 52, 53]}}},
        )
        return [
            {
                "name": c.name,
                "exit_code": 2 if c.name == "mutation_report" else 0,
                "passed": c.name != "mutation_report",
                "timed_out": False,
            }
            for c in commands
        ]

    monkeypatch.setattr(core, "run_commands", execute)
    report = core.negative_mutation(tmp_path)
    assert report["passed"] and report["deleted_test"] == "test_real_exception_routes"
    copied = Path(report["mutated_test_path"]).read_text()
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "test_real_exception_routes"
        for n in ast.parse(copied).body
    )


def test_replay_reductions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7954-TERMINAL: primitive drift is caught after custody checks."""
    value = core.base([], [], {})
    path = tmp_path / "experiment_7954_fixture.json"
    raw = tmp_path / "raw" / path.stem
    log = raw / "log"
    log.parent.mkdir(parents=True)
    log.write_text("passed")
    value.update(
        validation_receipts=[{"log_path": str(log), "log_sha256": sha256_file(log)}],
        source_artifact_hashes=[{"path": str(log), "sha256": sha256_file(log)}],
    )
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    value["validation_receipts"][0]["log_path"] = str(log.relative_to(tmp_path))
    monkeypatch.setattr(core, "ROOT", tmp_path)
    core.seal(value, path)
    core.replay(path)
    log.write_text("drift")
    with pytest.raises(ValueError, match="primitive hash drift"):
        core.replay(path)
    value["validation_receipts"] = []
    core.seal(value, path)
    with pytest.raises(ValueError, match="primitive hash drift"):
        core.replay(path)
    value["source_artifact_hashes"] = []
    value["training_coverage_ready_score"] = 1
    atomic_json(raw / "coverage.json", {"files": {}})
    core.seal(value, path)
    with pytest.raises(ValueError, match="readiness drift"):
        core.replay(path)


def test_qualification_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7954: the new parameterized wrapper qualifies all its branches."""
    from scripts.experiments import experiment_7954_v690_training_coverage as cli

    path = tmp_path / "experiment_7954_fixture.json"
    called = []
    monkeypatch.setattr(core, "qualify", lambda *a: called.append(a) or 0)
    monkeypatch.setattr(core, "replay", lambda *a: called.append(a))
    args = ["--date", "20260930"]
    assert cli.main(args + ["--output", str(path)]) == 0
    assert cli.main(args + ["--cold-replay", str(path)]) == 0
    assert len(called) == 2

    def fail(*args: object) -> None:
        raise ValueError("owned failure")

    monkeypatch.setattr(core, "replay", fail)
    assert cli.main(args + ["--cold-replay", str(path)]) == 2


def test_cold_readiness_and_consumer_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7954-TERMINAL: cold reduction rejects denominator and reader drift."""
    path = tmp_path / "experiment_7954_fixture.json"
    raw = tmp_path / "raw" / path.stem
    counts = {name: {"num_statements": 1, "missing_lines": 0} for name in core.COVERED}
    atomic_json(raw / "coverage.json", {"files": {n: {"summary": v} for n, v in counts.items()}})
    source = tmp_path / "source.json"
    atomic_json(source, {})
    selected = [{"passed": True, "gate_path": str(source), "gate_sha256": sha256_file(source)}] * 3
    value = core.base([], [], {})
    value.update(
        training_coverage_ready_score=1,
        validation_receipts=[
            {"passed": True, "log_path": str(source), "log_sha256": sha256_file(source)}
        ],
        coverage_statement_counts=counts,
        consumer_selection_rows=selected,
    )
    value["sample_size_budget"].update(completed=0, independent=0)
    monkeypatch.setattr(
        core, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    monkeypatch.setattr(fit, "replay", lambda *a: None)
    core.seal(value, path)
    core.replay(path)
    value["sample_size_budget"]["completed"] = 1
    core.seal(value, path)
    with pytest.raises(ValueError, match="readiness drift"):
        core.replay(path)
    value["sample_size_budget"]["completed"] = 0
    value["validation_receipts"] = [
        {
            "passed": True,
            "path": str(raw / "coverage.json"),
            "sha256": sha256_file(raw / "coverage.json"),
        }
    ]
    core.seal(value, path)
    source.write_text("drift")
    with pytest.raises(ValueError, match="route primary hash drift"):
        core.replay(path)
