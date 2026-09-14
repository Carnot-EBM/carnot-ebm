"""Tests for validation that separates required scope from repository health.

Spec refs: REQ-REPORT-7303 and SCENARIO-REPORT-7303-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7303_v642_validation_scope as experiment
from carnot.reporting import experiment_7303_validation_scope as scope


ROOT = Path(__file__).resolve().parents[2]


def _fake_toolchain(root: Path) -> tuple[Path, dict[str, str]]:
    """Create executable tools that expose real argv and controlled exit codes."""

    bin_dir = root / ".venv/bin"
    bin_dir.mkdir(parents=True)
    ledger = root / "tool-argv.jsonl"
    program = bin_dir / "tool"
    program.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

row = {"program": Path(sys.argv[0]).name, "argv": sys.argv[1:]}
with Path(os.environ["EXP7303_ARGV_LEDGER"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\\n")
if "--exp7303-resolve-imports" in sys.argv:
    marker = sys.argv.index("--exp7303-resolve-imports")
    repo = Path(sys.argv[marker + 1])
    resolved = {}
    for name in sys.argv[marker + 2:]:
        resolved[name] = str(repo / "python" / Path(*name.split("."))).replace(".py", "") + ".py"
    print(json.dumps({"resolved_imports": resolved}, sort_keys=True), flush=True)
needle = os.environ.get("EXP7303_FAIL_MATCH", "")
raise_code = int(os.environ.get("EXP7303_FAIL_CODE", "7"))
if needle and any(needle in item for item in sys.argv[1:]):
    print(f"controlled failure: {needle}", flush=True)
    raise SystemExit(raise_code)
print("controlled pass", flush=True)
""",
        encoding="utf-8",
    )
    program.chmod(0o755)
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        (bin_dir / name).symlink_to(program)
    environment = {
        "EXP7303_ARGV_LEDGER": str(ledger),
        "EXP7303_FAKE_ROOT": str(root),
    }
    return ledger, environment


def _private_sources(root: Path) -> tuple[list[str], list[str]]:
    """Create explicit test and implementation files for one isolated run."""

    tests = ["tests/python/test_affected.py"]
    modules = [
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/experiment_7303_v642_validation_scope.py",
    ]
    for relative in (
        *tests,
        *modules,
        "scripts/experiments/experiment_7303_v642_validation_scope.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# private fixture\n", encoding="utf-8")
    return tests, modules


def _passing_receipts() -> list[dict[str, object]]:
    """Build a complete named set for the reducer-only control."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": [name],
            "scope": "explicit_fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in scope.REQUIRED_CHECK_NAMES
    ]


def _write_historical_evidence(root: Path) -> None:
    """Write minimal hash-consistent V641 artifacts and collection logs."""

    error_text = (
        "_ ERROR collecting tests/python/test_old_failure.py _\n"
        "tests/python/test_old_failure.py:1: in <module>\n"
        "E   KeyError: 'historical-registry-entry'\n"
        "!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!\n"
    )
    for item in experiment.HISTORICAL_EVIDENCE:
        log_path = root / item.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(error_text, encoding="utf-8")
        artifact_path = root / item.artifact_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "schema": "historical.fixture.v1",
                    "experiment_id": item.experiment_id,
                    "milestone": "2026.09.641",
                    "status": "complete",
                    "verdict_class": "disqualified",
                    "honest_verdict": "complete_disqualified_historical_validation_failed",
                    "validation_receipts": [
                        {
                            "name": "focused_pytest",
                            "command": "pytest explicit_test.py",
                            "exit_code": 0,
                            "passed": True,
                            "log_path": "historical-focused.log",
                            "log_sha256": "sha256:" + "b" * 64,
                        },
                        {
                            "name": "full_python_suite",
                            "command": "pytest tests/python -q",
                            "exit_code": 2,
                            "passed": False,
                            "log_path": item.log_path.as_posix(),
                            "log_sha256": experiment.sha256_file(log_path),
                        },
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def test_scenario_report_7303_affected_failure_propagates_actual_exit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7303-AFFECTED keeps the real failed process receipt."""

    root = tmp_path / "affected"
    tests, modules = _private_sources(root)
    ledger, environment = _fake_toolchain(root)
    environment["EXP7303_FAIL_MATCH"] = tests[0]
    result = scope.run_scoped_validation(
        root,
        tests,
        modules,
        static_paths=["scripts/experiments/experiment_7303_v642_validation_scope.py"],
        basetemp=root / "pytest-private",
        coverage_file=root / ".coverage",
        extra_env=environment,
    )

    focused = next(row for row in result["validation_receipts"] if row["name"] == "focused_pytest")
    assert focused["exit_code"] == 7
    assert focused["passed"] is False
    assert tests[0] in focused["command_argv"]
    assert result["required_checks_passed"] is False
    assert "focused_pytest" in result["failed_required_commands"]
    assert ledger.read_text(encoding="utf-8").count("\n") == len(scope.REQUIRED_CHECK_NAMES)


def test_scenario_report_7303_unscoped_pytest_target_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7303-UNSCOPED rejects a directory before execution."""

    root = tmp_path / "unscoped"
    _tests, modules = _private_sources(root)
    ledger, environment = _fake_toolchain(root)
    with pytest.raises(ValueError, match="repository-wide pytest target"):
        scope.run_scoped_validation(
            root,
            ["tests/python"],
            modules,
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
            extra_env=environment,
        )
    with pytest.raises(ValueError, match="explicit test paths"):
        scope.build_scoped_commands(
            root,
            [],
            modules,
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
        )
    with pytest.raises(ValueError, match="explicit changed-module paths"):
        scope.build_scoped_commands(
            root,
            ["tests/python/test_affected.py"],
            [],
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
        )
    with pytest.raises(ValueError, match="leaves repository"):
        scope.build_scoped_commands(
            root,
            ["../test_escape.py"],
            modules,
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
        )
    with pytest.raises(ValueError, match="in-repository Python file"):
        scope.build_scoped_commands(
            root,
            ["tests/python/test_affected.py"],
            modules,
            static_paths=["README.md"],
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
        )
    with pytest.raises(ValueError, match="below python/carnot"):
        scope.build_scoped_commands(
            root,
            ["tests/python/test_affected.py"],
            ["scripts/not_a_module.py"],
            basetemp=root / "pytest-private",
            coverage_file=root / ".coverage",
        )
    outside = tmp_path / "outside.log"
    outside.write_text("outside\n", encoding="utf-8")
    assert scope._relative_or_absolute(outside, root) == str(outside)

    blocked_output = root / "results/blocked.json"
    blocked = experiment.run_experiment(
        root,
        experiment.RUN_DATE,
        output_path=blocked_output,
        raw_dir=root / "raw-blocked",
        checkpoint_path=root / "checkpoint-blocked.json",
    )
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["observed_value"] is False
    assert experiment.validate_artifact(blocked) == []
    with monkeypatch.context() as patch:
        patch.setattr(experiment, "validate_artifact", lambda _artifact: ["forced"])
        with pytest.raises(ValueError, match="blocked_artifact_invalid:forced"):
            experiment.run_experiment(
                root,
                experiment.RUN_DATE,
                output_path=root / "results/blocked-invalid.json",
                raw_dir=root / "raw-blocked-invalid",
                checkpoint_path=root / "checkpoint-blocked-invalid.json",
            )
    with pytest.raises(ValueError, match="--date must be"):
        experiment.run_experiment(root, "20260913", output_path=blocked_output)
    assert ledger.exists() is False


def test_scenario_report_7303_missing_expected_command_cannot_pass() -> None:
    """SCENARIO-REPORT-7303-MISSING names an absent required command."""

    receipts = _passing_receipts()
    receipts = [row for row in receipts if row["name"] != "changed_module_mypy"]
    reduced = scope.reduce_required_checks(receipts)
    assert reduced["required_checks_passed"] is False
    assert reduced["missing_required_commands"] == ["changed_module_mypy"]
    assert reduced["failed_required_commands"] == []
    duplicate = scope.reduce_required_checks([*_passing_receipts(), _passing_receipts()[0]])
    assert duplicate["required_checks_passed"] is False
    assert duplicate["duplicate_required_commands"] == [scope.REQUIRED_CHECK_NAMES[0]]


def test_scenario_report_7303_unrelated_history_does_not_become_pass_receipt(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7303-HEALTH keeps collection debt outside current checks."""

    history = [
        {
            "experiment_id": "exp7293-reuse-measurement",
            "milestone": "2026.09.641",
            "predates_current_milestone": True,
            "command": "pytest tests/python -q",
            "exit_code": 2,
            "log_path": "results/raw/old/full_python_suite.log",
            "log_sha256": "sha256:" + "c" * 64,
            "collection_errors": [
                {
                    "test_path": "tests/python/test_old.py",
                    "exception": "KeyError: historical",
                }
            ],
            "resolved": False,
        }
    ]
    reduced = scope.validation_outcome(_passing_receipts(), history)
    assert reduced["required_checks_passed"] is True
    assert reduced["repository_health"]["status"] == "degraded_open"
    assert reduced["repository_health"]["historical_failures"] == history
    assert all(row["name"] != "full_python_suite" for row in reduced["validation_receipts"])
    assert experiment.check_dependency(None, "missing", "ready_score", 1)["observed_value"] == (
        "missing_artifact"
    )
    assert (
        experiment.check_dependency(
            {"status": "complete", "flagged_adversarial": True, "ready_score": 1},
            "quarantined",
            "ready_score",
            1,
        )["observed_value"]
        == "quarantined"
    )
    assert (
        experiment.check_dependency(
            {"status": "complete", "verdict_class": "disqualified", "ready_score": 1},
            "disqualified",
            "ready_score",
            1,
        )["observed_value"]
        == "disqualified"
    )
    assert (
        experiment.check_dependency(
            {"status": "partial", "ready_score": 1}, "partial", "ready_score", 1
        )["field"]
        == "status"
    )
    assert (
        experiment.check_dependency(
            {"status": "complete", "ready_score": 0}, "false", "ready_score", 1
        )["passed"]
        is False
    )
    assert (
        experiment.check_dependency(
            {"status": "complete", "ready_score": 1}, "clean", "ready_score", 1
        )["passed"]
        is True
    )

    malformed_root = tmp_path / "malformed-history"
    _write_historical_evidence(malformed_root)
    first = experiment.HISTORICAL_EVIDENCE[0]
    (malformed_root / first.artifact_path).write_text("not json\n", encoding="utf-8")
    checks, _hashes, observed = experiment.authenticate_inputs(
        malformed_root, output_path=malformed_root / "results/out.json"
    )
    assert any(row["check"] == "historical_artifact_parse" for row in checks)
    assert len(observed) == len(experiment.HISTORICAL_EVIDENCE) - 1


def test_scenario_report_7303_clean_scoped_run_and_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7303-CLEAN and ARTIFACT execute and publish exact evidence."""

    root = tmp_path / "clean"
    tests, modules = _private_sources(root)
    ledger, environment = _fake_toolchain(root)
    _write_historical_evidence(root)
    for relative in experiment.REQUIRED_INPUT_PATHS:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture input: {relative}\n", encoding="utf-8")
    output = root / "results/experiment_7303_v642_validation_scope.json"
    raw_dir = root / "results/raw/experiment_7303"
    checkpoint = root / "results/checkpoints/experiment_7303.json"

    artifact = experiment.run_experiment(
        root,
        experiment.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        test_paths=tests,
        changed_modules=modules,
        static_paths=["scripts/experiments/experiment_7303_v642_validation_scope.py"],
        extra_env=environment,
    )

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["required_checks_passed"] is True
    assert artifact["validation_scope_ready_score"] == 1
    assert artifact["repository_health"]["status"] == "degraded_open"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["resolved_imports"][
        "carnot.reporting.experiment_7303_validation_scope"
    ].startswith(str(root / "python"))
    assert experiment.validate_artifact(artifact) == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    argv_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(argv_rows) == len(scope.REQUIRED_CHECK_NAMES) + 2
    assert all("tests/python" not in row["argv"] for row in argv_rows)
    assert checkpoint.exists() and output.exists()

    corrupted = deepcopy(artifact)
    corrupted["schema"] = "wrong"
    assert "identity" in experiment.validate_artifact(corrupted)

    with monkeypatch.context() as patch:
        patch.setattr(experiment, "validate_artifact", lambda _artifact: ["forced"])
        with pytest.raises(ValueError, match="terminal_artifact_invalid:forced"):
            experiment.run_experiment(
                root,
                experiment.RUN_DATE,
                output_path=root / "results/invalid-final.json",
                raw_dir=root / "results/raw/invalid-final",
                checkpoint_path=root / "results/checkpoints/invalid-final.json",
                test_paths=tests,
                changed_modules=modules,
                static_paths=["scripts/experiments/experiment_7303_v642_validation_scope.py"],
                extra_env=environment,
            )

    with monkeypatch.context() as patch:
        patch.setattr(
            experiment,
            "run_experiment",
            lambda _root, _date: {"status": "complete"},
        )
        assert experiment.main(["--date", experiment.RUN_DATE]) == 0
