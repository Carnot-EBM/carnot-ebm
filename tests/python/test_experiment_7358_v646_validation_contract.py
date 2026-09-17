"""Tests for the V646 experiment-only validation and classification boundary.

Spec refs: REQ-REPORT-7358 and SCENARIO-REPORT-7358-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_7358_v646_validation_contract as experiment
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _manifest(root: Path) -> experiment.AffectedManifest:
    """Create one explicit manifest with files that the plan can resolve."""

    paths = (
        "tests/python/test_affected.py",
        "python/carnot/experiment_affected.py",
        "scripts/experiments/experiment_affected.py",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return experiment.AffectedManifest(
        experiment_id="fixture",
        test_paths=(paths[0],),
        changed_modules=(paths[1],),
        static_paths=(paths[2],),
    )


def _passing_receipts(root: Path) -> list[dict[str, object]]:
    """Create a complete affected receipt set with a worktree import."""

    rows: list[dict[str, object]] = []
    for name in validation_scope.REQUIRED_CHECK_NAMES:
        row: dict[str, object] = {
            "name": name,
            "command": name,
            "command_argv": [name],
            "scope": "fixture",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        if name == "worktree_imports":
            row["resolved_imports"] = {
                "carnot.experiment_affected": str(root / "python/carnot/experiment_affected.py")
            }
        rows.append(row)
    return rows


def _command_rows(root: Path) -> list[dict[str, object]]:
    """Enrich passing receipts as required affected command rows."""

    return [
        experiment.command_row(
            row,
            category="required_validation",
            required=True,
            started_at_utc="2026-09-17T00:00:00+00:00",
            ended_at_utc="2026-09-17T00:00:01+00:00",
        )
        for row in _passing_receipts(root)
    ]


def test_scenario_report_7358_plan_rejects_broad_and_unrelated_targets(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7358-PLAN rejects expansion before child execution."""

    root = tmp_path / "repo"
    manifest = _manifest(root)
    private = tmp_path / "private"
    commands = experiment.build_command_plan(root, manifest, private)

    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert (private / "basetemp").is_dir()
    assert (private / "coverage").is_dir()
    assert experiment.validate_command_plan(root, manifest, commands) == []
    pytest_commands = [row for row in commands if "pytest" in " ".join(row.argv)]
    assert len(pytest_commands) == 2
    assert all(manifest.test_paths[0] in row.argv for row in pytest_commands)

    broad = validation_scope.CommandSpec(
        "full_python_suite",
        (str(root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository",
    )
    errors = experiment.validate_command_plan(root, manifest, [*commands, broad])
    assert "unscoped_test_target:tests/python" in errors
    assert "unexpected_command:full_python_suite" in errors

    unrelated = deepcopy(commands)
    focused_index = next(i for i, row in enumerate(unrelated) if row.name == "focused_pytest")
    original = unrelated[focused_index]
    unrelated[focused_index] = validation_scope.CommandSpec(
        original.name,
        tuple(
            "tests/python/test_unrelated.py" if item == manifest.test_paths[0] else item
            for item in original.argv
        ),
        original.scope,
    )
    assert (
        "test_outside_manifest:tests/python/test_unrelated.py"
        in experiment.validate_command_plan(root, manifest, unrelated)
    )
    assert "missing_command:worktree_imports" in experiment.validate_command_plan(
        root, manifest, commands[1:]
    )
    assert "duplicate_command:worktree_imports" in experiment.validate_command_plan(
        root, manifest, [*commands, commands[0]]
    )


def test_scenario_report_7358_scope_rejects_missing_parent_and_installed_import(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7358-SCOPE requires private parents and worktree imports."""

    root = tmp_path / "repo"
    manifest = _manifest(root)
    private = tmp_path / "private"
    commands = experiment.build_command_plan(root, manifest, private)
    for path in (private / "basetemp", private / "coverage"):
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    errors = experiment.validate_command_plan(root, manifest, commands)
    assert any(value.startswith("missing_basetemp_parent:") for value in errors)
    assert any(value.startswith("missing_coverage_parent:") for value in errors)

    receipts = _passing_receipts(root)
    assert experiment.reduce_affected_receipts(root, manifest, receipts)["passed"] is True
    receipts[0]["resolved_imports"] = {
        "carnot.experiment_affected": "/opt/site-packages/carnot/experiment_affected.py"
    }
    reduced = experiment.reduce_affected_receipts(root, manifest, receipts)
    assert reduced["passed"] is False
    assert reduced["installed_imports"] == ["/opt/site-packages/carnot/experiment_affected.py"]


def test_scenario_report_7358_receipts_keep_exit_hash_category_and_heartbeat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7358-RECEIPTS proves the reused runner handles silence."""

    commands = [
        experiment.PlannedCommand(
            validation_scope.CommandSpec(
                "passing_child",
                (sys.executable, "-u", "-c", "print('pass', flush=True)"),
                "temporary_control",
            ),
            "completion",
            False,
        ),
        experiment.PlannedCommand(
            validation_scope.CommandSpec(
                "failing_child",
                (sys.executable, "-u", "-c", "raise SystemExit(7)"),
                "temporary_control",
            ),
            "required_validation",
            False,
        ),
        experiment.PlannedCommand(
            validation_scope.CommandSpec(
                "silent_child",
                (
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; time.sleep(0.08)",
                ),
                "temporary_control",
            ),
            "safety",
            False,
        ),
    ]
    rows = experiment.run_categorized_commands(
        ROOT,
        commands,
        log_dir=tmp_path / "logs",
        heartbeat_s=0.02,
    )
    output = capsys.readouterr().out

    assert [row["exit_code"] for row in rows] == [0, 7, 0]
    assert [row["command_category"] for row in rows] == [
        "completion",
        "required_validation",
        "safety",
    ]
    assert all(row["required"] is False for row in rows)
    assert all(str(row["log_sha256"]).startswith("sha256:") for row in rows)
    assert all(row["started_at_utc"] <= row["ended_at_utc"] for row in rows)
    assert "subprocess_outstanding silent_child" in output
    experiment.progress(0.0, "test", "boundary", units=3)
    assert "phase=test event=boundary" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "expected_class", "ready", "capture", "value", "promotion"),
    [
        ({"efficacy_passed": False}, "null", 1, 1, 0, 0),
        ({"required_validation_passed": False}, "disqualified", 0, 0, 0, 0),
        ({"safety_passed": False}, "disqualified", 0, 0, 0, 0),
        ({"result_present": False}, "disqualified", 0, 0, 0, 0),
        ({"prerequisites_passed": False}, "blocked", 0, 0, 0, 0),
        ({"flagged_adversarial": True}, "disqualified", 0, 0, 0, 0),
        (
            {"result_present": False, "retryable_own_work_unfinished": True},
            "partial",
            0,
            0,
            0,
            0,
        ),
        ({"efficacy_passed": True}, "circular_positive", 1, 1, 1, 1),
    ],
)
def test_scenario_report_7358_classification_closed_semantics(
    overrides: dict[str, bool],
    expected_class: str,
    ready: int,
    capture: int,
    value: int,
    promotion: int,
) -> None:
    """SCENARIO-REPORT-7358-CLASSIFICATION keeps all gate families separate."""

    inputs = {
        "prerequisites_passed": True,
        "required_validation_passed": True,
        "safety_passed": True,
        "result_present": True,
        "result_complete": True,
        "efficacy_passed": False,
        "flagged_adversarial": False,
        "retryable_own_work_unfinished": False,
        **overrides,
    }
    result = experiment.classify_terminal(**inputs)
    assert result["verdict_class"] == expected_class
    assert result["fixture_ready_score"] == ready
    assert result["capture_complete_score"] == capture
    assert result["scientific_value_score"] == value
    assert result["promotion_score"] == promotion


def test_scenario_report_7358_historical_commands_retain_original_evidence() -> None:
    """SCENARIO-REPORT-7358-PLAN traces old commands without changing verdicts."""

    checks, hashes, artifacts = experiment.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks)
    assert set(artifacts) == {"exp7346", "exp7354"}
    assert all(value.startswith("sha256:") for value in hashes.values())
    rows = experiment.historical_command_rows(artifacts)
    broad = [row for row in rows if row["name"] == "full_python_suite"]
    assert len(broad) == 2
    assert all(row["command_argv"][1:] == ["tests/python", "-q"] for row in broad)
    assert all(row["exit_code"] == -15 and row["timed_out"] is True for row in broad)
    assert {row["caller"] for row in broad} == {
        "carnot.experiment_7346_v645_learning_adapter.main",
        "carnot.experiment_7354_v645_arc_transfer.e2e_command_specs",
    }
    assert artifacts["exp7346"]["verdict_class"] == "disqualified"
    assert artifacts["exp7354"]["flagged_adversarial"] is True
    assert experiment.historical_command_rows({"bad": {"validation_receipts": [None]}}) == []
    sidecars = experiment._historical_sidecars(artifacts, hashes)
    assert len(sidecars) == 2
    assert all(row["authorizes_current_inference"] is False for row in sidecars)


def test_scenario_report_7358_terminal_artifact_reduces_from_raw_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7358-TERMINAL recomputes readiness and null science."""

    root = tmp_path / "repo"
    manifest = _manifest(root)
    command_rows = _command_rows(root)
    classifications = experiment.run_classification_controls()
    artifact = experiment.build_artifact(
        preconditions=[experiment.precondition_row("fixture", "fixture", "ready", True, True)],
        source_hashes={"fixture": "sha256:" + "b" * 64},
        historical_artifacts={},
        historical_commands=[],
        command_plan_rows=command_rows,
        classification_rows=classifications,
        manifest=manifest,
        duration_s=1.0,
        phase_spans=[{"phase": "validation", "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0}],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        heartbeat_control_passed=True,
    )

    assert artifact["status"] == "complete_validation_contract_null_science"
    assert artifact["verdict_class"] == "null"
    assert artifact["validation_contract_ready_score"] == 1
    assert artifact["scientific_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert experiment.independent_reduce(artifact)["validation_contract_ready_score"] == 1
    assert experiment.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["command_plan_rows"][0]["exit_code"] = 9
    errors = experiment.validate_artifact(changed)
    assert "stored_reduction_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors

    changed = deepcopy(artifact)
    changed["validation_contract_ready_score"] = 0
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "stored_reduction_mismatch" in experiment.validate_artifact(changed)


def test_scenario_report_7358_blocked_artifact_names_exact_missing_input(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7358 blocks external absence without dependent work."""

    checks, hashes, artifacts = experiment.collect_preconditions(tmp_path)
    assert checks[0]["passed"] is False
    artifact = experiment.build_blocked_artifact(
        preconditions=checks,
        source_hashes=hashes,
        historical_artifacts=artifacts,
        duration_s=0.01,
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
    )
    first = artifact["gate_check_summary"]["first_failure"]
    assert artifact["verdict_class"] == "blocked"
    assert artifact["validation_contract_ready_score"] == 0
    assert first["upstream"] == "AGENTS.md"
    assert first["artifact_field"] == "bytes"
    assert first["expected"] == "readable_nonempty_bytes"
    assert first["observed"] is None
    assert experiment.validate_artifact(artifact) == []


def test_req_report_7358_argument_validation_and_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7358 keeps the thin CLI and writer deterministic."""

    output = tmp_path / "nested/result.json"
    experiment.atomic_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(SystemExit, match="--date must be 20260917"):
        experiment.run_experiment(ROOT, "20260916", output_path=output)
    with monkeypatch.context() as patch:
        patch.setattr(
            experiment, "run_experiment", lambda *_args, **_kwargs: {"status": "complete"}
        )
        assert experiment.main(["--date", experiment.RUN_DATE, "--output", str(output)]) == 0


def test_req_report_7358_validator_rejects_each_unsafe_field(tmp_path: Path) -> None:
    """REQ-REPORT-7358 rejects malformed identity, model, gates, and scores."""

    checks, hashes, artifacts = experiment.collect_preconditions(tmp_path)
    blocked = experiment.build_blocked_artifact(
        preconditions=checks,
        source_hashes=hashes,
        historical_artifacts=artifacts,
        duration_s=0.01,
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
    )
    broken = deepcopy(blocked)
    broken.update(
        {
            "schema": "wrong",
            "MODEL_SPECS": ["model"],
            "model_invoked": True,
            "invocation_counts": {"attempted": 1},
            "inference_substrate_class": "wrong",
            "field_principles": {},
            "command_plan_rows": [{"name": "unexpected"}],
            "validation_receipts": [{"name": "unexpected"}],
            "gate_check_summary": {"first_failure": None},
            "validation_contract_ready_score": 1,
            "fixture_ready_score": 1,
            "flagged_adversarial": True,
            "promotion_score": 1,
        }
    )
    errors = experiment.validate_artifact(broken)
    assert {
        "identity_mismatch",
        "current_model_declaration_mismatch",
        "current_invocation_counts_nonzero",
        "substrate_class_mismatch",
        "field_principles_incomplete",
        "blocked_artifact_has_dependent_work",
        "blocked_gate_summary_missing",
        "blocked_ready_score_nonzero",
        "failed_state_scores_nonzero",
        "adversarial_promotion_nonzero",
        "reproducibility_checksum_mismatch",
    }.issubset(errors)
    assert experiment.validate_artifact(1) == ["artifact_not_object"]
    assert (
        experiment._worktree_root_from_rows(experiment.AffectedManifest("empty", (), (), ()), [])
        == experiment.REPO_ROOT
    )

    invalid_class = deepcopy(blocked)
    invalid_class["verdict_class"] = "unknown"
    invalid_class["reproducibility_checksum"] = experiment.reproducibility_checksum(invalid_class)
    assert "verdict_class_invalid" in experiment.validate_artifact(invalid_class)
