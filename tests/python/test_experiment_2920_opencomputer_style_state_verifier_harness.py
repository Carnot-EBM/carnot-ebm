"""Tests for Exp 2920 OpenComputer-style local state verifier harness.

Spec: REQ-VERIFY-2920, SCENARIO-VERIFY-2920.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import opencomputer_state_verifier_harness as exp


def test_req_verify_2920_manifest_schema_has_state_tasks(tmp_path: Path) -> None:
    """REQ-VERIFY-2920: manifest defines 3-5 observable local state tasks."""

    tasks = exp.build_state_tasks()
    manifest_path = tmp_path / exp.MANIFEST_FILENAME
    manifest = exp.write_task_manifest(tasks, manifest_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["schema"] == "carnot.opencomputer_state_verifier_manifest.v1"
    assert manifest["run_date"] == "20260523"
    assert manifest["n_state_tasks"] == 4
    assert len(tasks) == 4
    assert {task.task_type for task in tasks} == {
        "filesystem_state",
        "json_file_transform",
        "jsonl_inventory",
        "sqlite_row_edit",
    }

    for row in manifest["tasks"]:
        assert row["verifier"]["entrypoint"] == (
            "carnot.eval.opencomputer_state_verifier_harness.verify_task_state"
        )
        assert row["observable_state"]["root_relative"] is True
        assert len(row["checks"]) >= 2
        assert all({"check_id", "points", "localized_to"} <= set(check) for check in row["checks"])


def test_scenario_verify_2920_golden_and_negative_states_return_partial_credit(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2920: golden states pass and negative states fail with locality."""

    tasks = exp.build_state_tasks()
    golden_results = []
    negative_results = []

    for task in tasks:
        golden_root = exp.materialize_task_state(task, "golden", tmp_path / "golden")
        negative_root = exp.materialize_task_state(task, "negative", tmp_path / "negative")
        golden = exp.verify_task_state(task, golden_root)
        negative = exp.verify_task_state(task, negative_root)
        rematerialized = exp.materialize_task_state(task, "golden", tmp_path / "golden")

        assert rematerialized == golden_root
        assert golden == exp.verify_task_state(task, golden_root)
        assert set(exp.PARTIAL_CREDIT_FIELDS) <= set(golden)
        assert set(exp.PARTIAL_CREDIT_FIELDS) <= set(negative)
        assert golden["passed"] is True
        assert golden["score"] == pytest.approx(1.0)
        assert golden["earned_points"] == golden["max_points"]
        assert golden["violations"] == []
        assert negative["passed"] is False
        assert 0.0 < negative["score"] < 1.0
        assert negative["earned_points"] < negative["max_points"]
        assert negative["violations"]
        assert all(item["localized_to"] for item in negative["violations"])

        golden_results.append(golden)
        negative_results.append(negative)

    assert exp.pass_rate(golden_results) == pytest.approx(1.0)
    assert exp.reject_rate(negative_results) == pytest.approx(1.0)


def test_req_verify_2920_failure_localization_names_observable_state(tmp_path: Path) -> None:
    """REQ-VERIFY-2920: negative states point to the failing file, row, or field."""

    by_type = {task.task_type: task for task in exp.build_state_tasks()}
    expected_locations = {
        "json_file_transform": "config/app.json:features.search.enabled",
        "sqlite_row_edit": "data/tasks.sqlite:tasks[ship-harness].status",
        "filesystem_state": "workspace/tmp/cache.bin",
        "jsonl_inventory": "inventory/items.jsonl:sku=widget-b",
    }

    for task_type, localized_to in expected_locations.items():
        state_root = exp.materialize_task_state(
            by_type[task_type],
            "negative",
            tmp_path / task_type,
        )
        result = exp.verify_task_state(by_type[task_type], state_root)
        assert any(item["localized_to"] == localized_to for item in result["violations"])


def test_req_verify_2920_artifact_writes_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-2920: deliverable JSON exposes required deterministic-harness fields."""

    output_path = tmp_path / exp.OUTPUT_FILENAME
    manifest_path = tmp_path / exp.MANIFEST_FILENAME
    artifact = exp.write_experiment_artifact(
        exp.ExperimentConfig(
            output_path=output_path,
            manifest_path=manifest_path,
            started_at=10.0,
            clock=lambda: 12.25,
            tests_run=("focused pytest",),
        )
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["n_state_tasks"] == 4
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["state_verifier_harness_ready"] is True
    assert Path(artifact["task_manifest_path"]).name == exp.MANIFEST_FILENAME
    assert artifact["n_state_tasks"] == 4
    assert artifact["verifier_source_paths"] == [
        "python/carnot/eval/opencomputer_state_verifier_harness.py"
    ]
    assert artifact["golden_state_pass_rate"] == pytest.approx(1.0)
    assert artifact["negative_state_reject_rate"] == pytest.approx(1.0)
    assert artifact["partial_credit_fields"] == list(exp.PARTIAL_CREDIT_FIELDS)
    assert artifact["failure_localization_examples"]
    assert artifact["llm_judge_used"] is False
    assert artifact["inference_substrate"] == "deterministic_state_verifier"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["run_date"] == "20260523"
    assert artifact["tests_run"] == ["focused pytest"]


def test_req_verify_2920_verifier_fails_closed_for_missing_state(tmp_path: Path) -> None:
    """REQ-VERIFY-2920: absent observable state is rejected with zero partial credit."""

    task = exp.build_state_tasks()[0]
    missing = exp.verify_task_state(task, tmp_path / "missing")

    assert missing["passed"] is False
    assert missing["score"] == 0.0
    assert missing["earned_points"] == 0.0
    assert missing["max_points"] == sum(check.points for check in task.checks)
    assert missing["violations"] == [
        {
            "check_id": "state_root_exists",
            "localized_to": str(tmp_path / "missing"),
            "detail": "state root does not exist",
        }
    ]
    with pytest.raises(ValueError, match="unknown state variant"):
        exp.materialize_task_state(task, "initial", tmp_path)
