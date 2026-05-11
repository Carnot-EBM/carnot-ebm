"""Tests for pipeline fail-fast checks.

Spec traces: REQ-PIPELINE-1826, SCENARIO-PIPELINE-1826
"""

import json
from pathlib import Path
from carnot.pipeline.fail_fast import pipeline_fail_fast_check


def test_pipeline_fail_fast_check_doomed(tmp_path: Path) -> None:
    """Test that a doomed task generates the correct artifact and returns True."""
    output_path = tmp_path / "experiment_1826_fail_fast.json"
    task = {
        "id": "exp_test_1826",
        "doomed_rerun": True,
        "doomed_reason": "test reason"
    }

    result = pipeline_fail_fast_check(task, output_path)

    assert result is True
    assert output_path.exists()
    
    with open(output_path, "r") as f:
        artifact = json.load(f)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_doomed_rerun"
    assert artifact["reason"] == "test reason"
    assert artifact["task_id"] == "exp_test_1826"


def test_pipeline_fail_fast_check_missing_priors(tmp_path: Path) -> None:
    """Test that a rerun task missing prior failures generates the correct artifact."""
    output_path = tmp_path / "experiment_1826_fail_fast.json"
    task = {
        "id": "exp_test_1826_2",
        "is_rerun": True
    }

    result = pipeline_fail_fast_check(task, output_path)

    assert result is True
    assert output_path.exists()
    
    with open(output_path, "r") as f:
        artifact = json.load(f)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_doomed_rerun"
    assert artifact["reason"] == "Missing prior_failures for rerun task"
    assert artifact["task_id"] == "exp_test_1826_2"


def test_pipeline_fail_fast_check_valid(tmp_path: Path) -> None:
    """Test that a valid task does not generate an artifact and returns False."""
    output_path = tmp_path / "experiment_1826_fail_fast.json"
    task = {
        "id": "exp_test_1826_3",
        "is_rerun": True,
        "prior_failures": {"some": "data"}
    }

    result = pipeline_fail_fast_check(task, output_path)

    assert result is False
    assert not output_path.exists()
