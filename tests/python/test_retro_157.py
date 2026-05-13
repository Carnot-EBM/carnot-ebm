"""
Tests for the milestone .157 retrospective generator.

REQ-REPORT-157: The retro generator MUST scan experiment_2008-2016 artifacts,
classify them as completed/blocked/failed, and emit a carnot.milestone_retro.v1
artifact with correct counts, verdicts, and at least one recommendation.

SCENARIO-REPORT-157-A: Milestone .157 Retrospective Handles Blocked and Missing Artifacts
"""

import json
import os
from pathlib import Path

from carnot.retro_157 import generate_retro, _classify_artifact


# --- unit tests for _classify_artifact ---

def test_classify_blocked_via_honest_verdict():
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"

def test_classify_blocked_via_status():
    assert _classify_artifact({"status": "blocked"}) == "blocked"

def test_classify_failed_via_verdict():
    assert _classify_artifact({"honest_verdict": "failed_due_to_error"}) == "failed"
    assert _classify_artifact({"honest_verdict": "Audit complete: Missing artifacts"}) == "failed"

def test_classify_failed_via_status():
    assert _classify_artifact({"status": "failure"}) == "failed"
    assert _classify_artifact({"status": "error"}) == "failed"

def test_classify_completed_on_complete_verdict():
    assert _classify_artifact({"honest_verdict": "complete: done"}) == "completed"

def test_classify_completed_on_success_status():
    assert _classify_artifact({"status": "success"}) == "completed"

def test_classify_completed_on_complete_status():
    assert _classify_artifact({"status": "complete"}) == "completed"

def test_classify_empty_artifact():
    assert _classify_artifact({}) == "completed"

# --- helpers for integration tests ---

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# --- integration tests for generate_retro ---

def test_generate_retro_schema_fields(tmp_path: Path) -> None:
    _write(tmp_path / "experiment_2009_ebm.json", {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
    })

    out = str(tmp_path / "experiment_2017_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.157"
    assert result["experiment_id"] == 2017
    assert result["status"] == "complete"
    assert result["retro_complete"] is True


def test_generate_retro_handles_missing_and_blocked(tmp_path: Path) -> None:
    # 2009, 2010, 2011, 2015 are blocked
    for exp_id in [2009, 2010, 2011, 2015]:
        _write(tmp_path / f"experiment_{exp_id}.json", {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        })
    # 2016 is failed pre_retro
    _write(tmp_path / "experiment_2016.json", {
        "status": "failure",
        "honest_verdict": "Audit complete: Missing artifacts",
    })
    # Let's say 2008 actually completed
    _write(tmp_path / "experiment_2008.json", {
        "status": "success",
        "honest_verdict": "complete: task finished",
    })
    # 2012, 2013, 2014 are missing

    out = str(tmp_path / "experiment_2017_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["completed_task_count"] == 1
    assert result["blocked_task_count"] == 4
    assert result["failed_task_count"] == 4  # 3 missing + 1 failed
    assert 2012 in result["failed_experiments"]
    assert 2016 in result["failed_experiments"]
    assert 2009 in result["blocked_experiments"]
    assert 2008 in result["completed_experiments"]


def test_generate_retro_bottlenecks_and_recommendations(tmp_path: Path) -> None:
    out = str(tmp_path / "experiment_2017_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0
    assert "bottlenecks_identified" in result
    assert isinstance(result["bottlenecks_identified"], list)
    assert len(result["bottlenecks_identified"]) > 0

def test_generate_retro_unreadable_artifact(tmp_path: Path) -> None:
    broken = tmp_path / "experiment_2012.json"
    broken.write_text("{ not valid json >>>")

    out = str(tmp_path / "experiment_2017_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 2012 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp2012"] == "UNREADABLE"

def test_generate_retro_ignores_out_of_range(tmp_path: Path) -> None:
    _write(tmp_path / "experiment_2007_retro.json", {"status": "complete"})
    _write(tmp_path / "experiment_2017_retro.json", {"status": "complete"})

    out = str(tmp_path / "experiment_2017_milestone_157_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    all_counted = (
        result["completed_experiments"]
        + result["blocked_experiments"]
        + result["failed_experiments"]
    )
    assert 2007 not in all_counted
    assert 2017 not in all_counted
