"""
Tests for Experiment 2104: Milestone 2026.05.165 Retrospective.

REQ-REPORT-009
"""
import json
from pathlib import Path

from carnot.reporting.experiment_2104_retro import (
    get_retro_results,
    write_results,
)


def test_experiment_2104_results():
    """Test the structure and values of the 2104 retrospective results."""
    res = get_retro_results()
    assert res["schema"] == "carnot.milestone_research_retro.v1"
    assert res["milestone"] == "2026.05.165"
    assert len(res["tasks_summary"]) == 4
    
    # Check that 2100-2102 are blocked, 2103 passed
    passed_count = sum(1 for t in res["tasks_summary"] if t["gate_passed"])
    assert passed_count == 1
    
    assert res["gates_passed_count"] == 1
    assert res["gates_failed_count"] == 3
    assert res["adversarial_verify_flag_count"] == 0
    assert res["honest_verdict"].startswith("complete:")


def test_experiment_2104_write(tmp_path: Path):
    """Test writing the results JSON."""
    output_file = tmp_path / "experiment_2104_retro.json"
    write_results(str(output_file))
    assert output_file.exists()

    with open(output_file, "r") as f:
        data = json.load(f)
    assert data["schema"] == "carnot.milestone_research_retro.v1"
