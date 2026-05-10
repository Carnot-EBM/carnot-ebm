"""Tests for self_play_discovery.

Spec: REQ-SELFPLAY-1683.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline import self_play_discovery

def test_self_play_constraint_discoverer_init():
    """Test initialization of SelfPlayConstraintDiscoverer."""
    discoverer = self_play_discovery.SelfPlayConstraintDiscoverer()
    assert discoverer is not None

def test_ingest_failing_traces():
    """Test ingestion of failing traces.
    
    Spec: REQ-SELFPLAY-1683.
    """
    discoverer = self_play_discovery.SelfPlayConstraintDiscoverer()
    traces = [
        {"trace_id": "test_1", "output": "invalid output"},
        {"trace_id": "test_2", "output": "output with secret"},
        {"trace_id": "test_3", "output": "normal failing"}
    ]
    results = discoverer.ingest_failing_traces(traces)
    assert len(results) == 3
    
    assert results[0]["trace_id"] == "test_1"
    assert results[0]["root_conflict"] == "invalid_format"
    assert results[0]["dsl_input"]["constraints"][0]["id"] == "c_no_fail"
    
    assert results[1]["trace_id"] == "test_2"
    assert results[1]["root_conflict"] == "logical_failure_detected"
    assert results[1]["dsl_input"]["constraints"][0]["id"] == "c_no_secret"
    
    assert results[2]["trace_id"] == "test_3"
    assert results[2]["root_conflict"] == "logical_failure_detected"
    assert results[2]["dsl_input"]["constraints"][0]["id"] == "c_no_fail"

def test_run_experiment(tmp_path: Path):
    """Test running the experiment and saving artifact.
    
    Spec: REQ-SELFPLAY-1683.
    """
    output_path = tmp_path / "experiment_1683_self_play.json"
    result = self_play_discovery.run_experiment(output_path)
    
    assert result["status"] == "complete"
    assert result["experiment_id"] == 1683
    assert len(result["results"]) == 2
    
    assert output_path.exists()
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["experiment_id"] == 1683
    assert saved_data["status"] == "complete"
    assert saved_data["run_date"] == self_play_discovery.RUN_DATE
