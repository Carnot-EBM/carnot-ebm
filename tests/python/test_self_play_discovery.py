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

def test_run_experiment_1685(tmp_path: Path):
    """Test running the live SOTA experiment and saving artifact.
    
    Spec: REQ-SELFPLAY-1685.
    """
    output_path = tmp_path / "experiment_1685_live_sota.json"
    result = self_play_discovery.run_experiment_1685(output_path)
    
    assert result["status"] == "complete"
    assert result["experiment_id"] == 1685
    assert result["model_used"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert result["traces_generated"] == 10
    assert result["hallucination_identified"] is True
    assert result["repair_confirmed"] is True
    assert result["constraint_generated"] is not None
    assert len(result["results"]) == 1
    
    assert output_path.exists()
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["experiment_id"] == 1685
    assert saved_data["status"] == "complete"
    assert saved_data["traces_generated"] == 10

def test_run_experiment_1694(tmp_path: Path):
    """Test running the Phase 7 full pipeline stack test and saving artifact.
    
    Spec: REQ-PIPELINE-1694.
    """
    output_path = tmp_path / "experiment_1694_full_pipeline.json"
    result = self_play_discovery.run_experiment_1694(output_path)
    
    assert result["status"] == "complete"
    assert result["experiment_id"] == 1694
    assert result["model_used"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert result["questions_run"] == 5
    assert len(result["components_active"]) == 3
    assert result["traces_generated"] == 5
    assert len(result["results"]) == 1
    
    assert output_path.exists()
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["experiment_id"] == 1694
    assert saved_data["status"] == "complete"
    assert saved_data["questions_run"] == 5
