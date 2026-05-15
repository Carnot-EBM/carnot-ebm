"""Tests for Experiment 1778 CSL Baseline.

Spec: REQ-CSL-1778, SCENARIO-CSL-1778
"""

import json
from pathlib import Path
import pytest
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_1778_csl_baseline as mod

def test_collect_traces():
    """Traces are collected."""
    traces = mod.collect_traces()
    assert isinstance(traces, list)

def test_evaluate_baseline():
    """Baseline soundness mistakes are evaluated (SCENARIO-CSL-1778)."""
    mistakes = mod.evaluate_baseline([{"trace_id": 1, "turns": 5}])
    assert isinstance(mistakes, int)
    assert mistakes >= 0

def test_experiment_1778_csl_baseline_artifact(tmp_path, monkeypatch):
    """End-to-end run writes deliverable JSON with required schema fields (SCENARIO-CSL-1778)."""
    original_deliverable = mod.DELIVERABLE
    temp_deliverable = tmp_path / "experiment_1778_csl_baseline.json"
    monkeypatch.setattr(mod, "DELIVERABLE", str(temp_deliverable))

    try:
        mod.main()
    finally:
        mod.DELIVERABLE = original_deliverable

    assert temp_deliverable.exists()
    
    with open(temp_deliverable) as f:
        artifact = json.load(f)
        
    assert artifact["schema"] == "carnot.csl.baseline.v1"
    assert "baseline_soundness_mistakes" in artifact
    assert isinstance(artifact["baseline_soundness_mistakes"], int)
