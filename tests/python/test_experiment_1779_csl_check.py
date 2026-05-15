"""Tests for Experiment 1779 CSL Check.

Spec: REQ-CSL-1779, SCENARIO-CSL-1779
"""

import json
from pathlib import Path
import pytest
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_1779_csl_check as mod

def test_collect_synthetic_traces():
    """Traces are collected."""
    traces = mod.collect_synthetic_traces()
    assert isinstance(traces, list)
    assert len(traces) > 0

def test_evaluate_soundness():
    """Non-forgetting soundness check is evaluated (SCENARIO-CSL-1779)."""
    passed = mod.evaluate_soundness([{"trace_id": 1, "turns": 3}])
    assert passed is True
    
    failed = mod.evaluate_soundness([])
    assert failed is False

def test_experiment_1779_csl_check_artifact(tmp_path, monkeypatch):
    """End-to-end run writes deliverable JSON with required schema fields (SCENARIO-CSL-1779)."""
    original_deliverable = mod.DELIVERABLE
    temp_deliverable = tmp_path / "experiment_1779_csl_nonforgetting.json"
    monkeypatch.setattr(mod, "DELIVERABLE", str(temp_deliverable))

    try:
        mod.main()
    finally:
        mod.DELIVERABLE = original_deliverable

    assert temp_deliverable.exists()
    
    with open(temp_deliverable) as f:
        artifact = json.load(f)
        
    assert artifact["schema"] == "carnot.csl.check.v1"
    assert "check_implemented" in artifact
    assert artifact["check_implemented"] is True
