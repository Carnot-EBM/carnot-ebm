"""Tests for Experiment 1780 CSL Loop.

Spec: REQ-CSL-1780, SCENARIO-CSL-1780
"""

import json
from pathlib import Path
import pytest
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_1780_csl_loop as mod

def test_run_csl_loop():
    """CSL loop produces required output structure."""
    artifact = mod.run_csl_loop()
    assert isinstance(artifact, dict)
    assert artifact["schema"] == "carnot.csl.loop.v1"
    assert "utility_delta" in artifact
    assert isinstance(artifact["utility_delta"], float)
    assert "soundness_mistakes" in artifact
    assert isinstance(artifact["soundness_mistakes"], int)

def test_experiment_1780_csl_loop_artifact(tmp_path, monkeypatch):
    """End-to-end run writes deliverable JSON with required schema fields (SCENARIO-CSL-1780)."""
    original_deliverable = mod.DELIVERABLE
    temp_deliverable = tmp_path / "experiment_1780_csl_loop.json"
    monkeypatch.setattr(mod, "DELIVERABLE", str(temp_deliverable))

    try:
        mod.main()
    finally:
        mod.DELIVERABLE = original_deliverable

    assert temp_deliverable.exists()
    
    with open(temp_deliverable) as f:
        artifact = json.load(f)
        
    assert artifact["schema"] == "carnot.csl.loop.v1"
    assert "utility_delta" in artifact
    assert "soundness_mistakes" in artifact
    assert artifact["model"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
