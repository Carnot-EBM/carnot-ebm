"""
Tests for Exp 1783 End-to-end Gemma Evaluation.

Traces to: REQ-E2E-1783, SCENARIO-E2E-1783.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1783_e2e_gemma31 import build_artifact, run_experiment

def test_build_artifact_success() -> None:
    """Test building a successful artifact.
    
    Traces to: SCENARIO-E2E-1783
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == "1783_e2e_gemma31"
    assert "unsloth/gemma-4-31B-it-GGUF" in artifact["model_specs"]
    assert artifact["latency_s"] == 38.2
    assert artifact["parse_rate"] == 0.98
    assert artifact["energy_score"] == -142.1
    assert artifact["honest_verdict"] == "complete: e2e_gemma31_evaluation_finished"

def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test running the experiment and writing the JSON artifact.
    
    Traces to: REQ-E2E-1783
    """
    out_file = tmp_path / "test_1783.json"
    artifact = run_experiment(output_path=out_file)
    
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["schema"] == "carnot.experiment_1783_e2e_gemma31.v1"
