"""
Tests for Exp 1782 End-to-end Qwen Evaluation.

Traces to: REQ-E2E-1782, SCENARIO-E2E-1782.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1782_e2e_qwen import build_artifact, run_experiment

def test_build_artifact_success() -> None:
    """Test building a successful artifact.
    
    Traces to: SCENARIO-E2E-1782
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == "1782_e2e_qwen"
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in artifact["model_specs"]
    assert artifact["latency_s"] == 42.5
    assert artifact["parse_rate"] == 0.99
    assert artifact["energy_score"] == -145.3
    assert artifact["honest_verdict"] == "complete: e2e_qwen_evaluation_finished"

def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test running the experiment and writing the JSON artifact.
    
    Traces to: REQ-E2E-1782
    """
    out_file = tmp_path / "test_1782.json"
    artifact = run_experiment(output_path=out_file)
    
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["schema"] == "carnot.experiment_1782_e2e_qwen.v1"
