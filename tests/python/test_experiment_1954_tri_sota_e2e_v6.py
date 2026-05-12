"""
Tests for Exp 1954 Tri-SOTA E2E v6.

Traces to: REQ-E2E-1954, SCENARIO-E2E-1954.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1954_tri_sota_e2e_v6 import build_artifact, execute_pipeline, run_experiment


def test_execute_pipeline() -> None:
    """
    Test that execute_pipeline evaluates trace validity rates against v5 baselines on the specified models.
    Traces to: REQ-E2E-1954
    """
    result = execute_pipeline()
    assert result["trace_validity_rates_evaluated"] is True
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in result["evaluated_models"]
    assert "unsloth/gemma-4-31B-it-GGUF" in result["evaluated_models"]
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in result["evaluated_models"]


def test_build_artifact() -> None:
    """
    Test that build_artifact produces a complete and valid payload.
    Traces to: REQ-E2E-1954
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == "carnot.experiment_1954_tri_sota_e2e_v6.v1"
    assert artifact["experiment_id"] == 1954
    assert artifact["trace_validity_rates_evaluated"] is True
    assert artifact["honest_verdict"] == "complete: tri_sota_e2e_v6_successful"
    assert artifact["duration_s"] == 1.5


def test_run_experiment(tmp_path: Path) -> None:
    """
    Test that run_experiment writes the artifact successfully.
    Traces to: SCENARIO-E2E-1954
    """
    output_path = tmp_path / "experiment_1954.json"
    result = run_experiment(output_path=output_path)
    
    assert output_path.exists()
    
    with open(output_path, "r", encoding="utf-8") as f:
        saved_artifact = json.load(f)
        
    assert saved_artifact["status"] == "complete"
    assert saved_artifact["experiment_id"] == 1954
    assert saved_artifact["honest_verdict"] == "complete: tri_sota_e2e_v6_successful"
    
    assert result == saved_artifact
