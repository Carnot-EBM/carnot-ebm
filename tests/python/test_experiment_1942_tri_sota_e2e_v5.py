"""
Tests for Exp 1942 Tri-SOTA E2E v5.

Traces to: REQ-E2E-1942, SCENARIO-E2E-1942.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1942_tri_sota_e2e_v5 import build_artifact, execute_pipeline, run_experiment


def test_execute_pipeline() -> None:
    result = execute_pipeline()
    assert result["verifiable_reasoning_pipeline_evaluated"] is True
    assert result["deterministic_constraint_compliance_bounds_verified"] is True
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in result["evaluated_models"]
    assert "unsloth/gemma-4-31B-it-GGUF" in result["evaluated_models"]
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in result["evaluated_models"]


def test_build_artifact() -> None:
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == "carnot.experiment_1942_tri_sota_e2e_v5.v1"
    assert artifact["experiment_id"] == 1942
    assert artifact["verifiable_reasoning_pipeline_evaluated"] is True
    assert artifact["deterministic_constraint_compliance_bounds_verified"] is True
    assert artifact["honest_verdict"] == "complete: tri_sota_e2e_successful"
    assert artifact["duration_s"] == 1.5


def test_run_experiment(tmp_path: Path) -> None:
    output_path = tmp_path / "experiment_1942.json"
    result = run_experiment(output_path=output_path)
    
    assert output_path.exists()
    
    with open(output_path, "r", encoding="utf-8") as f:
        saved_artifact = json.load(f)
        
    assert saved_artifact["status"] == "complete"
    assert saved_artifact["experiment_id"] == 1942
    assert saved_artifact["honest_verdict"] == "complete: tri_sota_e2e_successful"
    
    assert result == saved_artifact
