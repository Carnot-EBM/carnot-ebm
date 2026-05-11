"""
Tests for Exp 1810 Capstone E2E Gemma4-26B.

Traces to: REQ-E2E-1810, SCENARIO-E2E-1810.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1810_capstone_gemma26 import build_artifact, run_experiment


def test_build_artifact_success() -> None:
    """Test building a successful artifact.
    
    Traces to: SCENARIO-E2E-1810
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == "1810_capstone_gemma26"
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in artifact["model_specs"]
    assert artifact["accuracy"] == 0.92
    assert artifact["energy"] == 120.5
    assert artifact["honest_verdict"] == "complete: capstone_gemma26_evaluation_finished"


def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test running the experiment and writing the JSON artifact.
    
    Traces to: REQ-E2E-1810
    """
    out_file = tmp_path / "test_1810.json"
    artifact = run_experiment(output_path=out_file)
    
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["schema"] == "carnot.experiment_1810_capstone_gemma26.v1"
