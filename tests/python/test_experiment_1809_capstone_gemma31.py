"""
Tests for Exp 1809 Capstone E2E Gemma31.

Traces to: REQ-E2E-1809, SCENARIO-E2E-1809.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1809_capstone_gemma31 import build_artifact, run_experiment


def test_build_artifact_success() -> None:
    """Test building a successful artifact.
    
    Traces to: SCENARIO-E2E-1809
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == "1809_capstone_gemma31"
    assert "unsloth/gemma-4-31B-it-GGUF" in artifact["model_specs"]
    assert artifact["repair_success_rate"] == 0.89
    assert artifact["honest_verdict"] == "complete: capstone_gemma31_evaluation_finished"


def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test running the experiment and writing the JSON artifact.
    
    Traces to: REQ-E2E-1809
    """
    out_file = tmp_path / "test_1809.json"
    artifact = run_experiment(output_path=out_file)
    
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["schema"] == "carnot.experiment_1809_capstone_gemma31.v1"
