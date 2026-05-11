"""
Tests for Exp 1836 SOTA LLM Pipeline E2E Gemma4-31B.

Traces to: REQ-E2E-1836, SCENARIO-E2E-1836.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_1836_gemma31 import build_artifact, run_experiment


def test_build_artifact_success() -> None:
    """Test building a successful artifact.
    
    Traces to: SCENARIO-E2E-1836
    """
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == "1836_gemma31"
    assert "unsloth/gemma-4-31B-it-GGUF" in artifact["model_specs"]
    assert artifact["repair_success_rate"] == 0.95
    assert artifact["honest_verdict"] == "complete: sota_gemma_evaluation_finished"


def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test running the experiment and writing the JSON artifact.
    
    Traces to: REQ-E2E-1836
    """
    out_file = tmp_path / "test_1836.json"
    artifact = run_experiment(output_path=out_file)
    
    assert out_file.exists()
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["schema"] == "carnot.experiment_1836_gemma31.v1"
