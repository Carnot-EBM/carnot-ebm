"""Tests for Exp 1641 NSVIF Constraint Compiler SOTA Validation.

Spec: REQ-VERIFY-1641, SCENARIO-VERIFY-1641.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1641_nsvif_sota as mod


def test_req_verify_1641_artifact_generation(tmp_path: Path) -> None:
    """REQ-VERIFY-1641: artifact generates with required fields and zero false accepts."""
    
    output_path = tmp_path / "results" / "experiment_1641_nsvif_sota.json"
    
    artifact = mod.run_experiment(output_path=output_path)
    
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["experiment_id"] == 1641
    assert artifact["status"] == "complete"
    assert artifact["false_accepts"] == 0
    assert artifact["validation_rate"] == 1.0
    assert "complete:" in artifact["honest_verdict"]


def test_scenario_verify_1641_zero_false_accepts() -> None:
    """SCENARIO-VERIFY-1641: records false_accepts=0 and validation_rate metric."""
    
    artifact = mod.run_experiment(output_path=Path("/dev/null"))
    
    assert artifact["false_accepts"] == 0
    assert artifact["validation_rate"] == 1.0
