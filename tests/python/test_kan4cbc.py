"""Tests for KAN4CBC MILP Z3 Verification."""

import json
from pathlib import Path

from carnot.models.kan.kan4cbc import (
    build_experiment_2083_artifact,
    verify_milp_kan_robustness,
    write_experiment_2083_artifact,
)

def test_verify_milp_kan_robustness():
    """SCENARIO-KAN-2083: Verify KAN MILP robustness using Z3."""
    result = verify_milp_kan_robustness(input_lb=-1.0, input_ub=1.0, epsilon=0.1)
    assert result["status"] == "complete"
    assert result["is_robust"] is True
    assert isinstance(result["execution_time_s"], float)
    assert result["execution_time_s"] >= 0.0

def test_build_experiment_2083_artifact():
    """REQ-KAN-2083: Build artifact."""
    artifact = build_experiment_2083_artifact()
    assert artifact["schema"] == "carnot.kan4cbc.experiment_2083.v1"
    assert artifact["is_robust"] is True
    assert artifact["honest_verdict"] == "complete: kan4cbc_smt_robustness_verification"

def test_write_experiment_2083_artifact(tmp_path):
    """REQ-KAN-2083: Write artifact."""
    out_file = tmp_path / "experiment_2083_kan4cbc.json"
    artifact = write_experiment_2083_artifact(output_path=out_file)
    
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["experiment_id"] == 2083
    assert data["is_robust"] is True

def test_write_experiment_2083_artifact_default_path():
    """REQ-KAN-2083: Write artifact with default path."""
    artifact = write_experiment_2083_artifact()
    assert artifact["is_robust"] is True
