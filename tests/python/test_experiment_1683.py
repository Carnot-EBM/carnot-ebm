"""Tests for FR-11 Soundness Audit logic."""
import json
import os
import pytest
from pathlib import Path
from carnot.memory.audit import FR11Audit

def test_fr11_audit_blocked_experiment(tmp_path):
    """Test SCENARIO-FR11-1683: Audit a blocked artifact."""
    artifact_path = tmp_path / "experiment_1682_scg_ets_integration.json"
    artifact_data = {
        "experiment": 1682,
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed"
    }
    artifact_path.write_text(json.dumps(artifact_data))
    
    audit = FR11Audit()
    results = audit.audit_rollback_passing(str(artifact_path))
    
    # REQ-FR11-1683-2, REQ-FR11-1683-3
    assert results["soundness_mistakes"] == 0
    assert results["completeness_mistakes"] == 0
    assert results["nonforgetting_rate"] == 1.0

def test_fr11_audit_file_not_found():
    """Test FileNotFoundError is raised when artifact is missing."""
    audit = FR11Audit()
    with pytest.raises(FileNotFoundError):
        audit.audit_rollback_passing("non_existent_file.json")
