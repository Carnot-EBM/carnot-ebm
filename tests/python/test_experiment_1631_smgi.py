"""Tests for Exp 1631: SMGI Certified Update Policy.

Spec: REQ-LEARN-1631, SCENARIO-LEARN-1631
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1631_smgi as mod

def test_experiment_1631_smgi_writes_certified_update_success(tmp_path: Path) -> None:
    output_path = tmp_path / "experiment_1631_smgi.json"
    
    artifact = mod.main(output_path)
    
    assert output_path.exists()
    assert artifact["status"] == "success"
    assert artifact["schema"] == "smgi_certified_update_v1"
    assert artifact["experiment_id"] == 1631
    assert artifact["certified_update_success"] is True
    assert artifact["trace_stored"] is True
    assert artifact["honest_verdict"] == "smgi_certified_update_passed"

    data = json.loads(output_path.read_text("utf-8"))
    assert data["certified_update_success"] is True
