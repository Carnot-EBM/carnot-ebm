"""
Tests for Experiment 1904 activation contract.

REQ-AUTO-1904: Must generate an activation contract with required baseline fields.
SCENARIO-AUTO-1904: Generates valid JSON matching schema.
"""

import json
import os
import tempfile
from carnot.experiment_1904_activation import generate_activation_contract, save_activation_contract


def test_generate_activation_contract():
    """Test that the contract contains all required fields and baseline values."""
    contract = generate_activation_contract()
    
    required_keys = [
        "status",
        "honest_verdict",
        "milestone_148_archived",
        "live_sota_blocked_missing_models",
        "telemetry_missing_terminal_artifact",
        "next_gate_contract_ready",
        "tests_run"
    ]
    
    for key in required_keys:
        assert key in contract
        
    assert contract["milestone_148_archived"] is False
    assert contract["live_sota_blocked_missing_models"] is False
    assert contract["telemetry_missing_terminal_artifact"] is False
    assert contract["next_gate_contract_ready"] is False
    assert contract["tests_run"] is False


def test_save_activation_contract():
    """Test saving the contract to disk and verifying its contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_contract.json")
        save_activation_contract(output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data["milestone_148_archived"] is False
        assert data["tests_run"] is True
