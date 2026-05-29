"""
Tests for FR-11 LogicVault Z3 Integration.

Spec: REQ-LEARN-3357, SCENARIO-LEARN-3357, SCENARIO-LEARN-3357-REJECT
"""

import sys
import z3
import pytest
from pathlib import Path
from carnot.pipeline.session_memory import SessionMemory

# Add scripts directory to path to import the experiment script
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.experiment_3357_fr11_logicvault import run_experiment_3357

def test_logic_vault_admit_consistent(tmp_path):
    """
    SCENARIO-LEARN-3357: Consistent Facts Admitted
    """
    mem = SessionMemory(storage_dir=str(tmp_path), model_id="test")
    mem.init_logic_vault()
    
    x = z3.Int('x')
    mem.add_axiom(x > 0)
    
    # Check and admit a consistent fact
    admitted = mem.check_and_admit(x > 5)
    
    assert admitted is True
    assert mem._vault_total == 1
    assert mem._vault_accepted == 1
    assert mem.ledger_consistency_rate == 1.0


def test_logic_vault_reject_contradictory(tmp_path):
    """
    SCENARIO-LEARN-3357-REJECT: Contradictory Facts Rejected
    """
    mem = SessionMemory(storage_dir=str(tmp_path), model_id="test")
    mem.init_logic_vault()
    
    x = z3.Int('x')
    mem.add_axiom(x > 5)
    
    # Check and admit an inconsistent fact
    admitted = mem.check_and_admit(x < 0)
    
    assert admitted is False
    assert mem._vault_total == 1
    assert mem._vault_accepted == 0
    assert mem.ledger_consistency_rate == 0.0


def test_logic_vault_auto_init(tmp_path):
    """
    Test that the logic vault is automatically initialized if properties are accessed
    before explicitly calling init_logic_vault().
    """
    mem = SessionMemory(storage_dir=str(tmp_path), model_id="test")
    x = z3.Int('x')
    
    # Should auto-init
    mem.add_axiom(x > 0)
    admitted = mem.check_and_admit(x > 5)
    assert admitted is True


def test_experiment_3357_fr11_logicvault_runner():
    """
    REQ-LEARN-3357: Experiment runner test.
    """
    artifact = run_experiment_3357()
    
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["fr11_logicvault_ready"] is True
    assert "ledger_consistency_rate" in artifact
    assert artifact["accepted_queries"] == 2
    assert artifact["rejected_queries"] == 1
    
    # After 3 queries (2 consistent, 1 contradictory), rate is 2/3
    assert artifact["ledger_consistency_rate"] == 2.0 / 3.0
