"""
Tests for FR-11 LogicVault Concurrent Agent Beliefs.
"""

from pathlib import Path
import json

import pytest
import z3

from carnot.pipeline.session_memory import SessionMemory

def test_concurrent_agent_beliefs(tmp_path: Path):
    """
    Spec: REQ-LEARN-3373, SCENARIO-LEARN-3373
    """
    mem = SessionMemory(storage_dir=str(tmp_path), model_id="multi_agent_model")
    
    agent_A = "agent_A"
    agent_B = "agent_B"
    
    mem.init_logic_vault(agent_A)
    mem.init_logic_vault(agent_B)
    
    x = z3.Int('x')
    y = z3.Int('y')
    
    # Agent A adds x > 0
    mem.add_axiom(x > 0, agent_A)
    
    # Agent B adds y < 0
    mem.add_axiom(y < 0, agent_B)
    
    # Agent A checks x > 5 (consistent)
    admitted_A1 = mem.check_and_admit(x > 5, agent_A)
    assert admitted_A1 is True
    
    # Agent A checks x < 0 (contradictory)
    admitted_A2 = mem.check_and_admit(x < 0, agent_A)
    assert admitted_A2 is False
    
    # Agent B checks y < -5 (consistent)
    admitted_B1 = mem.check_and_admit(y < -5, agent_B)
    assert admitted_B1 is True
    
    # Agent B checks y > 0 (contradictory)
    admitted_B2 = mem.check_and_admit(y > 0, agent_B)
    assert admitted_B2 is False
    
    # Verify consistency rates are independent
    # A has 2 checks, 1 accepted -> 0.5
    # B has 2 checks, 1 accepted -> 0.5
    assert mem._ledger_consistency_rates[agent_A] == 0.5
    assert mem._ledger_consistency_rates[agent_B] == 0.5
