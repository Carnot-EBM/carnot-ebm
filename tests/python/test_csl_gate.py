import pytest
from carnot.pipeline.csl_gate import ZeroForgettingGate

def test_zero_forgetting_gate():
    """Test REQ-KONA-042 and SCENARIO-KONA-042"""
    gate = ZeroForgettingGate()
    
    # Pre test had failures {1, 2}, post test has failures {1}. No new failures -> Pass.
    assert gate.evaluate({1, 2}, {1}) is True
    
    # Pre test had failures {1}, post test has failures {1, 2}. New failure 2 -> Block.
    assert gate.evaluate({1}, {1, 2}) is False
    
    # Pre test had {}, post test has {1}. Block.
    assert gate.evaluate(set(), {1}) is False
    
    # Pre test had {1}, post test has {}. Pass.
    assert gate.evaluate({1}, set()) is True
