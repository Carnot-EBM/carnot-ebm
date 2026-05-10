"""Tests for EidokuGate.

Spec traces: REQ-VERIFY-1500
"""

from carnot.pipeline.eidoku_gate import EidokuGate, EidokuGateResult

def test_eidoku_gate_init():
    """Test initialization."""
    gate = EidokuGate(default_cost=1.0)
    assert gate.default_cost == 1.0

def test_eidoku_gate_compute_cost_no_violation():
    """Test computation without violation."""
    gate = EidokuGate()
    res = gate.compute_cost("question", "response is fine")
    assert isinstance(res, EidokuGateResult)
    assert res.violation_cost == 0.0
    assert res.runtime_ms >= 0

def test_eidoku_gate_compute_cost_with_violation():
    """Test computation with violation."""
    gate = EidokuGate()
    res = gate.compute_cost("question", "response has a VIOLATION")
    assert res.violation_cost == 10.0
    assert res.runtime_ms >= 0
