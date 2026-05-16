"""Test KAN Z3 MILP Verifier."""
from carnot.kan_z3 import verify_zero_false_accepts

def test_verify_zero_false_accepts():
    """Verify that Z3 solver passes with zero false accepts."""
    assert verify_zero_false_accepts() is True
