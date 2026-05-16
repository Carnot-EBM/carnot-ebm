"""Test Z3-compatible validator backend."""
import pytest
from carnot.pipeline.z3_validator import Z3Validator

def test_z3_validator():
    """Test Z3 validator backend routes constraints and evaluates them.
    
    References:
    - REQ-VERIFY-1975
    - SCENARIO-VERIFY-1975
    """
    validator = Z3Validator()
    
    constraints = [
        {"type": "lower_bound", "target": "temperature", "value": 20.0},
        {"type": "upper_bound", "target": "pressure", "value": 100.0}
    ]
    
    # Provide assignments that satisfy and violate
    valid_assignment = {"temperature": 25.0, "pressure": 50.0}
    invalid_assignment_1 = {"temperature": 15.0, "pressure": 50.0}
    invalid_assignment_2 = {"temperature": 25.0, "pressure": 150.0}
    
    assert validator.validate(constraints, valid_assignment) is True
    assert validator.validate(constraints, invalid_assignment_1) is False
    assert validator.validate(constraints, invalid_assignment_2) is False
    
    # Test false accept rate
    # If we pass an invalid assignment, it should not accept it.
    false_accepts = 0
    total_invalid = 2
    
    if validator.validate(constraints, invalid_assignment_1): false_accepts += 1
    if validator.validate(constraints, invalid_assignment_2): false_accepts += 1
    
    false_accept_rate = false_accepts / total_invalid
    assert false_accept_rate == 0.0

