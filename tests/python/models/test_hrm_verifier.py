"""
Tests for HRMVerifier.
REQ-VERIFY-1764
SCENARIO-HRM-001
"""
from carnot.models.hrm_verifier import HRMVerifier

def test_hrm_verifier_multi_level_evaluation():
    # Given an HRM verifier initialized with levels
    verifier = HRMVerifier(levels=3)
    
    # When constraints are evaluated
    constraints = [{"type": "format", "value": "json"}]
    result = verifier.evaluate(constraints)
    
    # Then it returns a multi-level verification score.
    assert "score" in result
    assert "details" in result
    assert result["levels_evaluated"] == 3
    assert result["details"]["level_3"] == 1.0

def test_hrm_verifier_init_defaults():
    verifier = HRMVerifier()
    assert verifier.levels == 3
