"""Tests for NEXUS framework pre-action defense.

Spec: REQ-NEXUS-2115, SCENARIO-NEXUS-2115
"""

from carnot.pipeline.nexus_grounding import NexusGroundingVerifier, NexusSafetyEvaluation

class MockVerifier:
    def __init__(self, score_val: float):
        self._score = score_val
    def score(self, text: str) -> float:
        return self._score

def test_nexus_safety_evaluation():
    """Verify evaluation record is created correctly (SCENARIO-NEXUS-2115)."""
    eval_result = NexusSafetyEvaluation(
        is_safe=True,
        risk_score=0.1,
        violations=[],
        feasibility_decoupled=True
    )
    assert eval_result.is_safe is True
    assert eval_result.risk_score == 0.1
    assert eval_result.feasibility_decoupled is True

def test_nexus_grounding_verifier_safe():
    """Evaluate safe pre-action trace with decoupled constraints (SCENARIO-NEXUS-2115)."""
    verifier = MockVerifier(0.2)
    nexus = NexusGroundingVerifier(verifier=verifier)
    result = nexus.evaluate_pre_action_safety("safe text", {"physical_constraint_1"})
    assert result.is_safe is True
    assert result.risk_score == 0.2
    assert result.feasibility_decoupled is True
    assert len(result.violations) == 0

def test_nexus_grounding_verifier_unsafe():
    """Evaluate unsafe pre-action trace without decoupled constraints (SCENARIO-NEXUS-2115)."""
    verifier = MockVerifier(0.8)
    nexus = NexusGroundingVerifier(verifier=verifier)
    result = nexus.evaluate_pre_action_safety("unsafe text", None)
    assert result.is_safe is False
    assert result.risk_score == 0.8
    assert result.feasibility_decoupled is False
    assert len(result.violations) == 1
    assert "safety violation detected" in result.violations[0]
