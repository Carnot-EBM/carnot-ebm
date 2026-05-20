"""Tests for the Tier 0f Semantic Calibrated Verifier."""

from carnot.verify import SemanticCalibratedVerifier

def test_semantic_calibrated_verifier_initialization():
    verifier = SemanticCalibratedVerifier()
    assert verifier is not None

def test_semantic_calibrated_verifier_verify():
    verifier = SemanticCalibratedVerifier()
    prob = verifier.verify("Test string to verify.")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
