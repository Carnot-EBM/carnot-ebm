import pytest
from carnot.verify.curry_howard import SoftCurryHowardVerifier

def test_structural_violation():
    verifier = SoftCurryHowardVerifier()
    # Number before think block -> high penalty
    score = verifier.score("2\n\n<think>\nHere is reasoning\n</think>")
    assert score >= 0.8

def test_type_violation():
    verifier = SoftCurryHowardVerifier()
    # Uses apples as count, then suddenly as rate
    score = verifier.score("<think>I have 5 apples. The speed is 5 apples m/s.</think>")
    assert score > 0.0

def test_no_violation():
    verifier = SoftCurryHowardVerifier()
    # Normal response with no specific heuristic triggers
    score = verifier.score("\n\n<think>\nReasoning here.\n</think>\n\n1")
    # Score should be very low (just length penalty)
    assert score < 0.2

def test_fallback_heuristic():
    verifier = SoftCurryHowardVerifier()
    # Contains "command" which is one of the fallback triggers
    score = verifier.score("\n\n<think>\nAnalyze user command.\n</think>\n")
    assert score >= 0.5
