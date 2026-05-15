"""Tests for the EB-SLE Hack Verifier.

Spec references:
- REQ-VERIFY-1742
- SCENARIO-VERIFY-1742
"""

from carnot.pipeline.eb_sle_hack_verifier import EBSLEHackVerifier

def test_detect_hack_empty_response():
    verifier = EBSLEHackVerifier()
    assert verifier.detect_hack("initial", "") is True
    assert verifier.detect_hack("initial", "   ") is True

def test_detect_hack_syntax_exploit():
    verifier = EBSLEHackVerifier(min_reasoning_length=10)
    # Less than 10 alphabetical chars
    assert verifier.detect_hack("initial response with lots of text", "{}") is True
    assert verifier.detect_hack("initial", "123 + 456 = 579") is True

def test_detect_hack_looping():
    verifier = EBSLEHackVerifier(loop_threshold=3)
    looping_resp = "I am sorry. I am sorry. I am sorry."
    assert verifier.detect_hack("initial", looping_resp) is True
    
def test_detect_hack_identical():
    verifier = EBSLEHackVerifier()
    assert verifier.detect_hack("same", "same") is True
    
def test_detect_hack_valid_repair():
    verifier = EBSLEHackVerifier()
    initial = "The answer is 5."
    repaired = "I realized my mistake. The correct calculation is 2+4=6, so the answer is 6."
    assert verifier.detect_hack(initial, repaired) is False
