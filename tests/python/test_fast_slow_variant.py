import pytest
from carnot.pipeline.fast_slow_variant import fast_slow_variant, VerifyResult

def test_fast_slow_variant_success():
    def mock_llm(prompt: str) -> str:
        if "Feedback from previous attempts" in prompt:
            return "correct_response"
        return "wrong_response"

    def mock_verifier(response: str) -> VerifyResult:
        if response == "correct_response":
            return VerifyResult(True, "")
        return VerifyResult(False, "Failed rule X")

    result = fast_slow_variant("solve this", mock_llm, mock_verifier, max_iters=3)
    
    assert result["passed"] is True
    assert result["iters"] == 2
    assert result["response"] == "correct_response"

def test_fast_slow_variant_fail():
    def mock_llm(prompt: str) -> str:
        return "always_wrong"

    def mock_verifier(response: str) -> VerifyResult:
        return VerifyResult(False, "Failed rule Y")

    result = fast_slow_variant("solve this", mock_llm, mock_verifier, max_iters=3)
    
    assert result["passed"] is False
    assert result["iters"] == 3
    assert result["response"] == "always_wrong"
