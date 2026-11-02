"""Fast-Slow Variant pipeline.

Spec: REQ-FAST-SLOW-1811
"""
from typing import Callable, Dict, Any, List

class VerifyResult:
    def __init__(self, is_correct: bool, failure_summary: str):
        self.is_correct = is_correct
        self.failure_summary = failure_summary

def fast_slow_variant(
    prompt: str, 
    base_llm: Callable[[str], str], 
    verifier_ensemble: Callable[[str], VerifyResult], 
    max_iters: int = 10
) -> Dict[str, Any]:
    """
    Fast-Slow Variant Pipeline.
    SLOW = base LLM (frozen at inference) + verifier ensemble.
    FAST = verifier-output-summary context buffer that re-prompts the LLM iteration-to-iteration.
    """
    context_buffer: List[str] = []
    
    # Initialize response to None to handle max_iters=0 case
    response = ""
    
    for i in range(max_iters):
        current_prompt = prompt
        if context_buffer:
            current_prompt += "\n\nFeedback from previous attempts:\n" + "\n".join(context_buffer)
            
        # SLOW weights never update, LLM is frozen.
        response = base_llm(current_prompt)
        verify_result = verifier_ensemble(response)
        
        if verify_result.is_correct:
            return {"response": response, "iters": i + 1, "passed": True}
        
        # FAST updates: verifier-failure-summary context buffer
        context_buffer.append(f"- {verify_result.failure_summary}")
        
    return {"response": response, "iters": max_iters, "passed": False}
