import re
from typing import Any
from z3 import Solver, Int, sat, unsat
from carnot.verify.fregelogic_hybrid import FregeLogicHybrid

class HierarchicalLogConsVerifier:
    """Verifier enforcing hierarchical instruction compliance via Z3 partial-order constraints."""
    def __init__(self, semantic_threshold: float = 0.5, laab_threshold: float = 0.5):
        self.fregelogic = FregeLogicHybrid(semantic_threshold, laab_threshold)

    def _structural_valid(self, text: str) -> bool:
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
        for char in text:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        if len(stack) > 0:
            return False
            
        text = text.strip()
        if text and not (text[-1] in ".!?" or text[-1].isalnum() or text.endswith(">")):
            return False
            
        return True

    def estimate_response_priority_level(self, response: str):
        response_lower = response.lower()
        import z3
        if "system:" in response_lower or "instruction:" in response_lower or "claim: system" in response_lower:
            return z3.Int('s0')
        elif "user:" in response_lower or "question:" in response_lower:
            return z3.Int('s1')
        return z3.Int('s2')

    def verify(self, entry: dict[str, Any]) -> dict[str, Any]:
        import z3
        response = str(entry.get("response_text", ""))
        prompt = str(entry.get("prompt", ""))
        
        # TruncProof structural pre-check
        structural_valid = self._structural_valid(response)
        structural_penalty = 0.0
        if not structural_valid:
            structural_penalty = 0.3
            
        hierarchy_violation = False
        priority_level_used = "fallback"
        z3_encoding_used = False
        
        system_content = entry.get("system", entry.get("instruction", prompt[:100]))
        user_content = entry.get("question", entry.get("prompt", prompt[100:200]))
        task_content = response[:100]
        
        s0, s1, s2 = z3.Ints('s0 s1 s2')
        solver = z3.Solver()
        solver.add(s0 > s1, s1 > s2)
        
        response_level = self.estimate_response_priority_level(response)
        solver.add(z3.Not(response_level >= s0))
        
        verdict = solver.check()
        z3_encoding_used = True
        priority_level_used = "z3_forced"
        
        if verdict == z3.unsat:
            hierarchy_violation = True
            base_score = 1.0
        else:
            base_score = 0.0
            
        logcons_score = base_score + structural_penalty
        logcons_score = min(1.0, max(0.0, logcons_score))
        
        if hierarchy_violation:
            logcons_score = 1.0
            
        return {
            "logcons_score": logcons_score,
            "hierarchy_violation": hierarchy_violation,
            "priority_level_used": priority_level_used,
            "structural_valid": structural_valid,
            "z3_encoding_used": z3_encoding_used
        }
