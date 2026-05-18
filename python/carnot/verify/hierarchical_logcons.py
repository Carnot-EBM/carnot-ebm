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

    def verify(self, entry: dict[str, Any]) -> dict[str, Any]:
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
        
        # Z3 Consistency Check
        claimed_match = re.search(r'claimed_level\s*=\s*(\d+)', response)
        required_match = re.search(r'required_level\s*=\s*(\d+)', prompt)
        
        if claimed_match and required_match:
            claimed_level = int(claimed_match.group(1))
            required_level = int(required_match.group(1))
            priority_level_used = f"level_{claimed_level}"
            z3_encoding_used = True
            
            solver = Solver()
            c = Int('claimed')
            r = Int('required')
            solver.add(c == claimed_level)
            solver.add(r == required_level)
            solver.add(c >= r)
            if solver.check() == unsat:
                hierarchy_violation = True
                base_score = 1.0
            else:
                base_score = 0.0
        else:
            frege_verdict = self.fregelogic.verify(entry)
            base_score = frege_verdict.get("fregelogic_risk_score", 0.5)
            # sentence-level logical flow bonus
            sentences = [s for s in response.split('.') if s.strip()]
            priority_consistency_bonus = 0.05 if len(sentences) > 1 else 0.0
            base_score = max(0.0, base_score - priority_consistency_bonus)
            
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
