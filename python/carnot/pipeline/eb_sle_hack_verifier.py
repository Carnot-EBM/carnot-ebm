"""EB-SLE Reward Hacking Prevention Prototype.

Detects if the repaired generation merely loops or exploits syntax vs solving the problem.

Spec: REQ-VERIFY-1742, SCENARIO-VERIFY-1742
"""

import re
from typing import Optional

class EBSLEHackVerifier:
    """Verifies that a repair generation hasn't reward-hacked."""
    
    def __init__(self, loop_threshold: int = 3, min_reasoning_length: int = 10) -> None:
        self.loop_threshold = loop_threshold
        self.min_reasoning_length = min_reasoning_length
        
    def detect_hack(self, initial_response: str, repaired_response: str) -> bool:
        """Detect if the repaired response is a reward hack.
        
        Returns:
            True if a hack is detected, False otherwise.
        """
        # 1. Trivial pass: no response
        if not repaired_response.strip():
            return True
            
        # 2. Syntax exploitation: removed all reasoning, just returns empty brackets/tags
        alpha_chars = len(re.sub(r'[^a-zA-Z]', '', repaired_response))
        if alpha_chars < self.min_reasoning_length:
            return True
            
        # 3. Looping: checks for repetitive phrases (a sign of degeneration)
        sentences = [s.strip() for s in repaired_response.split('.') if s.strip()]
        if len(sentences) >= self.loop_threshold:
            # Check if any sentence is repeated more than loop_threshold times
            sentence_counts = {}
            for s in sentences:
                sentence_counts[s] = sentence_counts.get(s, 0) + 1
                if sentence_counts[s] >= self.loop_threshold:
                    return True
                    
        # 4. Same response (no actual repair)
        if initial_response == repaired_response:
            return True

        return False
