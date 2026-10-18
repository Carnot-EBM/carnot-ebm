"""NCO Constraint Decoder

Negative Constraint Optimization (arXiv:2605.10065).
"""
from __future__ import annotations

class NCOConstraintDecoder:
    """Decodes tokens with Weighted Finite State Automata negative constraints."""

    def __init__(self, patterns: list[str]):
        self.patterns = patterns

    def decode(self, token_texts: list[str]) -> dict[str, int | list[str]]:
        """Process a sequence of tokens and return WFSA metrics."""
        wfsa_states: set[str] = set()
        rejection_score = 0
        patterns_fired: set[int] = set()
        
        text_so_far = ""
        for token in token_texts:
            text_so_far += token
            for k, p in enumerate(self.patterns):
                if k in patterns_fired:
                    continue
                if p in text_so_far:
                    patterns_fired.add(k)
                    rejection_score += 1
                    wfsa_states.add("REJECTED")
                else:
                    is_tracking = False
                    # Check if any suffix of text_so_far is a prefix of p
                    check_len = min(len(text_so_far), len(p) - 1)
                    for i in range(check_len, 0, -1):
                        if p.startswith(text_so_far[-i:]):
                            is_tracking = True
                            break
                    if is_tracking:
                        wfsa_states.add(f"TRACKING_{k}")
                    else:
                        wfsa_states.add("INITIAL")
                        
        return {
            "nco_rejection_score": rejection_score,
            "n_patterns_fired": len(patterns_fired),
            "wfsa_states_visited": sorted(list(wfsa_states))
        }
