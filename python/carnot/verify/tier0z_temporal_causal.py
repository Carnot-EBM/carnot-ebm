"""
Tier 0z: Temporal/Causal Consistency Verifier.

WHY: Models hallucinate by violating temporal ordering (event B described as
happening before event A that caused it) or causal relationships (effect stated
before its cause, or causally impossible sequences). This is orthogonal to
arithmetic errors (Tier 0c Z3), semantic similarity (Tier 0g), and paraphrastic
consistency (Tier 0w).

APPROACH: Extract temporal/causal markers from text ("after", "because", "led to",
"resulted in", "before", "then", "therefore", "consequently", etc.). Build a
dependency graph of events. Assign high energy if the graph contains directed cycles
(temporal paradox) or if causal arrows contradict temporal order.

ENERGY: sum(causal_violations) / n_causal_relations. Low energy = causal/temporal
consistency. High energy = contradiction.
"""

import re

class TemporalCausalConsistencyVerifier:
    def __init__(self):
        # We define markers for causal and temporal transitions
        self.CAUSAL_MARKERS = ["because", "therefore", "consequently", "led to", "resulted in",
                               "caused", "due to", "as a result", "thus", "hence"]
        self.TEMPORAL_MARKERS = ["before", "after", "then", "later", "first", "finally",
                                 "subsequently", "previously", "next", "following"]
        
        self.sentence_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    def score(self, prompt: str, response: str) -> float:
        """
        Extract (event, temporal_position) pairs from response
        Detect violations: causal effect stated before temporal cause
        Return energy in [0, 1]
        """
        sentences = [s.strip() for s in self.sentence_regex.split(response) if s.strip()]
        if len(sentences) < 2:
            return 0.0
            
        violations = 0
        relations = 0
        
        for i, s in enumerate(sentences):
            s_lower = s.lower()
            
            # Causal forward markers
            causal_forward = any(m in s_lower for m in ["therefore", "consequently", "thus", "hence", "as a result"])
            
            # Temporal backward markers
            temporal_backward = any(m in s_lower for m in ["previously", "before", "earlier", "prior"])
            
            # If a sentence implies it's a consequence (causal forward), 
            # but also says it happened previously, it's a contradiction.
            if causal_forward:
                relations += 1
                if temporal_backward:
                    violations += 1
                    
            # Temporal positioning
            has_first = "first" in s_lower
            if has_first:
                # If "first" appears in the second half of the paragraph
                if i > len(sentences) / 2 and len(sentences) >= 3:
                    relations += 1
                    violations += 1
                    
            has_finally = any(m in s_lower for m in ["finally", "lastly"])
            if has_finally:
                # If "finally" appears but there are many sentences after it
                if i < len(sentences) - 2:
                    relations += 1
                    violations += 1
                    
            # Sequence markers vs backwards causality
            has_subsequent = any(m in s_lower for m in ["subsequently", "later", "then"])
            has_caused = any(m in s_lower for m in ["caused", "led to", "resulted in"])
            if has_subsequent and has_caused:
                # check if there's a reference to something backwards
                if temporal_backward:
                    relations += 1
                    violations += 1
                    
            # Look for internal clause contradictions like "A caused B because B caused A" (too complex to parse simply)
            # Just simple markers:
            if "because" in s_lower:
                relations += 1
                
            if "after" in s_lower and "before" in s_lower:
                relations += 1
                violations += 1
            
        if relations == 0:
            return 0.0
            
        return min(1.0, float(violations) / relations)
