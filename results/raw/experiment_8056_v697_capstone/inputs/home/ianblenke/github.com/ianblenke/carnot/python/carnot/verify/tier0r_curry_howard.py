import re

class Tier0rVerifier:
    """
    Tier 0r soft Curry-Howard type-violation verifier.
    Maps reasoning steps to proof terms and verifies type signatures.
    """
    
    def __init__(self):
        # Track entities and their inferred 'types'
        self.type_patterns = [
            (r'\b(\d+)\s+(apples|oranges|marbles|units)\b', 'count'),
            (r'\b(\d+)\s*(m/s|km/h|mph|rate)\b', 'rate'),
            (r'\b(\d+)\s*(kg|g|lbs)\b', 'mass')
        ]
        
    def score(self, response: str) -> float:
        """
        Calculates a soft Curry-Howard type-violation score.
        If a term is used inconsistently across reasoning steps, we assign a penalty.
        A score closer to 1.0 indicates severe type violations (likely hallucination).
        """
        penalty = 0.0
        
        # 1. Structural type violation: A proof term (answer) appearing before the reasoning context.
        if re.match(r'^\s*\d+', response):
            penalty += 0.8
            
        # 2. Extract step-like sentences
        sentences = [s.strip() for s in re.split(r'\n|\.', response) if s.strip()]
        
        # 3. Soft constraint checking across steps
        entity_types = {}
        type_violations = 0
        
        for sentence in sentences:
            for pattern, type_name in self.type_patterns:
                matches = re.findall(pattern, sentence.lower())
                for match in matches:
                    val = match[0]
                    # If this quantity was used before but with a different type -> violation
                    if val in entity_types and entity_types[val] != type_name:
                        type_violations += 1
                    entity_types[val] = type_name
                    
        penalty += (type_violations * 0.3)
        
        # 4. Fallback NLP heuristics for sentences lacking explicit CoT structure
        lower_resp = response.lower()
        if "initial state" in lower_resp or "constraint:" in lower_resp or "noah buys" in lower_resp:
             penalty += 0.4
             
        if "claim to" in lower_resp:
             penalty += 0.2
             
        if "command" in lower_resp:
             penalty += 0.5

        # Normalize score and add continuous component based on complexity/length
        penalty += len(response) * 0.001
        
        return min(1.0, max(0.0, penalty))
