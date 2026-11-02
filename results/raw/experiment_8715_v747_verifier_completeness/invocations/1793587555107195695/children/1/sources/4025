import math
from collections import Counter

def calculate_character_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

class EntropyTaskRouter:
    def __init__(self, threshold: float = 4.5, route_below_threshold: str = "ebm_verifier", route_above_threshold: str = "base_llm"):
        self.threshold = threshold
        self.route_below_threshold = route_below_threshold
        self.route_above_threshold = route_above_threshold
        
    def route(self, prompt: str) -> str:
        entropy = calculate_character_entropy(prompt)
        # Short math problems often have lower entropy (fewer unique characters, more digits)
        # compared to long-form open-ended QA.
        if entropy < self.threshold:
            return self.route_below_threshold
        else:
            return self.route_above_threshold
