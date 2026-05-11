"""CPU-based HILED simulator that mimics FPGA-based Gibbs/Potts sampling.

Spec: REQ-SAMPLE-1869
"""

import time
from typing import List, Optional


class HiledSimulator:
    """CPU-based HILED simulator mimicking FPGA-based Gibbs/Potts sampling.

    This applies constraint enforcement and simulates hardware energy scoring
    during decoding.

    Spec: REQ-SAMPLE-1869
    """

    def __init__(
        self, penalty: float = 2.0, constraints: Optional[List[str]] = None
    ) -> None:
        self.penalty = penalty
        self.constraints = constraints or ["unsafe", "hallucination", "error"]
        self.simulated_steps = 0
        self.latency_ms = 0.0

    def score_candidate(self, text: str, initial_logprob: float) -> float:
        """Simulate hardware scoring of a candidate.

        Args:
            text: Candidate response text.
            initial_logprob: The logprob from the LLM.

        Returns:
            Adjusted logprob applying HILED constraints.
        """
        start = time.time()

        # Simulate hardware processing latency
        time.sleep(0.001)
        self.simulated_steps += 1

        score = initial_logprob
        text_lower = text.lower()

        # Apply Potts-like constraints
        for constraint in self.constraints:
            if constraint in text_lower:
                score -= self.penalty

        self.latency_ms += (time.time() - start) * 1000
        return score
