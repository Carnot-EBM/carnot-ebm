import math
from typing import List

def compute_spilled_energy(logprobs: List[float]) -> float:
    """
    Computes per-token spilled energy as defined in arXiv:2602.18671.
    spilled_energy = mean(1.0 - exp(p_i)) for all tokens.
    """
    if not logprobs:
        return 0.0
    spilled_t = [1.0 - math.exp(p) for p in logprobs]
    return sum(spilled_t) / len(spilled_t)

def compute_marginalized_energy(logprobs: List[float]) -> float:
    """
    Computes marginalized energy as defined in arXiv:2602.18671.
    marginalized_energy = -mean(p_i) for all tokens.
    """
    if not logprobs:
        return 0.0
    return -sum(logprobs) / len(logprobs)
