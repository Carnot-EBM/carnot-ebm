"""ODAR free energy proxy.

**Researcher summary:**
    Computes a proxy for expected free energy from raw logprobs.
    Surprise is mapped to negative mean logprob, and complexity is mapped
    to variance of logprobs.

Spec: REQ-TIER0-008-1
"""

import numpy as np


def compute_odar_energy_proxy(log_probs: list[float]) -> tuple[float, float, float]:
    """Compute ODAR free energy proxy from log probabilities.

    Returns:
        tuple[float, float, float]: (odar_energy, surprise, complexity)
    """
    if not log_probs:
        return 0.0, 0.0, 0.0

    arr = np.array(log_probs)
    surprise = -np.mean(arr)
    complexity = np.var(arr)

    odar_energy = float(surprise + 0.5 * complexity)
    return odar_energy, float(surprise), float(complexity)
