"""Suppressed Retrieval Probe (Tier 0o).

Detects off-manifold hallucinations by measuring the proxy self-consistency
of logprobs under paraphrase, combined with the logprob sequence entropy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SuppressedRetrievalProbe:
    """Tier 0o probe detecting suppressed retrieval signatures.
    
    This probe operates on the token_logprobs of a response. It proxies
    paraphrase self-consistency by splitting the sequence, normalizing,
    and measuring divergence, scaled by the total response entropy.
    """

    def compute_score(self, logprobs: list[float]) -> tuple[float, float]:
        """Compute the suppression score from a list of logprobs."""
        if not logprobs:
            return 0.0, 0.0
        n = len(logprobs)
        if n < 2:
            return 0.0, 0.0
        
        half1 = logprobs[:n//2]
        half2 = logprobs[n//2:]
        
        mean1 = float(np.mean(half1))
        std1 = float(np.std(half1)) + 1e-8
        mean2 = float(np.mean(half2))
        std2 = float(np.std(half2)) + 1e-8
        
        norm1 = [(lp - mean1)/std1 for lp in half1]
        norm2 = [(lp - mean2)/std2 for lp in half2[:len(half1)]]
        
        paraphrase_divergence = float(np.mean([(a - b)**2 for a, b in zip(norm1, norm2)]))
        
        exp_lps = [math.exp(min(lp, 0)) for lp in logprobs]
        total = sum(exp_lps) + 1e-8
        probs = [e / total for e in exp_lps]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        
        suppression_score = paraphrase_divergence * entropy
        return float(suppression_score), float(paraphrase_divergence)

    def verify(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Compute the suppression score for a telemetry entry."""
        logprobs = entry.get("token_logprobs", [])
        score, divergence = self.compute_score(logprobs)
        return {
            "suppression_score": score,
            "paraphrase_divergence": divergence,
        }
