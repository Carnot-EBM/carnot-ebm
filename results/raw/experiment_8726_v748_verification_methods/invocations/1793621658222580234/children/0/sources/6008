"""DiffuTruth reconstruction energy proxy verifier.

This module implements a CPU-feasible proxy for diffusion model reconstruction error
as an indicator of hallucination risk. Hallucinated text tends to have higher energy
(requiring more reconstruction steps), modeled here via token-level log-perplexity
variance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

JsonDict = dict[str, Any]

class DiffuTruthVerifier:
    """Tier 0k DiffuTruth energy proxy for hallucination detection."""

    def __init__(self) -> None:
        self.energy_proxy_method = "token_logprobs_std_times_mean_abs"

    def verify(self, entry: JsonDict) -> JsonDict:
        """Calculate the DiffuTruth score for a telemetry entry.
        
        Args:
            entry: A dictionary containing 'token_logprobs'.
            
        Returns:
            A dictionary with 'diffutruth_score' and 'energy_proxy_method'.
        """
        token_logprobs = entry.get("token_logprobs")
        if not token_logprobs or not isinstance(token_logprobs, list) or len(token_logprobs) == 0:
            return {
                "diffutruth_score": 0.0,
                "energy_proxy_method": self.energy_proxy_method,
            }

        valid_logprobs = [float(lp) for lp in token_logprobs if lp is not None and math.isfinite(float(lp))]
        
        if not valid_logprobs:
            return {
                "diffutruth_score": 0.0,
                "energy_proxy_method": self.energy_proxy_method,
            }
            
        logprobs_array = np.array(valid_logprobs)
        std_lp = np.std(logprobs_array, ddof=0)
        mean_abs_lp = np.mean(np.abs(logprobs_array))
        
        energy_score = float(std_lp * mean_abs_lp)
        
        return {
            "diffutruth_score": energy_score,
            "energy_proxy_method": self.energy_proxy_method,
        }
