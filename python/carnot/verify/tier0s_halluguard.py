"""Tier 0s arithmetic-consistency heuristic (HONEST HEURISTIC).

**What this is, honestly:** a 50-line text-statistical heuristic that
extracts numeric tokens from a response and computes simple arithmetic-
gap signals. It was originally framed as an NTK-based HalluGuard
(arXiv:2601.18753) approximation, but the implementation does NOT
invoke a model, does NOT compute kernels, and does NOT use GPU. It is
a regex-and-arithmetic pre-filter that happens to discriminate well on
FoVer-style math-step-error pairs because those inputs always contain
three or more numeric tokens.

**What we approximate vs the paper:**
- The HalluGuard NTK method (arXiv:2601.18753) computes kernel-distance
  signals on per-token hidden-state activations of a target model.
- This module does no such thing. It uses
  `re.findall(r'\\d+', response)` to extract integers, then computes
  `|num[0]+num[1] − num[2]|` as a "reasoning instability" proxy and
  `|num[2] − num[3]|` as a "semantic jump" proxy.
- On responses with fewer than 3 numeric tokens, both signals return
  0.0 unconditionally (the "fast-path" exp2727 documented). This is
  not a bug; it's a deliberate noise-suppression on non-arithmetic
  inputs — but it means the verifier contributes no signal outside the
  arithmetic-step input shape.

**Acceptable uses:** as a cheap pre-filter on math-reasoning corpora
(FoVer step pairs, GSM8K-style arithmetic CoT) where the input shape
guarantees ≥3 numeric tokens. Contribution to the production ensemble
v7b's 0.9857 AUROC is incidental — the headline is carried by other
verifiers (KAN, Z3, semantic-consistency).

**Not acceptable uses:** standalone hallucination detection on natural-
language outputs, code, or any input shape lacking numeric tokens. On
those inputs this verifier returns 0.0 for everything.

Per CLAUDE.md "Verifier Authenticity Discipline" (2026-05-21): this
docstring discloses the gap between the implementation and any cited
paper. It is the honest-heuristic pattern (cf. `pcib_probe.py`,
`rprm_step_reward.py`).
"""

import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Tier0sVerifier:
    """Tier 0s arithmetic-consistency heuristic. See module docstring.

    NOT a kernel-based / NTK / model-invoking verifier. Pure text-
    statistical pre-filter for math-reasoning step pairs.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.alpha = 0.5

    def halluguard_ntk_score(self, response: str) -> float:
        """Calculate the HalluGuard NTK score approximation.
        
        Args:
            response: The text response to analyze.
            
        Returns:
            The hallucination score.
        """
        # Extract numbers to mock the semantic evaluation of the reasoning trace
        nums = list(map(int, re.findall(r'\d+', response)))
        
        # 1. Compute logprob variance (mocked via reasoning instability / arithmetic deviation)
        if len(nums) >= 3:
            expected_sum = nums[0] + nums[1]
            actual_sum = nums[2]
            logprob_variance = float(abs(expected_sum - actual_sum))
        else:
            logger.debug("Tier0s fast-path (logprob_variance): < 3 numbers found, returning 0.0")
            logprob_variance = 0.0
            
        # 2. Compute sentence-boundary semantic jump magnitude (mocked via logical gap across sentences)
        if len(nums) >= 4:
            semantic_jump = float(abs(nums[2] - nums[3]))
        else:
            logger.debug("Tier0s fast-path (semantic_jump): < 4 numbers found, returning 0.0")
            semantic_jump = 0.0
            
        # Weighted combination
        score = self.alpha * logprob_variance + (1.0 - self.alpha) * semantic_jump
        return float(score)

    def detect(self, response: str) -> bool:
        """Detect if the response contains a hallucination based on the threshold."""
        return self.halluguard_ntk_score(response) > self.threshold
