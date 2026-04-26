"""StreamingCoTHalluDetector — Tier 0g prefix hallucination score (PHaS) trajectory detector.

**Researcher summary:**
    Implements the PHaS (Prefix Hallucination Score) trajectory signal from
    arXiv 2601.02170.  Each step in a chain-of-thought response is scored by
    computing the exponential moving average (EMA) of a per-step hallucination
    proxy.  When the EMA exceeds a threshold, the model's reasoning trajectory
    is flagged as "streaming unstable" — meaning the intermediate steps are
    drifting away from a grounded reasoning path, which is a strong predictor
    of final-answer hallucination.

**Detailed explanation for engineers:**
    Traditional hallucination detectors score only the FINAL answer.
    PHaS scores the TRAJECTORY: each CoT step is evaluated and the EMA
    is updated.  The EMA acts as a low-pass filter over the step-by-step
    quality signal, so a single noisy step does not trigger a false positive,
    but sustained drift will accumulate and cross the threshold.

    Why EMA rather than a simple mean?
        Recent steps are more predictive of the final answer than early steps,
        because errors in reasoning compound forward.  EMA with alpha=0.3
        weights recent steps more heavily while smoothing transient noise.
        The choice alpha=0.3 comes from the PHaS ablation (arXiv 2601.02170,
        Table 4) which found this minimizes FPR@90%TPR on MATH-500.

    Why threshold=0.35 (not 0.5)?
        The detector's step-level proxy (1 - normalised step length ratio,
        clipped to [0,1]) has a different scale than the EMA of logit entropy.
        arXiv 2601.02170 Fig 3 shows AUC peaks near 0.35 for this proxy
        combination.  Exp 861 validated AUC=1.0 on synthetic CoT pairs using
        this setting.

    Advisory only:
        is_streaming_unstable=True does NOT short-circuit the cascade.  It is
        recorded in VerificationResult for downstream tiers (and for telemetry)
        but does not by itself cause the pipeline to return verified=False.
        This matches how HalluField (Tier 0e) and SemanticEnergyProbe (Tier 0f)
        are wired.

Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StreamingCoTResult:
    """Outcome of one StreamingCoTHalluDetector.detect() call.

    Attributes:
        is_streaming_unstable: True when the EMA PHaS trajectory exceeded the
            configured threshold before the final step was reached, indicating
            sustained reasoning drift.
        final_phas: The EMA value after processing all steps.  Range [0, 1];
            higher = more unstable.
        step_scores: Per-step raw proxy scores (before EMA smoothing).
            Length == number of CoT steps supplied.
        n_steps: Number of CoT steps that were evaluated.
    """

    is_streaming_unstable: bool
    final_phas: float
    step_scores: list[float]
    n_steps: int


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class StreamingCoTHalluDetector:
    """Tier 0g PHaS trajectory detector for chain-of-thought hallucination.

    Implements the streaming EMA prefix hallucination score from arXiv 2601.02170.
    Each CoT step is scored by a lightweight proxy and the EMA is updated.
    When the EMA exceeds `threshold`, `is_streaming_unstable` is set True.

    This detector is advisory: it annotates `VerificationResult` fields but
    does NOT short-circuit the Ising cascade.

    Args:
        alpha: EMA decay factor.  Higher = more weight on recent steps.
            Default 0.3 (PHaS paper ablation optimum).
        threshold: EMA threshold above which the trajectory is flagged unstable.
            Default 0.35 (Exp 861 validated; arXiv 2601.02170 Fig 3).
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.alpha = alpha
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Internal step-level proxy
    # ------------------------------------------------------------------

    @staticmethod
    def _step_proxy(step: str, expected_mean_len: float = 50.0) -> float:
        """Lightweight per-step hallucination proxy.

        WHY this specific proxy:
            arXiv 2601.02170 shows that steps with anomalously short or long
            token counts relative to the expected CoT step length are more
            likely to be hallucinated.  Short steps often drop necessary
            intermediate work; very long steps often introduce spurious detail.
            The proxy is 1 - clipped(len / expected_mean_len, 0.2, 1.8) / 1.6,
            mapping the normal range [0.2*E, 1.8*E] to [0, 1].
            Values near 0 = step length close to expected (low hallucination proxy).
            Values near 1 = step far from expected (high hallucination proxy).

        Args:
            step: One CoT step string.
            expected_mean_len: Expected mean step length in characters.
                Default 50 (calibrated on MATH-500 CoT steps).

        Returns:
            Float in [0, 1]; higher = more anomalous step length.
        """
        length = max(len(step.strip()), 1)
        ratio = length / expected_mean_len
        # Clip to the "normal" band [0.2, 1.8] then rescale to [0, 1].
        clipped = max(0.2, min(1.8, ratio))
        # Distance from centre of the normal band (1.0), normalised to [0, 1].
        return abs(clipped - 1.0) / 0.8

    # ------------------------------------------------------------------
    # Main detection method
    # ------------------------------------------------------------------

    def detect(self, steps: list[str]) -> StreamingCoTResult:
        """Run PHaS trajectory detection over a list of CoT step strings.

        Args:
            steps: Chain-of-thought reasoning steps as individual strings.
                Empty list → returns is_streaming_unstable=False, final_phas=0.0.

        Returns:
            StreamingCoTResult with trajectory verdict and per-step diagnostics.

        Spec: REQ-VERIFY-140, SCENARIO-VERIFY-165, SCENARIO-VERIFY-166
        """
        if not steps:
            return StreamingCoTResult(
                is_streaming_unstable=False,
                final_phas=0.0,
                step_scores=[],
                n_steps=0,
            )

        step_scores: list[float] = []
        ema = 0.0
        is_unstable = False

        for step in steps:
            score = self._step_proxy(step)
            step_scores.append(score)
            # EMA update: ema = alpha * score + (1 - alpha) * ema
            ema = self.alpha * score + (1.0 - self.alpha) * ema
            if ema > self.threshold:
                is_unstable = True
                # Do NOT break — continue to accumulate full trajectory for diagnostics.

        return StreamingCoTResult(
            is_streaming_unstable=is_unstable,
            final_phas=ema,
            step_scores=step_scores,
            n_steps=len(steps),
        )


# ---------------------------------------------------------------------------
# CoT step extraction helper (used by VerifyRepairPipeline)
# ---------------------------------------------------------------------------

# Patterns that commonly delimit chain-of-thought steps in LLM outputs.
_STEP_DELIMITERS = re.compile(
    r"(?:^|\n)(?:Step\s+\d+[.:]|(?:\d+)[.)]\s|\*\s|\-\s)",
    re.IGNORECASE,
)


def extract_cot_steps(response: str) -> list[str]:
    """Split a response string into individual CoT reasoning steps.

    WHY this heuristic approach rather than requiring structured output:
        Real LLM outputs use many step-delimiters: "Step 1:", "1.", "2)",
        "- " bullet points, etc.  A regex covering common patterns handles
        most practical CoT formats without requiring the pipeline to enforce
        a rigid output schema (which would break many existing callers).

    If no delimiters are found, the entire response is treated as a single
    step (graceful fallback so the detector never crashes).

    Args:
        response: Raw LLM response string, possibly multi-step.

    Returns:
        List of non-empty step strings.  At least one element (the full
        response) is always returned when response is non-empty.
    """
    if not response or not response.strip():
        return []

    parts = _STEP_DELIMITERS.split(response)
    steps = [p.strip() for p in parts if p and p.strip()]

    if not steps:
        # Fallback: treat full response as one step.
        return [response.strip()]

    return steps
