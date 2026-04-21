"""CausalReasoningVerifier — step-to-step causal entailment checking for CoT responses.

**Why this module exists (arXiv 2601.21210, January 2026):**

    SymCodeVerifier (Exp 619) catches arithmetic errors WITHIN a single CoT step
    by generating Python and executing it.  But a different class of error escapes
    arithmetic checking entirely: a step that does its arithmetic correctly, then
    carries the wrong number forward into the next step.

    Example:
        Step k:   "47 + 28 = 75, so we have 75 items."
        Step k+1: "We started with 80 items, so we now have 80 - 10 = 70."

    Step k's arithmetic is correct (47+28=75).  Step k+1's arithmetic is correct
    (80-10=70).  But the causal link is broken: step k concluded 75, yet step k+1
    opened with 80.  This is a *causal break* — the numeric conclusion of one step
    does not match the numeric premise of the next step.

    CausalReasoningVerifier covers this orthogonal violation type by comparing the
    numeric conclusion of step k to the first numeric premise in step k+1.

Spec: REQ-VERIFY-139, REQ-VERIFY-140,
      SCENARIO-VERIFY-183, SCENARIO-VERIFY-184, SCENARIO-VERIFY-185
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from carnot.pipeline.symcode_verifier import SymCodeVerifier
from carnot.extraction.llm_extractor_v1 import LLMAsExtractorV1


# ---------------------------------------------------------------------------
# CausalEntailmentResult
# ---------------------------------------------------------------------------


@dataclass
class CausalEntailmentResult:
    """Result of checking whether step_k causally justifies step_(k+1).

    Fields:
        step_k_index     — Zero-based index of the *earlier* step in the response.
        step_k_text      — Full text of the earlier step.
        step_k1_text     — Full text of the later step.
        entailment_score — Magnitude of the causal break: 0.0 means no break,
                           positive values indicate how far the premise of step k+1
                           diverges from the conclusion of step k (relative to the
                           magnitude of the conclusion).
        causal_violation — True iff a violation was detected.
        violation_type   — One of 'arithmetic' (bad math within step_k),
                           'causal_break' (step_k conclusion != step_k+1 premise),
                           or 'none' (no violation detected).
    """

    step_k_index: int
    step_k_text: str
    step_k1_text: str
    entailment_score: float
    causal_violation: bool
    violation_type: Literal["arithmetic", "causal_break", "none"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches a number appearing at the start of a step, often after "We have",
# "There are", "Starting with", etc.  Used to find the *opening premise* of a step.
_OPENING_NUMBER_RE = re.compile(
    r"(?:we\s+(?:have|had|start(?:ed)?\s+with)|there\s+(?:are|were)|starting\s+with|"
    r"from\s+the\s+previous\s+step[,\s])\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Generic number extractor — used as fallback when the structured pattern misses.
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# CausalReasoningVerifier
# ---------------------------------------------------------------------------


class CausalReasoningVerifier:
    """Detect causal breaks between consecutive CoT steps.

    This verifier is *orthogonal* to SymCodeVerifier:
      - SymCodeVerifier checks arithmetic WITHIN a step.
      - CausalReasoningVerifier checks ENTAILMENT ACROSS step boundaries.

    Two-pass algorithm for each (step_k, step_k+1) pair:

    Pass 1 — arithmetic check:
        Run SymCodeVerifier on step_k.  If a violation is detected, classify as
        'arithmetic' and stop.  No point checking causal continuity when the
        source step is itself wrong.

    Pass 2 — causal break check (only when Pass 1 passes):
        Extract the last number stated in step_k (the conclusion).
        Extract the last number stated in step_k+1 (its opening premise).
        If both are present and differ by more than 0.01 in absolute terms,
        classify as 'causal_break'.  The entailment_score is the relative
        divergence: |conclusion - premise| / max(|conclusion|, 1.0).

    Args:
        symcode     : SymCodeVerifier instance used for arithmetic checking.
        llm_extractor : Optional LLMAsExtractorV1.  Currently unused (reserved for
                        richer premise extraction in a future experiment).

    Spec: REQ-VERIFY-139
    """

    def __init__(
        self,
        symcode: SymCodeVerifier,
        llm_extractor: Optional[LLMAsExtractorV1] = None,
    ) -> None:
        self.symcode = symcode
        self.extractor = llm_extractor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_numeric_conclusion(self, step_text: str) -> Optional[float]:
        """Extract the numeric conclusion of a step (the final number stated).

        Why 'last' number: CoT steps typically state their answer at the end
        ("so the total is 75").  Taking the last number maximises the chance
        that we grab the *result*, not an operand from mid-step arithmetic.

        Returns the float value, or None if the step contains no numbers.
        """
        numbers = _NUMBER_RE.findall(step_text)
        if not numbers:
            return None
        try:
            return float(numbers[-1])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Core entailment check
    # ------------------------------------------------------------------

    def check_entailment(self, step_k: str, step_k1: str, step_k_index: int = 0) -> CausalEntailmentResult:
        """Check whether step_k causally justifies step_(k+1).

        Algorithm (see class docstring for full description):
        1. Run SymCodeVerifier on step_k — if arithmetic violation, return 'arithmetic'.
        2. Compare numeric conclusion of step_k vs. numeric conclusion of step_k+1.
           If they differ by > 0.01, return 'causal_break'.
        3. Otherwise return 'none'.

        Args:
            step_k       : Text of the earlier reasoning step.
            step_k1      : Text of the later reasoning step.
            step_k_index : Zero-based index of step_k in the parent response.

        Returns:
            CausalEntailmentResult describing any detected violation.
        """
        # Pass 1: arithmetic check on step_k using SymCodeVerifier.
        cot = self.symcode.verify_step(step_k, step_index=step_k_index)
        if cot.violation_detected:
            return CausalEntailmentResult(
                step_k_index=step_k_index,
                step_k_text=step_k,
                step_k1_text=step_k1,
                entailment_score=1.0,
                causal_violation=True,
                violation_type="arithmetic",
            )

        # Pass 2: causal break check — does step_k's conclusion appear in step_k+1?
        conclusion_k = self._extract_numeric_conclusion(step_k)
        premise_k1 = self._extract_numeric_conclusion(step_k1)

        if conclusion_k is not None and premise_k1 is not None:
            delta = abs(conclusion_k - premise_k1)
            if delta > 0.01:
                score = delta / max(abs(conclusion_k), 1.0)
                return CausalEntailmentResult(
                    step_k_index=step_k_index,
                    step_k_text=step_k,
                    step_k1_text=step_k1,
                    entailment_score=score,
                    causal_violation=True,
                    violation_type="causal_break",
                )

        # No violation detected.
        return CausalEntailmentResult(
            step_k_index=step_k_index,
            step_k_text=step_k,
            step_k1_text=step_k1,
            entailment_score=0.0,
            causal_violation=False,
            violation_type="none",
        )

    # ------------------------------------------------------------------
    # Full response verification
    # ------------------------------------------------------------------

    def verify_response(self, response: str) -> list[CausalEntailmentResult]:
        """Verify all step-pair entailments in a CoT response.

        Segments the response using SymCodeVerifier.segment_steps() (the same
        segmenter used for arithmetic checking, ensuring consistency), then
        checks every consecutive (step_k, step_k+1) pair.

        Args:
            response: Full CoT response text from the model.

        Returns:
            List of CausalEntailmentResult, one per consecutive step pair.
            Empty list if the response has fewer than two steps.
        """
        steps = self.symcode.segment_steps(response)
        results: list[CausalEntailmentResult] = []
        for i in range(len(steps) - 1):
            result = self.check_entailment(steps[i], steps[i + 1], step_k_index=i)
            results.append(result)
        return results

    def detection_score(self, response: str) -> float:
        """Return the maximum entailment violation score across all step pairs.

        A score of 0.0 means no causal break was detected.  Positive values
        indicate the relative magnitude of the largest detected break.  This
        scalar can be thresholded (> 0.0) to classify a response as causally
        broken.

        Args:
            response: Full CoT response text.

        Returns:
            Float >= 0.0.  Returns 0.0 for empty or single-step responses.
        """
        results = self.verify_response(response)
        if not results:
            return 0.0
        return max(r.entailment_score for r in results)

    def any_violation(self, response: str) -> bool:
        """Return True if any causal or arithmetic violation is detected.

        Equivalent to detection_score(response) > 0.0, but more readable
        at call sites that only need a boolean signal.
        """
        return self.detection_score(response) > 0.0
