"""HermesVerifierAdapter — HERMES-style step-boundary verification for CoT responses.

**Why this module exists (arXiv 2511.18760, HERMES):**

    HERMES achieves a 67% accuracy improvement on AIME'25 by running a formal
    prover (Lean) ASYNCHRONOUSLY at step boundaries instead of waiting until the
    response is complete.  The four-module loop is:

        LLM generates step → Translator formalizes → Prover verifies →
        Feedback injected before the next step

    For Carnot we substitute two existing components:
    - Translator: LLMAsExtractorV1.extract() — extracts ArithmeticClaim objects
      from the step text.  This is cheaper than a full Lean formaliser and
      already handles the surface-variation problem that broke regex approaches.
    - Prover: SymCodeVerifier.verify_response() — executes Python expressions
      extracted from each step and compares to the stated result.  AUC=0.804
      on live Qwen3.5-0.8B responses (Exp 619).  Code execution is
      distribution-invariant: no training required.

    The step-boundary granularity comes from InterWhenMonitor.split_at_boundaries(),
    which is sentence-level (matching the Interwhen paper, arXiv 2602.11202).

    This module is a CPU prototype: no GPU is needed because SymCodeVerifier
    falls back to regex in CI mode and LLMAsExtractorV1 falls back to
    StepSegmentEvalChain when llm_caller=None.

Spec: REQ-VERIFY-136, SCENARIO-VERIFY-178, SCENARIO-VERIFY-179
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from carnot.extraction.llm_extractor_v1 import ArithmeticClaim, LLMAsExtractorV1
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# HermesVerificationStep — result for one step in the HERMES loop
# ---------------------------------------------------------------------------


@dataclass
class HermesVerificationStep:
    """Result of applying the HERMES four-module loop to one CoT step.

    This maps directly to the HERMES paper's per-step output: the step text
    that the LLM generated, the claims extracted by the translator, the
    verdict from the prover, and the feedback text that would be injected
    before the next step.

    Fields:
        step_index        — Zero-based position of this step in the response.
        step_text         — Original step text from the CoT response.
        translated_claims — ArithmeticClaims extracted by LLMAsExtractorV1.
                            May be empty when the step contains no detectable
                            arithmetic.
        prover_verdict    — 'violated' if SymCodeVerifier detected an error,
                            'correct' if no error, 'unknown' is reserved for
                            future use (e.g. prover timeout).
        feedback_injected — True iff a correction hint was generated (i.e.
                            the feedback that would be injected before the
                            next step in a live streaming setting).
        feedback_text     — The correction hint string, or None if no feedback
                            was generated.  Non-None iff feedback_injected=True.
    """

    step_index: int
    step_text: str
    translated_claims: list[ArithmeticClaim]
    prover_verdict: str  # 'correct' | 'violated' | 'unknown'
    feedback_injected: bool
    feedback_text: Optional[str]


# ---------------------------------------------------------------------------
# HermesVerifierAdapter
# ---------------------------------------------------------------------------


class HermesVerifierAdapter:
    """Implements the HERMES step-boundary feedback loop using Carnot components.

    HERMES (arXiv 2511.18760) inserts a formal verifier between each step of a
    chain-of-thought response and feeds back correction hints before the next
    step.  This class adapts that architecture to Carnot's existing pipeline:

        Translator  → LLMAsExtractorV1  (extract arithmetic claims)
        Prover      → SymCodeVerifier   (execute Python, compare to stated result)
        Feedback    → generate_feedback()

    In CI mode (llm_caller=None for both components): the extractor uses
    StepSegmentEvalChain (regex-based) and the verifier uses regex code
    extraction.  This is sufficient for unit testing without GPU or network.

    Args:
        extractor : LLMAsExtractorV1 instance (the translator module).
        verifier  : SymCodeVerifier instance (the prover module).
    """

    def __init__(
        self,
        extractor: LLMAsExtractorV1,
        verifier: SymCodeVerifier,
    ) -> None:
        self.extractor = extractor
        self.verifier = verifier

    # ------------------------------------------------------------------
    # Module 1: Translate (LLMAsExtractorV1)
    # ------------------------------------------------------------------

    def translate(self, step: str) -> list[ArithmeticClaim]:
        """Extractor module: extract arithmetic claims from one CoT step.

        Delegates to LLMAsExtractorV1.extract(), which runs up to three
        extraction strategies (JsonClaimExtractor, SymCodeExtractor,
        StepSegmentEvalChain) and returns only the claims where the extracted
        expression disagrees with the stated result.

        In CI mode (llm_caller=None): only StepSegmentEvalChain runs.

        Args:
            step: A single sentence from the CoT response.

        Returns:
            List of ArithmeticClaim objects flagged as violations.  Empty
            list when no arithmetic is detected or all detected arithmetic
            is correct.
        """
        return self.extractor.extract(step)

    # ------------------------------------------------------------------
    # Module 2: Prove (SymCodeVerifier)
    # ------------------------------------------------------------------

    def prove(self, step: str) -> str:
        """Prover module: run SymCodeVerifier and return a verdict string.

        SymCodeVerifier asks the LLM to write a Python expression for the
        arithmetic in the step, evaluates it with safe_eval(), and compares
        to the last stated numeric result.  A mismatch is a violation.

        In CI mode (llm_caller=None): uses regex to find N op M patterns.

        Args:
            step: A single sentence from the CoT response.

        Returns:
            'violated' if any CoTStep has violation_detected=True.
            'correct'  if all steps pass (or no arithmetic found).
        """
        results = self.verifier.verify_response(step)
        violated = any(r.violation_detected for r in results)
        return "violated" if violated else "correct"

    # ------------------------------------------------------------------
    # Module 3: Generate feedback
    # ------------------------------------------------------------------

    def generate_feedback(self, step: str, claims: list[ArithmeticClaim]) -> str:
        """Feedback module: produce a correction hint when a violation is detected.

        In the HERMES architecture, the feedback string is injected into the
        prompt before the LLM generates the next step.  A hint is generated
        when any extracted claim has low confidence (< 0.5) or a missing
        lhs_expr (the expression the LLM failed to formalise).

        Args:
            step:   The step text (not used for the hint text itself, but
                    preserved here so callers can extend this method).
            claims: ArithmeticClaims extracted by translate().

        Returns:
            A non-empty correction hint string if feedback should be injected.
            Empty string if no feedback is needed (all claims look plausible).
        """
        violated_claims = [
            c for c in claims if c.confidence < 0.5 or c.lhs_expr is None
        ]
        if not violated_claims:
            return ""
        return f"Re-check the calculation: {violated_claims[0].claim_text}"

    # ------------------------------------------------------------------
    # Full HERMES loop for one step
    # ------------------------------------------------------------------

    def process_step(self, step_text: str, step_index: int) -> HermesVerificationStep:
        """Run the full HERMES four-module loop for one CoT step.

        Sequence:
            1. translate(step_text) → claims (ArithmeticClaim list)
            2. prove(step_text)     → verdict ('violated' | 'correct')
            3. If violated: generate_feedback(step_text, claims) → feedback_text
               Else: feedback_text = None

        Args:
            step_text:  One sentence from the CoT response.
            step_index: Zero-based position in the full response.

        Returns:
            HermesVerificationStep with all fields populated.
        """
        claims = self.translate(step_text)
        verdict = self.prove(step_text)
        if verdict == "violated":
            feedback = self.generate_feedback(step_text, claims) or None
        else:
            feedback = None
        return HermesVerificationStep(
            step_index=step_index,
            step_text=step_text,
            translated_claims=claims,
            prover_verdict=verdict,
            feedback_injected=feedback is not None,
            feedback_text=feedback,
        )

    # ------------------------------------------------------------------
    # Full HERMES loop for a complete response
    # ------------------------------------------------------------------

    def process_response(self, response: str) -> list[HermesVerificationStep]:
        """Apply the HERMES loop to every step in a completed CoT response.

        Uses InterWhenMonitor.split_at_boundaries() to segment the response
        at sentence boundaries — the same granularity used by the Interwhen
        paper (arXiv 2602.11202) and Carnot's Exp 627 InterWhenMonitor.

        In a live streaming deployment this would be called each time the
        model emits a sentence-ending token.  Here it simulates that by
        replaying the completed response sentence-by-sentence.

        Args:
            response: Full CoT response text from the model.

        Returns:
            Ordered list of HermesVerificationStep, one per sentence fragment.
            Empty list if the response has no non-empty sentences.
        """
        from carnot.pipeline.interwhen_monitor import InterWhenMonitor

        sentences = InterWhenMonitor(self.verifier).split_at_boundaries(response)
        return [self.process_step(s, i) for i, s in enumerate(sentences)]
