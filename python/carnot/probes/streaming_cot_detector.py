"""StreamingCoTHalluDetector — Tier 0g streaming hallucination detector for long CoT.

**Researcher summary:**
    Implements arXiv 2601.02170 (Streaming Hallucination Detection in Long CoT).
    Maintains a per-prefix cumulative PHaS (Predictive Hallucination Score) as an
    exponential moving average of EORM step scores.  When PHaS drops below a
    threshold, the chain-of-thought is flagged as streaming-unstable.

**Detailed explanation for engineers:**
    Large language models that produce long chains of thought (CoT) often embed
    hallucinations *inside* intermediate steps rather than only in the final answer.
    Classic whole-response scoring (EORM Tier 2) waits until the full response
    is complete before flagging anything wrong.

    The streaming detector changes this by scoring each CoT step *as it arrives*
    and maintaining a rolling quality signal called PHaS (Predictive Hallucination
    Score).  The key formula is:

        phas_t = alpha * eorm_score_t + (1 - alpha) * phas_(t-1)

    This is an exponential moving average (EMA): recent scores matter more when
    alpha is large, past history matters more when alpha is small.  A default of
    alpha=0.3 keeps a long memory while staying responsive to local changes.

    **Why EORM score and not raw energy?**
    EORM outputs energy (lower = better).  The PHaS convention from the paper
    treats *higher* as better (like a reward), so we negate the energy to get a
    score: score = -energy.  This means:
        - A correct step → energy near 0 or negative → score near 0 or positive → PHaS stays high
        - A hallucinated step → large positive energy → large negative score → PHaS drops

    **Threshold logic:**
    When phas_t < threshold (default 0.35), the flag is_streaming_unstable fires.
    This is advisory: the pipeline records the flag but does NOT block repair.

    **No state sharing:**
    Each StreamingCoTHalluDetector instance is single-use per CoT response.
    Call reset() to re-use the same detector across multiple responses.

Spec: REQ-PROBE-040, SCENARIO-PROBE-050
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.models.eorm import EORMModel


class StreamingCoTHalluDetector:
    """Tier 0g streaming hallucination detector using rolling PHaS over CoT steps.

    **For engineers:**
        Wrap an EORMModel and call process_step() once per CoT reasoning step.
        The detector maintains a running PHaS history and sets is_streaming_unstable
        when the latest PHaS falls below the threshold.

        This is advisory-only (Tier 0g): it populates the streaming_cot_unstable
        flag in VerificationResult but does not stop the pipeline.

    Example::

        from carnot.models.eorm import EORMModel
        from carnot.probes.streaming_cot_detector import StreamingCoTHalluDetector

        eorm = EORMModel()
        detector = StreamingCoTHalluDetector(eorm)
        for step in cot_steps:
            result = detector.process_step(step)
            if result["is_unstable"]:
                print(f"PHaS={result['phas_t']:.3f} — possible hallucination")
        print(f"final unstable={detector.is_streaming_unstable()}")

    Spec: REQ-PROBE-040
    """

    def __init__(
        self,
        eorm_model: EORMModel,
        alpha: float = 0.3,
        threshold: float = 0.35,
    ) -> None:
        """Create a StreamingCoTHalluDetector.

        **For engineers:**
            alpha controls how quickly the EMA reacts to new steps.  Low alpha
            (e.g., 0.1) is sluggish but stable; high alpha (e.g., 0.7) reacts
            fast but is noisy.  The paper default of 0.3 is a balanced choice.

            threshold is the PHaS value below which the CoT is flagged.
            The paper uses 0.3 as the detection threshold; 0.35 gives slightly
            earlier detection at the cost of more false positives.

        Args:
            eorm_model: An EORMModel instance.  score() method calls energy()
                and negates the result so higher scores = better quality.
            alpha: EMA decay weight for PHaS update.  Range (0, 1).  Default 0.3.
            threshold: PHaS level below which is_streaming_unstable fires.  Default 0.35.
        """
        self.eorm = eorm_model
        self.alpha = alpha
        self.threshold = threshold
        self.phas_history: list[float] = []

    def score(self, cot_step: str) -> float:
        """Compute a quality score for a single CoT step using EORM.

        **For engineers:**
            EORM returns energy (lower = better).  We negate it to produce a
            score where higher = better, matching the PHaS convention from
            arXiv 2601.02170.  An empty question context is used because at
            streaming time we score each step in isolation — the full question
            is not available per step.

        Args:
            cot_step: A single step of chain-of-thought text.

        Returns:
            Float score (higher = model considers this step more correct).
        """
        from carnot.models.eorm import CoTEnergyInput

        # Score each step as a standalone response; question context not available
        # per-step so we use an empty question.  Negating energy → score: higher = better.
        energy = self.eorm.energy(CoTEnergyInput(question_text="", response_text=cot_step))
        return -energy

    def process_step(self, cot_step: str) -> dict[str, float | bool]:
        """Process one CoT step, update PHaS, and return the current detector state.

        **For engineers:**
            For the first step there is no prior PHaS, so we initialize phas_t = score.
            Subsequent steps apply the EMA formula:
                phas_t = alpha * score + (1 - alpha) * phas_(t-1)

            The returned dict carries:
                phas_t     — current PHaS value (EMA of EORM scores so far)
                eorm_score — raw EORM score for this step (negated energy)
                is_unstable — True when phas_t < threshold

        Args:
            cot_step: Text of one CoT reasoning step.

        Returns:
            Dict with keys "phas_t" (float), "eorm_score" (float), "is_unstable" (bool).
        """
        eorm_score = self.score(cot_step)

        if self.phas_history:
            phas_t = self.alpha * eorm_score + (1.0 - self.alpha) * self.phas_history[-1]
        else:
            # First step: bootstrap PHaS directly from the EORM score
            phas_t = eorm_score

        self.phas_history.append(phas_t)
        is_unstable = phas_t < self.threshold

        return {
            "phas_t": phas_t,
            "eorm_score": eorm_score,
            "is_unstable": is_unstable,
        }

    def is_streaming_unstable(self) -> bool:
        """Return True when the most recent PHaS value is below the detection threshold.

        **For engineers:**
            This is the flag that gets written to VerificationResult.streaming_cot_unstable.
            Returns False when no steps have been processed yet (no PHaS history).

        Returns:
            True if the last phas_t < threshold, False otherwise.

        Spec: REQ-PROBE-040
        """
        if not self.phas_history:
            return False
        return self.phas_history[-1] < self.threshold

    def reset(self) -> None:
        """Clear PHaS history to re-use this detector for a new CoT response.

        **For engineers:**
            Each CoT response is an independent scoring context.  Call reset()
            between responses when the same detector instance is reused.
        """
        self.phas_history = []
