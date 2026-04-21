"""SpecGuard Step-Level Verifier — log-prob and attention-grounding hallucination detection.

**Researcher summary:**
    arXiv 2604.15244 introduces SpecGuard: a spec-based guardrail system that detects
    hallucinated reasoning steps using two signals computed from the generation forward
    pass (zero extra model calls).  This module implements both signals and combines
    them into a single per-step rejection score suitable for use as Tier 0f in the
    ThreeTierPipeline cascade.

**Why two signals?**
    LLMs hallucinate in two distinct failure modes:
      (1) Uncertainty hallucination: the model generates a token it is not confident
          about — visible as a low log-probability.  LPBV (Log-Probability-Based
          Verification) captures this: mean log-prob of the step tokens near zero or
          positive (for very uncertain models) signals a hallucinated step.
      (2) Grounding hallucination: the model generates plausible text that is NOT
          anchored to the problem constraints — visible as diffuse attention weights
          that do not concentrate on spec-relevant tokens.  ABGV (Attention-Based
          Grounding Verification) captures this: a low maximum attention weight means
          the model was not looking at any single "spec anchor" when producing the step.

**How scores are interpreted:**
    Both LPBV and ABGV produce scores in [0, 1] where HIGHER = more suspicious.
    The combined_score is a 50/50 blend.  When combined_score >= combined_threshold,
    the step is rejected as likely hallucinated.

**CPU-safe design:**
    Both _compute_lpbv and _compute_abgv gracefully degrade when the relevant
    generation signals (token_logprobs, attention_weights) are absent.  The
    fallbacks use lightweight text heuristics so the verifier can run in CI or
    on CPU-only hosts without any GPU or model loaded.

Spec: REQ-VERIFY-152, REQ-VERIFY-153, REQ-VERIFY-154
SCENARIO-VERIFY-206, SCENARIO-VERIFY-207, SCENARIO-VERIFY-208
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SpecGuardStepResult
# ---------------------------------------------------------------------------


@dataclass
class SpecGuardStepResult:
    """Result of applying SpecGuard to a single reasoning step.

    **Detailed explanation for engineers:**
        Each reasoning step in an LLM response is scored independently.
        The two component scores (lpbv_score, abgv_score) capture different
        failure modes.  combined_score is the decision variable; when it
        crosses combined_threshold, step_rejected is set to True.

    Attributes
    ----------
    step_index : int
        Zero-based position of this step within the full response.
    step_text : str
        The raw text of this step (after splitting the response on sentence
        boundaries).
    lpbv_score : float
        Log-Probability-Based Verification score in [0, 1].
        Higher = model was LESS confident when generating this step.
    abgv_score : float
        Attention-Based Grounding Verification score in [0, 1].
        Higher = attention was MORE diffuse (model not grounded in spec).
    combined_score : float
        0.5 * lpbv_score + 0.5 * abgv_score.  Decision variable.
    step_rejected : bool
        True iff combined_score >= the verifier's combined_threshold.

    Spec: REQ-VERIFY-152, REQ-VERIFY-153
    """

    step_index: int
    step_text: str
    lpbv_score: float
    abgv_score: float
    combined_score: float
    step_rejected: bool


# ---------------------------------------------------------------------------
# SpecGuardVerifier
# ---------------------------------------------------------------------------


class SpecGuardVerifier:
    """SpecGuard step-level verifier combining LPBV + ABGV signals.

    **Detailed explanation for engineers:**
        Implements the two detection signals from arXiv 2604.15244:

        LPBV (Log-Probability-Based Verification):
            Uses the token log-probabilities that are already cached during
            generation.  Low mean log-prob = model was uncertain = elevated
            risk of hallucination.  Formula:
                score = clamp(-mean_logprob / 10.0, 0, 1)
            A mean_logprob of -10.0 nats maps to score=1.0 (maximally
            uncertain); a mean_logprob of 0.0 maps to score=0.0 (perfectly
            confident).  When token_logprobs is None (not available), a
            text-length proxy is used: short steps score low, long steps
            score high (naive uncertainty heuristic).

        ABGV (Attention-Based Grounding Verification):
            Uses the per-token attention weights from the last generation
            step.  Low max-attention = the model's focus was spread across
            many tokens rather than grounded in a specific spec clause.
            Formula:
                score = clamp(1.0 - max_attn, 0, 1)
            When attention_weights is None (offline/CI mode), a text
            heuristic is used: steps containing explicit numbers are treated
            as grounded (score=0.0); steps with no numbers are assumed
            ungrounded (score=0.5).

        Thresholds:
            lpbv_threshold  — kept for future per-signal gates; not currently
                              used in the rejection decision.
            abgv_threshold  — kept for future per-signal gates.
            combined_threshold — the active threshold: combined_score >= this
                                 value causes step_rejected=True.

    Parameters
    ----------
    lpbv_threshold : float
        Reserved; not used in current combined rejection logic.  Default 0.3.
    abgv_threshold : float
        Reserved; not used in current combined rejection logic.  Default 0.3.
    combined_threshold : float
        Steps with combined_score >= this value are rejected.  Default 0.5.

    Spec: REQ-VERIFY-152, REQ-VERIFY-153, REQ-VERIFY-154
    """

    def __init__(
        self,
        lpbv_threshold: float = 0.3,
        abgv_threshold: float = 0.3,
        combined_threshold: float = 0.5,
    ) -> None:
        self.lpbv_threshold = lpbv_threshold
        self.abgv_threshold = abgv_threshold
        self.combined_threshold = combined_threshold

    # ------------------------------------------------------------------
    # _compute_lpbv
    # ------------------------------------------------------------------

    def _compute_lpbv(
        self,
        step_text: str,
        token_logprobs: list[float] | None,
    ) -> float:
        """Log-prob verification: low mean logprob = high uncertainty = high score.

        **Detailed explanation for engineers:**
            Token log-probabilities are the natural log of the model's
            predicted probability for each output token.  A perfectly confident
            model assigns log-prob = 0.0 (probability 1.0) to every token it
            generates.  Real models produce values typically in [-3, -10] nats;
            values below -10 indicate the model was very uncertain.

            Negating and scaling by 1/10 maps the typical range to [0, 1]:
                score = -mean_logprob / 10.0   (clamped to [0, 1])

            When token_logprobs is None, we use a crude length proxy:
            short steps (< 200 chars) are assumed more focused / less likely
            to be hallucinations; longer steps score higher.  This is a weak
            signal but avoids crashing in CI.

        Parameters
        ----------
        step_text : str
            Text of the step (used only in the None-logprobs fallback).
        token_logprobs : list[float] or None
            Per-token log-probabilities from the generation pass.  Each
            value should be <= 0.0 (log of a probability).

        Returns
        -------
        float
            Score in [0, 1].  Higher = more suspicious.

        Spec: REQ-VERIFY-152
        """
        if token_logprobs is None:
            # Naive proxy: longer steps without logprobs get a higher score.
            return min(1.0, max(0.0, len(step_text) / 200.0))
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
        return max(0.0, min(1.0, -mean_logprob / 10.0))

    # ------------------------------------------------------------------
    # _compute_abgv
    # ------------------------------------------------------------------

    def _compute_abgv(
        self,
        step_text: str,
        attention_weights: list[float] | None,
    ) -> float:
        """Attention grounding: low max-attention = ungrounded = high score.

        **Detailed explanation for engineers:**
            Attention-Based Grounding Verification (ABGV) measures whether
            the model concentrated its attention on spec-relevant tokens while
            generating this step.  A high maximum attention weight means the
            model was "looking at" something specific (a number, a constraint,
            an operator).  Low max attention means attention was diffuse — the
            model was not grounded in any particular constraint.

            Formula:
                score = 1.0 - max(attention_weights)   (clamped to [0, 1])

            An empty attention list (no data) maps to max_attn=0.0, score=1.0
            (maximally ungrounded — conservative rejection trigger).

            When attention_weights is None, a heuristic based on digit presence
            in the step text is used:
              - Steps containing at least one digit are treated as grounded
                (score=0.0): "the total is 18" references a concrete number.
              - Steps with no digits are treated as partially ungrounded
                (score=0.5): "therefore the answer is correct" has no anchor.

        Parameters
        ----------
        step_text : str
            Text of the step (used only in the None-attention fallback).
        attention_weights : list[float] or None
            Flattened attention weights for the token positions in this step.
            Values should be in [0, 1] and typically sum to 1.0.

        Returns
        -------
        float
            Score in [0, 1].  Higher = more suspicious.

        Spec: REQ-VERIFY-153
        """
        if attention_weights is None:
            numbers = re.findall(r"\d+", step_text)
            return 0.0 if numbers else 0.5
        max_attn = max(attention_weights) if attention_weights else 0.0
        return max(0.0, min(1.0, 1.0 - max_attn))

    # ------------------------------------------------------------------
    # verify_step
    # ------------------------------------------------------------------

    def verify_step(
        self,
        step_index: int,
        step_text: str,
        token_logprobs: list[float] | None = None,
        attention_weights: list[float] | None = None,
    ) -> SpecGuardStepResult:
        """Verify one reasoning step and return a scored result.

        **Detailed explanation for engineers:**
            This is the per-step entry point.  It computes both signals,
            blends them 50/50, applies the combined threshold, and packages
            everything into a SpecGuardStepResult.

            Typical callers: detection_score() (batch), ThreeTierPipeline
            Tier 0f wrapper (future), and unit tests.

        Parameters
        ----------
        step_index : int
            Zero-based position of this step in the split response.
        step_text : str
            The text of this single reasoning step.
        token_logprobs : list[float] or None
            Per-token log-probabilities.  Pass None for CPU/CI fallback.
        attention_weights : list[float] or None
            Attention weights for this step.  Pass None for CPU/CI fallback.

        Returns
        -------
        SpecGuardStepResult

        Spec: REQ-VERIFY-152, REQ-VERIFY-153
        """
        lpbv = self._compute_lpbv(step_text, token_logprobs)
        abgv = self._compute_abgv(step_text, attention_weights)
        combined = 0.5 * lpbv + 0.5 * abgv
        rejected = combined >= self.combined_threshold
        return SpecGuardStepResult(
            step_index=step_index,
            step_text=step_text,
            lpbv_score=lpbv,
            abgv_score=abgv,
            combined_score=combined,
            step_rejected=rejected,
        )

    # ------------------------------------------------------------------
    # detection_score
    # ------------------------------------------------------------------

    def detection_score(
        self,
        response: str,
        all_logprobs: list[list[float]] | None = None,
        all_attentions: list[list[float]] | None = None,
    ) -> float:
        """Return the max combined score across all step segments of a response.

        **Detailed explanation for engineers:**
            Splits the response into sentence-level segments and scores each
            one independently.  The overall detection score is the MAXIMUM
            combined_score across all segments.

            Using the maximum rather than the mean is a conservative choice:
            a single highly suspicious step is enough to flag the whole
            response.  This matches the SpecGuard design in arXiv 2604.15244
            where one hallucinated step is sufficient to reject a chain.

            Step splitting uses the same regex as the original paper evaluation
            code: split on sentence-ending punctuation and newlines.

        Parameters
        ----------
        response : str
            The full LLM response text.
        all_logprobs : list[list[float]] or None
            One log-prob list per step segment, aligned by index.  Extra
            entries beyond len(steps) are ignored; missing entries default
            to None (fallback heuristic).
        all_attentions : list[list[float]] or None
            One attention-weight list per step segment.  Same alignment.

        Returns
        -------
        float
            Max combined_score in [0, 1].  Returns 0.0 for an empty response.

        Spec: REQ-VERIFY-154
        """
        steps = [s.strip() for s in re.split(r"[.!?\n]+", response) if s.strip()]
        scores = []
        for i, step in enumerate(steps):
            lp = all_logprobs[i] if all_logprobs and i < len(all_logprobs) else None
            at = all_attentions[i] if all_attentions and i < len(all_attentions) else None
            result = self.verify_step(i, step, lp, at)
            scores.append(result.combined_score)
        return max(scores) if scores else 0.0
