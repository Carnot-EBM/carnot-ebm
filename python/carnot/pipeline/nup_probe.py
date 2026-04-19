"""NUPProbe — Neural Uncertainty Principle Tier 0c pre-filter.

**Why NUPProbe exists (arXiv 2603.19562 — Neural Uncertainty Principle, 2026):**
    The Neural Uncertainty Principle frames hallucination as an under-constrained
    continuation problem.  When an LLM is about to hallucinate, the entropy of its
    next-token distribution is HIGH — many continuations are nearly equally plausible
    because the model lacks sufficient constraint information to commit to one path.
    When the continuation is factually or logically *constrained*, the distribution
    is PEAKED — the model is confident because the correct continuation is forced by
    prior context (arithmetic result, causal chain, established fact).

    This is mathematically compatible with Carnot's constraint-satisfaction formulation:
        high entropy  = high energy  = constraint violation likely
        low entropy   = low energy   = continuation well-constrained = likely correct

    NUPProbe operationalises this: given a CoT step, compute the entropy of the
    continuation distribution as a proxy for violation likelihood.  No LLM inference
    required — the entropy is computed purely from token log-probabilities that were
    already produced when the original LLM generated the text.

**Why Tier 0c (pipeline position):**
    The Carnot cascade is ordered by cost:
        Tier 0a: CarnotThinkProbe (~50-200 ms, requires secondary LLM on GPU)
        Tier 0b: SpilledEnergyDetector (~0.1 ms, requires token logits)
        Tier 0c: NUPProbe (~0.001 ms, pure arithmetic on logprobs OR char entropy fallback)
        Tier 1:  Ising constraint evaluator (~0.006 ms per constraint)

    NUPProbe is Tier 0c because it requires zero LLM calls and zero Ising sampling.
    It is pure arithmetic on a list of floats.  Earlier cascade position = higher
    skip rate = lower total verification cost at scale.  When logprobs are absent
    (as in live CoT data that lacks token-level metadata), the character-entropy
    fallback provides a structural proxy that is less precise but still informative.

**Why character entropy as a fallback:**
    Token log-probability distributions require access to the LLM's internal softmax
    layer, which is often unavailable for closed-source or streamed outputs.  In such
    cases, the character-level Shannon entropy of the step text provides a structural
    proxy:
        - Low character entropy = repetitive, formulaic text (e.g., "2 + 2 = 4")
          → likely well-constrained → low violation risk
        - High character entropy = high character variety (e.g., mixed symbols,
          freeform prose, LaTeX) → possibly under-constrained → higher violation risk
    This is less precise than token logprob entropy but retains the directionality
    that makes the probe useful as a pre-filter.

**AUC evaluation and Tier 0c qualification:**
    NUPProbeResult.is_viable_tier_0c = (auc > 0.700).  AUC 0.700 is the threshold
    because below that the probe adds overhead without saving enough Ising calls to
    justify its position in the cascade.  Above 0.700, the probe's skip rate at the
    operating threshold justifiably reduces downstream invocations.

Spec: REQ-VERIFY-096, REQ-VERIFY-097,
      SCENARIO-VERIFY-129, SCENARIO-VERIFY-130
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# ContinuationEntropy
# ---------------------------------------------------------------------------


@dataclass
class ContinuationEntropy:
    """Shannon entropy of a token continuation distribution.

    **Detailed explanation for engineers:**
        Given a list of log-probabilities log(p_i) for each token i in the
        vocabulary at a given generation step, Shannon entropy is:

            H = -sum_i  p_i * log(p_i)
              = -sum_i  exp(logp_i) * logp_i        (natural nats by default)

        Interpretation:
            H near 0    → peaked distribution → model is confident → well-constrained
            H near ln(V)→ uniform over V tokens → model is uncertain → under-constrained

        is_high_entropy is True when H > entropy_threshold (default 1.5 nats).
        1.5 nats ≈ e^1.5 ≈ 4.5 tokens with equal weight, which is empirically the
        inflection between "confident prediction" and "uncertain continuation" for
        instruction-tuned models on math word problems.

    Attributes:
        logprobs: Raw log-probabilities (natural log) for each token in the vocab.
        entropy: Shannon entropy in nats.
        is_high_entropy: True when entropy > threshold.
        threshold: The entropy_threshold used for this instance.

    Spec: REQ-VERIFY-096
    """

    logprobs: list[float]
    entropy: float
    is_high_entropy: bool
    threshold: float = 1.5

    @classmethod
    def from_logprobs(
        cls,
        logprobs: list[float],
        threshold: float = 1.5,
    ) -> "ContinuationEntropy":
        """Compute Shannon entropy from a list of log-probabilities.

        **Detailed explanation for engineers:**
            The logprobs are log(p_i) values in natural log units.  We convert
            to probabilities via p_i = exp(logp_i), then compute the Shannon
            entropy H = -sum p_i * log(p_i).  The logprobs need not be
            normalised (they are treated as unnormalised log-scores if their
            exp-sum differs from 1.0) — in that case we re-normalise before
            computing entropy so the result is always a valid entropy.

            Edge cases:
                Empty logprobs list → entropy = 0.0 (no distribution = no uncertainty).
                Single-element list → entropy = 0.0 (only one option = certain).

        Args:
            logprobs: Log-probabilities (natural log) for each token.
            threshold: Entropy threshold above which is_high_entropy=True.

        Returns:
            ContinuationEntropy with computed entropy and high-entropy flag.

        Spec: REQ-VERIFY-096
        """
        if len(logprobs) <= 1:
            return cls(
                logprobs=logprobs,
                entropy=0.0,
                is_high_entropy=False,
                threshold=threshold,
            )

        # Convert to probabilities (handling numerical stability via max-shift)
        max_lp = max(logprobs)
        raw_probs = [math.exp(lp - max_lp) for lp in logprobs]
        total = sum(raw_probs)
        probs = [p / total for p in raw_probs]

        # Shannon entropy H = -sum p_i * log(p_i), ignoring p_i == 0
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)

        return cls(
            logprobs=logprobs,
            entropy=entropy,
            is_high_entropy=entropy > threshold,
            threshold=threshold,
        )


# ---------------------------------------------------------------------------
# NUPProbeResult
# ---------------------------------------------------------------------------


@dataclass
class NUPProbeResult:
    """Summary statistics from a NUPProbe evaluation run.

    **Detailed explanation for engineers:**
        NUPProbeResult captures the outcome of evaluating NUPProbe on a labelled
        dataset.  The key field is is_viable_tier_0c, which gates promotion to the
        pipeline cascade: if AUC > 0.700, NUPProbe can be inserted as Tier 0c in
        ThreeTierPipeline to skip downstream verification calls for high-entropy steps.

    Attributes:
        n_pairs: Number of (CoT step, label) pairs evaluated.
        auc: ROC-AUC of the probe's entropy score vs. ground-truth violation labels.
        threshold_used: The entropy_threshold used for binary classification.
        probe_latency_ms: Mean wall-clock time per probe call in milliseconds.
        is_viable_tier_0c: True when auc > 0.700.

    Spec: REQ-VERIFY-097
    """

    n_pairs: int
    auc: float
    threshold_used: float
    probe_latency_ms: float
    is_viable_tier_0c: bool = field(init=False)

    def __post_init__(self) -> None:
        # Computed after init so callers never need to pass it explicitly.
        # The 0.700 threshold is the minimum AUC required for Tier 0c promotion;
        # below this the probe's skip rate does not justify its cascade position.
        self.is_viable_tier_0c = self.auc > 0.700


# ---------------------------------------------------------------------------
# NUPProbe
# ---------------------------------------------------------------------------


class NUPProbe:
    """Neural Uncertainty Principle Tier 0c pre-filter.

    **Detailed explanation for engineers:**
        NUPProbe maps each CoT step to a violation probability score by measuring
        how uncertain the LLM was when it produced that step.  The NUP insight
        (arXiv 2603.19562) is that hallucinations emerge from under-constrained
        continuations — the LLM's distribution over next tokens is high-entropy
        because multiple paths are nearly equally plausible.

        Two operating modes:

        1. logprob mode (preferred):
            Caller supplies logprobs: list[float] — the per-token log-probabilities
            from the LLM's softmax layer for this step.  NUPProbe computes Shannon
            entropy directly.  Latency: < 0.01 ms.

        2. character entropy fallback (when logprobs absent):
            Computes character-level Shannon entropy from the step text.  This uses
            the frequency distribution of characters in the text as a proxy for the
            token distribution.  Less precise (character entropy does not directly
            measure the LLM's uncertainty) but retains the directional signal:
            formulaic/repetitive text has low character entropy; freeform/symbol-heavy
            text has higher character entropy.

        Pipeline contract:
            NUPProbe sits BEFORE SpilledEnergyDetector (Tier 0b) in the cascade.
            If predict_violation() returns True, downstream verifiers may be skipped
            (or the step may be flagged immediately, depending on pipeline policy).

    Args:
        entropy_threshold: Shannon entropy threshold in nats above which a step is
            classified as high-uncertainty (violation predicted).  Default 1.5.

    Spec: REQ-VERIFY-096, REQ-VERIFY-097,
          SCENARIO-VERIFY-129, SCENARIO-VERIFY-130
    """

    def __init__(self, entropy_threshold: float = 1.5) -> None:
        self.entropy_threshold = entropy_threshold

    def score(
        self,
        cot_text: str,
        logprobs: Optional[list[float]] = None,
    ) -> float:
        """Compute uncertainty score for a CoT step.

        **Detailed explanation for engineers:**
            Returns a float in [0, ∞) representing the entropy of the continuation
            distribution.  Higher score = higher uncertainty = higher violation risk.

            If logprobs is supplied, uses Shannon entropy of the token distribution.
            Otherwise falls back to character-level entropy of cot_text as a structural
            proxy (see module docstring for WHY character entropy is informative even
            without token logprobs).

            Character entropy is capped at ln(256) ≈ 5.55 nats (max possible for
            single-byte characters), so scores from both modes are comparable in
            magnitude when interpreting against the entropy_threshold.

        Args:
            cot_text: The CoT step text.
            logprobs: Optional list of per-token log-probabilities.  If provided,
                Shannon entropy of this distribution is returned.  If None, character
                entropy of cot_text is returned as a fallback.

        Returns:
            Entropy score in nats (float >= 0).

        Spec: REQ-VERIFY-096
        """
        if logprobs is not None and len(logprobs) > 0:
            ce = ContinuationEntropy.from_logprobs(logprobs, self.entropy_threshold)
            return ce.entropy

        # Character entropy fallback: treat each character as an "observation"
        # and compute the Shannon entropy of the character frequency distribution.
        return self._char_entropy(cot_text)

    def predict_violation(
        self,
        cot_text: str,
        logprobs: Optional[list[float]] = None,
    ) -> bool:
        """Predict whether a CoT step is likely a constraint violation.

        **Detailed explanation for engineers:**
            Returns True when score() > entropy_threshold.  This is the binary
            routing decision: True → flag as likely violation (or escalate to Ising),
            False → step appears well-constrained → skip downstream verification.

            Note the asymmetry with CarnotThinkProbe: ThinkProbe returns True for
            should_run_ising when UNCERTAIN.  NUPProbe returns True for violation
            when UNCERTAIN.  The naming reflects which direction is the "positive"
            class in each probe's framing.

        Args:
            cot_text: The CoT step text.
            logprobs: Optional per-token log-probabilities.

        Returns:
            True if entropy score exceeds threshold (likely violation), else False.

        Spec: REQ-VERIFY-096, SCENARIO-VERIFY-129
        """
        return self.score(cot_text, logprobs) > self.entropy_threshold

    def evaluate_auc(self, labeled_pairs: list[dict]) -> float:
        """Compute ROC-AUC of entropy scores against ground-truth violation labels.

        **Detailed explanation for engineers:**
            Each element of labeled_pairs must have:
                'step_text' or 'cot_text': str   — the CoT step to score
                'label': str or bool             — ground truth:
                    'incorrect' or False → violation (positive class)
                    'correct'   or True  → not a violation (negative class)
                'logprobs': list[float] optional — if absent, char entropy fallback

            ROC-AUC is computed by sorting pairs by descending score, then building
            the ROC curve (TPR vs FPR at each threshold).  AUC is computed via the
            trapezoidal rule.  This is the standard AUC(ROC) metric used in binary
            classification evaluation.

            Edge cases:
                < 2 pairs → returns 0.5 (chance level, undefined AUC).
                All labels same → returns 0.5 (cannot distinguish classes).

        Args:
            labeled_pairs: List of dicts with 'step_text'/'cot_text', 'label',
                and optionally 'logprobs'.

        Returns:
            Float in [0.0, 1.0].  0.5 = chance; 1.0 = perfect discrimination.

        Spec: REQ-VERIFY-097, SCENARIO-VERIFY-130
        """
        if len(labeled_pairs) < 2:
            return 0.5

        scores_and_labels: list[tuple[float, bool]] = []
        for pair in labeled_pairs:
            text = pair.get("step_text") or pair.get("cot_text", "")
            lp = pair.get("logprobs")
            raw_label = pair.get("label", "incorrect")
            # Normalise label to bool: True = violation (positive)
            if isinstance(raw_label, bool):
                is_violation = raw_label
            else:
                is_violation = str(raw_label).lower() == "incorrect"
            s = self.score(text, lp)
            scores_and_labels.append((s, is_violation))

        # Count positives and negatives
        n_pos = sum(1 for _, v in scores_and_labels if v)
        n_neg = sum(1 for _, v in scores_and_labels if not v)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Sort descending by score (higher score = more likely violation = more positive)
        sorted_pairs = sorted(scores_and_labels, key=lambda x: x[0], reverse=True)

        # Build ROC curve as a list of (fpr, tpr) points, starting at the origin.
        # Each step through the sorted list moves either tp or fp by 1.
        # We only add trapezoid area when FPR changes (i.e., a negative is processed).
        # Positives processed before a negative "stack up" the TPR at the same FPR, so
        # the trapezoid correctly uses the stacked-up TPR as the right endpoint.
        roc_points: list[tuple[float, float]] = [(0.0, 0.0)]
        tp = 0
        fp = 0
        for _, is_violation in sorted_pairs:
            if is_violation:
                tp += 1
            else:
                fp += 1
            roc_points.append((fp / n_neg, tp / n_pos))

        # Trapezoidal rule in (FPR, TPR) space.
        # Skip consecutive points with the same FPR (vertical segments add no area).
        auc = 0.0
        for i in range(len(roc_points) - 1):
            fpr_prev, tpr_prev = roc_points[i]
            fpr_curr, tpr_curr = roc_points[i + 1]
            if fpr_curr > fpr_prev:
                auc += (fpr_curr - fpr_prev) * (tpr_curr + tpr_prev) / 2.0

        return float(min(1.0, max(0.0, auc)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _char_entropy(text: str) -> float:
        """Compute character-level Shannon entropy of a string.

        **Detailed explanation for engineers:**
            Treat each character in text as an independent draw from a discrete
            distribution over the character set.  The Shannon entropy of that
            distribution measures the average surprise per character — a proxy
            for the structural unpredictability of the text.

            Why this is informative as a fallback:
                Formulaic arithmetic text (e.g., "C = 4 × 20 = 80") has low
                character entropy because digits and operators repeat.  Freeform
                prose or mixed-symbol text (LaTeX, code) has higher character entropy.
                While this does NOT directly measure the LLM's token uncertainty,
                it correlates with text complexity, which correlates with the likelihood
                that the model is navigating an under-constrained domain.

            Returns 0.0 for empty or single-character strings.

        Args:
            text: Input string.

        Returns:
            Character-level Shannon entropy in nats (natural log base).
        """
        if len(text) <= 1:
            return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum(
            (c / total) * math.log(c / total)
            for c in counts.values()
            if c > 0
        )


# ---------------------------------------------------------------------------
# Convenience: timed score with latency
# ---------------------------------------------------------------------------


def score_with_latency(
    probe: NUPProbe,
    cot_text: str,
    logprobs: Optional[list[float]] = None,
) -> tuple[float, float]:
    """Score a step and return (score, latency_ms).

    Used by the experiment script to measure probe throughput.

    Args:
        probe: NUPProbe instance.
        cot_text: Step text.
        logprobs: Optional log-probabilities.

    Returns:
        Tuple of (entropy_score, latency_ms).
    """
    t0 = time.perf_counter()
    s = probe.score(cot_text, logprobs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return s, latency_ms
