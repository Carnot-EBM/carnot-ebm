"""PCIB Hallucination Probe — Predictive Coding + Information Bottleneck signals.

**What this is and why it exists:**
    arXiv 2601.15652 (PCIB, January 2026) combines two neuroscience-inspired ideas:

    1. Predictive Coding: the brain constantly predicts the next sensory input. When
       reality surprises the prediction, a "prediction error" (surprise) signal fires.
       Applied to LLMs, a token with high surprise (-log p(token|context)) is one
       the model didn't expect given the context — often a hallucinated entity.

    2. Information Bottleneck: compress a representation to only what's relevant for
       the output. When an LLM makes a claim that carries information NOT bottlenecked
       through the input context, that claim is likely fabricated.

    PCIB's paper achieves 0.8669 AUROC using a <1M parameter classifier trained on
    75× less data and running 1000× faster than LLM-judge baselines.

**What we're approximating here:**
    We don't have access to per-token logits from an external LLM at inference time.
    Instead, we implement two text-statistical proxies that capture the same
    information-theoretic intuitions:

    - Entity Uptake (entity_uptake): Measures what fraction of numeric/factual entities
      in the response are "novel" — not derivable from what came before in the context.
      In a hallucinated step, the model introduces numbers that didn't exist in the
      source. We detect this via token-novelty ratio on numerical tokens.

    - Falsifiability Score (falsifiability_score): Detects confident declarative claims
      ("Therefore X = 366") that cannot be verified from the visible computation trail.
      In an incorrect reasoning step, the stated conclusion often doesn't follow from
      the arithmetic shown above it. We detect this by cross-checking conclusion-marked
      sentences against visible intermediate values.

    Combined score = weighted average of both signals.

**Research context:**
    This is implemented as Tier 0f (PCIB energy probe) in Carnot's fast-path
    verification cascade. It operates entirely on text — no GPU needed, sub-1ms latency.
    Used as an advisory signal before the more expensive constraint-based tiers fire.

Spec: REQ-VERIFY-162, REQ-VERIFY-163
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# --- Number extraction helpers ---


def _extract_numbers(text: str) -> list[float]:
    """Return all numeric values found in text (integers and decimals).

    We use a broad pattern that captures LaTeX-wrapped numbers like \\( 80 \\)
    as well as plain numbers. This is intentionally liberal — false positives
    are acceptable; false negatives would miss real entity tokens.
    """
    # Strip LaTeX wrappers so \( 80 \) gives us 80
    stripped = re.sub(r"\\[()[\]]", " ", text)
    stripped = re.sub(r"\\\w+\{?", " ", stripped)  # remove \frac, \times etc.
    pattern = r"-?\d+(?:\.\d+)?"
    matches = re.findall(pattern, stripped)
    result = []
    for m in matches:
        try:
            result.append(float(m))
        except ValueError:
            pass
    return result


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?', or newlines."""
    parts = re.split(r"[.!?\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_conclusion_sentence(sentence: str) -> bool:
    """Return True if a sentence appears to be drawing a definitive conclusion.

    Conclusion sentences are the ones most likely to contain falsifiable claims:
    "Therefore, X = 260", "The answer is 366", "Total = 80", etc.
    """
    lowered = sentence.lower()
    conclusion_markers = (
        "therefore",
        "thus",
        "hence",
        "so the",
        "answer is",
        "result is",
        "total is",
        "total =",
        "sum is",
        "= answer",
        "in total",
        "altogether",
    )
    return any(m in lowered for m in conclusion_markers)


# --- Arithmetic consistency helper ---


def _values_reachable(target: float, source_values: Sequence[float], tol: float = 0.01) -> bool:
    """Return True if `target` can be produced by simple arithmetic on `source_values`.

    We check: identity (target in sources), addition of any two, multiplication of any
    two, subtraction of any two. This covers most single-step arithmetic conclusions.
    We don't do exhaustive search — speed > recall here (runs in <1ms for small N).
    """
    src = list(source_values)
    # Identity
    if any(abs(target - v) <= tol for v in src):
        return True
    # Pairwise ops
    for i, a in enumerate(src):
        for b in src[i:]:
            if abs(target - (a + b)) <= tol:
                return True
            if abs(target - (a * b)) <= tol:
                return True
            if abs(target - abs(a - b)) <= tol:
                return True
            # division guard
            if abs(b) > tol and abs(target - a / b) <= tol:
                return True
            if abs(a) > tol and abs(target - b / a) <= tol:
                return True
    return False


# --- PCIBProbe ---


class PCIBProbe:
    """Lightweight text-statistical probe implementing PCIB-inspired hallucination signals.

    This probe approximates the two key PCIB signals from arXiv 2601.15652 using
    only text statistics — no LLM logits required. It is designed to run in <1ms
    per step and require zero GPU resources, making it appropriate as a Tier 0
    fast-path pre-filter.

    The two signals:

    entity_uptake — "surprise of entity-focused tokens":
        Fraction of numeric entities in the response portion that are NOT present
        in the context portion. High value = model introduced unexpected numbers =
        higher hallucination risk.

    falsifiability_score — "detects confident claims that contradict source context":
        Fraction of conclusion-marked sentences whose numeric claim cannot be
        derived from the numbers visible earlier in the text. High value =
        the stated conclusion is arithmetically ungrounded = higher hallucination risk.

    Combined PCIB energy score: weighted average of both signals with equal weight.
    The combined score is in [0, 1]; higher = more likely hallucinating.

    Args:
        entity_weight: Weight for entity_uptake in combined score (default 0.5).
        falsifiability_weight: Weight for falsifiability_score in combined score (default 0.5).
        min_numbers_for_falsifiability: Minimum number of numeric entities required
            before the falsifiability check fires; below this threshold the score is 0.0
            (not enough arithmetic to check). Default: 2.
    """

    def __init__(
        self,
        entity_weight: float = 0.5,
        falsifiability_weight: float = 0.5,
        min_numbers_for_falsifiability: int = 2,
    ) -> None:
        self.entity_weight = entity_weight
        self.falsifiability_weight = falsifiability_weight
        self.min_numbers_for_falsifiability = min_numbers_for_falsifiability

    def compute_entity_uptake(self, response_text: str, context: str) -> float:
        """Compute the entity-uptake surprise signal.

        In the PCIB paper, entity uptake measures how surprising the entity-focused
        tokens in a response are relative to the prior distribution established by
        the context. High surprise = entities NOT in the context = hallucinated.

        Our approximation: treat the numbers in `context` as the "prior" and the
        numbers in `response_text` as the "posterior". Surprise = fraction of
        response numbers not seen in the context number set.

        Args:
            response_text: The model's reasoning step or output to evaluate.
            context: The source context (question or problem statement). When context
                is just an ID or empty, falls back to using the first half of
                response_text as context and the second half as response.

        Returns:
            float in [0, 1]. 0 = fully grounded (no novel entities), 1 = entirely novel.
        """
        # When context carries real content (more than just an ID), use it directly.
        # Otherwise split the response into premise / conclusion halves.
        if len(context.strip()) > 20:  # noqa: PLR2004 — magic number is intentional threshold
            ctx_numbers = set(_extract_numbers(context))
            resp_numbers = _extract_numbers(response_text)
        else:
            # Fallback: use the first half of the step as context
            midpoint = len(response_text) // 2
            ctx_numbers = set(_extract_numbers(response_text[:midpoint]))
            resp_numbers = _extract_numbers(response_text[midpoint:])

        if not resp_numbers:
            return 0.0  # No numeric entities to evaluate

        # Count how many response numbers are truly novel (not in context)
        novel = sum(
            1
            for n in resp_numbers
            if not any(abs(n - c) < 0.01 for c in ctx_numbers)  # noqa: PLR2004
        )
        return novel / len(resp_numbers)

    def compute_falsifiability_score(self, response_text: str, context: str) -> float:
        """Compute the falsifiability score — confident claims not anchored to context.

        In PCIB, the falsifiability score captures when a model states a definitive
        claim that cannot be verified from the source material. For math CoT steps,
        this manifests as a conclusion sentence ("Therefore X = 366") whose stated
        value is arithmetically inconsistent with the numbers visible in the step.

        Our approximation:
          1. Split into "premise sentences" (before the conclusion) and "conclusion
             sentences" (those matching conclusion marker patterns).
          2. Collect all numbers from premise sentences as the reachable set.
          3. For each conclusion sentence, check if its numbers can be derived from
             the premise numbers via single-step arithmetic.
          4. Falsifiability score = fraction of conclusion numbers NOT reachable.

        Args:
            response_text: The model's reasoning step to evaluate.
            context: Source context (used to extend the reachable value set).

        Returns:
            float in [0, 1]. 0 = all conclusions grounded, 1 = fully ungrounded.
        """
        sentences = _extract_sentences(response_text)
        if len(sentences) < 2:  # noqa: PLR2004
            return 0.0  # Too short to evaluate

        # Collect numbers from the source context as additional grounded values
        ctx_numbers = _extract_numbers(context) if len(context.strip()) > 20 else []  # noqa: PLR2004

        conclusion_sentences = [s for s in sentences if _is_conclusion_sentence(s)]
        premise_sentences = [s for s in sentences if not _is_conclusion_sentence(s)]

        premise_numbers = _extract_numbers(" ".join(premise_sentences)) + ctx_numbers

        if len(premise_numbers) < self.min_numbers_for_falsifiability:
            return 0.0  # Not enough arithmetic to check

        if not conclusion_sentences:
            return 0.0  # No definitive claims to falsify

        # Check each number in each conclusion sentence
        ungrounded_count = 0
        total_conclusion_numbers = 0
        for sent in conclusion_sentences:
            nums = _extract_numbers(sent)
            for n in nums:
                total_conclusion_numbers += 1
                if not _values_reachable(n, premise_numbers):
                    ungrounded_count += 1

        if total_conclusion_numbers == 0:
            return 0.0

        return ungrounded_count / total_conclusion_numbers

    def score(self, response_text: str, context: str) -> float:
        """Compute the combined PCIB energy score for a response.

        This is the primary interface for the verification pipeline. Higher scores
        indicate higher hallucination probability.

        The combined score is a weighted sum of entity_uptake and falsifiability_score.
        Both components are in [0, 1], so the combined score is also in [0, 1].

        Args:
            response_text: The model's response text to evaluate.
            context: The source context (question, problem statement, or prior text).

        Returns:
            float in [0, 1]. Threshold for Tier 0f: >= 0.5 flagged as likely hallucinated.
        """
        eu = self.compute_entity_uptake(response_text, context)
        fs = self.compute_falsifiability_score(response_text, context)
        return self.entity_weight * eu + self.falsifiability_weight * fs
