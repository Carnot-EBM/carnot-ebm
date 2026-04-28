"""SpilledEnergyDetector — training-free logit-spill hallucination probe (Tier 0b).

**What "spilled energy" means:**
    In thermodynamics, energy "spills" when it is not efficiently contained by a
    system's structure.  In LLM terms, a model's probability mass "spills" across
    many tokens when it is hallucinating — it is less certain, more diffuse.
    Without access to raw logits, this module approximates logit spill using
    token-level text statistics that correlate with distribution spread.

**The three spill signals:**
    1. Numerical novelty spill — fraction of numeric tokens in the conclusion that
       are absent from the premise/context.  A hallucinating step introduces new
       numbers without grounding them in what was given.

    2. Vocabulary entropy spill — normalized Shannon entropy of numeric tokens in
       the full response.  Correct steps use a small, internally consistent set
       of numbers; hallucinated steps introduce diverse, poorly-coordinated values.

    3. Arithmetic inconsistency spill — fraction of conclusion values that cannot
       be produced by single-step arithmetic on the values established in the
       premises.  This catches "magical" transitions where the answer appears
       without a visible computation trail.

    Combined spill = weighted sum of all three signals (equal weights by default).
    Score is in [0, 1]; higher = more energy spilled = more likely hallucinating.

**Why this achieves AUROC=1.0 on synthetic (Exp 949):**
    The synthetic corpus has very short (20-token) responses.  On such tiny inputs
    the numerical novelty signal alone is perfectly discriminative — a hallucinated
    response introduces entirely different numbers than the source context.
    On real CoT steps (FOVER corpus, 57 pairs), the probe's AUROC is lower (~0.65-0.75)
    because real LLM outputs contain legitimate paraphrasing and intermediate values
    not present in the question text.

Spec: REQ-TIER0-002, SCENARIO-TIER0-002
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Shared helpers (reused from pcib_probe pattern)
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    """Return all numeric values in text (integers and decimals).

    Strips LaTeX wrappers like \\( 80 \\) so we see 80, not the markup.
    Liberal by design — false positives are cheap; missing a real entity
    inflates the spill score unfairly.
    """
    stripped = re.sub(r"\\[()[\]]", " ", text)
    stripped = re.sub(r"\\\w+\{?", " ", stripped)
    matches = re.findall(r"-?\d+(?:\.\d+)?", stripped)
    result = []
    for m in matches:
        try:
            result.append(float(m))
        except ValueError:
            pass
    return result


def _split_premise_conclusion(text: str) -> tuple[str, str]:
    """Split a CoT step into (premise, conclusion) halves.

    Tries to find a natural split at conclusion markers ("therefore", "thus",
    "so the answer is", etc.).  Falls back to a 60/40 character split if no
    marker is found — the first 60% is the working-out, the last 40% is the
    stated answer.

    Why 60/40 rather than 50/50: CoT steps tend to be front-loaded with
    arithmetic and back-loaded with the stated result.  A 60/40 split gives
    the premise set more numbers to work with, reducing false-spill signals.
    """
    lowered = text.lower()
    for marker in ("therefore", "thus", "hence", "so the", "answer is", "result is", "total is"):
        idx = lowered.find(marker)
        if idx > 0:
            return text[:idx], text[idx:]
    # fallback split at 60%
    split = int(len(text) * 0.6)
    return text[:split], text[split:]


def _values_reachable(target: float, sources: Sequence[float], tol: float = 0.05) -> bool:
    """Return True if target can be produced by simple arithmetic on sources.

    Checks identity, addition, subtraction, multiplication, division.  Tolerance
    is 0.05 (5%) rather than 0.01 to handle floating-point approximations in
    text-extracted numbers (e.g. "33.3" for "100/3").

    Why this tolerance is intentionally loose:
        Text-extracted numbers lose precision (e.g. "0.33" for 1/3).  A tight
        tolerance would produce false spill signals for legitimately grounded values.
    """
    src = list(sources)
    if any(abs(target - v) <= tol * max(abs(target), 1.0) for v in src):
        return True
    for i, a in enumerate(src):
        for b in src[i:]:
            if abs(target - (a + b)) <= tol * max(abs(target), 1.0):
                return True
            if abs(target - abs(a - b)) <= tol * max(abs(target), 1.0):
                return True
            prod = a * b
            if abs(target - prod) <= tol * max(abs(target), abs(prod), 1.0):
                return True
            if abs(b) > 1e-9 and abs(target - a / b) <= tol * max(abs(target), 1.0):
                return True
            if abs(a) > 1e-9 and abs(target - b / a) <= tol * max(abs(target), 1.0):
                return True
    return False


# ---------------------------------------------------------------------------
# SpilledEnergyDetector — public API
# ---------------------------------------------------------------------------


class SpilledEnergyDetector:
    """Training-free probe that estimates logit spill from text statistics.

    The probe requires no training data and produces a deterministic score
    for any response text.  It is appropriate as a Tier 0b pre-filter before
    the more expensive trained probes.

    Design contract:
        - score() returns a float in [0, 1].
        - 0.0 = all output energy is perfectly contained by the input context.
        - 1.0 = all output energy spilled — nothing is grounded.
        - The empirical decision threshold on FOVER is ~0.38 (from Exp 949 synthetic).

    Args:
        novelty_weight: Weight of the numerical novelty signal (default 0.45).
        entropy_weight: Weight of the vocabulary entropy signal (default 0.25).
        consistency_weight: Weight of the arithmetic consistency signal (default 0.30).
    """

    def __init__(
        self,
        novelty_weight: float = 0.45,
        entropy_weight: float = 0.25,
        consistency_weight: float = 0.30,
    ) -> None:
        self.novelty_weight = novelty_weight
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight

    # ------------------------------------------------------------------
    # Signal 1: numerical novelty spill
    # ------------------------------------------------------------------

    def numerical_novelty_spill(self, response_text: str, context: str = "") -> float:
        """Fraction of numeric tokens in conclusion that are absent from premises/context.

        A hallucinating step introduces numbers not grounded in what was given.
        Correct steps use only values derived from the context or computed from
        values that were already established.

        Returns:
            float in [0, 1].  0 = all numbers in conclusion were in the premise.
        """
        premise, conclusion = _split_premise_conclusion(response_text)
        ctx_numbers = set(_extract_numbers(context)) if len(context.strip()) > 10 else set()
        premise_numbers = set(_extract_numbers(premise)) | ctx_numbers
        conclusion_numbers = _extract_numbers(conclusion)
        if not conclusion_numbers:
            return 0.0
        novel = sum(
            1
            for n in conclusion_numbers
            if not any(abs(n - p) <= 0.05 * max(abs(n), 1.0) for p in premise_numbers)
        )
        return novel / len(conclusion_numbers)

    # ------------------------------------------------------------------
    # Signal 2: vocabulary entropy spill
    # ------------------------------------------------------------------

    def vocabulary_entropy_spill(self, response_text: str) -> float:
        """Normalized entropy of the numeric token distribution.

        A correct step has a small, internally consistent vocabulary of numbers
        (low entropy).  A hallucinating step sprays diverse, unrelated values
        across the text (high entropy).

        Why normalize by log2(N+1): this maps the raw entropy onto [0, 1] so it
        is directly comparable to the other two signals.  Adding 1 avoids log(0)
        for single-number responses.

        Returns:
            float in [0, 1].  0 = single unique value (perfectly focused).
        """
        numbers = _extract_numbers(response_text)
        if len(numbers) < 2:
            return 0.0  # not enough numbers to compute meaningful entropy
        # Round to 2 decimal places to bin "close" values together, preventing
        # noise in text extraction from inflating the entropy.
        buckets: dict[float, int] = {}
        for n in numbers:
            key = round(n, 2)
            buckets[key] = buckets.get(key, 0) + 1
        total = sum(buckets.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in buckets.values() if c > 0)
        max_entropy = math.log2(len(buckets) + 1)  # +1 avoids log(0) for single-bucket case
        return min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Signal 3: arithmetic inconsistency spill
    # ------------------------------------------------------------------

    def arithmetic_inconsistency_spill(self, response_text: str, context: str = "") -> float:
        """Fraction of conclusion values not reachable by arithmetic on premises.

        This is the most direct measure of energy spill: the model stated a result
        that is not derivable from the values it showed in its working.  This
        manifests as a "magic" answer that appears without a computable path.

        Returns:
            float in [0, 1].  0 = all conclusions derivable.  1 = nothing is derivable.
        """
        premise, conclusion = _split_premise_conclusion(response_text)
        ctx_numbers = _extract_numbers(context) if len(context.strip()) > 10 else []
        premise_numbers = _extract_numbers(premise) + ctx_numbers
        conclusion_numbers = _extract_numbers(conclusion)
        if not conclusion_numbers or len(premise_numbers) < 1:
            return 0.0
        unreachable = sum(
            1 for n in conclusion_numbers if not _values_reachable(n, premise_numbers)
        )
        return unreachable / len(conclusion_numbers)

    # ------------------------------------------------------------------
    # Combined score (primary API)
    # ------------------------------------------------------------------

    def spill_score(self, response_text: str, context: str = "") -> float:
        """Compute the combined spilled-energy score for a response.

        This is the primary interface for the verification cascade.  Returns a
        score in [0, 1] where higher means more energy spilled (more likely
        hallucinating).

        Decision threshold: 0.372 (optimal on Exp 949 synthetic corpus).

        Args:
            response_text: The model's full response or reasoning step.
            context: The source question/prompt (empty string if unavailable).

        Returns:
            float in [0, 1].
        """
        s1 = self.numerical_novelty_spill(response_text, context)
        s2 = self.vocabulary_entropy_spill(response_text)
        s3 = self.arithmetic_inconsistency_spill(response_text, context)
        return self.novelty_weight * s1 + self.entropy_weight * s2 + self.consistency_weight * s3

    def score(self, response_text: str, context: str = "") -> float:
        """Alias for spill_score() — drop-in compatible with PCIBProbe.score()."""
        return self.spill_score(response_text, context)

    def is_violation(self, response_text: str, context: str = "", threshold: float = 0.372) -> bool:
        """Return True if the spill score exceeds the decision threshold.

        The default threshold (0.372) is the optimal value found in Exp 949 on
        the synthetic corpus.  Callers may override it after calibrating on
        their own labeled data.

        Args:
            response_text: Response text to evaluate.
            context: Source question/context.
            threshold: Decision boundary.  Default 0.372 from Exp 949.

        Returns:
            True if the response likely contains hallucinated content.
        """
        return self.spill_score(response_text, context) >= threshold
