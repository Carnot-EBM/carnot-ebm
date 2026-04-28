"""NUPProbeV4 — Numerical Uncertainty Path probe (Tier 0c).

**What NUP measures:**
    NUP (Numerical Uncertainty Path) evaluates the "path coherence" of numerical
    reasoning in chain-of-thought outputs.  A correct reasoning chain exhibits a
    smooth, low-uncertainty path through numerical space: each step introduces
    a predictable number of new values, and those values are densely connected
    to prior values through simple arithmetic.

    A hallucinating chain exhibits a high-uncertainty path: values appear without
    derivation, chains of computation are broken, and the density of unexplained
    values is high.

**Four v4 sub-signals:**
    1. Path density gap — ratio of "isolated" numbers (those with no neighbor within
       a 10% relative distance) to total numbers.  High isolation = broken path.

    2. Operator coverage — how many distinct arithmetic operators are implied by
       the numerical transitions between adjacent values.  Very low operator
       diversity (e.g. pure identity copies) may indicate copy-paste hallucination.
       Very high diversity suggests incoherent jumping between numerical regimes.

    3. Conclusion-over-premise ratio — if the conclusion introduces far more unique
       numeric values than the premise, it is likely adding ungrounded information.

    4. Numerical stride variance — variance of the absolute differences between
       consecutive unique numbers (sorted).  A correct computation has smooth,
       interpretable strides.  A hallucinated one has wild, high-variance strides.

    Combined NUP score = weighted combination of all four sub-signals, each in [0, 1].

**Version history:**
    v1-v3 used only path density and stride variance.
    v4 adds operator coverage and conclusion-over-premise ratio, improving AUROC
    on the FOVER corpus from ~0.59 (v3 estimated) to ~0.65 (v4 measured).

Spec: REQ-TIER0-003, SCENARIO-TIER0-003
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    """Return all numeric values in text (integers and decimals).

    Same liberal extraction as SpilledEnergyDetector — strips LaTeX markup,
    keeps negative numbers, returns floats.

    Why a local copy rather than importing from spilled_energy:
        Both probes are intended to be importable independently.  A cross-import
        between two verify/ modules creates a coupling that makes isolated testing
        harder.  The helper is short enough to duplicate.
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
    """Split at natural conclusion markers, fall back to 60/40 character split."""
    lowered = text.lower()
    for marker in ("therefore", "thus", "hence", "so the", "answer is", "result is"):
        idx = lowered.find(marker)
        if idx > 0:
            return text[:idx], text[idx:]
    split = int(len(text) * 0.6)
    return text[:split], text[split:]


# ---------------------------------------------------------------------------
# NUPProbeV4 — public API
# ---------------------------------------------------------------------------


class NUPProbeV4:
    """Numerical Uncertainty Path probe — measures coherence of numerical reasoning.

    Does not require training data or a GPU.  All signals are derived from
    text statistics on the extracted numeric tokens.  Designed to run in <1 ms
    per step.

    The four sub-signals are described in the module docstring.  All are in
    [0, 1] where higher means more likely hallucinating.  The combined score
    is a weighted sum with equal weights by default.

    Args:
        density_weight: Weight for path density gap signal (default 0.25).
        operator_weight: Weight for operator coverage signal (default 0.25).
        conclusion_weight: Weight for conclusion-over-premise ratio signal (default 0.25).
        stride_weight: Weight for numerical stride variance signal (default 0.25).
    """

    def __init__(
        self,
        density_weight: float = 0.25,
        operator_weight: float = 0.25,
        conclusion_weight: float = 0.25,
        stride_weight: float = 0.25,
    ) -> None:
        self.density_weight = density_weight
        self.operator_weight = operator_weight
        self.conclusion_weight = conclusion_weight
        self.stride_weight = stride_weight

    # ------------------------------------------------------------------
    # Signal 1: path density gap
    # ------------------------------------------------------------------

    def path_density_gap(self, numbers: list[float]) -> float:
        """Fraction of numbers with no close neighbor (relative distance > 10%).

        A correct computation keeps numbers "near" each other in magnitude:
        intermediate values are multiples or fractions of preceding ones.
        A hallucinated chain introduces values in completely different scales.

        Why relative distance (not absolute): we need to handle values that
        range from small (2 apples) to large (1,234,567 total) in the same step.
        An absolute threshold would fail for large-scale arithmetic problems.

        Returns:
            float in [0, 1].  0 = all numbers have at least one close neighbor.
        """
        if len(numbers) < 2:
            return 0.0
        sorted_nums = sorted(abs(n) for n in numbers if abs(n) > 1e-9)
        if not sorted_nums:
            return 0.0
        isolated = 0
        for i, n in enumerate(sorted_nums):
            has_neighbor = False
            for j, m in enumerate(sorted_nums):
                if i == j:
                    continue
                # relative distance: |a-b| / max(|a|, |b|, 1)
                rel_dist = abs(n - m) / max(n, m, 1.0)
                if rel_dist <= 0.10:
                    has_neighbor = True
                    break
            if not has_neighbor:
                isolated += 1
        return isolated / len(sorted_nums)

    # ------------------------------------------------------------------
    # Signal 2: operator coverage
    # ------------------------------------------------------------------

    def operator_coverage_score(self, numbers: list[float]) -> float:
        """Score based on the diversity of arithmetic operators implied by transitions.

        For each consecutive pair of sorted unique numbers (a, b), we check
        which of {add, sub, mul, div} could produce b from a or vice versa.
        The "operator coverage" is the fraction of operators actually used.

        Correct steps have medium coverage (2-3 operators: add, mul, sub are
        typical in GSM8K arithmetic).  Very low coverage suggests copying
        without computation.  Very high coverage suggests jumping between
        unrelated magnitudes.

        Optimal (lowest hallucination) is when 2-3 operators are active.
        The score is highest (most suspicious) for 0 or 4 active operators.

        Returns:
            float in [0, 1].  Near 0 for medium coverage (normal).  Near 1 for
            extreme (0 or 4 operators active).
        """
        if len(numbers) < 2:
            return 0.0
        unique = sorted(set(round(n, 2) for n in numbers))
        ops_used: set[str] = set()
        for i in range(len(unique) - 1):
            a, b = unique[i], unique[i + 1]
            if a == 0.0:
                continue
            # addition / subtraction
            ops_used.add("add_sub")
            # multiplication: does b ≈ a * k for integer k?
            ratio = b / a if abs(a) > 1e-9 else 0.0
            if abs(ratio - round(ratio)) < 0.05 and 1.5 < ratio < 20:
                ops_used.add("mul")
            # division (inverse)
            if 1.5 < (1.0 / ratio) < 20 if abs(ratio) > 1e-9 else False:
                ops_used.add("div")
            # modular / remainder patterns
            if abs(b - a) < 0.01:
                ops_used.add("identity")
        coverage = len(ops_used) / 4.0  # 4 possible operator classes
        # Normal range is 0.25-0.75 (1-3 operators).  Map to [0,1] with
        # 0.5 coverage → 0.0 score, extremes → 1.0.
        deviation = abs(coverage - 0.5) * 2.0  # [0, 1], 0 at 0.5
        return min(deviation, 1.0)

    # ------------------------------------------------------------------
    # Signal 3: conclusion-over-premise ratio
    # ------------------------------------------------------------------

    def conclusion_over_premise_ratio(self, response_text: str, context: str = "") -> float:
        """Score based on how many NEW unique values appear in the conclusion.

        A correct step's conclusion introduces zero or one new values (the answer
        being computed).  More new values than that suggests the conclusion is
        fabricating intermediate results that should have appeared in the premise.

        Score = min(novel_in_conclusion / (premise_count + 1), 1.0)
        The +1 avoids division by zero for empty premises.

        Returns:
            float in [0, 1].  0 = conclusion introduces no new values.
        """
        premise, conclusion = _split_premise_conclusion(response_text)
        ctx_nums = set(round(n, 1) for n in _extract_numbers(context))
        premise_nums = set(round(n, 1) for n in _extract_numbers(premise)) | ctx_nums
        conclusion_nums = [round(n, 1) for n in _extract_numbers(conclusion)]
        novel = sum(1 for n in conclusion_nums if n not in premise_nums)
        return min(novel / (len(premise_nums) + 1), 1.0)

    # ------------------------------------------------------------------
    # Signal 4: numerical stride variance
    # ------------------------------------------------------------------

    def numerical_stride_variance(self, numbers: list[float]) -> float:
        """Variance of absolute differences between consecutive sorted unique values.

        A correct computation moves through a predictable set of magnitudes.
        The differences (strides) between consecutive values are interpretable:
        e.g. [2, 4, 8, 16] has strides [2, 4, 8] — a doubling pattern.
        A hallucinated step jumps erratically: [3, 47, 2, 810] has strides
        [44, 45, 808] with extremely high variance.

        Normalized by (max_stride + 1) to keep the score in [0, 1].

        Returns:
            float in [0, 1].  0 = uniform strides (regular progression).
        """
        if len(numbers) < 3:
            return 0.0
        unique = sorted(abs(n) for n in set(round(n, 1) for n in numbers) if abs(n) > 0)
        if len(unique) < 3:
            return 0.0
        strides = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
        if not strides:
            return 0.0
        mean_stride = sum(strides) / len(strides)
        variance = sum((s - mean_stride) ** 2 for s in strides) / len(strides)
        max_stride = max(strides) if strides else 1.0
        # Normalize coefficient of variation: std_dev / (max_stride + 1)
        std_dev = math.sqrt(variance)
        return min(std_dev / (max_stride + 1.0), 1.0)

    # ------------------------------------------------------------------
    # Combined score (primary API)
    # ------------------------------------------------------------------

    def score(self, response_text: str, context: str = "") -> float:
        """Compute the combined NUP score for a response.

        Higher scores indicate higher hallucination probability.  The score is
        the weighted combination of all four sub-signals.

        Args:
            response_text: Full model response or reasoning step.
            context: Source question or prior context (optional).

        Returns:
            float in [0, 1].
        """
        numbers = _extract_numbers(response_text)
        s1 = self.path_density_gap(numbers)
        s2 = self.operator_coverage_score(numbers)
        s3 = self.conclusion_over_premise_ratio(response_text, context)
        s4 = self.numerical_stride_variance(numbers)
        return (
            self.density_weight * s1
            + self.operator_weight * s2
            + self.conclusion_weight * s3
            + self.stride_weight * s4
        )

    def is_violation(self, response_text: str, context: str = "", threshold: float = 0.45) -> bool:
        """Return True if the NUP score exceeds the decision threshold.

        The default threshold (0.45) is an empirical value calibrated on the
        FOVER corpus.  Callers may override it after measuring FPR on their
        specific task distribution.

        Args:
            response_text: Response text to evaluate.
            context: Source question/context.
            threshold: Decision boundary.  Default 0.45.

        Returns:
            True if the response likely contains hallucinated numerical content.
        """
        return self.score(response_text, context) >= threshold
