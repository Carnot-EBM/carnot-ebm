"""Confidence-weighted violation filtering for arithmetic extractors.

**Why confidence filtering matters (root cause of low extraction recall):**

    Exp 554 revealed that VeriCoT and VPRM find zero violations on 25 real
    IT model responses — not because there are no errors, but because:

    1. VPRM's regex patterns require specific prose syntax ('A plus B gives C').
       IT models write 'we compute 47 + 28 = 75' (equation style).  Zero matches.

    2. VeriCoT's Z3 path requires well-formed FOL premises from an LLM extraction
       step.  When the LLM cannot extract premises, violations cannot be detected.

    The hypothesis tested here: even when a base extractor does produce violations,
    some will be false positives arising from:
        - Approximate or hedged statements ('approximately 75', 'roughly 30')
        - Intermediate computation steps (not final answers, inherently imprecise)
        - Format mismatches (the result is correct but the pattern matched wrong tokens)

    Confidence-weighted filtering attaches a score (0.0-1.0) to each violation
    and only passes violations above a configurable threshold to the repair loop.

    This separates 'definitive arithmetic error detected' (high confidence)
    from 'something looks off but it might be a prose ambiguity' (low confidence).

**How confidence is scored (heuristics, not neural):**

    The scorer uses lightweight regex heuristics to classify violation text.
    This is intentional: adding a neural judge would re-introduce the reward-hacking
    vulnerability that VPRM was designed to eliminate.

    Heuristic hierarchy (highest priority wins):
        1. Equation-style error ('47 + 28 = 76', >5% magnitude off) → 0.95
        2. Explicit final answer error ('the answer is X' with clear mismatch) → 0.90
        3. Approximate/hedged statement ('approximately', 'roughly', 'about') → 0.20
        4. Intermediate computation step marker ('step N', 'first we') → 0.40
        5. Default (no pattern matched) → 0.60

Spec: REQ-EXTRACT-031, REQ-EXTRACT-032,
      SCENARIO-EXTRACT-058, SCENARIO-EXTRACT-059, SCENARIO-EXTRACT-060
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# ViolationConfidence — one violation with its confidence score
# ---------------------------------------------------------------------------


@dataclass
class ViolationConfidence:
    """A single detected violation paired with a confidence score.

    Fields
    ------
    violation_text : str
        The natural-language text (or structured repr) of the violation,
        as returned by the base extractor.  Used for diagnostic output.
    confidence_score : float
        0.0 = certainly a false positive (approximate statement, hedged phrasing)
        1.0 = certainly a true violation (definitive arithmetic error)
        In practice, scores fall in four bands:
            [0.0, 0.25) — almost certainly noise (approximate phrases)
            [0.25, 0.55) — likely intermediate computation artifact
            [0.55, 0.75) — ambiguous; depends on threshold policy
            [0.75, 1.0] — high-confidence arithmetic error
    violation_type : str
        Short classifier label from the scorer: 'equation_error', 'final_answer_error',
        'approximate', 'intermediate', or 'default'.
    is_definitive : bool
        True iff confidence_score >= 0.80.  Convenience flag for downstream logic
        that wants a binary 'take action / do not take action' decision without
        tuning the threshold themselves.

    Spec: REQ-EXTRACT-031, SCENARIO-EXTRACT-058
    """

    violation_text: str
    confidence_score: float
    violation_type: str
    is_definitive: bool = field(init=False)

    def __post_init__(self) -> None:
        """Derive is_definitive from confidence_score after construction."""
        self.is_definitive = self.confidence_score >= 0.80


# ---------------------------------------------------------------------------
# Protocol: base extractor interface
# ---------------------------------------------------------------------------


class BaseExtractor(Protocol):
    """Minimal protocol any base extractor must satisfy for confidence wrapping.

    Any extractor that has a detect_violations(text) method returning a list
    satisfies this protocol.  Compatible with VPRMArithmeticVerifier and
    VeriCoTStepValidator without modification.
    """

    def detect_violations(self, text: str) -> list[Any]:
        """Detect violations in text; return empty list when none found."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Confidence heuristics — regex patterns for each confidence band
# ---------------------------------------------------------------------------

# 'approximately', 'roughly', 'about', 'around' → low confidence
_APPROXIMATE_RE = re.compile(
    r"\b(?:approximately|roughly|about|around|nearly|~)\b",
    re.IGNORECASE,
)

# 'step N', 'first we', 'next we', 'then we' → intermediate step marker
_INTERMEDIATE_RE = re.compile(
    r"\b(?:step\s+\d+|first\s+we|next\s+we|then\s+we|initially|to\s+begin)\b",
    re.IGNORECASE,
)

# 'the answer is', 'the result is', 'the total is' → final answer claim
_FINAL_ANSWER_RE = re.compile(
    r"\b(?:the\s+answer\s+is|the\s+result\s+is|the\s+total\s+is|therefore|thus,?)\b",
    re.IGNORECASE,
)

# Equation-style 'A + B = C' or 'A - B = C', etc. (catches equation-format errors)
_EQUATION_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
)

# 5% relative error threshold for calling an equation-style error "definitive"
# Any equation mismatch beyond floating-point noise is a definitive arithmetic error.
# We use 1e-6 relative tolerance (same as VPRM's _FLOAT_TOL) to forgive decimal rounding
# while catching any integer-level mistake ('47 + 28 = 76' is off by 1, clearly wrong).
_EQUATION_FLOAT_TOL = 1e-6


def _score_equation_error(violation_text: str) -> float | None:
    """Return 0.95 if the text contains a clearly wrong equation (any mismatch beyond rounding).

    Returns None if no equation pattern is found or the equation is correct within tolerance.

    Why not a percentage threshold?
        A 5% threshold would miss off-by-one integer errors like '47 + 28 = 76'
        (only 1.3% relative error) while still being a definitive arithmetic mistake.
        Using 1e-6 relative tolerance (same as VPRM) catches any real error while
        forgiving IEEE 754 floating-point rounding artifacts.
    """
    m = _EQUATION_RE.search(violation_text)
    if not m:
        return None
    a = float(m.group(1))
    op = m.group(2)
    b = float(m.group(3))
    stated = float(m.group(4))

    try:
        if op == "+":
            computed = a + b
        elif op == "-":
            computed = a - b
        elif op == "*":
            computed = a * b
        elif op == "/" and abs(b) > 1e-12:
            computed = a / b
        else:
            return None
    except ZeroDivisionError:
        return None

    abs_diff = abs(computed - stated)
    if abs_diff < 1e-9:
        return None  # exact match
    denom = max(abs(computed), abs(stated), 1.0)
    if abs_diff / denom < _EQUATION_FLOAT_TOL:
        return None  # within floating-point tolerance
    return 0.95  # equation is wrong — definitive arithmetic error


def score_violation(violation_text: str) -> tuple[float, str]:
    """Assign a confidence score to a violation string using lightweight heuristics.

    Parameters
    ----------
    violation_text : str
        The string representation of one violation (e.g., the step text that
        triggered the rule, or a structured repr from the base extractor).

    Returns
    -------
    (confidence_score, violation_type) : tuple[float, str]
        confidence_score: float in [0.0, 1.0]
        violation_type: short label describing which heuristic fired

    Heuristic precedence (first match wins):
        1. equation_error  (0.95) — clear wrong equation in the text
        2. final_answer    (0.90) — 'the answer is', 'therefore', etc.
        3. approximate     (0.20) — 'approximately', 'roughly', etc.
        4. intermediate    (0.40) — step markers ('step 1', 'first we')
        5. default         (0.60) — no heuristic matched

    Spec: REQ-EXTRACT-031, SCENARIO-EXTRACT-058
    """
    eq_score = _score_equation_error(violation_text)
    if eq_score is not None:
        return eq_score, "equation_error"

    if _FINAL_ANSWER_RE.search(violation_text):
        return 0.90, "final_answer_error"

    if _APPROXIMATE_RE.search(violation_text):
        return 0.20, "approximate"

    if _INTERMEDIATE_RE.search(violation_text):
        return 0.40, "intermediate"

    return 0.60, "default"


# ---------------------------------------------------------------------------
# ConfidenceWeightedExtractor — wraps any base extractor with scoring
# ---------------------------------------------------------------------------


class ConfidenceWeightedExtractor:
    """Wrap a base extractor and attach confidence scores to each violation.

    Usage
    -----
    ::

        base = VPRMArithmeticVerifier()
        extractor = ConfidenceWeightedExtractor(base, confidence_threshold=0.7)
        all_violations = extractor.extract(response_text)
        high_conf = extractor.above_threshold(all_violations)
        should_repair = len(high_conf) > 0

    Parameters
    ----------
    base_extractor : BaseExtractor
        Any extractor with detect_violations(text) -> list[Any].
    confidence_threshold : float
        Violations with confidence_score >= this threshold are returned by
        above_threshold().  Default 0.7 is a balanced starting point:
        - Above 0.7: includes definitive arithmetic errors and final-answer mismatches
        - Excludes: approximate statements (0.20) and intermediate steps (0.40)

    Spec: REQ-EXTRACT-031, REQ-EXTRACT-032,
          SCENARIO-EXTRACT-058, SCENARIO-EXTRACT-059, SCENARIO-EXTRACT-060
    """

    def __init__(
        self, base_extractor: BaseExtractor, confidence_threshold: float = 0.7
    ) -> None:
        self.base_extractor = base_extractor
        self.confidence_threshold = confidence_threshold

    def extract(self, response: str) -> list[ViolationConfidence]:
        """Run the base extractor and score each violation it finds.

        Parameters
        ----------
        response : str
            Full response text to analyze (CoT trace or prose answer).

        Returns
        -------
        list[ViolationConfidence]
            One ViolationConfidence per violation found by the base extractor.
            Empty if the base extractor found no violations.

        Why convert to string before scoring?
            The base extractor may return typed objects (RuleVerdict, StepVerdict).
            str() gives us a normalized text representation that the scoring
            heuristics can operate on uniformly, regardless of the extractor type.

        Spec: REQ-EXTRACT-031, SCENARIO-EXTRACT-058
        """
        raw_violations = self.base_extractor.detect_violations(response)
        result: list[ViolationConfidence] = []
        for raw in raw_violations:
            vtext = str(raw)
            score, vtype = score_violation(vtext)
            result.append(
                ViolationConfidence(
                    violation_text=vtext,
                    confidence_score=score,
                    violation_type=vtype,
                )
            )
        return result

    def above_threshold(
        self, violations: list[ViolationConfidence]
    ) -> list[ViolationConfidence]:
        """Return only violations whose confidence_score >= confidence_threshold.

        This is the gate that prevents low-confidence noise from triggering repair.
        Only violations that pass this gate should be sent to the repair loop.

        Parameters
        ----------
        violations : list[ViolationConfidence]
            Output from extract().

        Returns
        -------
        list[ViolationConfidence]
            Subset where confidence_score >= self.confidence_threshold.

        Spec: REQ-EXTRACT-031, SCENARIO-EXTRACT-059
        """
        return [v for v in violations if v.confidence_score >= self.confidence_threshold]
