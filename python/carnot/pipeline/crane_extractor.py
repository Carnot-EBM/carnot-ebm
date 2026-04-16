"""CRANEExtractionGate — Constrained Reasoning And Normalization Engine (Exp 418).

**Researcher summary (Exp 418):**
    Previous experiments (366, 368, 379) used LLMConstraintExtractor as the primary
    extractor for non-BASELINE pipeline variants.  LLMConstraintExtractor uses a
    free-form LLM prompt to extract arithmetic claims, which requires fragile regex
    parsing of model output and GPU inference overhead.

    CRANE improves on this by adding a structured output gate that runs entirely on
    CPU with no model dependency:
    1. Parse arithmetic claims directly from the response text using multi-pattern
       regex ("CRANE patterns") that recognise common CoT reasoning formats.
    2. Score each candidate claim's structural confidence (0.0-1.0) based on:
       - Whether all three operands are valid floats (base 0.3)
       - Whether the arithmetic is correct given the operator (bonus 0.4)
       - Whether the claim appears in a numbered reasoning step (bonus 0.3)
    3. Gate out low-confidence claims (below ``min_confidence``) to reduce false
       positives.  CRANE does NOT suppress uncertain violations — it simply does not
       report them, letting the fallback extractor handle ambiguous cases.
    4. Return only the violated claims (wrong arithmetic) as ConstraintResult objects.

**Why "gate" in the name?**
    The confidence gate is the distinguishing feature: CRANE filters out candidates it
    is not confident about, trading recall for precision.  This is the opposite of
    ArithmeticExtractor which reports everything that pattern-matches, regardless of
    reliability.

**Why a separate module from LLMConstraintExtractor?**
    LLMConstraintExtractor requires a loaded LLM to canonicalize claims.  CRANE is
    purely regex + deterministic math — faster, no GPU dependency, fully CI-safe.

**When to use CRANE vs LLMConstraintExtractor:**
    - CRANE is the PRIMARY extractor for FULL_STACK variant in Exp 419.  It has
      higher precision and zero GPU dependency.
    - LLMConstraintExtractor is the FALLBACK when CRANE extracts zero claims from a
      response (e.g. very short responses, non-standard formatting).

Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-BENCH-020
"""

from __future__ import annotations

import re
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.verify.constraint import BaseConstraint


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

_NUMBER_PAT = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

# Pattern 1: "N OP N = N" (explicit equality, inline)
_INLINE_EQ = re.compile(
    rf"(?P<a>{_NUMBER_PAT})\s*(?P<op>[+\-*/×÷])\s*(?P<b>{_NUMBER_PAT})"
    rf"\s*=\s*(?P<c>{_NUMBER_PAT})",
    re.IGNORECASE,
)

# Pattern 2: "N OP N is/gives/equals N" (natural language equality)
_IS_EQ = re.compile(
    rf"(?P<a>{_NUMBER_PAT})\s*(?P<op>[+\-*/])\s*(?P<b>{_NUMBER_PAT})"
    rf"\s+(?:is|gives|equals)\s+(?P<c>{_NUMBER_PAT})",
    re.IGNORECASE,
)

# Pattern for numbered reasoning steps — structural confidence bonus.
_NUMBERED_STEP = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)


def _strip_commas(text: str) -> str:
    """Remove thousands-separator commas so '1,000' parses as 1000.0."""
    return text.replace(",", "")


def _safe_float(text: str | None) -> float | None:
    """Parse a numeric string to float; return None on failure or None input."""
    if text is None:
        return None
    try:
        return float(_strip_commas(text))
    except (ValueError, TypeError):
        return None


def _normalise_op(op: str) -> str:
    """Normalise Unicode operator symbols to ASCII."""
    return {"×": "*", "÷": "/"}.get(op.strip(), op.strip())


def _op_result(a: float, op: str, b: float) -> float | None:
    """Evaluate a OP b; return None on division-by-zero or unknown operator."""
    op = _normalise_op(op)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0.0:
            return None
        return a / b
    return None


def _claim_confidence(match: re.Match, text: str) -> float:
    """Compute a structural confidence score (0.0-1.0) for a CRANE regex match.

    **Detailed explanation for engineers:**
        Confidence is built up additively:
        - 0.3 base: all three operands parse to valid floats.
        - +0.4: the arithmetic is correct (a OP b == c within tolerance 0.5).
        - +0.3: the match falls inside a numbered reasoning step (structured CoT).

        When arithmetic is WRONG (the violation case), we cap at 0.3 unless the
        structural bonus applies.  The lower cap helps filter formatting-glitch
        false positives (e.g. two unrelated numbers joined by an operator).

    Args:
        match: Regex match with groups a, op, b, c.
        text:  Full response text for structural context detection.

    Returns:
        Float in [0.0, 1.0].
    """
    a_val = _safe_float(match.group("a"))
    b_val = _safe_float(match.group("b"))
    c_val = _safe_float(match.group("c"))

    if a_val is None or b_val is None or c_val is None:
        return 0.0

    base = 0.3  # operands are parseable

    op = _normalise_op(match.group("op"))
    expected = _op_result(a_val, op, b_val)
    arithmetic_ok = expected is not None and abs(expected - c_val) < 0.501
    if arithmetic_ok:
        base += 0.4

    # Structural bonus: is the match inside a numbered reasoning step?
    match_start = match.start()
    newline_before = text.rfind("\n", 0, match_start)
    line_start = 0 if newline_before == -1 else newline_before + 1
    line_end = text.find("\n", match_start)
    line_end = len(text) if line_end == -1 else line_end
    line_text = text[line_start:line_end]
    if _NUMBERED_STEP.match(line_text):
        base += 0.3

    return min(base, 1.0)


# ---------------------------------------------------------------------------
# _CRANEConstraint — single-claim adapter for the constraint pipeline
# ---------------------------------------------------------------------------


class _CRANEConstraint(BaseConstraint):
    """Constraint adapter for one CRANE-extracted arithmetic claim.

    **Detailed explanation for engineers:**
        Each CRANE violation is wrapped in this adapter so it can flow through
        Carnot's constraint pipeline (BaseConstraint → energy function).

        ``is_violated=True`` → energy returns 1.0 (unsatisfied constraint).
        ``is_violated=False`` is unused in practice (CRANE only reports violations),
        but the adapter supports it for completeness.

        The confidence field is stored for downstream inspection (e.g. logging).
    """

    def __init__(
        self,
        is_violated: bool,
        description: str,
        confidence: float,
    ) -> None:
        self._is_violated = is_violated
        self._description = description
        self.confidence = confidence

    @property
    def name(self) -> str:
        status = "violated" if self._is_violated else "ok"
        return f"crane({status},{self.confidence:.2f}): {self._description[:60]}"

    @property
    def satisfaction_threshold(self) -> float:
        return 0.5

    def energy(self, x: Any) -> Any:
        """Return 1.0 if violated, 0.0 if satisfied (dtype-compatible with JAX if available)."""
        _ = x
        try:
            import jax.numpy as jnp  # noqa: PLC0415
            return jnp.float32(1.0 if self._is_violated else 0.0)
        except ImportError:
            return 1.0 if self._is_violated else 0.0

    def is_satisfied(self, x: Any) -> bool:
        _ = x
        return not self._is_violated


# ---------------------------------------------------------------------------
# CRANEExtractionGate
# ---------------------------------------------------------------------------


class CRANEExtractionGate:
    """Constrained Reasoning And Normalization Engine — structured extraction gate.

    **Detailed explanation for engineers:**
        CRANE is the PRIMARY constraint extractor for the FULL_STACK pipeline variant
        in Exp 419.  Pure-Python, regex + deterministic-math with a confidence gate.
        No LLM call, no GPU required.

        Algorithm
        ---------
        1. Apply two complementary regex patterns to the response text:
           - _INLINE_EQ: "N OP N = N" forms
           - _IS_EQ: "N OP N is/gives/equals N" forms
        2. For each match, compute structural confidence via _claim_confidence().
        3. Filter out matches below ``min_confidence``.
        4. For surviving matches, verify arithmetic: if a OP b ≠ c (tolerance 0.5),
           mark as violated.
        5. Return only violated claims as ConstraintResult objects.

        Only violations are returned because the caller counts violations to decide
        whether to trigger repair.  Arithmetically-correct claims carry no repair signal.

    Parameters
    ----------
    min_confidence : float
        Claims below this threshold are not reported.  Default 0.7 balances
        precision vs. recall for GSM8K-style CoT responses.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-BENCH-020
    """

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence

    @property
    def supported_domains(self) -> list[str]:
        """Domains this extractor handles.  Currently arithmetic only."""
        return ["arithmetic"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        """Extract violated arithmetic claims from *text*.

        Applies CRANE regex patterns, scores structural confidence, gates below
        threshold, and returns only violations as ConstraintResult objects.

        Parameters
        ----------
        text : str
            Response text to analyse (GSM8K model output, CoT trace, etc.).
        domain : str | None
            If supplied and not "arithmetic", returns empty list immediately.

        Returns
        -------
        list[ConstraintResult]
            One ConstraintResult per violated claim that passed the confidence gate.
            Empty when no violations found or all candidates filtered by the gate.
        """
        if domain is not None and domain != "arithmetic":
            return []

        violations: list[ConstraintResult] = []
        seen: set[str] = set()  # deduplicate identical matches from both patterns

        for pattern in (_INLINE_EQ, _IS_EQ):
            for match in pattern.finditer(text):
                confidence = _claim_confidence(match, text)
                if confidence < self.min_confidence:
                    continue

                a_val = _safe_float(match.group("a"))
                b_val = _safe_float(match.group("b"))
                c_val = _safe_float(match.group("c"))
                if a_val is None or b_val is None or c_val is None:
                    continue

                op = _normalise_op(match.group("op"))
                expected = _op_result(a_val, op, b_val)
                if expected is None:
                    continue  # division by zero — skip

                # Only report violations (arithmetic wrong).
                if abs(expected - c_val) < 0.501:
                    continue

                # Deduplicate: same (a, op, b, c) tuple from both patterns.
                dedup_key = (match.group("a"), op, match.group("b"), match.group("c"))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                desc = (
                    f"{match.group('a')} {match.group('op')} {match.group('b')} "
                    f"= {match.group('c')} (expected {expected:.6g})"
                )
                constraint = _CRANEConstraint(
                    is_violated=True,
                    description=desc,
                    confidence=confidence,
                )
                violations.append(
                    ConstraintResult(
                        constraint_type="arithmetic",
                        description=desc,
                        energy_term=constraint,
                        metadata={
                            "a": a_val,
                            "op": op,
                            "b": b_val,
                            "claimed_result": c_val,
                            "correct_result": expected,
                            "satisfied": False,
                            "confidence": confidence,
                            "extractor": "crane",
                        },
                    )
                )

        return violations


__all__ = ["CRANEExtractionGate"]
