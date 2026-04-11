"""Memory-driven constraint generation: Tier 2 → constraint ADDITION (Exp 141).

**Researcher summary:**
    Exp 134 showed that reweighting existing constraints produced no measurable
    improvement (fixed=0.53, adaptive=0.55). The root cause: you cannot close
    gaps caused by MISSING constraint types by adjusting weights of constraints
    that already exist. This module fixes that: when ConstraintMemory has seen
    a pattern 3+ times in a domain, ConstraintGenerator ADDs new constraints
    of the appropriate type to the extraction results, rather than just
    reweighting what was already there.

    Three new constraint types are generated from memory patterns:
    - "arithmetic_carry" → CarryChainConstraint v2: catches multi-step carry
      propagation errors in addition (e.g., 99 + 1, 999 + 11), subtraction
      borrow chains (1000 - 1), digit-count violations, and negative-result
      errors in B > A subtractions.
    - "comparison_boundary" → BoundConstraint: catches numeric inequality
      claims (X < Y, X >= 0) embedded in prose that LogicExtractor ignores
      because it only handles "if/then" and "but not" patterns.
    - "negation_scope" → NegationConstraint v2: catches "X is not Y",
      "not all A are B", "no A are B" patterns AND checks for violations
      (positive assertion of the negated claim elsewhere in the text).

**Detailed explanation for engineers:**
    How memory patterns map to new constraints:
        ConstraintMemory stores (domain, error_type) frequency counts.
        The error_type is the constraint_type of violations that were
        caught -- e.g., "arithmetic_carry" means the pipeline detected a
        carry-chain violation. When that pattern is seen 3+ times for a
        domain, ConstraintGenerator applies the corresponding extractor to
        the text being verified.

    Why generate rather than always extract?
        We want to be conservative: the three new extractors are more
        aggressive and produce more false positives than the base extractors.
        By gating them on memory frequency, we only enable them when there
        is accumulated evidence they are useful for this domain.

    Integration with AutoExtractor:
        AutoExtractor.extract() accepts an optional memory= parameter.
        If provided, it instantiates a ConstraintGenerator and calls
        generate(), then merges the results with the static extraction
        output. Deduplication is by constraint_type -- if ArithmeticExtractor
        already returned an "arithmetic" constraint, we do not add a second
        "arithmetic" constraint from generation. But we DO add a new
        "arithmetic_carry" constraint type alongside it.

    Backward compatibility:
        AutoExtractor.extract(text, domain=None) with no memory= argument
        produces identical output to the pre-Exp-141 version. The memory=
        parameter defaults to None and is a purely additive code path.

Spec: REQ-LEARN-003, REQ-LEARN-004, SCENARIO-LEARN-003
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.memory import PATTERN_THRESHOLD, ConstraintMemory


# ---------------------------------------------------------------------------
# Pattern → error_type constants
# These string constants are the error_type keys stored in ConstraintMemory
# that trigger each of the three new constraint generators.
# ---------------------------------------------------------------------------

#: Memory error_type that triggers carry-chain constraint generation.
PATTERN_ARITHMETIC_CARRY: str = "arithmetic_carry"

#: Memory error_type that triggers comparison boundary constraint generation.
PATTERN_COMPARISON_BOUNDARY: str = "comparison_boundary"

#: Memory error_type that triggers negation scope constraint generation.
PATTERN_NEGATION_SCOPE: str = "negation_scope"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_carries(a: int, b: int) -> int:
    """Count carry operations in the addition a + b.

    **Detailed explanation for engineers:**
        Simulates column-by-column addition from least significant digit.
        Each time a digit column sums to >= 10, a carry propagates to the
        next column -- this is the error source in multi-step problems like
        99 + 1 = 100 (two carries: units column carries to tens, tens
        column carries to hundreds). We count these propagation events.

    Args:
        a: First non-negative integer operand.
        b: Second non-negative integer operand.

    Returns:
        Number of carry operations (0 means no carries, 1 means one column
        overflowed, 2+ means cascading carries).
    """
    # Work with absolute values; sign does not affect carry count.
    a, b = abs(a), abs(b)
    count = 0
    carry = 0
    # Process until both operands and any pending carry are exhausted.
    while a > 0 or b > 0 or carry > 0:
        digit_a = a % 10
        digit_b = b % 10
        total = digit_a + digit_b + carry
        carry = total // 10
        if carry:
            count += 1
        a //= 10
        b //= 10
    return count


def _count_borrows(a: int, b: int) -> int:
    """Count borrow operations in the subtraction |a| - |b|.

    **Detailed explanation for engineers:**
        Simulates column-by-column subtraction from least significant digit,
        operating on absolute values. When the digit of the minuend is smaller
        than the digit of the subtrahend plus any pending borrow, the algorithm
        borrows from the next column -- incrementing the count. Cascade borrows
        occur in cases like 1000 - 1 = 999 (three borrows: units, tens,
        hundreds all need to borrow from the next column in turn).

        If |a| < |b| we swap them so the algorithm always subtracts the smaller
        from the larger; the negative-result detection is handled separately
        by the caller (CarryChainConstraint) via the sign check.

    Args:
        a: Minuend (may be negative; absolute value is used).
        b: Subtrahend (may be negative; absolute value is used).

    Returns:
        Number of borrow operations (0 = no borrows, 2+ = cascade borrows).
    """
    a, b = abs(a), abs(b)
    # Always subtract smaller from larger for borrow-counting purposes.
    if a < b:
        a, b = b, a
    count = 0
    borrow = 0
    while b > 0 or borrow > 0:
        digit_a = a % 10
        digit_b = b % 10
        if digit_a < digit_b + borrow:
            count += 1
            borrow = 1
        else:
            borrow = 0
        a //= 10
        b //= 10
    return count


def _digit_count(n: int) -> int:
    """Return the number of decimal digits in the absolute value of n.

    **Detailed explanation for engineers:**
        Used by CarryChainConstraint to check whether a claimed result has
        a plausible digit count. The addition A + B can produce at most
        max(digits(A), digits(B)) + 1 digits -- if the LLM writes a result
        with far more digits, that is a carry-propagation error.

    Args:
        n: Any integer (sign ignored).

    Returns:
        Number of digits (e.g., _digit_count(0) = 1, _digit_count(999) = 3).
    """
    return len(str(abs(n))) if n != 0 else 1


def _eval_comparison(left: float, op: str, right: float) -> bool:
    """Evaluate a numeric comparison operator.

    Args:
        left: Left-hand side value.
        op: One of "<", "<=", ">", ">=", "==", "!=".
        right: Right-hand side value.

    Returns:
        True if the comparison holds, False otherwise.
    """
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    return False


# ---------------------------------------------------------------------------
# CarryChainConstraint extractor
# ---------------------------------------------------------------------------


@dataclass
class CarryChainConstraint:
    """Extract carry-chain and borrow-chain arithmetic violations from text (v2).

    **Researcher summary (v2):**
        Extended in Exp 173 to also catch subtraction borrow chains
        (1000 - 1 = 999), digit-count violations in additions, and sign
        errors in B > A subtractions. Original carry detection retained.

    **Detailed explanation for engineers:**
        The base ArithmeticExtractor verifies "a + b = c" by simple
        arithmetic: correct = a + b, satisfied = (c == correct). But
        carry-chain errors have a distinct pattern in reasoning traces:
        LLMs often correctly handle the ones column and tens column but
        fail to propagate the final carry, e.g.:
            "99 + 1: units: 9+1=10, write 0 carry 1. Tens: 9+0+1=10,
             write 0 carry 1. Result: 100." -- correct
            "99 + 1: units: 9+1=0, tens: 9+0=9. Result: 90." -- wrong

        We identify carry-chain cases by checking whether _count_carries
        >= 2 (two or more cascading carries), then verifying the claimed
        result. Subtraction borrow chains are detected via _count_borrows.
        Negative-result errors (A - B where B > A claimed as positive) are
        flagged regardless of borrow count.

        This is a SEPARATE constraint type ("arithmetic_carry") so it
        coexists with "arithmetic" without duplicate deduplication.

    Spec: REQ-LEARN-004
    """

    #: Minimum number of cascading carries to classify as a carry-chain.
    min_carries: int = 2

    def extract(self, text: str) -> list[ConstraintResult]:
        """Extract carry-chain and borrow-chain arithmetic claims from text.

        **Detailed explanation for engineers (v2 additions):**
            In addition to detecting cascade carries in addition, this version
            also handles three new checks introduced in Exp 173:

            1. Subtraction borrow chains: "a - b = c" patterns where the
               subtraction requires >= min_carries consecutive borrows (e.g.,
               1000 - 1 needs 3 borrows). Detected via _count_borrows().

            2. Digit count check (addition only): the sum A + B can have at
               most max(digits(A), digits(B)) + 1 digits. A claimed result
               with far more digits indicates a carry-propagation error where
               the LLM copied intermediate carry bits into the result.

            3. Negative result check (subtraction only): if the minuend A is
               less than the subtrahend B, the result must be negative. If the
               LLM claims a non-negative result, that is a sign error -- flagged
               as satisfied=False regardless of borrow count.

        Args:
            text: Arbitrary text that may contain "a + b = c" or "a - b = c".

        Returns:
            List of ConstraintResult with constraint_type="arithmetic_carry"
            for qualified expressions. Empty list if none are found.
        """
        results: list[ConstraintResult] = []

        # ---- Addition: cascade carry detection + digit count check -----------
        add_pattern = r"(-?\d+)\s*\+\s*(\d+)\s*=\s*(-?\d+)"
        for m in re.finditer(add_pattern, text):
            a = int(m.group(1))
            b = int(m.group(2))
            claimed = int(m.group(3))

            carry_count = _count_carries(a, b)
            correct = a + b

            # Digit count check: sum can have at most max(digits(a), digits(b))+1.
            max_expected_digits = max(_digit_count(a), _digit_count(b)) + 1
            claimed_digits = _digit_count(claimed)
            digit_count_ok = claimed_digits <= max_expected_digits

            # Only flag if there are cascade carries OR a digit-count violation.
            # Single-carry / no-carry additions without digit issues are left
            # to ArithmeticExtractor.
            if carry_count < self.min_carries and digit_count_ok:
                continue

            satisfied = (claimed == correct) and digit_count_ok
            results.append(
                ConstraintResult(
                    constraint_type=PATTERN_ARITHMETIC_CARRY,
                    description=(
                        f"carry chain {carry_count}x: {a} + {b} = {claimed}"
                        + ("" if satisfied else f" (correct: {correct})")
                    ),
                    metadata={
                        "a": a,
                        "b": b,
                        "op": "+",
                        "claimed_result": claimed,
                        "correct_result": correct,
                        "carry_count": carry_count,
                        "digit_count_ok": digit_count_ok,
                        "claimed_digits": claimed_digits,
                        "max_expected_digits": max_expected_digits,
                        "satisfied": satisfied,
                    },
                )
            )

        # ---- Subtraction: borrow chain + negative result detection -----------
        # Pattern avoids matching "a + -b" by requiring literal minus between
        # two digit groups (not preceded by another operator).
        sub_pattern = r"(-?\d+)\s*-\s*(\d+)\s*=\s*(-?\d+)"
        for m in re.finditer(sub_pattern, text):
            a = int(m.group(1))
            b = int(m.group(2))
            claimed = int(m.group(3))
            correct = a - b

            borrow_count = _count_borrows(a, b)
            # Negative-result case: when B > A the true result is negative.
            # We ALWAYS flag these (to verify the sign) regardless of borrow count.
            is_negative_case = a < b
            # Negative-result violation: B > A and LLM claims non-negative (wrong sign).
            negative_violated = is_negative_case and (claimed >= 0) and (claimed != correct)

            # Only flag if there are cascade borrows OR this is a B > A subtraction.
            if borrow_count < self.min_carries and not is_negative_case:
                continue

            satisfied = claimed == correct
            if negative_violated:
                desc_suffix = f" (should be negative: {correct})"
            elif not satisfied:
                desc_suffix = f" (correct: {correct})"
            else:
                desc_suffix = ""
            results.append(
                ConstraintResult(
                    constraint_type=PATTERN_ARITHMETIC_CARRY,
                    description=(
                        f"borrow chain {borrow_count}x: {a} - {b} = {claimed}"
                        + desc_suffix
                    ),
                    metadata={
                        "a": a,
                        "b": b,
                        "op": "-",
                        "claimed_result": claimed,
                        "correct_result": correct,
                        "borrow_count": borrow_count,
                        "negative_violated": negative_violated,
                        "satisfied": satisfied,
                    },
                )
            )
        return results


# ---------------------------------------------------------------------------
# BoundConstraint extractor
# ---------------------------------------------------------------------------


class BoundConstraint:
    """Extract numeric comparison/boundary claims from text.

    **Researcher summary:**
        Catches inequality violations embedded in prose -- e.g., "the answer
        is 5 < 3" or "score >= 100" -- that the base extractors miss because
        they focus on "a + b = c" or "if/then" patterns.

    **Detailed explanation for engineers:**
        Scans for numeric comparisons using <, <=, >, >=, ==, != operators.
        Evaluates each claim directly (both operands must be numeric literals)
        and marks it satisfied or not. The constraint_type is
        "comparison_boundary" so it does not conflict with "bound" constraints
        from CodeExtractor (which uses "bound" for loop variables).

    Spec: REQ-LEARN-004
    """

    def extract(self, text: str) -> list[ConstraintResult]:
        """Extract and evaluate numeric comparison expressions from text.

        Args:
            text: Arbitrary text that may contain comparison expressions.

        Returns:
            List of ConstraintResult with constraint_type="comparison_boundary".
            Empty list if no numeric comparisons are found.
        """
        results: list[ConstraintResult] = []
        # Match patterns like "5 < 10", "3.14 >= 3.0", "x == 42" (numeric only).
        pattern = r"(-?\d+(?:\.\d+)?)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)"
        for m in re.finditer(pattern, text):
            left = float(m.group(1))
            op = m.group(2)
            right = float(m.group(3))
            satisfied = _eval_comparison(left, op, right)
            results.append(
                ConstraintResult(
                    constraint_type=PATTERN_COMPARISON_BOUNDARY,
                    description=f"comparison: {left} {op} {right}",
                    metadata={
                        "left": left,
                        "op": op,
                        "right": right,
                        "satisfied": satisfied,
                    },
                )
            )
        return results


# ---------------------------------------------------------------------------
# NegationConstraint extractor
# ---------------------------------------------------------------------------


class NegationConstraint:
    """Extract negation-scope claims and detect violations (v2).

    **Researcher summary (v2):**
        Extended in Exp 173 to ADD violation detection: satisfied=True when
        the negation is respected, satisfied=False when the text first denies
        "X is Y" but then asserts it anyway. Also adds "not all A are B" and
        "no A are/is/has B" universal-negation patterns.

    **Detailed explanation for engineers:**
        The LogicExtractor handles "cannot/can't/does not" patterns using
        full-sentence regex. NegationConstraint is complementary: it handles
        shorter inline negation phrases that appear mid-sentence.

        Extracts four negation sub-patterns (v2):
        1. "X is not Y"      → subject X, negated predicate Y.
           Violated if "X is Y" appears outside the negation span.
        2. "not all A are B" → negated universal. Violated if "all A are B"
           appears outside the span.
        3. "no A are/is/has B" → universal negation. Violated if "A are B"
           appears outside the span.
        4. "X ≠ Y" / "X != Y" → mathematical not-equal. Violated if
           "X = Y" or "X == Y" appears outside the span.

        All results carry ``satisfied`` in metadata (True/False) reflecting
        whether the negation was upheld in the surrounding text.

    Spec: REQ-LEARN-004
    """

    def extract(self, text: str) -> list[ConstraintResult]:
        """Extract negation-scope patterns from text and check for violations.

        **Detailed explanation for engineers (v2 additions):**
            The original implementation only DETECTED negation phrases; it did
            not check whether the surrounding text VIOLATED them. This v2
            version adds a ``satisfied`` field to every result:

            - satisfied=True  → the negation was respected (no positive
              assertion of the negated claim found elsewhere in the text).
            - satisfied=False → the negation was violated (the text first
              says "X is not Y" but later asserts "X is Y", or says
              "no A are B" but later asserts "A is B").

            Violation check method:
                For each extracted negation claim, we build the "positive form"
                of the claim (strip "not"/"no"), then search for it in the text
                OUTSIDE the negation phrase's span. We remove the negation span
                from the text before searching to avoid false-positive matches
                on the negation phrase itself (since "cats are black" is a
                substring of "no cats are black").

            New patterns added in v2:
                3. "not all A are B" → negated universal claim. Violated if
                   "all A are B" appears elsewhere.
                4. "no A are/is/has B" → universal negation. Violated if
                   "A are/is/has B" appears elsewhere.

        Args:
            text: Arbitrary text that may contain negation phrases.

        Returns:
            List of ConstraintResult with constraint_type="negation_scope".
            Each result has ``satisfied`` in metadata (True/False).
            Empty list if no negation patterns are found.
        """
        results: list[ConstraintResult] = []
        seen: set[str] = set()  # Deduplicate within a single text.

        # ------------------------------------------------------------------
        # Pattern 1: "X is not Y" (subject + negated predicate).
        # Violation: text outside this span asserts "X is Y" (positive form).
        # ------------------------------------------------------------------
        for m in re.finditer(
            r"\b([\w\s]{1,40}?)\s+is\s+not\s+([\w\s]{1,40}?)(?=[.,;!?]|$)",
            text,
            re.IGNORECASE,
        ):
            subject = m.group(1).strip()
            negated = m.group(2).strip()
            desc = f"negation: '{subject}' is not '{negated}'"
            if desc not in seen:
                seen.add(desc)
                # Remove the negation span from text before searching for
                # the positive form, to avoid matching against itself.
                text_outside = text[: m.start()] + text[m.end() :]
                positive_re = re.compile(
                    r"\b" + re.escape(subject) + r"\s+is\s+" + re.escape(negated) + r"\b",
                    re.IGNORECASE,
                )
                violated = bool(positive_re.search(text_outside))
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "is_not",
                            "subject": subject,
                            "negated_predicate": negated,
                            "raw": m.group(0).strip(),
                            "satisfied": not violated,
                        },
                    )
                )

        # ------------------------------------------------------------------
        # Pattern 2: "not all A are B" → negated universal claim.
        # Violation: text outside this span asserts "all A are B".
        # ------------------------------------------------------------------
        for m in re.finditer(
            r"\bnot\s+all\s+([\w]+(?:\s+[\w]+)?)\s+are\s+([\w]+(?:\s+[\w]+)?)\b",
            text,
            re.IGNORECASE,
        ):
            subject = m.group(1).strip()
            predicate = m.group(2).strip()
            desc = f"negation: not all {subject} are {predicate}"
            if desc not in seen:
                seen.add(desc)
                text_outside = text[: m.start()] + text[m.end() :]
                positive_re = re.compile(
                    r"\ball\s+" + re.escape(subject) + r"\s+are\s+" + re.escape(predicate) + r"\b",
                    re.IGNORECASE,
                )
                violated = bool(positive_re.search(text_outside))
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "not_all",
                            "subject": subject,
                            "negated_predicate": predicate,
                            "raw": m.group(0).strip(),
                            "satisfied": not violated,
                        },
                    )
                )

        # ------------------------------------------------------------------
        # Pattern 3: "no A are/is/has B" → universal negation.
        # Violation: text outside this span asserts "A are/is/has B".
        # ------------------------------------------------------------------
        for m in re.finditer(
            r"\bno\s+([\w]+(?:\s+[\w]+)?)\s+(are|is|has)\s+([\w]+(?:\s+[\w]+)?)\b",
            text,
            re.IGNORECASE,
        ):
            subject = m.group(1).strip()
            verb = m.group(2).strip().lower()
            predicate = m.group(3).strip()
            desc = f"negation: no {subject} {verb} {predicate}"
            if desc not in seen:
                seen.add(desc)
                text_outside = text[: m.start()] + text[m.end() :]
                positive_re = re.compile(
                    r"\b" + re.escape(subject) + r"\s+(?:are|is|has)\s+" + re.escape(predicate) + r"\b",
                    re.IGNORECASE,
                )
                violated = bool(positive_re.search(text_outside))
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "no_A_are_B",
                            "subject": subject,
                            "negated_predicate": predicate,
                            "raw": m.group(0).strip(),
                            "satisfied": not violated,
                        },
                    )
                )

        # ------------------------------------------------------------------
        # Pattern 4: "X ≠ Y" or "X != Y" (mathematical not-equal assertion).
        # Violation: text also asserts "X == Y" or "X = Y" (positive equality).
        # ------------------------------------------------------------------
        for m in re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*(?:≠|!=)\s*(-?\d+(?:\.\d+)?)",
            text,
        ):
            left = m.group(1).strip()
            right = m.group(2).strip()
            desc = f"negation: {left} ≠ {right}"
            if desc not in seen:
                seen.add(desc)
                text_outside = text[: m.start()] + text[m.end() :]
                positive_re = re.compile(
                    re.escape(left) + r"\s*(?:==|(?<!=)=(?!=))\s*" + re.escape(right)
                )
                violated = bool(positive_re.search(text_outside))
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "not_equal",
                            "left": left,
                            "right": right,
                            "raw": m.group(0).strip(),
                            "satisfied": not violated,
                        },
                    )
                )

        return results


# ---------------------------------------------------------------------------
# ConstraintGenerator
# ---------------------------------------------------------------------------

#: Mapping from memory error_type → extractor class.
#: When a pattern is mature (frequency >= PATTERN_THRESHOLD), the corresponding
#: extractor is applied to generate new constraints from the input text.
_PATTERN_EXTRACTOR_MAP: dict[str, type] = {
    PATTERN_ARITHMETIC_CARRY: CarryChainConstraint,
    PATTERN_COMPARISON_BOUNDARY: BoundConstraint,
    PATTERN_NEGATION_SCOPE: NegationConstraint,
}


class ConstraintGenerator:
    """Generate new ConstraintResults from learned memory patterns (Tier 2).

    **Researcher summary:**
        Reads ConstraintMemory to find mature patterns (seen 3+ times in
        a domain), then applies the corresponding specialized extractor to
        the input text. Returns new ConstraintResult objects that AutoExtractor
        merges with its static extraction output.

    **Detailed explanation for engineers:**
        This is the Tier 2 "constraint addition" fix for Exp 134. Whereas
        Tier 1 (ConstraintTracker + AdaptiveWeighter) adjusts the weights
        of existing constraints, Tier 2 ADDS new constraints discovered
        from accumulated pattern evidence.

        Lifecycle:
        1. Build from memory: ``ConstraintGenerator.from_memory(memory)``
        2. For each text being verified: ``generator.generate(text, domain)``
        3. AutoExtractor merges returned ConstraintResults with its own.

        Threshold gate:
        - Only generates constraints for error_types where
          ``memory._patterns[domain][error_type].frequency >= PATTERN_THRESHOLD``
          (default 3). This prevents over-generating from noisy data.
        - Additionally, only generates for error_types that have a known
          extractor in ``_PATTERN_EXTRACTOR_MAP``. Unknown error_types are
          silently skipped (forward-compatible: new error types added to
          memory don't break the generator).

        Pattern → extractor mapping:
        - "arithmetic_carry"    → CarryChainConstraint
        - "comparison_boundary" → BoundConstraint
        - "negation_scope"      → NegationConstraint

    Spec: REQ-LEARN-003, REQ-LEARN-004, SCENARIO-LEARN-003
    """

    def __init__(self, memory: ConstraintMemory) -> None:
        """Create a ConstraintGenerator backed by the given memory.

        Args:
            memory: Populated ConstraintMemory instance. Patterns are read
                lazily in generate() -- the memory can continue growing
                after construction.
        """
        self._memory = memory

    @classmethod
    def from_memory(cls, memory: ConstraintMemory) -> "ConstraintGenerator":
        """Construct a ConstraintGenerator from an existing ConstraintMemory.

        **Detailed explanation for engineers:**
            Factory method providing a named constructor. Equivalent to
            ``ConstraintGenerator(memory)`` but reads more clearly in
            pipeline code where the intent (building from learned memory)
            is more important than the constructor mechanics.

        Args:
            memory: Populated or empty ConstraintMemory instance.

        Returns:
            New ConstraintGenerator backed by this memory.
        """
        return cls(memory)

    def generate(self, text: str, domain: str) -> list[ConstraintResult]:
        """Generate new constraints for text based on mature memory patterns.

        **Detailed explanation for engineers:**
            1. Look up all (error_type, record) pairs for this domain.
            2. For each mature pattern (frequency >= PATTERN_THRESHOLD),
               check if there is a known extractor in _PATTERN_EXTRACTOR_MAP.
            3. If so, instantiate the extractor and call extract(text).
            4. Collect all returned ConstraintResult objects.
            5. Return the full list -- deduplication is handled by the caller
               (AutoExtractor).

            If the domain has no patterns in memory, or no mature patterns,
            or no patterns with known extractors, returns an empty list.

        Args:
            text: The response text to extract new constraints from.
            domain: The verification domain (e.g., "arithmetic", "code").
                Used to look up which patterns are mature for this domain.

        Returns:
            List of new ConstraintResult objects from memory-triggered
            extractors. May be empty if no mature patterns are known.
        """
        generated: list[ConstraintResult] = []

        # Access the internal pattern table directly for efficiency.
        # ConstraintMemory stores _patterns[domain][error_type] -> _PatternRecord.
        domain_patterns = self._memory._patterns.get(domain, {})

        for error_type, record in domain_patterns.items():
            # Threshold gate: only act on mature patterns.
            if record.frequency < PATTERN_THRESHOLD:
                continue

            # Only generate for error_types with a known extractor.
            extractor_cls = _PATTERN_EXTRACTOR_MAP.get(error_type)
            if extractor_cls is None:
                continue

            # Instantiate extractor and extract from text.
            extractor = extractor_cls()
            generated.extend(extractor.extract(text))

        return generated
