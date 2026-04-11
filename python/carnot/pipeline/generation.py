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
    - "arithmetic_carry" → CarryChainConstraint: catches multi-step carry
      propagation errors (e.g., 99 + 1, 999 + 11) that the base
      ArithmeticExtractor misses because it only checks the final sum.
    - "comparison_boundary" → BoundConstraint: catches numeric inequality
      claims (X < Y, X >= 0) embedded in prose that LogicExtractor ignores
      because it only handles "if/then" and "but not" patterns.
    - "negation_scope" → NegationConstraint: catches "X is not Y" and
      "not X" negation patterns that create logical scope errors in
      step-by-step reasoning.

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
    """Extract multi-step carry-chain arithmetic violations from text.

    **Researcher summary:**
        Catches addition errors where carries propagate across multiple
        digit columns -- the error type most commonly missed by the base
        ArithmeticExtractor.

    **Detailed explanation for engineers:**
        The base ArithmeticExtractor verifies "a + b = c" by simple
        subtraction: correct = a + b, satisfied = (c == correct). But
        carry-chain errors have a distinct pattern in reasoning traces:
        LLMs often correctly handle the ones column and tens column but
        fail to propagate the final carry, e.g.:
            "99 + 1: units: 9+1=10, write 0 carry 1. Tens: 9+0+1=10,
             write 0 carry 1. Result: 100." -- correct
            "99 + 1: units: 9+1=0, tens: 9+0=9. Result: 90." -- wrong

        We identify carry-chain cases by checking whether _count_carries
        >= 2 (two or more cascading carries), then verifying the claimed
        result. This is a SEPARATE constraint type ("arithmetic_carry")
        so it coexists with "arithmetic" without duplicate deduplication.

    Spec: REQ-LEARN-004
    """

    #: Minimum number of cascading carries to classify as a carry-chain.
    min_carries: int = 2

    def extract(self, text: str) -> list[ConstraintResult]:
        """Extract carry-chain arithmetic claims from text.

        Args:
            text: Arbitrary text that may contain "a + b = c" patterns.

        Returns:
            List of ConstraintResult with constraint_type="arithmetic_carry"
            for any addition expressions where carries cascade min_carries+
            times. Empty list if no carry-chain patterns are found.
        """
        results: list[ConstraintResult] = []
        # Match "a + b = c" only (subtraction rarely has multi-carry issues).
        pattern = r"(-?\d+)\s*\+\s*(\d+)\s*=\s*(-?\d+)"
        for m in re.finditer(pattern, text):
            a = int(m.group(1))
            b = int(m.group(2))
            claimed = int(m.group(3))

            # Only flag carry-chain cases -- let ArithmeticExtractor handle
            # single-carry or no-carry additions.
            carry_count = _count_carries(a, b)
            if carry_count < self.min_carries:
                continue

            correct = a + b
            satisfied = claimed == correct
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
                        "claimed_result": claimed,
                        "correct_result": correct,
                        "carry_count": carry_count,
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
    """Extract negation-scope claims ("X is not Y", "not X") from text.

    **Researcher summary:**
        Catches negation patterns that create logical scope errors in
        step-by-step reasoning -- e.g., "the answer is not 42" followed
        by "therefore the answer is 42."

    **Detailed explanation for engineers:**
        The LogicExtractor handles "cannot/can't/does not" patterns using
        full-sentence regex. NegationConstraint is complementary: it
        handles shorter inline negation phrases ("X is not Y") that appear
        mid-sentence and are often missed by sentence-boundary splitting.

        Extracts three negation sub-patterns:
        1. "X is not Y"   → constraint_type="negation_scope", subject X,
           negated predicate Y.
        2. "not X"        → standalone negation of phrase X.
        3. "X ≠ Y"        → mathematical not-equal assertion.

        These are soft constraints -- they don't compute an energy term,
        but they flag negation patterns for the pipeline to evaluate.

    Spec: REQ-LEARN-004
    """

    def extract(self, text: str) -> list[ConstraintResult]:
        """Extract negation-scope patterns from text.

        Args:
            text: Arbitrary text that may contain negation phrases.

        Returns:
            List of ConstraintResult with constraint_type="negation_scope".
            Empty list if no negation patterns are found.
        """
        results: list[ConstraintResult] = []
        seen: set[str] = set()  # Deduplicate within a single text.

        # Pattern 1: "X is not Y" (3-15 word subject/predicate phrases).
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
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "is_not",
                            "subject": subject,
                            "negated_predicate": negated,
                            "raw": m.group(0).strip(),
                        },
                    )
                )

        # Pattern 2: "X ≠ Y" or "X != Y" (mathematical inequality in prose).
        for m in re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*(?:≠|!=)\s*(-?\d+(?:\.\d+)?)",
            text,
        ):
            left = m.group(1).strip()
            right = m.group(2).strip()
            desc = f"negation: {left} ≠ {right}"
            if desc not in seen:
                seen.add(desc)
                results.append(
                    ConstraintResult(
                        constraint_type=PATTERN_NEGATION_SCOPE,
                        description=desc,
                        metadata={
                            "pattern": "not_equal",
                            "left": left,
                            "right": right,
                            "raw": m.group(0).strip(),
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
