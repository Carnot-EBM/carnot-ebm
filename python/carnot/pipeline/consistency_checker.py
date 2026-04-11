"""Global consistency checker for multi-step ConstraintStateMachine sequences (Exp 172).

**Researcher summary:**
    ConstraintStateMachine (Exp 125) checks each step locally — a step passes if
    its constraints are satisfied in isolation. But step 3's verified output can
    contradict step 1's verified facts even if each step passed locally.

    This module implements the pairwise consistency framework from arxiv 2601.13600:
    "Foundations of Global Consistency Checking with Noisy LLM Oracles." That paper
    proves that checking all (i, j) output pairs for pairwise consistency — using
    a possibly-noisy verifier — converges to correct global consistency judgments
    even with a 30% oracle error rate, via majority vote.

    Here we apply that insight without an oracle: we use deterministic text-level
    extraction to find three contradiction types across step pairs:

        1. Numeric: same entity name, different numeric value
           Example: "item costs $50" (step 1) vs "item costs $75" (step 3)

        2. Arithmetic: same arithmetic equation operands, different claimed result
           Example: "3 + 5 = 8" (step 1) vs "3 + 5 = 10" (step 3)

        3. Factual: same (subject, predicate) triple, different object value
           Example: "Paris is the capital of France" (step 1) vs
                    "Berlin is the capital of France" (step 3)

**Detailed explanation for engineers:**
    The GlobalConsistencyChecker.check() method:
    1. Retrieves the full step history from the ConstraintStateMachine.
    2. For every pair (i, j) with i < j, extracts and compares claims
       from step_i.output_text and step_j.output_text.
    3. Each extraction uses deterministic regex (no LLM oracle needed):
       - _extract_numeric_claims() finds (entity → value) mappings.
       - _extract_arithmetic_claims() finds ((a, op, b) → claimed) mappings.
       - _extract_factual_triples() uses factual_extractor.extract_claims()
         for (subject, predicate) → object mappings.
    4. A contradiction is recorded whenever the same key appears in both steps
       with different values.
    5. Returns a GlobalConsistencyReport with: consistent flag, list of
       (i, j, contradiction_type, description) tuples, severity level, and
       recommended_rollback_step (the earliest step i from any contradicting pair,
       meaning the last known-good state before the inconsistency was introduced).

    Severity levels:
        "none"     — no contradictions found
        "warning"  — exactly 1 non-factual contradiction pair
        "critical" — 2+ contradiction pairs, or any factual contradiction

    Integration:
        ConstraintStateMachine.check_global_consistency() calls
        GlobalConsistencyChecker().check(self) internally. This method is
        available after verify_step() (i.e., step()) has been called at least twice.

Target models: Qwen3.5-0.8B, google/gemma-4-E4B-it (as per Exp 172)
Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from carnot.pipeline.state_machine import ConstraintStateMachine


# ---------------------------------------------------------------------------
# GlobalConsistencyReport — result of a global consistency check
# ---------------------------------------------------------------------------


@dataclass
class GlobalConsistencyReport:
    """Report from a global consistency check across all step pairs.

    **Detailed explanation for engineers:**
        Summarises the result of checking every (i, j) step pair for
        cross-step contradictions. Each inconsistency is recorded as a tuple
        (i, j, contradiction_type, description) in inconsistent_pairs. The
        severity and recommended_rollback_step provide actionable guidance
        for agent frameworks: if severity is "critical", the framework should
        roll back to recommended_rollback_step and re-generate from there.

    Attributes:
        consistent: True iff no cross-step contradictions were detected.
        inconsistent_pairs: List of (i, j, contradiction_type, description) where:
            - i, j: Zero-based step indices (i < j)
            - contradiction_type: "numeric", "arithmetic", or "factual"
            - description: Human-readable explanation of the contradiction
        severity: "none" if consistent; "warning" if 1 non-factual pair;
            "critical" if 2+ pairs or any factual contradiction.
        recommended_rollback_step: The step index of the earliest step i
            involved in any contradiction. Roll back to this step to restore
            the last known-good state. None when consistent.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    consistent: bool
    inconsistent_pairs: list[tuple[int, int, str, str]]
    severity: str  # "none" | "warning" | "critical"
    recommended_rollback_step: int | None


# ---------------------------------------------------------------------------
# Module-level extraction patterns
# ---------------------------------------------------------------------------

#: Matches "entity [verb] [$]N" patterns.
#: Group "entity": the thing being described (1–3 words, starts with a letter).
#: Group "value": the numeric value (integer or decimal).
#: Examples: "widget costs $50", "price is 75", "answer was 100"
_NUMERIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?P<entity>[A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){0,2})"
        r"\s+(?:costs?|is|was|equals?)\s+\$?(?P<value>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
]

#: Matches arithmetic equations like "3 + 5 = 8" or "-3 - 2 = -5".
#: Groups: (a, op, b, claimed_result)
_ARITH_PATTERN: re.Pattern[str] = re.compile(
    r"(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)"
)

#: Stop words stripped from entity names during normalisation.
#: Articles and common sentence starters that appear before noun phrases
#: should not be part of the entity key used for cross-step comparison.
_ENTITY_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "this", "that", "its", "our", "their",
        "is", "was", "are", "were", "has", "have",
    }
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _normalize_entity(text: str) -> str:
    """Lowercase and strip leading articles / stop words from an entity name.

    **Detailed explanation for engineers:**
        Converts "The Widget" and "widget" to the same normalised key "widget",
        so that cross-step comparisons are robust to minor surface-form
        differences caused by how the LLM phrases the same entity.

    Args:
        text: Raw entity name captured by a regex group.

    Returns:
        Normalised entity string (lowercase, leading stop words removed).
        Returns empty string if nothing remains after stripping.

    Spec: REQ-VERIFY-001
    """
    tokens = text.lower().split()
    # Strip leading stop words only (trailing ones are usually part of the name).
    while tokens and tokens[0] in _ENTITY_STOP_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _extract_numeric_claims(text: str) -> dict[str, float]:
    """Extract (normalised_entity → numeric_value) pairs from text.

    **Detailed explanation for engineers:**
        Applies each pattern in _NUMERIC_PATTERNS to find "entity verb value"
        phrases. The entity is normalised via _normalize_entity() to remove
        articles and stop words, enabling cross-step matching of the same
        entity even when phrased differently.

        If the same entity appears multiple times in one step's text with
        different values, the last occurrence wins. This is acceptable for the
        cross-step consistency check: we want to compare what a step ultimately
        asserts about an entity.

    Args:
        text: Output text from one reasoning step.

    Returns:
        Dictionary mapping normalised entity name → float value.
        Empty dict if no matching patterns found.

    Spec: REQ-VERIFY-001
    """
    result: dict[str, float] = {}
    for pattern in _NUMERIC_PATTERNS:
        for m in pattern.finditer(text):
            entity = _normalize_entity(m.group("entity"))
            if entity:
                result[entity] = float(m.group("value"))
    return result


def _extract_arithmetic_claims(text: str) -> dict[tuple[int, str, int], int]:
    """Extract {(a, op, b) → claimed_result} mappings from arithmetic equations.

    **Detailed explanation for engineers:**
        Scans text for patterns like "3 + 5 = 8" and returns a dict keyed by
        the equation operands. Addition is normalised to (min, "+", max) so
        that "3 + 5 = 8" and "5 + 3 = 8" map to the same key, making the
        cross-step comparison robust to operand reordering.

        Subtraction is NOT commutative, so (3, "-", 5) and (5, "-", 3) are
        kept as separate keys.

    Args:
        text: Output text from one reasoning step.

    Returns:
        Dict mapping (a, op, b) tuple → claimed integer result.
        Empty dict if no arithmetic equations found.

    Spec: REQ-VERIFY-001
    """
    result: dict[tuple[int, str, int], int] = {}
    for m in _ARITH_PATTERN.finditer(text):
        a = int(m.group(1))
        op = m.group(2)
        b = int(m.group(3))
        claimed = int(m.group(4))
        # Normalise commutative addition: (min, "+", max)
        if op == "+":
            key: tuple[int, str, int] = (min(a, b), op, max(a, b))
        else:
            key = (a, op, b)
        result[key] = claimed
    return result


def _extract_factual_triples(text: str) -> dict[tuple[str, str], str]:
    """Extract {(subject_lower, predicate_key) → object_lower} factual triples.

    **Detailed explanation for engineers:**
        Delegates to factual_extractor.extract_claims() which uses the pattern
        library from Exp 158 to extract (subject, predicate, object) triples
        from natural language. Normalises all three parts to lowercase for
        cross-step comparison.

        If the same (subject, predicate) pair appears multiple times in one
        step's text with different objects, the last occurrence wins.

    Args:
        text: Output text from one reasoning step.

    Returns:
        Dict mapping (subject_lower, predicate_key_lower) → object_lower.
        Empty dict if no claim patterns match.

    Spec: REQ-VERIFY-001
    """
    # Lazy import to avoid circular dependency at module load time.
    from carnot.pipeline.factual_extractor import extract_claims

    triples = extract_claims(text)
    result: dict[tuple[str, str], str] = {}
    for subject, predicate_key, obj in triples:
        key = (subject.lower().strip(), predicate_key.lower().strip())
        result[key] = obj.lower().strip()
    return result


# ---------------------------------------------------------------------------
# GlobalConsistencyChecker
# ---------------------------------------------------------------------------


class GlobalConsistencyChecker:
    """Check cross-step contradictions that local per-step verification misses.

    **Researcher summary:**
        Implements the pairwise consistency framework from arxiv 2601.13600.
        Checks all (i, j) step pairs (i < j) in a ConstraintStateMachine's
        history for three types of contradiction: numeric value changes for
        the same entity, conflicting arithmetic results, and contradictory
        factual triples. Returns a GlobalConsistencyReport.

    **Detailed explanation for engineers:**
        The three contradiction types map to three extraction methods:

        Numeric contradiction:
            Both step i and step j mention the same entity with a numeric
            value (e.g., price, count, age). The values differ → contradiction.
            Detection: _extract_numeric_claims() on both outputs; compare
            by normalised entity key.

        Arithmetic contradiction:
            Both steps contain an arithmetic equation with the same operands
            (a op b) but different claimed results. Detection:
            _extract_arithmetic_claims() on both; compare by (a, op, b) key.

        Factual contradiction:
            Both steps assert different values for the same (subject, predicate)
            pair (e.g., "capital of France" is "Paris" in step 1, "Berlin" in
            step 3). Detection: _extract_factual_triples() on both; compare
            by (subject, predicate) key.

        Severity rules:
            - "critical": 2+ contradiction pairs, OR any factual contradiction
            - "warning":  exactly 1 non-factual contradiction pair
            - "none":     no contradictions

        recommended_rollback_step:
            The smallest step index i from all contradicting (i, j) pairs.
            This is the last step where the machine was globally consistent —
            rolling back to here discards all steps that introduced the
            contradiction.

    Usage::

        machine = ConstraintStateMachine()
        machine.step("Q1", "The price is $50.")
        machine.step("Q2", "Confirming the price.")
        machine.step("Q3", "Actually the price is $75.")  # contradiction!

        checker = GlobalConsistencyChecker()
        report = checker.check(machine)
        assert not report.consistent
        assert report.recommended_rollback_step == 0

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def check(self, state_machine: "ConstraintStateMachine") -> GlobalConsistencyReport:
        """Check all completed step pairs for global consistency.

        **Detailed explanation for engineers:**
            Iterates over all (i, j) pairs where 0 <= i < j < len(history).
            For each pair, calls _check_pair() which runs all three
            contradiction detection methods and collects any found.
            After all pairs are checked, computes severity and
            recommended_rollback_step from the collected contradictions.

        Args:
            state_machine: A ConstraintStateMachine with 0 or more completed
                steps. Each step must have been run via step() so that
                output_text is stored in the StepResult.

        Returns:
            GlobalConsistencyReport with consistent flag, inconsistent_pairs
            list, severity, and recommended_rollback_step. With fewer than
            2 steps, always returns consistent=True (no pairs to compare).

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        history = state_machine.history()

        # With 0 or 1 steps there are no pairs to compare — trivially consistent.
        if len(history) < 2:
            return GlobalConsistencyReport(
                consistent=True,
                inconsistent_pairs=[],
                severity="none",
                recommended_rollback_step=None,
            )

        # Check all (i, j) pairs where i < j.
        inconsistent_pairs: list[tuple[int, int, str, str]] = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                text_i = history[i].output_text
                text_j = history[j].output_text
                pairs = self._check_pair(i, j, text_i, text_j)
                inconsistent_pairs.extend(pairs)

        # No contradictions found → globally consistent.
        if not inconsistent_pairs:
            return GlobalConsistencyReport(
                consistent=True,
                inconsistent_pairs=[],
                severity="none",
                recommended_rollback_step=None,
            )

        # Severity: "critical" if any factual contradiction or 2+ pairs.
        # "warning" for exactly 1 non-factual pair.
        has_factual = any(ctype == "factual" for _, _, ctype, _ in inconsistent_pairs)
        severity = (
            "critical"
            if (len(inconsistent_pairs) >= 2 or has_factual)
            else "warning"
        )

        # Recommend rolling back to the earliest step i involved in any pair.
        # Step i is the last state known to be globally consistent before
        # the contradiction was introduced by step j.
        recommended_step = min(i for i, _, _, _ in inconsistent_pairs)

        return GlobalConsistencyReport(
            consistent=False,
            inconsistent_pairs=inconsistent_pairs,
            severity=severity,
            recommended_rollback_step=recommended_step,
        )

    def _check_pair(
        self,
        i: int,
        j: int,
        text_i: str,
        text_j: str,
    ) -> list[tuple[int, int, str, str]]:
        """Check one (step_i, step_j) pair for all three contradiction types.

        Args:
            i: Index of the earlier step.
            j: Index of the later step.
            text_i: Output text from step i.
            text_j: Output text from step j.

        Returns:
            List of (i, j, contradiction_type, description) tuples.
            Empty list if no contradictions found between this pair.

        Spec: REQ-VERIFY-001
        """
        results: list[tuple[int, int, str, str]] = []
        results.extend(self._check_numeric(i, j, text_i, text_j))
        results.extend(self._check_arithmetic(i, j, text_i, text_j))
        results.extend(self._check_factual(i, j, text_i, text_j))
        return results

    def _check_numeric(
        self,
        i: int,
        j: int,
        text_i: str,
        text_j: str,
    ) -> list[tuple[int, int, str, str]]:
        """Detect numeric contradictions: same entity, different values.

        **Detailed explanation for engineers:**
            Extracts (entity → value) from both texts. For each entity that
            appears in both, compares the float values with a 1e-9 tolerance
            to handle floating-point representations. A contradiction is
            reported only when the same entity has a meaningfully different
            value across the two steps.

        Args:
            i: Earlier step index.
            j: Later step index.
            text_i: Output text from step i.
            text_j: Output text from step j.

        Returns:
            List of (i, j, "numeric", description) tuples. Empty if no
            numeric contradictions found between this pair.

        Spec: REQ-VERIFY-001
        """
        claims_i = _extract_numeric_claims(text_i)
        claims_j = _extract_numeric_claims(text_j)
        contradictions: list[tuple[int, int, str, str]] = []
        for entity, val_i in claims_i.items():
            if entity in claims_j:
                val_j = claims_j[entity]
                if abs(val_i - val_j) > 1e-9:
                    desc = (
                        f"Entity '{entity}' has value {val_i} in step {i} "
                        f"but {val_j} in step {j}"
                    )
                    contradictions.append((i, j, "numeric", desc))
        return contradictions

    def _check_arithmetic(
        self,
        i: int,
        j: int,
        text_i: str,
        text_j: str,
    ) -> list[tuple[int, int, str, str]]:
        """Detect arithmetic contradictions: same equation, different claimed result.

        **Detailed explanation for engineers:**
            Extracts ((a, op, b) → claimed) from both texts. For each
            (a, op, b) key that appears in both texts, checks whether the
            claimed results agree. Differing claimed results mean the chain
            has internally inconsistent arithmetic — one step says the sum
            is X, another says it is Y.

        Args:
            i: Earlier step index.
            j: Later step index.
            text_i: Output text from step i.
            text_j: Output text from step j.

        Returns:
            List of (i, j, "arithmetic", description) tuples. Empty if no
            arithmetic contradictions found.

        Spec: REQ-VERIFY-001
        """
        eqs_i = _extract_arithmetic_claims(text_i)
        eqs_j = _extract_arithmetic_claims(text_j)
        contradictions: list[tuple[int, int, str, str]] = []
        for key, claimed_i in eqs_i.items():
            if key in eqs_j:
                claimed_j = eqs_j[key]
                if claimed_i != claimed_j:
                    a, op, b = key
                    desc = (
                        f"Arithmetic {a} {op} {b} = {claimed_i} (step {i}) "
                        f"contradicts = {claimed_j} (step {j})"
                    )
                    contradictions.append((i, j, "arithmetic", desc))
        return contradictions

    def _check_factual(
        self,
        i: int,
        j: int,
        text_i: str,
        text_j: str,
    ) -> list[tuple[int, int, str, str]]:
        """Detect factual contradictions: same (subject, predicate), different object.

        **Detailed explanation for engineers:**
            Uses extract_claims() from factual_extractor (Exp 158) to parse
            structured (subject, predicate, object) triples from natural
            language. Compares all (subject, predicate) keys present in both
            texts. When the same (subject, predicate) pair has different object
            values, a factual contradiction is recorded.

            Only predicates in factual_extractor._PREDICATE_TO_PROPERTY are
            extracted, so the check is limited to verifiable factual domains
            (capitals, birth places, currencies, etc.). Unknown predicates are
            silently skipped (same behaviour as FactualExtractor.extract()).

        Args:
            i: Earlier step index.
            j: Later step index.
            text_i: Output text from step i.
            text_j: Output text from step j.

        Returns:
            List of (i, j, "factual", description) tuples. Empty if no
            factual contradictions found.

        Spec: REQ-VERIFY-001
        """
        triples_i = _extract_factual_triples(text_i)
        triples_j = _extract_factual_triples(text_j)
        contradictions: list[tuple[int, int, str, str]] = []
        for sp_key, obj_i in triples_i.items():
            if sp_key in triples_j:
                obj_j = triples_j[sp_key]
                if obj_i != obj_j:
                    subject, predicate = sp_key
                    desc = (
                        f"Factual: '{subject}' {predicate} is '{obj_i}' (step {i}) "
                        f"but '{obj_j}' (step {j})"
                    )
                    contradictions.append((i, j, "factual", desc))
        return contradictions


__all__ = [
    "GlobalConsistencyChecker",
    "GlobalConsistencyReport",
]
