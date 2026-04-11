"""Tests for carnot.pipeline.generation -- ConstraintGenerator (Exp 141).

Covers Tier 2 constraint ADDITION from memory patterns:
- CarryChainConstraint extracts multi-carry arithmetic errors
- BoundConstraint extracts numeric comparison claims
- NegationConstraint extracts negation-scope patterns
- ConstraintGenerator threshold gating (< 3 patterns → no generation)
- ConstraintGenerator.from_memory() factory method
- AutoExtractor memory= parameter (additive, backward-compatible)
- Deduplication by constraint_type in AutoExtractor

Spec: REQ-LEARN-003, REQ-LEARN-004, SCENARIO-LEARN-003
"""

from __future__ import annotations

import pytest

from carnot.pipeline.extract import AutoExtractor
from carnot.pipeline.generation import (
    PATTERN_ARITHMETIC_CARRY,
    PATTERN_COMPARISON_BOUNDARY,
    PATTERN_NEGATION_SCOPE,
    BoundConstraint,
    CarryChainConstraint,
    ConstraintGenerator,
    NegationConstraint,
    _count_carries,
    _eval_comparison,
)
from carnot.pipeline.memory import PATTERN_THRESHOLD, ConstraintMemory


# ---------------------------------------------------------------------------
# Helper: build a ConstraintMemory with N records for a given domain/error_type
# ---------------------------------------------------------------------------


def _make_memory(domain: str, error_type: str, frequency: int) -> ConstraintMemory:
    """Build a ConstraintMemory with `frequency` records for domain/error_type."""
    mem = ConstraintMemory()
    for i in range(frequency):
        mem.record_pattern(domain, error_type, f"example_{i}")
    return mem


# ---------------------------------------------------------------------------
# _count_carries helper tests
# REQ-LEARN-004: carry detection
# ---------------------------------------------------------------------------


class TestCountCarries:
    """REQ-LEARN-004: _count_carries counts cascading carry operations."""

    def test_no_carry(self) -> None:
        """SCENARIO-LEARN-003: 1 + 1 = 2 has no carry."""
        assert _count_carries(1, 1) == 0

    def test_single_carry(self) -> None:
        """REQ-LEARN-004: 5 + 7 = 12 has exactly one carry."""
        assert _count_carries(5, 7) == 1

    def test_two_carries(self) -> None:
        """REQ-LEARN-004: 99 + 1 = 100 has two cascading carries."""
        assert _count_carries(99, 1) == 2

    def test_three_carries(self) -> None:
        """REQ-LEARN-004: 999 + 1 = 1000 has three cascading carries."""
        assert _count_carries(999, 1) == 3

    def test_zero_operands(self) -> None:
        """REQ-LEARN-004: 0 + 0 = 0 has no carry."""
        assert _count_carries(0, 0) == 0

    def test_negative_operands_use_abs(self) -> None:
        """REQ-LEARN-004: Negative operands treated as absolute values for carry count."""
        # abs(-99) + abs(1) = 99 + 1 → 2 carries
        assert _count_carries(-99, -1) == 2

    def test_large_numbers(self) -> None:
        """REQ-LEARN-004: Large numbers with multiple carries handled correctly."""
        # 9999 + 1 = 10000: 4 carries
        assert _count_carries(9999, 1) == 4


# ---------------------------------------------------------------------------
# _eval_comparison helper tests
# REQ-LEARN-004: comparison evaluation
# ---------------------------------------------------------------------------


class TestEvalComparison:
    """REQ-LEARN-004: _eval_comparison evaluates numeric inequality operators."""

    def test_less_than_true(self) -> None:
        assert _eval_comparison(3.0, "<", 5.0) is True

    def test_less_than_false(self) -> None:
        assert _eval_comparison(5.0, "<", 3.0) is False

    def test_less_equal_true(self) -> None:
        assert _eval_comparison(5.0, "<=", 5.0) is True

    def test_less_equal_false(self) -> None:
        assert _eval_comparison(6.0, "<=", 5.0) is False

    def test_greater_than_true(self) -> None:
        assert _eval_comparison(5.0, ">", 3.0) is True

    def test_greater_equal_true(self) -> None:
        assert _eval_comparison(5.0, ">=", 5.0) is True

    def test_equal_true(self) -> None:
        assert _eval_comparison(4.0, "==", 4.0) is True

    def test_equal_false(self) -> None:
        assert _eval_comparison(4.0, "==", 5.0) is False

    def test_not_equal_true(self) -> None:
        assert _eval_comparison(4.0, "!=", 5.0) is True

    def test_not_equal_false(self) -> None:
        assert _eval_comparison(4.0, "!=", 4.0) is False

    def test_unknown_op_returns_false(self) -> None:
        """REQ-LEARN-004: Unknown operator returns False (safe default)."""
        assert _eval_comparison(1.0, "~", 2.0) is False


# ---------------------------------------------------------------------------
# CarryChainConstraint tests
# REQ-LEARN-004: carry-chain extraction
# ---------------------------------------------------------------------------


class TestCarryChainConstraint:
    """REQ-LEARN-004: CarryChainConstraint extracts multi-carry arithmetic."""

    def setup_method(self) -> None:
        self.ext = CarryChainConstraint()

    def test_carry_chain_correct(self) -> None:
        """REQ-LEARN-004: Correct carry-chain sum is marked satisfied."""
        results = self.ext.extract("99 + 1 = 100")
        assert len(results) == 1
        r = results[0]
        assert r.constraint_type == PATTERN_ARITHMETIC_CARRY
        assert r.metadata["satisfied"] is True
        assert r.metadata["carry_count"] == 2
        assert r.metadata["correct_result"] == 100

    def test_carry_chain_wrong(self) -> None:
        """REQ-LEARN-004: Wrong carry-chain sum is marked unsatisfied."""
        results = self.ext.extract("99 + 1 = 90")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False
        assert r.metadata["correct_result"] == 100
        assert "correct: 100" in r.description

    def test_single_carry_skipped(self) -> None:
        """REQ-LEARN-004: Single-carry addition (5+7=12) is NOT extracted (< min_carries)."""
        results = self.ext.extract("5 + 7 = 12")
        assert results == []

    def test_no_carry_skipped(self) -> None:
        """REQ-LEARN-004: No-carry addition (2+3=5) is NOT extracted."""
        results = self.ext.extract("2 + 3 = 5")
        assert results == []

    def test_multiple_carry_chains(self) -> None:
        """REQ-LEARN-004: Multiple carry-chain expressions in one text."""
        text = "First: 99 + 1 = 100. Then: 999 + 1 = 1000."
        results = self.ext.extract(text)
        assert len(results) == 2
        assert all(r.constraint_type == PATTERN_ARITHMETIC_CARRY for r in results)

    def test_empty_text(self) -> None:
        """REQ-LEARN-004: Empty text returns empty list."""
        assert self.ext.extract("") == []

    def test_no_arithmetic(self) -> None:
        """REQ-LEARN-004: Text without arithmetic returns empty list."""
        assert self.ext.extract("The sky is blue.") == []

    def test_description_no_correction_when_correct(self) -> None:
        """REQ-LEARN-004: Description has no '(correct:)' suffix for correct sums."""
        results = self.ext.extract("999 + 1 = 1000")
        assert len(results) == 1
        assert "(correct:" not in results[0].description

    def test_metadata_fields(self) -> None:
        """REQ-LEARN-004: Metadata contains a, b, claimed_result, correct_result, carry_count."""
        results = self.ext.extract("99 + 1 = 99")
        assert len(results) == 1
        m = results[0].metadata
        assert m["a"] == 99
        assert m["b"] == 1
        assert m["claimed_result"] == 99
        assert m["correct_result"] == 100
        assert m["carry_count"] == 2


# ---------------------------------------------------------------------------
# BoundConstraint tests
# REQ-LEARN-004: comparison boundary extraction
# ---------------------------------------------------------------------------


class TestBoundConstraint:
    """REQ-LEARN-004: BoundConstraint extracts numeric comparison expressions."""

    def setup_method(self) -> None:
        self.ext = BoundConstraint()

    def test_less_than_satisfied(self) -> None:
        """REQ-LEARN-004: True inequality is marked satisfied."""
        results = self.ext.extract("3 < 5")
        assert len(results) == 1
        assert results[0].constraint_type == PATTERN_COMPARISON_BOUNDARY
        assert results[0].metadata["satisfied"] is True

    def test_less_than_violated(self) -> None:
        """REQ-LEARN-004: False inequality is marked unsatisfied."""
        results = self.ext.extract("5 < 3")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False

    def test_greater_equal(self) -> None:
        """REQ-LEARN-004: >= operator is handled."""
        results = self.ext.extract("10 >= 10")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_not_equal(self) -> None:
        """REQ-LEARN-004: != operator is handled."""
        results = self.ext.extract("4 != 5")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_equality_false(self) -> None:
        """REQ-LEARN-004: == operator when false is unsatisfied."""
        results = self.ext.extract("3 == 4")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False

    def test_floating_point(self) -> None:
        """REQ-LEARN-004: Decimal numbers are supported."""
        results = self.ext.extract("3.14 > 3.0")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_multiple_comparisons(self) -> None:
        """REQ-LEARN-004: Multiple comparisons in one text."""
        results = self.ext.extract("x: 3 < 5, y: 10 >= 10")
        assert len(results) == 2

    def test_no_comparison(self) -> None:
        """REQ-LEARN-004: Text without comparisons returns empty list."""
        assert self.ext.extract("The answer is 42.") == []

    def test_metadata_fields(self) -> None:
        """REQ-LEARN-004: Metadata contains left, op, right, satisfied."""
        results = self.ext.extract("7 > 3")
        assert len(results) == 1
        m = results[0].metadata
        assert m["left"] == 7.0
        assert m["op"] == ">"
        assert m["right"] == 3.0
        assert m["satisfied"] is True


# ---------------------------------------------------------------------------
# NegationConstraint tests
# REQ-LEARN-004: negation scope extraction
# ---------------------------------------------------------------------------


class TestNegationConstraint:
    """REQ-LEARN-004: NegationConstraint extracts negation scope patterns."""

    def setup_method(self) -> None:
        self.ext = NegationConstraint()

    def test_is_not_pattern(self) -> None:
        """REQ-LEARN-004: 'X is not Y' pattern is extracted."""
        results = self.ext.extract("The answer is not 42.")
        assert len(results) >= 1
        types = [r.constraint_type for r in results]
        assert PATTERN_NEGATION_SCOPE in types

    def test_not_equal_unicode(self) -> None:
        """REQ-LEARN-004: 'X ≠ Y' Unicode operator is extracted."""
        results = self.ext.extract("5 ≠ 3")
        assert len(results) == 1
        assert results[0].constraint_type == PATTERN_NEGATION_SCOPE
        assert results[0].metadata["pattern"] == "not_equal"

    def test_not_equal_ascii(self) -> None:
        """REQ-LEARN-004: 'X != Y' ASCII operator is extracted."""
        results = self.ext.extract("5 != 3")
        assert len(results) == 1
        assert results[0].metadata["pattern"] == "not_equal"

    def test_no_negation(self) -> None:
        """REQ-LEARN-004: Text without negation returns empty list."""
        assert self.ext.extract("The sky is blue.") == []

    def test_is_not_metadata(self) -> None:
        """REQ-LEARN-004: 'is_not' pattern metadata has subject and negated_predicate."""
        results = self.ext.extract("The result is not correct.")
        assert len(results) >= 1
        r = next(x for x in results if x.metadata.get("pattern") == "is_not")
        assert "subject" in r.metadata
        assert "negated_predicate" in r.metadata

    def test_deduplication_within_text(self) -> None:
        """REQ-LEARN-004: Same negation phrase appearing twice is extracted once."""
        text = "The answer is not 42. The answer is not 42."
        results = self.ext.extract(text)
        # Should have only one entry for "is not 42" pattern
        is_not_descs = [r.description for r in results if "42" in r.description]
        assert len(set(is_not_descs)) == len(is_not_descs)

    def test_empty_text(self) -> None:
        """REQ-LEARN-004: Empty text returns empty list."""
        assert self.ext.extract("") == []


# ---------------------------------------------------------------------------
# ConstraintGenerator tests
# REQ-LEARN-003, REQ-LEARN-004: threshold gating and generation
# ---------------------------------------------------------------------------


class TestConstraintGeneratorThreshold:
    """REQ-LEARN-003: ConstraintGenerator only fires for mature patterns (>=3)."""

    def test_below_threshold_no_generation(self) -> None:
        """REQ-LEARN-003: Patterns seen < PATTERN_THRESHOLD times produce no constraints."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD - 1)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("99 + 1 = 100", "arithmetic")
        assert results == []

    def test_at_threshold_generates(self) -> None:
        """REQ-LEARN-003: Pattern seen exactly PATTERN_THRESHOLD times triggers generation."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("99 + 1 = 100", "arithmetic")
        # Should generate at least one carry-chain constraint.
        assert len(results) >= 1
        assert any(r.constraint_type == PATTERN_ARITHMETIC_CARRY for r in results)

    def test_above_threshold_generates(self) -> None:
        """REQ-LEARN-003: Pattern seen more than threshold times still generates."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD + 5)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("99 + 1 = 90", "arithmetic")
        assert len(results) >= 1

    def test_unknown_error_type_skipped(self) -> None:
        """REQ-LEARN-003: Unknown error_type (not in mapping) is silently skipped."""
        mem = _make_memory("arithmetic", "unknown_future_pattern", PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("99 + 1 = 100", "arithmetic")
        assert results == []

    def test_wrong_domain_no_generation(self) -> None:
        """REQ-LEARN-003: Mature pattern for domain X does not trigger on domain Y."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        # Ask for "code" domain -- arithmetic patterns should not fire.
        results = gen.generate("99 + 1 = 100", "code")
        assert results == []

    def test_empty_memory_no_generation(self) -> None:
        """REQ-LEARN-003: Empty memory returns empty list."""
        mem = ConstraintMemory()
        gen = ConstraintGenerator.from_memory(mem)
        assert gen.generate("99 + 1 = 100", "arithmetic") == []


class TestConstraintGeneratorPatternMapping:
    """REQ-LEARN-004: ConstraintGenerator maps each error_type to correct extractor."""

    def test_arithmetic_carry_mapping(self) -> None:
        """REQ-LEARN-004: 'arithmetic_carry' error_type triggers CarryChainConstraint."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("999 + 1 = 1000", "arithmetic")
        types = [r.constraint_type for r in results]
        assert PATTERN_ARITHMETIC_CARRY in types

    def test_comparison_boundary_mapping(self) -> None:
        """REQ-LEARN-004: 'comparison_boundary' error_type triggers BoundConstraint."""
        mem = _make_memory("arithmetic", PATTERN_COMPARISON_BOUNDARY, PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("the score is 5 < 10", "arithmetic")
        types = [r.constraint_type for r in results]
        assert PATTERN_COMPARISON_BOUNDARY in types

    def test_negation_scope_mapping(self) -> None:
        """REQ-LEARN-004: 'negation_scope' error_type triggers NegationConstraint."""
        mem = _make_memory("arithmetic", PATTERN_NEGATION_SCOPE, PATTERN_THRESHOLD)
        gen = ConstraintGenerator.from_memory(mem)
        results = gen.generate("the answer is not 42.", "arithmetic")
        types = [r.constraint_type for r in results]
        assert PATTERN_NEGATION_SCOPE in types

    def test_multiple_mature_patterns_generate_multiple_types(self) -> None:
        """REQ-LEARN-004: Multiple mature patterns all trigger their extractors."""
        mem = ConstraintMemory()
        for _ in range(PATTERN_THRESHOLD):
            mem.record_pattern("arithmetic", PATTERN_ARITHMETIC_CARRY, "ex")
            mem.record_pattern("arithmetic", PATTERN_COMPARISON_BOUNDARY, "ex")
        gen = ConstraintGenerator.from_memory(mem)
        text = "99 + 1 = 90, and 5 < 3"
        results = gen.generate(text, "arithmetic")
        types = {r.constraint_type for r in results}
        assert PATTERN_ARITHMETIC_CARRY in types
        assert PATTERN_COMPARISON_BOUNDARY in types


class TestConstraintGeneratorFromMemory:
    """REQ-LEARN-003: ConstraintGenerator.from_memory() factory method."""

    def test_from_memory_returns_generator(self) -> None:
        """REQ-LEARN-003: from_memory() returns a ConstraintGenerator instance."""
        mem = ConstraintMemory()
        gen = ConstraintGenerator.from_memory(mem)
        assert isinstance(gen, ConstraintGenerator)

    def test_from_memory_and_constructor_equivalent(self) -> None:
        """REQ-LEARN-003: from_memory() and __init__ produce equivalent instances."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        gen1 = ConstraintGenerator(mem)
        gen2 = ConstraintGenerator.from_memory(mem)
        r1 = gen1.generate("99 + 1 = 100", "arithmetic")
        r2 = gen2.generate("99 + 1 = 100", "arithmetic")
        assert len(r1) == len(r2)
        assert [r.constraint_type for r in r1] == [r.constraint_type for r in r2]


# ---------------------------------------------------------------------------
# AutoExtractor memory= parameter tests
# REQ-LEARN-003, REQ-LEARN-004: backward compatibility and additive behavior
# ---------------------------------------------------------------------------


class TestAutoExtractorWithMemory:
    """REQ-LEARN-003: AutoExtractor memory= parameter is additive and backward-compatible."""

    def test_no_memory_unchanged(self) -> None:
        """REQ-LEARN-003: memory=None produces identical results to no-memory call."""
        ext = AutoExtractor()
        text = "47 + 28 = 75"
        results_no_mem = ext.extract(text, domain="arithmetic")
        results_mem_none = ext.extract(text, domain="arithmetic", memory=None)
        assert len(results_no_mem) == len(results_mem_none)
        descs_no_mem = {r.description for r in results_no_mem}
        descs_mem_none = {r.description for r in results_mem_none}
        assert descs_no_mem == descs_mem_none

    def test_memory_adds_carry_constraint(self) -> None:
        """REQ-LEARN-004: Mature carry-chain memory pattern adds new constraint type."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        ext = AutoExtractor()
        text = "99 + 1 = 100"
        results_no_mem = ext.extract(text, domain="arithmetic")
        results_with_mem = ext.extract(text, domain="arithmetic", memory=mem)
        # Should have more constraints with memory.
        assert len(results_with_mem) > len(results_no_mem)
        types_with = {r.constraint_type for r in results_with_mem}
        assert PATTERN_ARITHMETIC_CARRY in types_with

    def test_memory_deduplication_by_constraint_type(self) -> None:
        """REQ-LEARN-003: Memory does not add a duplicate constraint_type already present."""
        # The static extractors produce "arithmetic" type constraints.
        # If memory has an "arithmetic" pattern, we should NOT see two
        # "arithmetic" type constraints in the output -- the dedup logic
        # should prevent it.
        mem = _make_memory("arithmetic", "arithmetic", PATTERN_THRESHOLD)
        ext = AutoExtractor()
        text = "47 + 28 = 75"
        results = ext.extract(text, domain="arithmetic", memory=mem)
        arithmetic_types = [r for r in results if r.constraint_type == "arithmetic"]
        # Only one "arithmetic" type constraint (not duplicated from memory).
        assert len(arithmetic_types) == 1

    def test_memory_without_domain_no_generation(self) -> None:
        """REQ-LEARN-003: When domain=None, memory generation is skipped (requires domain)."""
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        ext = AutoExtractor()
        text = "99 + 1 = 100"
        results_no_domain = ext.extract(text, memory=mem)
        results_with_domain = ext.extract(text, domain="arithmetic", memory=mem)
        # With domain specified, we get more (memory-generated) constraints.
        # Without domain, same as static extraction only.
        types_no_domain = {r.constraint_type for r in results_no_domain}
        assert PATTERN_ARITHMETIC_CARRY not in types_no_domain
        types_with_domain = {r.constraint_type for r in results_with_domain}
        assert PATTERN_ARITHMETIC_CARRY in types_with_domain

    def test_memory_below_threshold_no_extra_constraints(self) -> None:
        """REQ-LEARN-003: Immature memory patterns do NOT add constraints."""
        mem = _make_memory(
            "arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD - 1
        )
        ext = AutoExtractor()
        text = "99 + 1 = 100"
        results_no_mem = ext.extract(text, domain="arithmetic")
        results_with_mem = ext.extract(text, domain="arithmetic", memory=mem)
        # Should produce identical outputs (immature pattern not yet triggered).
        assert len(results_no_mem) == len(results_with_mem)

    def test_memory_keeps_multiple_generated_constraints_of_same_type(self) -> None:
        """REQ-LEARN-004: Multiple violations of a NEW type are all kept (not deduplicated away).

        Regression test for adversarial review finding: the original dedup logic
        added the generated type to existing_types on first add, so subsequent
        violations of the same new type (e.g., two carry-chain errors in one
        response) were silently dropped. The fix: only block types from STATIC
        extraction, not from generation.
        """
        mem = _make_memory("arithmetic", PATTERN_ARITHMETIC_CARRY, PATTERN_THRESHOLD)
        ext = AutoExtractor()
        # Two carry-chain violations in the same text.
        text = "First: 99 + 1 = 90. Second: 999 + 1 = 990."
        results = ext.extract(text, domain="arithmetic", memory=mem)
        carry_results = [r for r in results if r.constraint_type == PATTERN_ARITHMETIC_CARRY]
        # Should have two separate carry-chain constraint results (one per expression).
        assert len(carry_results) == 2

    def test_memory_multiple_domains_only_target_fires(self) -> None:
        """REQ-LEARN-003: Memory patterns for other domains don't fire on target domain."""
        mem = ConstraintMemory()
        for _ in range(PATTERN_THRESHOLD):
            mem.record_pattern("code", PATTERN_ARITHMETIC_CARRY, "code carry ex")
        ext = AutoExtractor()
        text = "99 + 1 = 100"
        results = ext.extract(text, domain="arithmetic", memory=mem)
        # code-domain patterns should not fire for arithmetic domain
        types = {r.constraint_type for r in results}
        assert PATTERN_ARITHMETIC_CARRY not in types
